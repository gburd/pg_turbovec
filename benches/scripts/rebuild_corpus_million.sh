#!/usr/bin/env bash
# Rebuild the million-row corpus from scratch.
# Parent agent's loader had an uncorrelated subquery that emitted the
# same vector for every row, so docs.emb was a degenerate constant.
#
# This rebuild:
#   1. drops the existing turbovec + hnsw indexes
#   2. TRUNCATEs docs and re-inserts 1M proper random vectors
#   3. l2-normalises
#   4. refreshes query_set.emb from the new docs
#   5. recomputes gt_top10 (50 brute-force scans)
#   6. rebuilds HNSW + tv_4bit + tv_2bit indexes
#
# Wall time on arnold (1M × 384, single backend): ~10 min total.

set -euo pipefail
export LD_LIBRARY_PATH=/lib64
PG=$HOME/.pgrx/17.9/pgrx-install/bin
DB="-h /scratch/pg_turbovec-bench -p 28815 -d bench -X -P pager=off"

ts() { date -u +'%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "=== STEP 1: drop existing indexes (keeps docs_pkey) ==="
$PG/psql $DB -c '
DROP INDEX IF EXISTS docs_pgv_hnsw;
DROP INDEX IF EXISTS docs_tv_4bit;
DROP INDEX IF EXISTS docs_tv_2bit;
'

log "=== STEP 2: rebuild docs (1M rows × 384 dims, random) ==="
$PG/psql $DB -c '
TRUNCATE docs;
SET maintenance_work_mem = "1GB";
'
log "    inserting 1M rows..."
$PG/psql $DB <<'SQL'
\timing on
-- Volatile SQL function: PG cannot constant-fold the random vector
-- across the 1M outer rows the way it can with an uncorrelated
-- subquery, so each id gets a freshly drawn 384-d vector.
CREATE OR REPLACE FUNCTION rand_vec(d int) RETURNS vector
LANGUAGE sql VOLATILE AS $f$
  SELECT (array_agg(((random() - 0.5) * 2.0)::float4))::vector
  FROM generate_series(1, d) AS k;
$f$;
SELECT setseed(0.42);
INSERT INTO docs (id, emb)
SELECT gs, rand_vec(384)
FROM generate_series(1, 1000000) gs;
SELECT count(*) FROM docs;
SELECT count(distinct emb) AS distinct_doc_embs FROM docs;
SQL

log "=== STEP 3: l2-normalise ==="
$PG/psql $DB <<'SQL'
\timing on
UPDATE docs SET emb = l2_normalize(emb);
VACUUM ANALYZE docs;
SQL

log "=== STEP 4: refresh query_set.emb from docs ==="
$PG/psql $DB <<'SQL'
\timing on
UPDATE query_set qs SET emb = d.emb FROM docs d WHERE d.id = qs.doc_id;
SELECT count(distinct emb) AS distinct_query_embs FROM query_set;
SQL

log "=== STEP 5: recompute gt_top10 (brute-force, 50 queries × 1M rows) ==="
$PG/psql $DB <<'SQL'
\timing on
TRUNCATE gt_top10;
SET enable_indexscan      = off;
SET enable_indexonlyscan  = off;
SET enable_bitmapscan     = off;
SET max_parallel_workers_per_gather = 4;
INSERT INTO gt_top10 (qid, hit_id, rk)
SELECT q.qid, t.id, t.rk
FROM query_set q,
LATERAL (
    SELECT id, row_number() OVER (ORDER BY emb <=> q.emb) AS rk
    FROM docs
    ORDER BY emb <=> q.emb
    LIMIT 10
) t;
SELECT qid, array_agg(hit_id ORDER BY rk) AS top10
  FROM gt_top10 GROUP BY qid ORDER BY qid LIMIT 5;
SQL

log "=== STEP 6: rebuild HNSW (m=16, efc=64) ==="
$PG/psql $DB <<'SQL'
\timing on
CREATE INDEX docs_pgv_hnsw ON docs
    USING hnsw (emb vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
SELECT pg_size_pretty(pg_relation_size('docs_pgv_hnsw')) AS hnsw_size;
SQL

log "=== STEP 7: rebuild turbovec 4-bit ==="
$PG/psql $DB <<'SQL'
\timing on
CREATE INDEX docs_tv_4bit ON docs
    USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 4);
SELECT bit_width, dim, n_vectors,
       pg_size_pretty(octet_length(payload)::bigint) AS payload
  FROM turbovec.am_storage WHERE indexrelid = 'docs_tv_4bit'::regclass;
SQL

log "=== STEP 8: rebuild turbovec 2-bit ==="
$PG/psql $DB <<'SQL'
\timing on
CREATE INDEX docs_tv_2bit ON docs
    USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 2);
SELECT bit_width, dim, n_vectors,
       pg_size_pretty(octet_length(payload)::bigint) AS payload
  FROM turbovec.am_storage WHERE indexrelid = 'docs_tv_2bit'::regclass;
SQL

log "=== DONE ==="
