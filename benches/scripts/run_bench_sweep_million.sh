#!/usr/bin/env bash
# Drives the full latency + recall sweep for the million-row corpus.
# Each phase runs in its own psql session and writes results into
# the bench_runs table, then prints a human summary line.
#
# IMPORTANT: turbovec re-loads its 195MB / 103MB payload from
# am_storage on every scan (cache not wired into the index AM
# scan path in v1.0). That means every turbovec query is dominated
# by deserialize, not the kernel — see docs/RECALL.md.
#
# Phases:
#   A. HNSW ef_search=40           (50q, ~5s total)
#   B. HNSW ef_search=200          (50q, ~10s total)
#   C. turbovec 2-bit, search_k=100 (with both indexes present, planner picks 2-bit)
#   D. turbovec 2-bit, search_k=200
#   E. turbovec 4-bit, search_k=100 (drop 2-bit so 4-bit is the only candidate)
#   F. turbovec 4-bit, search_k=200
#
# At the end the 2-bit index is rebuilt (kept for the next run).

set -euo pipefail
export LD_LIBRARY_PATH=/lib64
PG=$HOME/.pgrx/17.9/pgrx-install/bin
DB="-h /scratch/pg_turbovec-bench -p 28815 -d bench -X -q -P pager=off"

ts() { date -u +'%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Ensure helper functions exist.
$PG/psql $DB -f /scratch/pg_turbovec-bench/scripts/bench_setup.sql > /dev/null 2>&1

run_phase() {
    local label="$1"
    local engine="$2"        # pgv or tv
    local guc_setup="$3"
    local note="$4"
    log "phase $label ($note)"
    $PG/psql $DB <<SQL
LOAD 'pg_turbovec';
$guc_setup
-- Warmup: prime PG buffer cache for am_storage / hnsw pages.
SELECT * FROM bench_one_query_${engine}(1);
SELECT * FROM bench_one_query_${engine}(1);
-- Timed sweep.
SELECT bench_run_config('$label', '$engine');
SELECT * FROM bench_summary('$label');
SQL
}

log "=== A: HNSW ef_search=40 ==="
run_phase 'hnsw_ef40' pgv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET hnsw.ef_search=40;" \
  "pgvector hnsw, ef_search=40"

log "=== B: HNSW ef_search=200 ==="
run_phase 'hnsw_ef200' pgv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET hnsw.ef_search=200;" \
  "pgvector hnsw, ef_search=200"

log "=== C: turbovec 2-bit, search_k=100 ==="
run_phase 'tv2_k100' tv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET search_path=turbovec,public; SET turbovec.search_k=100;" \
  "tv 2-bit (planner picks smaller index)"

log "=== D: turbovec 2-bit, search_k=200 ==="
run_phase 'tv2_k200' tv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET search_path=turbovec,public; SET turbovec.search_k=200;" \
  "tv 2-bit"

log "=== E: drop 2-bit so planner uses 4-bit ==="
$PG/psql $DB -c 'DROP INDEX docs_tv_2bit;'

log "=== E: turbovec 4-bit, search_k=100 ==="
run_phase 'tv4_k100' tv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET search_path=turbovec,public; SET turbovec.search_k=100;" \
  "tv 4-bit"

log "=== F: turbovec 4-bit, search_k=200 ==="
run_phase 'tv4_k200' tv \
  "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET search_path=turbovec,public; SET turbovec.search_k=200;" \
  "tv 4-bit"

log "=== rebuild 2-bit index ==="
$PG/psql $DB <<'SQL'
\timing on
CREATE INDEX docs_tv_2bit ON docs
    USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 2);
SQL

log "=== final summary ==="
$PG/psql $DB <<'SQL'
SELECT label, n, round(min_ms::numeric,1) AS min,
       round(p50_ms::numeric,1)  AS p50,
       round(p95_ms::numeric,1)  AS p95,
       round(max_ms::numeric,1)  AS max,
       round(mean_ms::numeric,1) AS mean,
       r_at_10
  FROM (
    SELECT * FROM bench_summary('hnsw_ef40')   UNION ALL
    SELECT * FROM bench_summary('hnsw_ef200')  UNION ALL
    SELECT * FROM bench_summary('tv2_k100')    UNION ALL
    SELECT * FROM bench_summary('tv2_k200')    UNION ALL
    SELECT * FROM bench_summary('tv4_k100')    UNION ALL
    SELECT * FROM bench_summary('tv4_k200')
  ) s;
SQL
log "=== DONE ==="
