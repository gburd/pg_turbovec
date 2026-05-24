#!/usr/bin/env bash
# Phase J: dbpedia-entities-openai-1M head-to-head sweep.
# Builds query_set + gt_top10, runs warm-cache sweep across 6 configs,
# emits results.tsv. JSON post-processed by emit_dbpedia_json.py.
set -euo pipefail

export PGHOST=/scratch/pg_turbovec-bench
export PGPORT=28815
export PGUSER=gburd
export PGDATABASE=bench_dbpedia
export LD_LIBRARY_PATH=/lib64
export PATH=/home/gburd/.pgrx/17.9/pgrx-install/bin:$HOME/.local/bin:$PATH
export PSQLRC=/dev/null

OUT=/scratch/pg_turbovec-bench/dbpedia_sweep
mkdir -p "$OUT"
LOG="$OUT/sweep.log"

ts(){ date -u +'%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
psql_q="psql -X -q -P pager=off -v ON_ERROR_STOP=1"

phase=${1:-all}

build_indexes() {
    log "build pgvector HNSW (m=16, ef_construction=64)"
    T=$(date +%s)
    $psql_q <<'SQL' >>"$LOG" 2>&1
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 16;
DROP INDEX IF EXISTS docs_pgv_hnsw;
CREATE INDEX docs_pgv_hnsw ON docs USING hnsw (emb vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
SQL
    echo "pgv_hnsw_build_s=$(($(date +%s)-T))" >> "$OUT/build_times.txt"
    log "  HNSW done"

    log "build pg_turbovec 4-bit"
    T=$(date +%s)
    $psql_q <<'SQL' >>"$LOG" 2>&1
SET search_path = turbovec, public;
SET maintenance_work_mem = '8GB';
DROP INDEX IF EXISTS docs_tv_4bit;
CREATE INDEX docs_tv_4bit ON docs
  USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
  WITH (bit_width = 4);
SQL
    echo "tv_4bit_build_s=$(($(date +%s)-T))" >> "$OUT/build_times.txt"
    log "  4-bit done"

    log "build pg_turbovec 2-bit"
    T=$(date +%s)
    $psql_q <<'SQL' >>"$LOG" 2>&1
SET search_path = turbovec, public;
SET maintenance_work_mem = '8GB';
DROP INDEX IF EXISTS docs_tv_2bit;
CREATE INDEX docs_tv_2bit ON docs
  USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
  WITH (bit_width = 2);
SQL
    echo "tv_2bit_build_s=$(($(date +%s)-T))" >> "$OUT/build_times.txt"
    log "  2-bit done"
}

setup_helpers() {
    log "install bench helpers + query_set"
    $psql_q -f /scratch/pg_turbovec-bench/scripts/bench_setup.sql >>"$LOG" 2>&1
    $psql_q <<'SQL' >>"$LOG" 2>&1
DROP TABLE IF EXISTS query_set;
CREATE TABLE query_set AS
SELECT row_number() OVER ()::int AS qid, id AS doc_id, ext_id, emb
FROM docs WHERE id <= 50;
CREATE INDEX ON query_set(qid);
SQL
}

ground_truth() {
    log "build brute-force gt_top10 (50 queries x 1M, parallel seqscan)"
    T=$(date +%s)
    $psql_q <<'SQL' >>"$LOG" 2>&1
DROP TABLE IF EXISTS gt_top10;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET max_parallel_workers_per_gather = 12;
SET max_parallel_workers = 16;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET min_parallel_table_scan_size = 0;
SET work_mem = '256MB';
CREATE TABLE gt_top10 AS
SELECT q.qid, k.id AS hit_id, k.rk
FROM query_set q
CROSS JOIN LATERAL (
  SELECT d.id, row_number() OVER (ORDER BY d.emb <=> q.emb) AS rk
  FROM docs d ORDER BY d.emb <=> q.emb LIMIT 10
) k;
CREATE INDEX ON gt_top10(qid);
SQL
    echo "gt_top10_build_s=$(($(date +%s)-T))" >> "$OUT/build_times.txt"
    log "  gt_top10 in $(($(date +%s)-T))s"
    $psql_q -c "SELECT count(*) FROM gt_top10;" >>"$LOG" 2>&1
}

storage() {
    log "storage report"
    $psql_q -tAF $'\t' >"$OUT/storage.tsv" <<'SQL'
SELECT 'docs_pgv_hnsw', pg_relation_size('docs_pgv_hnsw');
SELECT c.relname, octet_length(s.payload)::bigint
FROM turbovec.am_storage s JOIN pg_class c ON c.oid = s.indexrelid
ORDER BY c.relname;
SELECT 'docs_heap', pg_relation_size('docs');
SQL
    cat "$OUT/storage.tsv" >> "$LOG"
}

warm_phase() {
    local label="$1" engine="$2" guc="$3"
    local sp
    if [ "$engine" = "pgv" ]; then
        # pgvector path: keep search_path = public so plpgsql DECLARE
        # `vector` resolves to public.vector and `<=>` to pgvector's.
        sp="SET search_path=public;"
    else
        sp="SET search_path=turbovec,public;"
    fi
    log "warm sweep: $label"
    $psql_q <<SQL >>"$LOG" 2>&1
$sp
$guc
SELECT bench_reset('$label');
SELECT * FROM bench_one_query_${engine}(1);  -- discard warmup
SELECT * FROM bench_one_query_${engine}(1);  -- discard warmup
DO \$\$
DECLARE q int;
BEGIN
    FOR q IN SELECT qid FROM query_set ORDER BY qid LIMIT 50 LOOP
        PERFORM bench_record('$label','$engine',q);
    END LOOP;
END\$\$;
SELECT * FROM bench_summary('$label');
SQL
}

run_sweep() {
    # HNSW phases: hide turbovec indexes (rename out of way) so planner picks HNSW.
    log "RENAME tv indexes off for hnsw sweeps"
    $psql_q -c "ALTER INDEX docs_tv_4bit RENAME TO docs_tv_4bit_off;
                ALTER INDEX docs_tv_2bit RENAME TO docs_tv_2bit_off;" >>"$LOG" 2>&1
    warm_phase hnsw_ef40  pgv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET hnsw.ef_search=40;"
    warm_phase hnsw_ef200 pgv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET hnsw.ef_search=200;"

    log "RENAME hnsw off, restore 4-bit, keep 2-bit off"
    $psql_q -c "ALTER INDEX docs_pgv_hnsw RENAME TO docs_pgv_hnsw_off;
                ALTER INDEX docs_tv_4bit_off RENAME TO docs_tv_4bit;" >>"$LOG" 2>&1
    warm_phase tv_4bit_k100 tv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET turbovec.search_k=100;"
    warm_phase tv_4bit_k500 tv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET turbovec.search_k=500;"

    log "swap to 2-bit"
    $psql_q -c "ALTER INDEX docs_tv_4bit RENAME TO docs_tv_4bit_off;
                ALTER INDEX docs_tv_2bit_off RENAME TO docs_tv_2bit;" >>"$LOG" 2>&1
    warm_phase tv_2bit_k100 tv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET turbovec.search_k=100;"
    warm_phase tv_2bit_k500 tv "SET enable_seqscan=off; SET enable_indexonlyscan=off; SET turbovec.search_k=500;"

    log "restore all index names"
    $psql_q -c "ALTER INDEX docs_pgv_hnsw_off RENAME TO docs_pgv_hnsw;
                ALTER INDEX docs_tv_4bit_off RENAME TO docs_tv_4bit;" >>"$LOG" 2>&1

    log "dump bench_runs summary"
    $psql_q -tAF $'\t' >"$OUT/results.tsv" <<'SQL'
WITH s AS (
    SELECT * FROM bench_summary('hnsw_ef40')     UNION ALL
    SELECT * FROM bench_summary('hnsw_ef200')    UNION ALL
    SELECT * FROM bench_summary('tv_4bit_k100')  UNION ALL
    SELECT * FROM bench_summary('tv_4bit_k500')  UNION ALL
    SELECT * FROM bench_summary('tv_2bit_k100')  UNION ALL
    SELECT * FROM bench_summary('tv_2bit_k500')
)
SELECT label, n, min_ms, p50_ms, p95_ms, max_ms, mean_ms, r_at_10 FROM s;
SQL
    cat "$OUT/results.tsv" | tee -a "$LOG"
}

case "$phase" in
    build)   build_indexes; storage ;;
    helpers) setup_helpers ;;
    gt)      ground_truth ;;
    sweep)   run_sweep ;;
    storage) storage ;;
    all)     build_indexes; storage; setup_helpers; ground_truth; run_sweep ;;
    *) echo "usage: $0 {build|helpers|gt|sweep|storage|all}"; exit 1 ;;
esac
log "phase=$phase done"
