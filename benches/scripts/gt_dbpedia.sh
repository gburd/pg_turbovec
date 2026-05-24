#!/usr/bin/env bash
set -uo pipefail
export PGHOST=/scratch/pg_turbovec-bench PGPORT=28815 PGUSER=gburd LD_LIBRARY_PATH=/lib64
export PATH=/home/gburd/.pgrx/17.9/pgrx-install/bin:$PATH
export PSQLRC=/dev/null
LOG=/scratch/pg_turbovec-bench/dbpedia_sweep/gt.log
T=$(date +%s)
echo "[$(date -u +%H:%M:%S)] gt_top10 brute-force start" | tee -a "$LOG"
psql -d bench_dbpedia -X -q -P pager=off -v ON_ERROR_STOP=1 <<SQL >>"$LOG" 2>&1
TRUNCATE gt_top10;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET max_parallel_workers_per_gather = 12;
SET max_parallel_workers = 16;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET min_parallel_table_scan_size = 0;
SET work_mem = '256MB';
INSERT INTO gt_top10
SELECT q.qid, k.id AS hit_id, k.rk
FROM query_set q
CROSS JOIN LATERAL (
  SELECT d.id, row_number() OVER (ORDER BY d.emb <=> q.emb) AS rk
  FROM docs d ORDER BY d.emb <=> q.emb LIMIT 10
) k;
CREATE INDEX IF NOT EXISTS gt_top10_qid ON gt_top10(qid);
SELECT count(*) FROM gt_top10;
SQL
rc=$?
echo "[$(date -u +%H:%M:%S)] gt_top10 done in $(($(date +%s)-T))s rc=$rc" | tee -a "$LOG"
[ $rc -eq 0 ] && echo "gt_top10_build_s=$(($(date +%s)-T))" >> /scratch/pg_turbovec-bench/dbpedia_sweep/build_times.txt
