#!/bin/bash
# Post-fix regression benchmark: build time, warm kNN latency, sustained
# insert throughput. Run AFTER the corruption run completes (needs the box
# idle). Compares the FIXED binary against recorded expectations; the A/B
# vs the base binary is optional (the fix only adds locking around existing
# reads/writes + a reconcile splice, so any regression shows up here).
set -u
PGBIN=/data/.pgrx/18.4/pgrx-install/bin
PORT=28818; DB=postgres
PSQL="$PGBIN/psql -p $PORT -d $DB -v ON_ERROR_STOP=0"
OUT=${OUT:-/data/repro/bench-$(date +%s)}
mkdir -p "$OUT"
DIM=768
NBUILD=${NBUILD:-1000000}

echo "[bench] === build a fresh $NBUILD x ${DIM}d corpus ===" | tee "$OUT/bench.log"
$PSQL -q -c "DROP TABLE IF EXISTS bench_corpus CASCADE;"
$PSQL -q -c "CREATE TABLE bench_corpus (id bigint PRIMARY KEY, emb turbovec.vector);"
i=0; BATCH=20000
t0=$(date +%s)
while [ $i -lt $NBUILD ]; do
  end=$(( i + BATCH )); [ $end -gt $NBUILD ] && end=$NBUILD
  $PSQL -q -c "INSERT INTO bench_corpus (id, emb) SELECT g, turbovec.array_to_vec((SELECT array_agg((sin(g*0.001+s*0.01)+random()*0.01)::real) FROM generate_series(1,$DIM) AS s)) FROM generate_series($i,$end-1) AS g;"
  i=$end
done
t1=$(date +%s)
echo "[bench] load $NBUILD rows: $((t1-t0))s" | tee -a "$OUT/bench.log"

echo "[bench] === index build time ===" | tee -a "$OUT/bench.log"
tb0=$(date +%s%3N)
$PSQL -q -c "CREATE INDEX bench_idx ON bench_corpus USING turbovec (emb turbovec.vec_l2_ops);"
tb1=$(date +%s%3N)
echo "[bench] build ${NBUILD} rows: $(( (tb1-tb0) ))ms" | tee -a "$OUT/bench.log"
$PSQL -c "SELECT n_vectors,is_corrupt FROM turbovec.turbovec_check('bench_idx'::regclass);" | tee -a "$OUT/bench.log"

echo "[bench] === warm kNN latency (100 queries, probes=16, k=10) ===" | tee -a "$OUT/bench.log"
cat > "$OUT/knn.sql" <<SQL
SET search_path = turbovec, public;
SET turbovec.probes = 16;
SELECT id FROM bench_corpus
ORDER BY emb <-> turbovec.array_to_vec((SELECT array_agg((sin(x*0.5)+random())::real) FROM generate_series(1,$DIM) AS x))
LIMIT 10;
SQL
# warm the cache
$PSQL -q -f "$OUT/knn.sql" >/dev/null
# time 100 runs
tk0=$(date +%s%3N)
for q in $(seq 1 100); do $PSQL -q -f "$OUT/knn.sql" >/dev/null; done
tk1=$(date +%s%3N)
echo "[bench] 100 warm kNN queries: $(( tk1-tk0 ))ms total, $(( (tk1-tk0)/100 ))ms/query (incl psql startup)" | tee -a "$OUT/bench.log"
# tighter: single-connection pgbench-style loop
$PSQL -q -c "\timing on" >/dev/null 2>&1
echo "[bench] server-side timing (single connection, 20 queries):" | tee -a "$OUT/bench.log"
{ echo "SET search_path=turbovec,public; SET turbovec.probes=16; \timing on";
  for q in $(seq 1 20); do
    echo "SELECT id FROM bench_corpus ORDER BY emb <-> turbovec.array_to_vec((SELECT array_agg((sin(x*0.5+$q)+random())::real) FROM generate_series(1,$DIM) AS x)) LIMIT 10;";
  done; } | $PSQL 2>&1 | grep -E "^Time:" | tee -a "$OUT/bench.log"

echo "[bench] === sustained insert throughput (single writer, 30s) ===" | tee -a "$OUT/bench.log"
$PSQL -q -c "DROP SEQUENCE IF EXISTS bench_seq; CREATE SEQUENCE bench_seq START $((NBUILD+1));"
ti0=$(date +%s); N=0
while [ $(( $(date +%s) - ti0 )) -lt 30 ]; do
  $PSQL -q -c "INSERT INTO bench_corpus (id, emb) SELECT nextval('bench_seq'), turbovec.array_to_vec((SELECT array_agg((sin(gg*0.9+s*0.03)+random())::real) FROM generate_series(1,$DIM) AS s)) FROM generate_series(1,128) AS gg;"
  N=$(( N + 128 ))
done
ti1=$(date +%s)
echo "[bench] inserted $N rows in $((ti1-ti0))s = $(( N / (ti1-ti0) )) rows/s (batch=128, single writer)" | tee -a "$OUT/bench.log"
$PSQL -c "SELECT n_vectors,is_corrupt FROM turbovec.turbovec_check('bench_idx'::regclass);" | tee -a "$OUT/bench.log"
echo "[bench] DONE -> $OUT/bench.log"
