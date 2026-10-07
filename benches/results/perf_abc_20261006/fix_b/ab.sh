#!/bin/bash
# A/B the distance-fn per-call cost: base vs fix-b .so, alternated, own cluster.
# usage: ab.sh <rounds>
set -euo pipefail
B=/work/pg16rel_b/bin; D=/work/fixb/data; S=/work/fixb; P=55437
LIB=$($B/pg_config --pkglibdir)
ROUNDS=${1:-5}
export PGOPTIONS="-c search_path=turbovec,public"
PSQL="$B/psql -X -h $S -p $P -d bench -qAt"
trap "$B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true" EXIT
start() { cp /work/fixb/$1.so $LIB/pg_turbovec.so; taskset -c 6,7 $B/pg_ctl -D $D -l $S/pg.log -w start >/dev/null; }
stop()  { $B/pg_ctl -D $D -m fast -w stop >/dev/null; }
# run <sql>: 6 EXPLAIN ANALYZE executions in one session, drop the first, print sorted ms
run() {
  local args=(-c "PREPARE x AS $1")
  for i in 1 2 3 4 5 6; do args+=(-c "EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON) EXECUTE x"); done
  $PSQL "${args[@]}" | grep "Execution Time" | awk "{print \$3}" | tail -5 | sort -n | tr "\n" " "
}
QLIT=$($PSQL -c "SELECT tv::text FROM q" 2>/dev/null || true)
for r in $(seq 1 $ROUNDS); do
  for v in base new; do
    start $v
    [ -n "$QLIT" ] || QLIT=$($PSQL -c "SELECT tv::text FROM q")
    $PSQL -c "SELECT count(*) FROM t" -c "SELECT count(*) FROM tp" >/dev/null
    for tab in t tp; do
      for op in "<=>" "<->" "<#>" "<+>"; do
        echo "{\"round\":$r,\"v\":\"$v\",\"tab\":\"$tab\",\"op\":\"$op\",\"ms\":\"$(run "SELECT sum(tv $op '$QLIT'::vector) FROM $tab")\"}"
      done
      echo "{\"round\":$r,\"v\":\"$v\",\"tab\":\"$tab\",\"op\":\"vector_dims\",\"ms\":\"$(run "SELECT sum(vector_dims(tv)) FROM $tab")\"}"
      echo "{\"round\":$r,\"v\":\"$v\",\"tab\":\"$tab\",\"op\":\"pg_column_size\",\"ms\":\"$(run "SELECT sum(pg_column_size(tv)) FROM $tab")\"}"
    done
    stop
  done
done
