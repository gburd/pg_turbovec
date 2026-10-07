#!/bin/bash
# A/B the per-candidate ORDER BY recheck: flat 4-bit index, search_k=1024, LIMIT 10.
set -euo pipefail
B=/work/pg16rel_b/bin; D=/work/fixb/data; S=/work/fixb; P=55437
LIB=$($B/pg_config --pkglibdir); ROUNDS=${1:-5}
export PGOPTIONS="-c search_path=turbovec,public"
PSQL="$B/psql -X -h $S -p $P -d bench -qAt"
trap "$B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true" EXIT
start() { cp /work/fixb/$1.so $LIB/pg_turbovec.so; taskset -c 6,7 $B/pg_ctl -D $D -l $S/pg.log -w start >/dev/null; }
stop()  { $B/pg_ctl -D $D -m fast -w stop >/dev/null; }
start new
for tab in t tp; do $PSQL -c "CREATE INDEX IF NOT EXISTS ${tab}_idx ON $tab USING turbovec (tv vec_cosine_ops)"; done
QS=$($PSQL -c "SELECT string_agg(id::text, \$\$ \$\$) FROM (SELECT id FROM t ORDER BY id LIMIT 20 OFFSET 100) s")
stop
for r in $(seq 1 $ROUNDS); do
  for v in base new; do
    start $v
    for tab in t tp; do
      args=(-c "SET turbovec.search_k = 1024" -c "SET enable_seqscan = off")
      # warm: each query once, then timed: 20 distinct queries x 3, take per-query min
      for id in $QS; do args+=(-c "SELECT id FROM $tab ORDER BY tv <=> (SELECT tv FROM t WHERE id = $id) LIMIT 10"); done
      for rep in 1 2 3; do for id in $QS; do
        args+=(-c "EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON) SELECT id FROM $tab ORDER BY tv <=> (SELECT tv FROM t WHERE id = $id) LIMIT 10")
      done; done
      ms=$($PSQL "${args[@]}" | grep "Execution Time" | awk "{print \$3}" | tr "\n" " ")
      echo "{\"round\":$r,\"v\":\"$v\",\"tab\":\"$tab\",\"op\":\"knn1024\",\"ms\":\"$ms\"}"
    done
    stop
  done
done
