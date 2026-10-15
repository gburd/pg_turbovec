#!/usr/bin/env bash
# P5 closed-loop QPS via pgbench. Both systems at their 0.95 headline point + turbovec scan_parallelism=1.
# clients {1,16,96}; 20s warm (discarded) + 120s timed; -M prepared; \gset PK-lookup (identical both systems).
# pgbench pinned to 4 whole cores' HTs (x86) / 8 cores (arm); server gets the rest.
# usage: p5_qps.sh <tag> <kind flat|ivf> <tv_knob_name> <tv_knob_val> <pv_nprobe>
set -uxo pipefail
export PATH=/opt/pg18/bin:$PATH; R=~/results/p5; mkdir -p $R
TAG=$1; KIND=$2; TVK=$3; TVV=$4; PVN=$5
if [ "$KIND" = flat ]; then TVTBL=dtv_flat_$TAG; TVEXTRA=";turbovec.hi_dim_rerank=off"; else TVTBL=dtv_ivf_$TAG; TVEXTRA=""; fi
PVTBL=dpv_prism_$TAG; NQMAX=$(( $(psql -h /tmp -X -d db_pv -Atc "SELECT count(*) FROM queries_$TAG") - 1 ))
ARCH=$(uname -m)
if [ "$ARCH" = x86_64 ]; then CLIENT_CPUS="44,92,45,93,46,94,47,95"; SERVER_CPUS="0-43,48-91"; else CLIENT_CPUS="88-95"; SERVER_CPUS="0-87"; fi
cat > /tmp/q_$TAG.tmpl <<SQL
\set qid random(0, $NQMAX)
SELECT q::text AS qv FROM queries_$TAG WHERE qid = :qid \gset
SELECT id FROM TBL ORDER BY emb <=> :qv::VTYPE LIMIT 10;
SQL
run_sys() { # sysname db table vtype sets
  local sysn=$1 db=$2 tbl=$3 vt=$4 sets=$5
  sed "s/TBL/$tbl/; s/VTYPE/$vt/" /tmp/q_$TAG.tmpl > /tmp/q_${TAG}_${sysn}.sql
  pg_ctl -D /data/pgdata -m fast -w stop; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  env taskset -c $SERVER_CPUS pg_ctl -D /data/pgdata -l ~/pg.log -w start
  psql -h /tmp -X -d $db -Atc "SELECT pg_prewarm('$tbl'),pg_prewarm('queries_$TAG')" >/dev/null
  for kv in $(echo "$sets" | tr ';' ' '); do [ -n "$kv" ] && psql -h /tmp -X -d $db -c "ALTER DATABASE $db SET ${kv/=/ = }"; done
  for c in 1 16 96; do
    local j; j=$([ $c -lt 16 ] && echo $c || echo 16)
    taskset -c $CLIENT_CPUS pgbench -h /tmp -d $db -n -M prepared -f /tmp/q_${TAG}_${sysn}.sql -c $c -j $j -T 20 >/dev/null 2>&1 || true
    mpstat 1 125 > $R/${TAG}_${sysn}_c${c}.mpstat 2>/dev/null & local MP=$!
    taskset -c $CLIENT_CPUS pgbench -h /tmp -d $db -n -M prepared -f /tmp/q_${TAG}_${sysn}.sql \
      -c $c -j $j -T 120 -r -l --log-prefix=$R/${TAG}_${sysn}_c${c} > $R/${TAG}_${sysn}_c${c}.pgbench 2>&1 || true
    kill $MP 2>/dev/null || true
    grep -E "tps|latency average|failed" $R/${TAG}_${sysn}_c${c}.pgbench | sed "s/^/${sysn} c=$c: /"
  done
  for kv in $(echo "$sets" | tr ';' ' '); do [ -n "$kv" ] && psql -h /tmp -X -d $db -c "ALTER DATABASE $db RESET ${kv%%=*}"; done
}
run_sys tv  db_tv $TVTBL turbovec.vector "$TVK=$TVV$TVEXTRA"
run_sys tv1 db_tv $TVTBL turbovec.vector "$TVK=$TVV$TVEXTRA;turbovec.scan_parallelism=1"
run_sys pv  db_pv $PVTBL vector "prism.nprobe=$PVN"
echo "P5_QPS_DONE_$TAG"
