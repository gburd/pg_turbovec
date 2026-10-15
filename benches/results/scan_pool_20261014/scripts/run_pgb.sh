#!/usr/bin/env bash
# Multi-client sanity: pgbench closed loop, flat docs (all GUC defaults = search_k 32, hi_dim_rerank auto),
# 16 and 96 clients, -T 75: first 15 s = warm-up (every new backend cold-opens its cache), measured
# window = [15, 75) s from the aggregated log. CPU: /proc/stat (whole host incl. pgbench) and
# postmaster-children thread-group CPU, snapshots at +15 s and +75 s. No pinning.
set -uo pipefail
export PATH=/opt/pg18/bin:$PATH; R=~/results/q/pgb; mkdir -p $R
printf '\\set qid random(0, 5199)\nSELECT q::text AS qv FROM queries WHERE qid = :qid \\gset\nSELECT id FROM docs ORDER BY emb <=> :qv::turbovec.vector LIMIT 10;\n' > $R/pgb_tv.sql
run() { # label clients pgoptions
  local L=$1 C=$2 O=$3 J=$2; [ $J -gt 32 ] && J=32
  echo "== $(date -u +%T) $L clients=$C jobs=$J PGOPTIONS='$O'" >> $R/pgb.log
  PGOPTIONS="$O" pgbench -h /tmp -n -M prepared -T 75 -c $C -j $J -P 5 -l --aggregate-interval=1 --log-prefix=$R/$L -f $R/pgb_tv.sql db_tv > $R/$L.out 2>&1 &
  local PB=$!
  mpstat 5 15 > $R/$L.mpstat 2>&1 &
  sleep 15; python3 ~/pgsnap.py > $R/$L.snap0
  sleep 60; python3 ~/pgsnap.py > $R/$L.snap1
  wait $PB; echo "rc=$?" >> $R/$L.out; wait
  grep -E 'tps|latency average|failed' $R/$L.out >> $R/pgb.log
}
for rnd in ${ROUNDS:-1 2}; do
  bash ~/swap.sh old >> $R/swaps.txt 2>&1
  for c in 16 96; do run p${rnd}_old_c$c $c ""; done
  bash ~/swap.sh new >> $R/swaps.txt 2>&1
  for c in 16 96; do run p${rnd}_new_c$c $c ""; done
  for c in 16 96; do run p${rnd}_newsp1_c$c $c "-c turbovec.scan_parallelism=1"; done
done
echo PGB_DONE
