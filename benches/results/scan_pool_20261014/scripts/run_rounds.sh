#!/usr/bin/env bash
# Single-client warm cells. Rounds 0 (discarded) 1 2 3; each round runs OLD then NEW. Per arm-pass: swap.sh
# (install .so, restart, prewarm heap+indexes+queries), then every cell in a fresh connection (bench_q.py:
# 50 warm-up qids 0..49 incl. the per-backend cache cold-open, then 1,000 timed qids 50..1049).
set -uo pipefail
export PATH=/opt/pg18/bin:$PATH
R=~/results/q/warm; mkdir -p $R
B() { echo "== $(date -u +%T) $*" >> $R/bench.log
      python3 ~/bench_q.py --db db_tv --sys tv --warm 50 --n 1000 --out $R "$@" >> $R/bench.log 2>&1 || echo "FAIL $*" >> $R/bench.log; }
HDR="turbovec.hi_dim_rerank=off"
for rnd in ${ROUNDS:-0 1 2 3}; do for arm in old new; do
  bash ~/swap.sh $arm >> $R/swaps.txt 2>&1
  L=r${rnd}_${arm}
  for sk in 16 32 128 1024; do
    B --label ${L}_flat_sk${sk}_spdef --set "$HDR;turbovec.search_k=$sk" --expect-index docs_tv_flat
    if [ $arm = new ]; then for sp in 1 88 96; do
      B --label ${L}_flat_sk${sk}_sp${sp} --set "$HDR;turbovec.search_k=$sk;turbovec.scan_parallelism=$sp" --expect-index docs_tv_flat; done; fi
  done
  # OLD: show scan_parallelism did not bound flat in 2.12.1
  [ $arm = old ] && B --label ${L}_flat_sk32_sp1 --set "$HDR;turbovec.search_k=32;turbovec.scan_parallelism=1" --expect-index docs_tv_flat
  B --label ${L}_flatdef --expect-index docs_tv_flat
  for pr in 16 64; do
    B --label ${L}_ooc_p${pr} --table docs_ivf --set "turbovec.out_of_core=on;turbovec.probes=$pr" --expect-index docs_ivf_tv; done
  B --label ${L}_ooc_p16_sp8 --table docs_ivf --set "turbovec.out_of_core=on;turbovec.probes=16;turbovec.scan_parallelism=8" --expect-index docs_ivf_tv
  B --label ${L}_whole_p16 --table docs_ivf --set "turbovec.out_of_core=off;turbovec.cache_size_mb=2048;turbovec.probes=16" --expect-index docs_ivf_tv
  bash ~/idx_sha.sh $L >> $R/swaps.txt 2>&1
  grep -cE 'ERROR|WARNING|PANIC|FATAL' /data/pgdata/log/*.log | sed "s/^/$L pglog_err_count /" >> $R/swaps.txt
done; done
echo ROUNDS_DONE
