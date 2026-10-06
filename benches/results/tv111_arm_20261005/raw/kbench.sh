#!/bin/bash
# Build kbench at OLD (47a26a3 = what v2.10.3 ships) and NEW (ba92958); run at
# 1 and 32 threads, bits 4 and 2; NEW also with planes off. PG stopped.
set -euo pipefail
. ~/.cargo/env; . ~/pgenv.sh
pg_ctl -D /mnt/nvme/pgdata -m fast -w stop >/dev/null || true
cd /mnt/nvme
[ -f corpus_1000000/base.f32 ] || /mnt/nvme/venv/bin/python - <<'P'
import numpy as np, pyarrow.parquet as pq, glob
out=open('/mnt/nvme/corpus_1000000/base.f32','wb'); n=0
for p in sorted(glob.glob('/mnt/nvme/hf/en/*.parquet')):
    a=pq.read_table(p,columns=['emb']).column('emb').combine_chunks().values.to_numpy().reshape(-1,1024).astype(np.float32)
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-30)
    take=min(len(a),1_000_000-n); out.write(a[:take].tobytes()); n+=take
    if n>=1_000_000: break
out.close()
np.load('/mnt/nvme/corpus_1000000/q.npy').astype(np.float32).tofile('/mnt/nvme/corpus_1000000/q.f32')
print('wrote', n)
P
TAG=$(date +%s)
for arm in old:47a26a393b9b1e470de353c60a990afac09933fa new:ba9295846f3f18eb5a403004e3f9ff6448ec5de0; do
  name=${arm%%:*}; rev=${arm#*:}
  d=kb_${name}_$TAG; cp -r ~/kbench $d; sed -i "s/REV/$rev/" $d/Cargo.toml
  (cd $d && cargo build --release -q)
  ln -sfn $d kb_$name
done
B=corpus_1000000/base.f32; Q=corpus_1000000/q.f32
for r in 1 2; do
 for bits in 4 2; do
  for th in 1 32; do
    for spec in old new new_off; do
      case $spec in old) bin=kb_old; env=X=1;; new) bin=kb_new; env=X=1;; new_off) bin=kb_new; env="TURBOVEC_4BIT_PLANES=0 TURBOVEC_2BIT_PLANES=0";; esac
      out=$(env $env ./$bin/target/release/kbench $B $Q 1000000 $bits $th /mnt/nvme/ids_${spec}_b${bits}_t${th}.txt)
      echo "{\"arm\":\"$spec\",\"round\":$r,${out:1}" | tee -a /mnt/nvme/kbench.jsonl
    done
  done
 done
done
pg_ctl -D /mnt/nvme/pgdata -l /mnt/nvme/pg.log -w start >/dev/null
echo KBENCH_DONE
