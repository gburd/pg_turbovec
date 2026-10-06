#!/bin/bash
# Full A/B on the non-assert server: flat bw4 then flat bw2 (end-to-end),
# kernel-isolated (node time), cold-backend. One index at a time.
set -uo pipefail
. ~/pgenv.sh
for f in ab kernel cold; do : > /mnt/nvme/$f.jsonl; done
psql -d bench -qc "DROP INDEX IF EXISTS docs_tv_old"
for bw in 4 2; do
  psql -d bench -qc "DROP INDEX IF EXISTS docs_tv" -c "SET maintenance_work_mem='8GB'; SET max_parallel_maintenance_workers=8; CREATE INDEX docs_tv ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width=$bw)"
  FAMS=flat_bw$bw ROUNDS=3 bash ~/ab.sh
  if [ $bw = 4 ]; then bash ~/kab.sh; bash ~/cold.sh; fi
done
echo ALL_AB_DONE
