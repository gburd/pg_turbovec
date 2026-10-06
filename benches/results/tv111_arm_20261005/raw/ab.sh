#!/bin/bash
# A/B on ONE index per family (same bytes for both arms), arms alternated
# old,new,old,new... ROUNDS times, plus a NEW+planes_off control.
set -uo pipefail
. ~/pgenv.sh
export OUT=/mnt/nvme/ab.jsonl NQ=${NQ:-200}
ROUNDS=${ROUNDS:-3}
FAMS=${FAMS:-flat_bw4}
for fam in $FAMS; do
  export FAM=$fam
  for r in $(seq 1 $ROUNDS); do
    for arm in old new; do
      bash ~/swap.sh $arm
      ARM=$arm ENVSET=default ROUND=$r /mnt/nvme/venv/bin/python ~/bench.py || echo "BENCH FAIL $arm $fam"
    done
    bash ~/swap.sh new TURBOVEC_4BIT_PLANES=0 TURBOVEC_2BIT_PLANES=0
    ARM=new ENVSET=planes_off ROUND=$r /mnt/nvme/venv/bin/python ~/bench.py || echo "BENCH FAIL planes_off $fam"
  done
done
bash ~/swap.sh new
echo AB_DONE
