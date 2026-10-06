#!/bin/bash
. ~/pgenv.sh; export OUT=/mnt/nvme/kernel.jsonl
for r in 1 2 3; do
  for arm in old new; do bash ~/swap.sh $arm; ARM=$arm /mnt/nvme/venv/bin/python ~/kernel.py; done
  bash ~/swap.sh new TURBOVEC_4BIT_PLANES=0 TURBOVEC_2BIT_PLANES=0; ARM=new ENVSET=planes_off /mnt/nvme/venv/bin/python ~/kernel.py
done
bash ~/swap.sh new; echo KAB_DONE
