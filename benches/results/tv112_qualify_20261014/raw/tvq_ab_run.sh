#!/bin/bash
# old,new alternated ROUNDS times; round 1 kept but reported separately.
set -uo pipefail
W=/work/ab; export PATH=$W/pg16r/bin:$PATH OUT=$W/ab.jsonl NQ=200 CORPUS=$W/corpus_1000000 FAM=flat_bw4
for r in 1 2 3 4; do for arm in old new; do bash ~/tvq_swap.sh $arm; ARM=$arm ENVSET=default ROUND=$r $W/venv/bin/python $W/bench.py || echo "BENCH FAIL $arm r$r"; done; done
echo AB_DONE
