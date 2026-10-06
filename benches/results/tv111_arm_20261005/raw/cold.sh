#!/bin/bash
# Cold-backend latency (fresh psql per query => per-backend cache cold =>
# relfile read + cache build repaid) vs warm, per arm. Same index bytes.
# OS page cache warm (regime R2 of rebench_20260925).
. ~/pgenv.sh
OUT=/mnt/nvme/cold.jsonl
PY=/mnt/nvme/venv/bin/python
$PY -c "import numpy as np; Q=np.load('/mnt/nvme/corpus_1000000/q.npy'); open('/tmp/cq.txt','w').write('\n'.join('['+','.join(f'{x:.6f}' for x in q)+']' for q in Q[:20]))"
mapfile -t QV < /tmp/cq.txt
q(){ echo "SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.search_k=100; EXPLAIN (ANALYZE, FORMAT JSON) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '$1'::turbovec.vector LIMIT 10;"; }
et(){ $PY -c "import sys,json; print(json.load(sys.stdin)[0]['Execution Time'])"; }
for r in 1 2 3; do
 for spec in "old" "new" "new TURBOVEC_4BIT_PLANES=0 TURBOVEC_2BIT_PLANES=0"; do
  bash ~/swap.sh $spec >/dev/null
  # warm the OS page cache with one throwaway backend
  q "${QV[0]}" | psql -d bench -qAt >/dev/null
  : > /tmp/c.txt
  for i in $(seq 0 19); do q "${QV[$i]}" | psql -d bench -qAt | et >> /tmp/c.txt; done
  label=$(echo $spec | sed 's/ TURBOVEC_4BIT_PLANES=0 TURBOVEC_2BIT_PLANES=0/+planes_off/')
  $PY - "$label" "$r" <<'P'
import sys, statistics, json
a=[float(x) for x in open('/tmp/c.txt') if x.strip()]
row=dict(arm=sys.argv[1], round=int(sys.argv[2]), cold_p50_ms=round(statistics.median(a),1), cold_min_ms=round(min(a),1), n=len(a))
print(json.dumps(row)); open('/mnt/nvme/cold.jsonl','a').write(json.dumps(row)+'\n')
P
 done
done
bash ~/swap.sh new; echo COLD_DONE
