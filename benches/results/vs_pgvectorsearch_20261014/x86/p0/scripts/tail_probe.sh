#!/bin/bash
# Characterise the turbovec IVF out-of-core ~100 ms tail (P0 follow-up).
set -uo pipefail
R=~/results/tailprobe; mkdir -p $R
export PATH=/opt/pg18/bin:$PATH
run() { python3 ~/bench.py --db db_tv --sys tv --label "$1" --table docs_ivf --set "$2" --mode ids --warm 20 --n 300 --out $R --expect-index docs_ivf_tv > $R/$1.log 2>&1; python3 - "$R/$1.ids.jsonl" "$1" <<'PY'
import json,sys
w=sorted(json.loads(l)['wall_us']/1000 for l in open(sys.argv[1]))
slow=[x for x in w if x>60]
print(f"{sys.argv[2]:40s} n={len(w)} p50={w[len(w)//2]:.1f} p95={w[int(.95*len(w))]:.1f} slow(>60ms)={len(slow)} slow_median={slow[len(slow)//2] if slow else 0:.1f}")
PY
}
B="turbovec.cache_size_mb=256;turbovec.out_of_core=on;turbovec.probes=16"
run ooc_p16_sp4           "$B"
run ooc_p16_sp1           "$B;turbovec.scan_parallelism=1"
run ooc_p16_sp2           "$B;turbovec.scan_parallelism=2"
run ooc_p16_sp8           "$B;turbovec.scan_parallelism=8"
# same query repeated: deterministic per query or per call?
echo "--- repeat the 5 slowest qids of ooc_p16_sp4 three times each"
python3 - $R/ooc_p16_sp4.ids.jsonl > $R/slow_qids.txt <<'PY'
import json,sys
r=sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda x:-x['wall_us'])[:5]
print(" ".join(str(x['qid']) for x in r))
PY
cat $R/slow_qids.txt
echo TAIL_PROBE_DONE
