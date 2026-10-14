#!/usr/bin/env python3
"""Per-arm ids for the 200 held-out queries at search_k in (32, 1024); writes ids_<arm>.json."""
import json, os, subprocess, sys, numpy as np
C = "/work/ab/corpus_1000000"; Q = np.load(f"{C}/q.npy"); ARM = sys.argv[1]
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
out = {}
for k in (32, 1024):
    L = ["SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off;", f"SET turbovec.search_k={k};"]
    for q in lit:
        L += [r"\echo @@", f"SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{q}'::turbovec.vector LIMIT 10;"]
    r = subprocess.run(["psql", "-h", "/tmp", "-d", "bench", "-qAt"], input="\n".join(L) + "\n", capture_output=True, text=True, check=True)
    out[k] = [[int(x) for x in b.split()] for b in r.stdout.split("@@\n")[1:]]
json.dump(out, open(f"/work/ab/ids_{ARM}.json", "w"))
print(ARM, {k: len(v) for k, v in out.items()})
