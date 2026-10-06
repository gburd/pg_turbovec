#!/usr/bin/env python3
"""Isolate the turbovec search kernel from the PG heap recheck: per query,
`SELECT count(*) FROM (SELECT id ... ORDER BY tv <=> q LIMIT 1) s` makes the
index return its candidate batch (search_k) then stop after 1 tuple; with
TIMING ON the Index Scan node's actual time is ~ amrescan+first amgettuple =
turbovec search + translate (heap work ~1 fetch). Reports p50 node time."""
import json, os, statistics, subprocess, numpy as np
Q = np.load("/mnt/nvme/corpus_1000000/q.npy"); NQ = int(os.environ.get("NQ", "200"))
ARM = os.environ["ARM"]; ENVSET = os.environ.get("ENVSET", "default"); OUT = os.environ.get("OUT", "/mnt/nvme/kernel.jsonl")
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q[:NQ]]
for k in (32, 100, 256, 1024):
    L = [f"SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k={k};"]
    for i in range(10):
        L.append(f"EXPLAIN (ANALYZE, TIMING ON) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{lit[i]}'::turbovec.vector LIMIT 1;")
    for i in range(NQ):
        L += [r"\echo @@T", f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING ON) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{lit[i]}'::turbovec.vector LIMIT 1;"]
    r = subprocess.run(["psql", "-d", "bench", "-qAt", "-v", "ON_ERROR_STOP=1"], input="\n".join(L) + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    t = []
    for b in r.stdout.split("@@T\n")[1:]:
        p = json.loads(b)[0]["Plan"]
        while p.get("Node Type") != "Index Scan": p = p["Plans"][0]
        t.append(p["Actual Startup Time"])
    row = dict(arm=ARM, envset=ENVSET, k=k, n=len(t), kernel_p50_ms=round(statistics.median(t), 3), kernel_mean_ms=round(statistics.mean(t), 3))
    print(json.dumps(row), flush=True); open(OUT, "a").write(json.dumps(row) + "\n")
