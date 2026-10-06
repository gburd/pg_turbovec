#!/usr/bin/env python3
"""turbovec 1.1.1 A/B bench driver (adapted from benches/results/rebench_20260925/rb_driver.py).

Per arm: ONE psql session (warm-ups discarded), latency = top-level
EXPLAIN (ANALYZE) Execution Time (whole query incl. exact recheck), recall@10
from the same arm's returned ids vs exact numpy GT. Query vectors inlined as
literals. Seq-scan-on-docs fallback flagged. Writes JSONL rows to $OUT.

env: ARM=old|new  FAM=flat_bw4|flat_bw2|ivf_bw4  NQ (queries, default 200)
     ENVSET (label of extra env, e.g. planes_off)  CORPUS=/mnt/nvme/corpus_1000000
"""
import json, os, statistics, subprocess, sys, time
import numpy as np
C = os.environ.get("CORPUS", "/mnt/nvme/corpus_1000000")
Q = np.load(f"{C}/q.npy"); GT = np.load(f"{C}/gt100.npy")[:, :10]
NQ = int(os.environ.get("NQ", "200")); NW = int(os.environ.get("NW", "10"))
ARM = os.environ["ARM"]; FAM = os.environ["FAM"]; ENVSET = os.environ.get("ENVSET", "default")
OUT = os.environ.get("OUT", "/mnt/nvme/results.jsonl")
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q[:NQ]]
T = "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '%s'::turbovec.vector LIMIT 10;"
def arms():
    base = "SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off;"
    if FAM.startswith("flat"):
        for k in (32, 100, 256, 1024):
            yield f"{FAM}_k{k}", base + f" SET turbovec.search_k={k};"
    else:
        for p in (16, 64, 256):
            for k in (100, 256):
                yield f"{FAM}_p{p}_k{k}", base + f" SET turbovec.probes={p}; SET turbovec.search_k={k};"
def run(setup):
    L = [setup]
    for i in range(NW):
        L.append("EXPLAIN (ANALYZE, TIMING OFF) " + T % lit[i % NQ])
    for i in range(NQ):
        L += [r"\echo @@T", "EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + T % lit[i], r"\echo @@I", T % lit[i]]
    r = subprocess.run(["psql", "-d", "bench", "-v", "ON_ERROR_STOP=1", "-qAt"], input="\n".join(L) + "\n",
                       capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-800:])
    out = []
    for b in r.stdout.split("@@T\n")[1:]:
        j, rest = b.split("@@I\n", 1)
        plan = json.loads(j)[0]; seq = [False]
        def walk(n):
            if n.get("Node Type") == "Seq Scan" and n.get("Relation Name") == "docs": seq[0] = True
            for c in n.get("Plans", []): walk(c)
        walk(plan["Plan"])
        ids = [int(x) for x in rest.split() if x.strip().lstrip("-").isdigit()]
        out.append((plan["Execution Time"], ids, seq[0]))
    return out
for label, setup in arms():
    la0 = os.getloadavg()[0]; res = run(setup); la1 = os.getloadavg()[0]
    t = [x[0] for x in res]
    rec = [len(set(ids) & set(GT[i].tolist())) / 10 for i, (_, ids, _) in enumerate(res)]
    row = dict(arm=ARM, envset=ENVSET, label=label, n=len(t), p50_ms=round(statistics.median(t), 3),
               p95_ms=round(sorted(t)[int(0.95 * len(t)) - 1], 3), mean_ms=round(statistics.mean(t), 3),
               recall10=round(statistics.mean(rec), 4), seq_fallback=any(x[2] for x in res),
               loadavg=round(max(la0, la1), 2), ts=time.strftime("%H:%M:%S"))
    print(json.dumps(row), flush=True)
    open(OUT, "a").write(json.dumps(row) + "\n")
