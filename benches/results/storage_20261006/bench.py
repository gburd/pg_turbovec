#!/usr/bin/env python3
"""Warm kNN latency per storage variant, A/B alternated in rounds.
Real query: SELECT id FROM t ORDER BY tv <=> q LIMIT 10 on a flat 4-bit index,
search_k=K, oversample=1.0, hi_dim_rerank=off. Metric: EXPLAIN (ANALYZE,
TIMING OFF) "Execution Time". One backend per variant (warm per-backend index
cache); every query is run once untimed first to warm shared_buffers + cache.
Variant order rotates every round to cancel drift and order effects.
Also records the backend's minor page faults per query (/proc/<pid>/stat):
~2,000/query means glibc is returning the per-query memory to the OS and
faulting it back in (see FINDINGS "allocator state").
DEFAULTS=1 in the environment leaves oversample / hi_dim_rerank at their
defaults (hi_dim_rerank=auto widens the window to min(dim,1024) candidates
for dim >= 256), i.e. measures an untuned install.
usage: bench.py <dim> <rounds> <nq> <k,k,...> <variant>...   -> JSON on stdout"""
import os
import sys, json, statistics as st, numpy as np, psycopg
d, R, NQ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
KS = [int(x) for x in sys.argv[4].split(",")]; VS = sys.argv[5:]
Q = np.load(f"/work/fixc/q_{d}.npy")[:NQ]
lits = ["[" + ",".join("%.9g" % x for x in q) + "]" for q in Q]
def minflt(pid): return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[7])
conn, pids = {}, {}
for v in VS:
    c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
    c.execute("SET enable_seqscan = off; SET jit = off")
    if os.environ.get("DEFAULTS") != "1":
        c.execute("SET turbovec.oversample = 1.0; SET turbovec.hi_dim_rerank = off")
    conn[v] = c; pids[v] = c.execute("SELECT pg_backend_pid()").fetchone()[0]
def run(v, k, lit):
    c = conn[v]; c.execute(f"SET turbovec.search_k = {k}")
    p = c.execute(f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '{lit}'::turbovec.vector LIMIT 10").fetchone()[0][0]
    assert p["Plan"]["Plans"][0]["Node Type"] == "Index Scan", p["Plan"]
    return p["Execution Time"]
for v in VS:                      # warm-up: every (k, query) once
    for k in KS:
        for l in lits: run(v, k, l)
raw = {f"{v}/k{k}": [] for v in VS for k in KS}; flt = {f"{v}/k{k}": [] for v in VS for k in KS}
for r in range(R):
    order = VS[r % len(VS):] + VS[:r % len(VS)]
    for v in order:
        for k in KS:
            f0 = minflt(pids[v])
            raw[f"{v}/k{k}"].append([run(v, k, l) for l in lits])
            flt[f"{v}/k{k}"].append(round((minflt(pids[v]) - f0) / NQ, 1))
summary = {key: {"median_ms": st.median([x for rr in rounds for x in rr]),
                 "round_medians_ms": [round(st.median(rr), 3) for rr in rounds],
                 "backend_minor_faults_per_query": flt[key]}
           for key, rounds in raw.items()}
print(json.dumps({"defaults": os.environ.get("DEFAULTS") == "1", "dim": d, "rounds": R, "nq": NQ, "ks": KS, "variants": VS,
                  "summary": summary, "raw_ms": raw}))
