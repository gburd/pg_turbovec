#!/usr/bin/env python3
"""Does backend history change the MAIN/PLAIN cost? Same backend: k=1024 block,
then a k=32 + k=256 block (like bench.py's per-round sequence), then k=1024
again. Reports EXPLAIN Execution Time median and minor faults/query per block.
usage: faults2.py <dim> <nq> <variant>..."""
import sys, statistics as st, numpy as np, psycopg
d, NQ = int(sys.argv[1]), int(sys.argv[2]); VS = sys.argv[3:]
Q = np.load(f"/work/fixc/q_{d}.npy")[:NQ]
lits = ["[" + ",".join("%.9g" % x for x in q) + "]" for q in Q]
def minflt(pid): return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[7])
for v in VS:
    c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
    c.execute("SET enable_seqscan = off; SET jit = off; SET turbovec.oversample = 1.0; SET turbovec.hi_dim_rerank = off")
    pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
    def block(k):
        c.execute(f"SET turbovec.search_k = {k}"); f0 = minflt(pid)
        ex = [c.execute(f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '{l}'::turbovec.vector LIMIT 10").fetchone()[0][0]["Execution Time"] for l in lits]
        return st.median(ex), (minflt(pid) - f0) / NQ
    for step, k in [("fresh", 1024), ("again", 1024), ("k32", 32), ("k256", 256), ("after k32+k256", 1024), ("again", 1024)]:
        m, f = block(k)
        print(f"{v:6s} k={k:<5d} {step:15s} exec median {m:7.3f} ms  minor faults/query {f:7.1f}", flush=True)
    c.close()
