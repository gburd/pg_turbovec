#!/usr/bin/env python3
"""Same backend, same queries: client wall + backend minor faults for
(a) EXPLAIN (ANALYZE, TIMING OFF) and (b) the plain SELECT, to see whether
EXPLAIN's "Execution Time" misses a per-query cost that the wall clock pays.
usage: faults.py <dim> <nq> <k> <variant>..."""
import sys, time, statistics as st, numpy as np, psycopg
d, NQ, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); VS = sys.argv[4:]
Q = np.load(f"/work/fixc/q_{d}.npy")[:NQ]
lits = ["[" + ",".join("%.9g" % x for x in q) + "]" for q in Q]
def minflt(pid): return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[7])
for v in VS:
    c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
    c.execute(f"SET enable_seqscan = off; SET jit = off; SET turbovec.oversample = 1.0; SET turbovec.hi_dim_rerank = off; SET turbovec.search_k = {K}")
    pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
    sel = [f"SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '{l}'::turbovec.vector LIMIT 10" for l in lits]
    for s in sel: c.execute(s).fetchall()
    for mode in ("explain", "select", "explain", "select"):
        f0 = minflt(pid); wall = []; ex = []
        for s in sel:
            t0 = time.perf_counter()
            if mode == "explain":
                ex.append(c.execute("EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) " + s).fetchone()[0][0]["Execution Time"])
            else:
                c.execute(s).fetchall()
            wall.append((time.perf_counter() - t0) * 1e3)
        print(f"{v:6s} {mode:8s} wall {st.median(wall):7.3f} ms" + (f"  exec {st.median(ex):7.3f} ms" if ex else " " * 20) +
              f"  minor faults/query {(minflt(pid) - f0) / NQ:7.1f}", flush=True)
    c.close()
