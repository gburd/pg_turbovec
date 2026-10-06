#!/usr/bin/env python3
"""Client-side wall clock of the plain kNN query (no EXPLAIN), warm backends,
variants alternated per round. Also reports the backend's minor page faults
per query (from /proc/<pid>/stat) to expose allocator/first-touch costs.
usage: wall.py <dim> <rounds> <nq> <k> <variant>...   -> JSON on stdout"""
import sys, json, time, statistics as st, numpy as np, psycopg
d, R, NQ, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); VS = sys.argv[5:]
Q = np.load(f"/work/fixc/q_{d}.npy")[:NQ]
lits = ["[" + ",".join("%.9g" % x for x in q) + "]" for q in Q]
def minflt(pid): return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[7])
conn, pids = {}, {}
for v in VS:
    c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
    c.execute(f"SET enable_seqscan = off; SET jit = off; SET turbovec.oversample = 1.0; SET turbovec.hi_dim_rerank = off; SET turbovec.search_k = {K}")
    conn[v] = c; pids[v] = c.execute("SELECT pg_backend_pid()").fetchone()[0]
sql = {v: [f"SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '{l}'::turbovec.vector LIMIT 10" for l in lits] for v in VS}
for v in VS:
    for _ in range(2):
        for s in sql[v]: conn[v].execute(s).fetchall()
raw = {v: [] for v in VS}; flt = {v: [] for v in VS}
for r in range(R):
    for v in VS[r % len(VS):] + VS[:r % len(VS)]:
        f0 = minflt(pids[v]); ts = []
        for s in sql[v]:
            t0 = time.perf_counter(); conn[v].execute(s).fetchall(); ts.append((time.perf_counter() - t0) * 1e3)
        raw[v].append(ts); flt[v].append((minflt(pids[v]) - f0) / len(ts))
print(json.dumps({"dim": d, "k": K, "rounds": R, "nq": NQ,
  "summary": {v: {"median_wall_ms": st.median([x for rr in raw[v] for x in rr]),
                  "round_medians_ms": [round(st.median(rr), 3) for rr in raw[v]],
                  "minor_faults_per_query": [round(x, 1) for x in flt[v]]} for v in VS},
  "raw_ms": raw}))
