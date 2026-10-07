#!/usr/bin/env python3
"""Step-5 end-to-end A/B for the recheck work.

Arms are pg_turbovec .so builds (old = perf/recheck-abc, i.e. v2.11.0 code;
new = A+B integrated), installed one at a time into ONE non-assert PG
(/work/pg16rel), postmaster restarted between arms, arms alternated
ROUNDS times. Storage variants (default EXTENDED vs MAIN) are separate tables
with identical rows + identical index bytes.

Per (arm, table, search_k): 200 queries in ONE warm psql session after 20
warm-ups; latency = top-level EXPLAIN (ANALYZE, TIMING OFF) Execution Time;
recall@10 vs exact GT. Also captures backend on-CPU per candidate via
perf task-clock attribution on one pass (attrib mode).

usage: tvp_ab.py run <arm> <round>      (assumes arm already installed+running)
       tvp_ab.py summary
"""
import json, os, statistics as st, subprocess, sys, time
import numpy as np, psycopg
OUT = os.environ.get("OUT", "/work/ab/results.jsonl"); os.makedirs("/work/ab", exist_ok=True)
Q = np.load("/work/corpus/q.npy"); GT = np.load("/work/corpus/gt.npy")
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
TABLES = os.environ.get("TABLES", "docs_ext,docs_main").split(",")
KS = [int(x) for x in os.environ.get("KS", "32,100,256,1024").split(",")]
def conn():
    c = psycopg.connect("host=/tmp port=5440 dbname=bench", autocommit=True)
    c.execute("SET search_path=turbovec,public; SET enable_seqscan=off; SET jit=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off")
    return c
def run(arm, rnd):
    c = conn()
    for t in TABLES:
        for k in KS:
            c.execute(f"SET turbovec.search_k={k}")
            q = lambda l: f"SELECT id FROM {t} ORDER BY tv <=> '{l}'::turbovec.vector LIMIT 10"
            for l in lit[:20]: c.execute(q(l)).fetchall()
            ts, rec = [], []
            for i, l in enumerate(lit):
                ts.append(c.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + q(l)).fetchone()[0][0]["Execution Time"])
                ids = [r[0] for r in c.execute(q(l)).fetchall()]
                rec.append(len(set(ids) & set(GT[i].tolist())) / 10)
            row = dict(arm=arm, round=rnd, table=t, k=k, n=len(ts), p50_ms=round(st.median(ts), 3),
                       p95_ms=round(sorted(ts)[int(0.95 * len(ts)) - 1], 3), recall10=round(st.mean(rec), 4),
                       load=os.getloadavg()[0], ts=time.strftime("%H:%M:%S"))
            print(json.dumps(row), flush=True); open(OUT, "a").write(json.dumps(row) + "\n")
def summary():
    rows = [json.loads(l) for l in open(OUT)]
    arms = sorted({r["arm"] for r in rows})
    for t in TABLES:
        print(f"\n{t}: p50 ms (median over rounds), recall@10 (min over rounds)")
        print("search_k " + "".join(f"{a:>22}" for a in arms))
        for k in KS:
            cells = []
            for a in arms:
                v = [r for r in rows if r["arm"] == a and r["table"] == t and r["k"] == k]
                cells.append(f"{st.median(x['p50_ms'] for x in v):10.2f} R{min(x['recall10'] for x in v):.3f}" if v else f"{'-':>22}")
            print(f"{k:8d} " + "".join(f"{c:>22}" for c in cells))
if sys.argv[1] == "run": run(sys.argv[2], int(sys.argv[3]))
else: summary()
