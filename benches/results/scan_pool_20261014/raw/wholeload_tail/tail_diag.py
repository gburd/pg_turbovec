#!/usr/bin/env python3
"""Per-query whole-load vs OOC IVF latency on the same 1M x 1024-d index, plus
the probed-row count per query (from a SQL-side reconstruction is not possible,
so we time and compare ids, then rerun the slow ones with TURBOVEC_4BIT_PLANES
toggled via a separate postmaster env -- see tail_planes.sh)."""
import json, os, sys, time, numpy as np, psycopg
Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = sys.argv[2] if len(sys.argv) > 2 else '/home/fedora/diag.jsonl'
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q[:N]]
conn = psycopg.connect("host=/tmp dbname=bench", autocommit=True, prepare_threshold=0); cur = conn.cursor()
cur.execute("SET search_path = turbovec, public"); cur.execute("SET enable_seqscan = off")
sql = "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) %s::turbovec.vector LIMIT 10"
def run(setting):
    for s in setting: cur.execute(s)
    for i in range(20): cur.execute(sql, (lit[i],)).fetchall()
    res = []
    for i in range(N):
        t = time.perf_counter_ns(); ids = [r[0] for r in cur.execute(sql, (lit[i],))]; t = (time.perf_counter_ns() - t) / 1e6
        res.append((t, ids))
    return res
whole = run(["SET turbovec.out_of_core = off", "SET turbovec.cache_size_mb = 4096", "SET turbovec.probes = 16"])
ooc = run(["SET turbovec.out_of_core = on", "SET turbovec.probes = 16"])
with open(OUT, 'w') as f:
    for i in range(N):
        f.write(json.dumps(dict(qid=i, whole_ms=whole[i][0], ooc_ms=ooc[i][0], same_ids=whole[i][1] == ooc[i][1], same_set=set(whole[i][1]) == set(ooc[i][1]))) + "\n")
w = np.array([x[0] for x in whole]); o = np.array([x[0] for x in ooc])
print(f"whole p50 {np.median(w):.1f} p95 {np.percentile(w,95):.1f} slow>60 {(w>60).sum()}/{N} | ooc p50 {np.median(o):.1f} p95 {np.percentile(o,95):.1f} slow>60 {(o>60).sum()}")
print("same top-10 set whole vs ooc:", sum(set(whole[i][1]) == set(ooc[i][1]) for i in range(N)), "/", N)
