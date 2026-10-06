#!/usr/bin/env python3
"""Attribute the REAL query's backend CPU time per stage.
perf -e task-clock -c 100000  => each sample = 100 us of on-CPU time, so
samples are absolute time, independent of the hybrid-core PMUs. Only the
backend thread is attributed (the recheck is serial on it); the turbovec
scan's rayon workers are reported separately as wall-clock 'scan'.
usage: attrib.py <table> <search_k> <nq>"""
import os, signal, subprocess, sys, time, collections, numpy as np, psycopg
TBL, K, NQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = psycopg.connect("host=/tmp/rck port=55432 dbname=bench", autocommit=True)
pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
c.execute(f"SET search_path=turbovec,public; SET enable_seqscan=off; SET jit=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k={K}")
Q = np.load("/tmp/rck/q.npy"); lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
sql = [f"SELECT id FROM {TBL} ORDER BY tv <=> '{l}'::turbovec.vector LIMIT 10" for l in lit]
data = f"/tmp/rck/attr_{TBL}_{K}.data"
REUSE = os.environ.get("REUSE") == "1"
if not REUSE:
  for s in sql[:20]: c.execute(s).fetchall()                       # warm cache + TOAST pages
if REUSE:
  wall = float(os.environ["WALL_MS"]) / 1e3 * NQ
else:
 p = subprocess.Popen(["perf", "record", "-q", "-e", "task-clock", "-c", "100000", "-g", "--call-graph=fp", "-t", str(pid), "-o", data], stderr=subprocess.DEVNULL)
 time.sleep(0.5)
 t0 = time.perf_counter()
 for i in range(NQ): c.execute(sql[i % len(sql)]).fetchall()
 wall = time.perf_counter() - t0
 p.send_signal(signal.SIGINT); p.wait()
buckets = [  # (name, frame substrings) -- first match walking the stack from LEAF up
  ("TOAST fetch (detoast_attr ...)", ("toast_fetch_datum", "heap_fetch_toast_slice", "detoast_attr", "toast_open_indexes", "toast_close_indexes")),
  ("CBOR decode of vector (pgrx serde)", ("serde_cbor",)),
  ("exact cosine kernel (3 serial f64 sums)", ("pg_turbovec::distance::cosine_distance", "pg_turbovec::kernels::")),
  ("heap fetch of candidate (core)", ("index_fetch_heap", "heapam_index_fetch_tuple")),
  ("reorder queue push/pop (core)", ("reorderqueue_push", "reorderqueue_pop", "pairingheap", "ExecCopySlotHeapTuple", "ExecForceStoreHeapTuple", "datumCopy")),
  ("turbovec scan, backend share", ("turbovec::search", "turbovec::pack", "pg_turbovec::index::scan", "pg_turbovec::cache::")),
  ("other IndexNextWithReorder / expr eval (core)", ("IndexNextWithReorder", "ExecInterpExpr", "EvalOrderByExpressions", "cosine_distance_wrapper")),
]
out = subprocess.run(["perf", "script", "-i", data, "-F", "tid,ip,sym"], capture_output=True, text=True).stdout
cnt = collections.Counter(); total = 0
for blk in out.split("\n\n"):
    lines = [l.strip() for l in blk.strip().splitlines()]
    if len(lines) < 2: continue
    total += 1
    stack = [l.split(None, 1)[1] if " " in l else l for l in lines[1:]]  # drop ip; leaf first
    hit = "other (parse/plan/executor startup/libpq)"
    for name, keys in buckets:
        if any(any(k in fr for k in keys) for fr in stack):
            hit = name; break
    cnt[hit] += 1
us = lambda n: n * 100.0  # 100 us per sample
print(f"{TBL} search_k={K}: {NQ} queries, wall {wall*1e3/NQ:.2f} ms/query, backend on-CPU {us(total)/NQ/1e3:.2f} ms/query")
for name, _ in buckets + [("other (parse/plan/executor startup/libpq)", ())]:
    n = cnt[name]
    print(f"   {name:46s} {us(n)/NQ/1e3:7.3f} ms/query  {us(n)/NQ/K:7.2f} us/candidate")
print(f"   {'(wall - backend on-CPU = waiting on scan workers)':46s} {(wall*1e3/NQ) - us(total)/NQ/1e3:7.3f} ms/query")
