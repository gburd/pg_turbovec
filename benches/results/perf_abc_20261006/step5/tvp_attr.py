#!/usr/bin/env python3
"""Per-candidate backend on-CPU attribution for the CURRENTLY INSTALLED arm
(same method as benches/results/recheck_20261006/attrib.py): perf task-clock
-c 100000 (each sample = 100 us on-CPU) on the backend thread only, stacks
bucketed by first matching frame. usage: tvp_attr.py <arm> <table> <k> <nq>"""
import json, os, signal, subprocess, sys, time, collections, numpy as np, psycopg
ARM, TBL, K, NQ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = psycopg.connect("host=/tmp port=5440 dbname=bench", autocommit=True)
pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
c.execute(f"SET search_path=turbovec,public; SET enable_seqscan=off; SET jit=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k={K}")
Q = np.load("/work/corpus/q.npy"); lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
sql = [f"SELECT id FROM {TBL} ORDER BY tv <=> '{l}'::turbovec.vector LIMIT 10" for l in lit]
for s in sql[:20]: c.execute(s).fetchall()
data = f"/work/ab/attr_{ARM}_{TBL}_{K}.data"
p = subprocess.Popen(["perf", "record", "-q", "-e", "task-clock", "-c", "100000", "-g", "--call-graph=fp", "-t", str(pid), "-o", data], stderr=subprocess.DEVNULL)
time.sleep(0.5); t0 = time.perf_counter()
for i in range(NQ): c.execute(sql[i % len(sql)]).fetchall()
wall = time.perf_counter() - t0
p.send_signal(signal.SIGINT); p.wait()
buckets = [
  ("toast", ("toast_fetch_datum", "heap_fetch_toast_slice", "detoast_attr", "toast_open_indexes", "toast_close_indexes")),
  ("cbor_decode", ("serde_cbor",)),
  ("reorder_queue_core", ("reorderqueue_push", "reorderqueue_pop", "pairingheap", "ExecCopySlotHeapTuple", "ExecForceStoreHeapTuple", "datumCopy")),
  ("heap_fetch_core", ("index_fetch_heap", "heapam_index_fetch_tuple")),
  ("turbovec_scan_backend", ("turbovec::search", "rayon", "ReadOnlyIndex", "amgettuple", "amrescan", "ambeginscan")),
  ("distance_fn_ours", ("cosine_distance_wrapper", "l2_distance_wrapper", "distance::Slot", "with_slots", "with_operands")),
  ("executor_core", ("IndexNextWithReorder", "ExecInterpExpr", "ExecScan")),
]
out = subprocess.run(["perf", "script", "-i", data, "-F", "tid,ip,sym"], capture_output=True, text=True).stdout
cnt = collections.Counter()
for blk in out.split("\n\n"):
    lines = [l.strip() for l in blk.strip().splitlines()]
    if len(lines) < 2: continue
    stack = [l.split(None, 1)[1] if " " in l else l for l in lines[1:]]
    for name, keys in buckets:
        if any(any(x in fr for x in keys) for fr in stack): cnt[name] += 1; break
    else: cnt["other"] += 1
tot = sum(cnt.values())
row = dict(arm=ARM, table=TBL, k=K, nq=NQ, wall_ms_per_q=round(wall * 1e3 / NQ, 3), backend_cpu_ms_per_q=round(tot * 100 / NQ / 1e3, 3),
           us_per_candidate={n: round(cnt[n] * 100 / NQ / K, 3) for n in [b[0] for b in buckets] + ["other"]})
row["us_per_candidate"]["total"] = round(tot * 100 / NQ / K, 3)
print(json.dumps(row)); open("/work/ab/attr.jsonl", "a").write(json.dumps(row) + "\n")
