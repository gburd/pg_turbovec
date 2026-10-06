#!/usr/bin/env python3
"""Decompose the per-candidate cost of a turbovec ORDER BY ... LIMIT 10 query.
All in one warm backend, 100 queries, median ms.
 E2E(k)        : the real query, search_k = k (index scan + recheck of k candidates)
 SEARCH_ONLY   : turbovec.knn()-free proxy: index returns k tids, we stop after 1 row
                 -- not possible (reorder queue drains all). Instead measure pieces:
 HEAPFETCH(k)  : SELECT count(*) FROM docs WHERE ctid = ANY(<k tids>)  (heap fetch only, no detoast)
 DETOAST(k)    : same tids, SELECT sum(length(tv::text))? no -> pg_column_size forces no detoast;
                 use SELECT sum(turbovec.vector_dims(tv))  (detoast + CBOR decode, no distance)
 DIST(k)       : same tids, SELECT sum(tv <=> q)  (detoast + decode + distance)
"""
import json, statistics as st, subprocess, numpy as np, psycopg, time, sys
c = psycopg.connect("host=/tmp/rck port=55432 dbname=bench", autocommit=True)
c.execute("SET search_path=turbovec,public; SET enable_seqscan=off; SET enable_bitmapscan=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET jit=off")
Q = np.load("/tmp/rck/q.npy")[:100]
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
def t(sql, *a):
    t0 = time.perf_counter(); c.execute(sql, a).fetchall(); return (time.perf_counter() - t0) * 1e3
def et(sql):
    return json.loads(c.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + sql).fetchone()[0][0]["Execution Time"]) if False else c.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + sql).fetchone()[0][0]["Execution Time"]
out = {}
for k in (32, 100, 256, 1024):
    c.execute(f"SET turbovec.search_k={k}")
    for i in range(5): et(f"SELECT id FROM docs ORDER BY tv <=> '{lit[i]}'::turbovec.vector LIMIT 10")
    e2e = [et(f"SELECT id FROM docs ORDER BY tv <=> '{l}'::turbovec.vector LIMIT 10") for l in lit]
    # candidate tids for each query: the k tids the index would return. Use the
    # scan itself with LIMIT k to collect them (cost not timed).
    # NB SELECT ctid under the reorder queue projects (4294967295,0) (bug6) -> collect ids
    ids = [[r[0] for r in c.execute(f"SELECT id FROM docs ORDER BY tv <=> '{l}'::turbovec.vector LIMIT {k}").fetchall()] for l in lit[:30]]
    rtids = [[r[0] for r in c.execute("SELECT ctid::text FROM docs WHERE id = ANY(%s)", (x,)).fetchall()] for x in ids]
    c.execute("SET enable_indexscan=off; SET enable_tidscan=on")
    hf = [et(f"SELECT count(*) FROM docs WHERE ctid = ANY(ARRAY[{','.join("'"+t+"'::tid" for t in tl)}])") for tl in rtids]
    dt = [et(f"SELECT sum(vector_dims(tv)) FROM docs WHERE ctid = ANY(ARRAY[{','.join("'"+t+"'::tid" for t in tl)}])") for tl in rtids]
    ds = [et(f"SELECT sum(tv <=> '{lit[j]}'::turbovec.vector) FROM docs WHERE ctid = ANY(ARRAY[{','.join("'"+t+"'::tid" for t in tl)}])") for j, tl in enumerate(rtids)]
    c.execute("RESET enable_indexscan")
    out[k] = dict(e2e=st.median(e2e), heapfetch=st.median(hf), detoast_decode=st.median(dt), detoast_decode_dist=st.median(ds))
    print(k, {a: round(b, 3) for a, b in out[k].items()}, flush=True)
json.dump(out, open(f"/tmp/rck/decomp_{sys.argv[1] if len(sys.argv)>1 else 'x'}.json", "w"), indent=1)
