#!/usr/bin/env python3
"""Per-candidate cost of each recheck stage, fixed work, no index scan.
For 30 queries x K=1024 real candidate TIDs (what the index returns), time
in ONE warm backend (EXPLAIN ANALYZE Execution Time, TIMING OFF):
  fetch    : SELECT count(*)            WHERE ctid = ANY(tids)       heap fetch only
  dims     : SELECT sum(vector_dims(tv)) ...                         + detoast + CBOR decode
  dist     : SELECT sum(tv <=> q)        ...                         + 2nd decode (q) + distance
  rawlen   : SELECT sum(octet_length(tv::bytea))? not castable -> use pg_column_size(tv) (no detoast)
Report us per candidate, min-of-medians over 3 rounds (robust to the noisy shared box)."""
import statistics as st, numpy as np, psycopg, sys
TBL = sys.argv[1]; K = 1024
c = psycopg.connect("host=/tmp/rck port=55432 dbname=bench", autocommit=True)
c.execute("SET search_path=turbovec,public; SET jit=off; SET enable_seqscan=off; SET enable_bitmapscan=off")
Q = np.load("/tmp/rck/q.npy")[:30]
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q]
c.execute(f"SET turbovec.search_k={K}; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off")
tids = []
for l in lit:
    ids = [r[0] for r in c.execute(f"SELECT id FROM {TBL} ORDER BY tv <=> '{l}'::turbovec.vector LIMIT {K}").fetchall()]
    tids.append("ARRAY[" + ",".join(f"'{r[0]}'::tid" for r in c.execute(f"SELECT ctid::text FROM {TBL} WHERE id = ANY(%s)", (ids,)).fetchall()) + "]")
c.execute("SET enable_indexscan=off; SET enable_tidscan=on")
def et(sql): return c.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + sql).fetchone()[0][0]["Execution Time"]
stages = {
  "fetch": lambda i: f"SELECT count(*) FROM {TBL} WHERE ctid = ANY({tids[i]})",
  "colsize_nodetoast": lambda i: f"SELECT sum(pg_column_size(tv)) FROM {TBL} WHERE ctid = ANY({tids[i]})",
  "dims_detoast_decode": lambda i: f"SELECT sum(vector_dims(tv)) FROM {TBL} WHERE ctid = ANY({tids[i]})",
  "dist_full_recheck": lambda i: f"SELECT sum(tv <=> '{lit[i]}'::turbovec.vector) FROM {TBL} WHERE ctid = ANY({tids[i]})",
}
res = {}
for name, f in stages.items():
    rounds = []
    for _ in range(3):
        rounds.append(st.median(et(f(i)) for i in range(len(lit))))
    res[name] = min(rounds) * 1000 / K  # us per candidate
print(TBL, {k: round(v, 2) for k, v in res.items()})
