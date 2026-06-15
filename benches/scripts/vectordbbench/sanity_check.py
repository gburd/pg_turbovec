#!/usr/bin/env python3
"""Correctness sanity gate for the pre-AVX2-fixed v1.8.0 build on meh.

10k x 128-d DISTINCT random unit vectors. Build a turbovec 4-bit index.
For 20 fresh held-out probes, compare index top-10 vs brute-force top-10.
Pass iff: ids are 10 DISTINCT values (not one repeated), and mean recall@10
is clearly > 0 (we require >= 0.5 here; the AVX2 bug produced 0.0).
"""
import sys, numpy as np, psycopg

DSN = "host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=tv_sanity"
N, D, NQ = 10_000, 128, 20
rng = np.random.default_rng(42)

def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)

corpus = unit(rng.standard_normal((N, D)).astype(np.float32))
probes = unit(rng.standard_normal((NQ, D)).astype(np.float32))
assert len(set(corpus[i].tobytes() for i in range(N))) == N, "corpus not distinct"

# brute-force GT (cosine == 1 - dot for unit vectors)
sims = probes @ corpus.T
gt = np.argsort(-sims, axis=1)[:, :10]

with psycopg.connect("host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=postgres",
                     autocommit=True) as c, c.cursor() as cur:
    cur.execute("DROP DATABASE IF EXISTS tv_sanity")
    cur.execute("CREATE DATABASE tv_sanity")

with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
    cur.execute("DROP TABLE IF EXISTS s")
    cur.execute("CREATE TABLE s (id int, emb turbovec.vector)")
    with cur.copy("COPY s (id, emb) FROM STDIN") as cp:
        for i in range(N):
            cp.write_row((i, "[" + ",".join(f"{x:.6f}" for x in corpus[i]) + "]"))
    cur.execute("SET maintenance_work_mem='2GB'")
    cur.execute("CREATE INDEX s_tv ON s USING turbovec (emb turbovec.vec_cosine_ops) WITH (bit_width=4)")
    cur.execute("SET enable_seqscan=off")
    cur.execute("SET turbovec.search_k=500")

    recalls, distinct_ok = [], True
    for qi in range(NQ):
        v = "[" + ",".join(f"{x:.6f}" for x in probes[qi]) + "]"
        cur.execute("SELECT id FROM s ORDER BY emb OPERATOR(turbovec.<=>) %s::turbovec.vector LIMIT 10", (v,))
        ids = [r[0] for r in cur.fetchall()]
        if len(set(ids)) != 10:
            distinct_ok = False
        rec = len(set(ids) & set(int(x) for x in gt[qi])) / 10.0
        recalls.append(rec)
        if qi < 3:
            print(f"  q{qi}: ids={ids}  recall={rec:.2f}")

mean = float(np.mean(recalls))
print(f"\nmean recall@10 over {NQ} probes = {mean:.4f}")
print(f"all top-10 sets distinct (10 ids each): {distinct_ok}")
ok = distinct_ok and mean >= 0.5
print("SANITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
