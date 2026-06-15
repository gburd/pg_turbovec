#!/usr/bin/env python3
"""Load q1000.npy held-out queries into public.query_set(qid, emb)."""
import numpy as np, psycopg
Q = np.load("/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy")
QID = np.load("/scratch/pg_turbovec-bench/cohere-wiki/q1000_ids.npy")
print("queries:", Q.shape, "ids:", QID[0], "..", QID[-1])
with psycopg.connect("host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_wiki",
                     autocommit=True) as c, c.cursor() as cur:
    cur.execute("TRUNCATE public.query_set")
    with cur.copy("COPY public.query_set (qid, emb) FROM STDIN") as cp:
        for i in range(Q.shape[0]):
            # qid = sequential 0..N-1 for GT indexing; store original held id mapping is identity here
            cp.write_row((i, "[" + ",".join(f"{x:.7f}" for x in Q[i]) + "]"))
    cur.execute("SELECT count(*), vector_dims(min(emb)) FROM public.query_set")
    print("loaded query_set:", cur.fetchone())
