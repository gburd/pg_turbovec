#!/usr/bin/env python3
"""Fast 10M loader via COPY (text). Dual-column emb pgvector(960)+embt turbovec."""
import time, io
import numpy as np
import psycopg2

SOCK = "/mnt/nvme/pg"; DB = "vecbench"; TAB = "gist10m_corpus"; DIM = 960


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ubuntu", port=28818)
    cur = c.cursor(); cur.execute("SET search_path = public, turbovec"); c.commit(); cur.close()
    return c


conn = connect(); conn.autocommit = True; cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
cur.execute(f"DROP TABLE IF EXISTS {TAB} CASCADE")
cur.execute(f"CREATE TABLE {TAB} (id int, emb public.vector({DIM}), embt turbovec.vector)")
big = np.load("/mnt/nvme/data/gist10m.npy", mmap_mode="r")
n = big.shape[0]
batch = 100000
t0 = time.time()
for start in range(0, n, batch):
    end = min(start + batch, n)
    chunk = np.asarray(big[start:end], dtype=np.float32)
    buf = io.StringIO()
    for i in range(end - start):
        lit = "[" + ",".join("%.6f" % x for x in chunk[i]) + "]"
        buf.write(f"{start+i}\t{lit}\t{lit}\n")
    buf.seek(0)
    cur.copy_expert(f"COPY {TAB} (id, emb, embt) FROM STDIN", buf)
    if start % 1000000 == 0:
        print(f"  {end}/{n} ({time.time()-t0:.0f}s)", flush=True)
cur.execute(f"SELECT count(*), pg_total_relation_size('{TAB}') FROM {TAB}")
got, sz = cur.fetchone()
conn.close()
print(f"  loaded {got} rows in {time.time()-t0:.0f}s heap={sz/1e9:.2f}GB", flush=True)
print("LOAD10M_DONE", flush=True)
