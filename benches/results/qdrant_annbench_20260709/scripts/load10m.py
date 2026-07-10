#!/usr/bin/env python3
"""Load the 10M semi-synthetic GIST corpus (gist10m.npy) into PG dual-column
(emb public.vector(960) for HNSW, embt turbovec.vector for pg_turbovec)."""
import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

SOCK = "/mnt/nvme/pg"; DB = "vecbench"; TAB = "gist10m_corpus"; DIM = 960


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor(); cur.execute("SET search_path = public, turbovec"); c.commit(); cur.close()
    return c


def vlit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


conn = connect(); conn.autocommit = True; cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
cur.execute(f"DROP TABLE IF EXISTS {TAB} CASCADE")
cur.execute(f"CREATE TABLE {TAB} (id int, emb public.vector({DIM}), embt turbovec.vector)")
big = np.load("/mnt/nvme/data/gist10m.npy", mmap_mode="r")
n = big.shape[0]
batch = 20000
t0 = time.time()
for start in range(0, n, batch):
    end = min(start + batch, n)
    chunk = np.asarray(big[start:end], dtype=np.float32)
    rows = [(start + i, vlit(chunk[i]), vlit(chunk[i])) for i in range(end - start)]
    execute_values(cur, f"INSERT INTO {TAB} (id, emb, embt) VALUES %s",
                   rows, template="(%s, %s::public.vector, %s::turbovec.vector)")
    if start % 1000000 == 0:
        print(f"  {end}/{n} ({time.time()-t0:.0f}s)", flush=True)
cur.execute(f"SELECT count(*), pg_total_relation_size('{TAB}') FROM {TAB}")
got, sz = cur.fetchone()
conn.close()
print(f"  loaded {got} rows in {time.time()-t0:.0f}s heap={sz/1e9:.2f}GB", flush=True)
print("LOAD10M_DONE", flush=True)
