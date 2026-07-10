#!/usr/bin/env python3
"""Load SIFT-1M + GIST-960 1M into dual-column corpus tables.
emb public.vector(dim) (pgvector HNSW), embt turbovec.vector (pg_turbovec).
Only vector + pg_turbovec extensions (no vectorscale/vchord for this run).
GT from HDF5 test/neighbors at query time (g0_driver.load_gt)."""
import sys, time
import numpy as np
import h5py
import psycopg2
from psycopg2.extras import execute_values

SOCK = "/mnt/nvme/pg"
DB = "vecbench"
SPECS = {
    "sift1m": ("/mnt/nvme/data/sift-128-euclidean.hdf5", "sift_corpus", 128),
    "gist1m": ("/mnt/nvme/data/gist-960-euclidean.hdf5", "gist_corpus", 960),
}


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor(); cur.execute("SET search_path = public, turbovec")
    c.commit(); cur.close(); return c


def vlit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def load(corpus):
    h5, tab, dim = SPECS[corpus]
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
    cur.execute(f"DROP TABLE IF EXISTS {tab} CASCADE")
    cur.execute(f"CREATE TABLE {tab} (id int, emb public.vector({dim}), embt turbovec.vector)")
    t0 = time.time()
    with h5py.File(h5, "r") as h:
        train = h["train"]; n = train.shape[0]
        batch = 20000
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = np.asarray(train[start:end], dtype=np.float32)
            rows = []
            for i in range(end - start):
                lit = vlit(chunk[i]); rows.append((start + i, lit, lit))
            execute_values(cur,
                f"INSERT INTO {tab} (id, emb, embt) VALUES %s",
                rows, template="(%s, %s::public.vector, %s::turbovec.vector)")
            if start % 200000 == 0:
                print(f"  {corpus}: {end}/{n} ({time.time()-t0:.0f}s)", flush=True)
    cur.execute(f"SELECT count(*), pg_total_relation_size('{tab}') FROM {tab}")
    got, sz = cur.fetchone()
    conn.close()
    print(f"  {corpus}: loaded {got} rows in {time.time()-t0:.0f}s heap={sz/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["sift1m", "gist1m"]):
        load(c)
    print("LOAD_DONE", flush=True)
