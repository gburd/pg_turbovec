#!/usr/bin/env python3
"""Load a corpus into a shared-heap table with BOTH emb (pgvector) and
embt (turbovec) columns, via COPY (fast). One table per corpus so HNSW
and IVF benchmark the identical rows.

Usage:
  load_corpus.py gist1m
  load_corpus.py sift1m
  load_corpus.py syn5m
"""
import sys, time, io, subprocess, os
import numpy as np

PSQL = "/mnt/nvme/pg/install/bin/psql"
SOCK = "/mnt/nvme/pg"
DB = "vecbench"

SPEC = {
    "gist1m": ("gist_corpus", 960, "/mnt/nvme/data/gist-960-euclidean.hdf5", None),
    "sift1m": ("sift_corpus", 128, "/mnt/nvme/data/sift-128-euclidean.hdf5", None),
    "syn5m":  ("syn_corpus",  768, "/mnt/nvme/data/syn5m_train.npy", 5_000_000),
}


def psql(sql, db=DB):
    return subprocess.run([PSQL, "-h", SOCK, "-U", "ec2-user", "-d", db,
                           "-v", "ON_ERROR_STOP=1", "-c", sql],
                          check=True, capture_output=True, text=True).stdout


def ensure_db():
    subprocess.run([PSQL, "-h", SOCK, "-U", "ec2-user", "-d", "postgres",
                    "-c", f"CREATE DATABASE {DB}"], capture_output=True, text=True)
    psql("CREATE EXTENSION IF NOT EXISTS vector")
    psql("CREATE EXTENSION IF NOT EXISTS pg_turbovec")


def get_train(corpus):
    tab, dim, path, limit = SPEC[corpus]
    if path.endswith(".hdf5"):
        import h5py
        h = h5py.File(path, "r")
        return h["train"], dim, tab, limit
    else:
        arr = np.load(path, mmap_mode="r")
        return arr, dim, tab, limit


def main(corpus):
    ensure_db()
    train, dim, tab, limit = get_train(corpus)
    n = train.shape[0] if limit is None else min(limit, train.shape[0])
    print(f"loading {corpus}: {n} x {dim} into {tab}", flush=True)
    psql(f"DROP TABLE IF EXISTS {tab} CASCADE")
    psql(f"CREATE TABLE {tab} (id int, emb vector({dim}), embt turbovec.vector)")

    t0 = time.time()
    BATCH = 20000
    proc = subprocess.Popen(
        [PSQL, "-h", SOCK, "-U", "ec2-user", "-d", DB, "-v", "ON_ERROR_STOP=1",
         "-c", f"COPY {tab} (id, emb, embt) FROM STDIN"],
        stdin=subprocess.PIPE, text=True)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        chunk = np.asarray(train[start:end], dtype=np.float32)
        buf = io.StringIO()
        for i in range(end - start):
            lit = "[" + ",".join(f"{x:.6f}" for x in chunk[i]) + "]"
            buf.write(f"{start+i}\t{lit}\t{lit}\n")
        proc.stdin.write(buf.getvalue())
        if start % 200000 == 0:
            print(f"  {end}/{n} {time.time()-t0:.0f}s", flush=True)
    proc.stdin.close()
    rc = proc.wait()
    got = psql(f"SELECT count(*) FROM {tab}").strip()
    sz = psql(f"SELECT pg_size_pretty(pg_total_relation_size('{tab}'))").strip()
    print(f"COPY rc={rc} loaded={got} heap={sz} in {time.time()-t0:.0f}s", flush=True)
    print("LOAD_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
