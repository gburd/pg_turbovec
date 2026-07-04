#!/usr/bin/env python3
"""A/B/C vector index benchmark harness — shared heap, per-engine index.

One heap table `corpus` holds the SAME rows for all three engines:
  emb    public.vector(dim)     -- pgvector type (engines A, B)
  embt   turbovec.vector(dim)   -- pg_turbovec type (engine C)
Both columns are the identical embedding (loaded from the same HDF5 train
matrix). Query set + exact GT come from the HDF5 test/neighbors datasets.

Only ONE index exists at a time (each engine's build drops the others).
Recall@10 is measured against the dataset's published exact `neighbors`.
"""
import os, sys, time, json, statistics, threading, queue
import numpy as np
import h5py
import psycopg2
from psycopg2.extras import execute_values

SOCK = "/mnt/nvme/pg"
DB = "vecbench"


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor()
    cur.execute("SET search_path = public, turbovec")
    c.commit()
    cur.close()
    return c


def vlit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def load_corpus(h5path, dim, limit=None, normalize=False):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
    cur.execute("DROP TABLE IF EXISTS corpus CASCADE")
    cur.execute(f"CREATE TABLE corpus (id int, emb public.vector({dim}), embt turbovec.vector)")
    with h5py.File(h5path, "r") as h:
        train = h["train"]
        n = train.shape[0] if limit is None else min(limit, train.shape[0])
        batch = 20000
        t0 = time.time()
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = np.asarray(train[start:end], dtype=np.float32)
            if normalize:
                nn = np.linalg.norm(chunk, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                chunk = chunk / nn
            rows = []
            for i in range(end - start):
                lit = vlit(chunk[i])
                rows.append((start + i, lit, lit))
            execute_values(cur,
                "INSERT INTO corpus (id, emb, embt) VALUES %s",
                rows, template="(%s, %s::public.vector, %s::turbovec.vector)")
            if start % 200000 == 0:
                print(f"  loaded {end}/{n} ({time.time()-t0:.0f}s)", flush=True)
    cur.execute("SELECT count(*) FROM corpus")
    got = cur.fetchone()[0]
    cur.execute("SELECT pg_total_relation_size('corpus')")
    heap_sz = cur.fetchone()[0]
    conn.close()
    print(f"  loaded {got} rows in {time.time()-t0:.0f}s heap={heap_sz/1e9:.2f}GB", flush=True)
    return got, heap_sz


def load_queries(h5path, limit=None, normalize=False):
    with h5py.File(h5path, "r") as h:
        test = np.asarray(h["test"][:], dtype=np.float32)
        gt = np.asarray(h["neighbors"][:], dtype=np.int64)
    if normalize:
        nn = np.linalg.norm(test, axis=1, keepdims=True); nn[nn == 0] = 1.0
        test = test / nn
    if limit:
        test = test[:limit]; gt = gt[:limit]
    return test, gt


def drop_all_indexes():
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='corpus'")
    for (nm,) in cur.fetchall():
        cur.execute(f"DROP INDEX IF EXISTS {nm} CASCADE")
    conn.close()


def build_index(create_sql, idxname):
    drop_all_indexes()
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SET max_parallel_maintenance_workers = 32")
    cur.execute("SET maintenance_work_mem = '32GB'")
    t0 = time.time()
    cur.execute(create_sql)
    build_s = time.time() - t0
    cur.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s)", (idxname, idxname))
    rel, tot = cur.fetchone()
    conn.close()
    return {"build_s": round(build_s, 2), "idx_bytes": rel, "idx_total_bytes": tot}


def _recall_pass(cur, col, op, k, test, gt):
    hits = 0; total = 0; lats = []
    for i in range(len(test)):
        q = vlit(test[i])
        t0 = time.perf_counter()
        cur.execute(
            f"SELECT id FROM corpus ORDER BY {col} {op} %s LIMIT {k}", (q,))
        res = [r[0] for r in cur.fetchall()]
        lats.append((time.perf_counter() - t0) * 1000.0)
        truth = set(int(x) for x in gt[i][:k])
        hits += len(truth & set(res)); total += k
    return hits / total, lats


def measure(col, op, k, test, gt, setup_sql, repeats=3, cast=""):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    for s in setup_sql:
        cur.execute(s)
    opq = f"{op} %s{cast}"
    # warm
    for i in range(len(test)):
        cur.execute(f"SELECT id FROM corpus ORDER BY {col} {opq} LIMIT {k}", (vlit(test[i]),))
        cur.fetchall()
    recall = None; best = None
    for r in range(repeats):
        hits = 0; total = 0; lats = []
        for i in range(len(test)):
            q = vlit(test[i])
            t0 = time.perf_counter()
            cur.execute(f"SELECT id FROM corpus ORDER BY {col} {opq} LIMIT {k}", (q,))
            res = [row[0] for row in cur.fetchall()]
            lats.append((time.perf_counter() - t0) * 1000.0)
            if r == 0:
                truth = set(int(x) for x in gt[i][:k])
                hits += len(truth & set(res)); total += k
        if r == 0:
            recall = hits / total
        lats.sort()
        m = statistics.mean(lats)
        cand = {"p50": lats[len(lats)//2], "p95": lats[int(len(lats)*0.95)],
                "mean": m, "qps_1conn": 1000.0/m}
        if best is None or cand["mean"] < best["mean"]:
            best = cand
    conn.close()
    best["recall"] = round(recall, 4)
    for kk in ("p50", "p95", "mean", "qps_1conn"):
        best[kk] = round(best[kk], 3)
    return best


def measure_qps(col, op, k, test, setup_sql, nconn, cast="", duration=8.0):
    """Throughput at N concurrent conns. Each worker loops over queries for
    `duration` seconds; total completed / elapsed = QPS."""
    opq = f"{op} %s{cast}"
    stop = time.time() + duration
    counts = [0] * nconn
    def worker(wid):
        conn = connect(); conn.autocommit = True; cur = conn.cursor()
        for s in setup_sql:
            cur.execute(s)
        idx = wid; c = 0
        while time.time() < stop:
            q = vlit(test[idx % len(test)]); idx += 1
            cur.execute(f"SELECT id FROM corpus ORDER BY {col} {opq} LIMIT {k}", (q,))
            cur.fetchall(); c += 1
        counts[wid] = c
        conn.close()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(nconn)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.time() - t0
    return round(sum(counts) / elapsed, 1)


if __name__ == "__main__":
    print("harness ok")
