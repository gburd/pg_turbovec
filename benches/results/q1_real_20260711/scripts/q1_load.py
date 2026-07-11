#!/usr/bin/env python3
"""Q-1 loader v2: download HF parquet embedding shards, load into a
turbovec.vector corpus table, hold out N queries, compute EXACT top-100 L2
ground truth INLINE from the in-memory numpy chunks (no PG text round-trip,
so it scales to 10M). Writes queries+GT to .npy.

Usage:
  q1_load.py <hf_repo> <emb_col> <dim> <table> <max_rows> <n_queries> \
             [normalize] [prefix=en/] [pershard=100000]
"""
import sys, os, time
import numpy as np
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_values
from huggingface_hub import hf_hub_download, list_repo_files

SOCK = "/mnt/nvme/pg"; DB = "vecbench"; DATADIR = "/mnt/nvme/data"


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor(); cur.execute("SET search_path=public,turbovec"); c.commit(); cur.close()
    return c


def vlit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def download_shards(repo, subdir, max_rows, per_shard, prefix):
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    files.sort()
    need = max(1, (max_rows + per_shard - 1) // per_shard) + 1
    files = files[:need]
    local = os.path.join(DATADIR, subdir)
    paths = []
    for f in files:
        paths.append(hf_hub_download(repo, f, repo_type="dataset", local_dir=local))
        print(f"  dl {f}", flush=True)
    return paths


def load(repo, emb_col, dim, table, max_rows, n_queries, normalize, prefix, per_shard):
    paths = download_shards(repo, table + "_pq", max_rows, per_shard, prefix)
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    cur.execute(f"CREATE TABLE {table} (id int, embt turbovec.vector)")
    t0 = time.time(); loaded = 0
    queries = []; query_ids = []
    BUF = 200; K = 100
    Qmat = None; best_d = None; best_i = None; pending = []

    def gt_accum(ids_arr, mat):
        nonlocal best_d, best_i
        d = (mat * mat).sum(1)[None, :] - 2.0 * (Qmat @ mat.T)
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(ids_arr, (Qmat.shape[0], len(ids_arr)))], axis=1)
        part = np.argpartition(cand_d, BUF, axis=1)[:, :BUF]
        best_d = np.take_along_axis(cand_d, part, axis=1)
        best_i = np.take_along_axis(cand_i, part, axis=1)

    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            chunk = np.asarray(pf.read_row_group(rg, columns=[emb_col]).column(emb_col).to_pylist(),
                               dtype=np.float32)
            if normalize:
                nn = np.linalg.norm(chunk, axis=1, keepdims=True); nn[nn == 0] = 1.0
                chunk = chunk / nn
            take = chunk if loaded + chunk.shape[0] <= max_rows else chunk[:max_rows - loaded]
            start = loaded
            for i in range(take.shape[0]):
                if len(queries) < n_queries:
                    queries.append(take[i]); query_ids.append(start + i)
            rows = [(start + i, vlit(take[i])) for i in range(take.shape[0])]
            execute_values(cur, f"INSERT INTO {table} (id, embt) VALUES %s",
                           rows, template="(%s, %s::turbovec.vector)", page_size=5000)
            loaded += take.shape[0]
            ids_arr = np.arange(start, start + take.shape[0], dtype=np.int64)
            if Qmat is None:
                pending.append((ids_arr, take))
                if len(queries) >= n_queries:
                    Qmat = np.asarray(queries, dtype=np.float32)
                    best_d = np.full((n_queries, BUF), np.inf, dtype=np.float32)
                    best_i = np.full((n_queries, BUF), -1, dtype=np.int64)
                    for pa, pm in pending:
                        gt_accum(pa, pm)
                    pending = []
            else:
                gt_accum(ids_arr, take)
            if loaded % 500000 < take.shape[0]:
                print(f"  loaded {loaded}/{max_rows} ({time.time()-t0:.0f}s)", flush=True)
            if loaded >= max_rows:
                break
        if loaded >= max_rows:
            break
    cur.execute(f"SELECT count(*), pg_total_relation_size('{table}') FROM {table}")
    got, heap = cur.fetchone()
    print(f"  {table}: loaded {got} rows in {time.time()-t0:.0f}s heap={heap/1e9:.2f}GB dim={dim}", flush=True)
    conn.close()

    print(f"  finalizing exact top-100 GT for {n_queries} queries...", flush=True)
    qids = np.asarray(query_ids, dtype=np.int64)
    order = np.argsort(best_d, axis=1)
    gt_sorted = np.take_along_axis(best_i, order, axis=1)
    gt_out = np.full((n_queries, K), -1, dtype=np.int64)
    for r in range(n_queries):
        row = [x for x in gt_sorted[r] if x != qids[r] and x >= 0][:K]
        gt_out[r, :len(row)] = row
    np.save(f"{DATADIR}/{table}_queries.npy", Qmat)
    np.save(f"{DATADIR}/{table}_gt.npy", gt_out)
    valid = (gt_out >= 0).sum(1)
    print(f"  GT done: valid/row min/med/max {valid.min()}/{int(np.median(valid))}/{valid.max()}", flush=True)
    print("LOAD_DONE", flush=True)


if __name__ == "__main__":
    repo, emb_col, dim, table, max_rows, n_queries = sys.argv[1:7]
    rest = sys.argv[7:]
    normalize = "normalize" in rest
    prefix = next((a.split("=", 1)[1] for a in rest if a.startswith("prefix=")), None)
    per_shard = next((int(a.split("=", 1)[1]) for a in rest if a.startswith("pershard=")), 100000)
    load(repo, emb_col, int(dim), table, int(max_rows), int(n_queries), normalize, prefix, per_shard)
