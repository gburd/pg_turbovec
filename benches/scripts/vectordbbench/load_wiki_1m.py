#!/usr/bin/env python3
"""Load N corpus rows + H held-out query rows from the Cohere-wiki parquet
shards into public.docs (pgvector `vector` column), via binary COPY pipe.

Corpus ids: 0..N-1. Held-out query ids: N..N+H-1 (NOT in the index).
Embeddings are L2-normalized (cosine corpus). Saves held-out queries +
their ids to .npy for ground-truth / sweep.
"""
import argparse, glob, os, struct, subprocess, sys, time
import numpy as np
import pyarrow.parquet as pq

LOCAL = "/scratch/pg_turbovec-bench/cohere-wiki/en"
DIM = 1024
BATCH = 8000

COPY_HEADER  = b"PGCOPY\n\xff\r\n\0" + struct.pack(">II", 0, 0)
COPY_TRAILER = struct.pack(">h", -1)

REC_DTYPE = np.dtype([
    ('nfields',  '>i2'), ('id_len', '>i4'), ('id_val', '>i8'),
    ('vec_len',  '>i4'), ('vec_dim', '>i2'), ('vec_pad', '>i2'),
    ('vec_data', f'>{DIM}f4'),
])

def normalize_inplace(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    np.maximum(n, 1e-30, out=n)
    v /= n

def write_rows(fp, ids, vecs):
    """vecs: float32 (n, DIM), already normalized. ids: int64 (n,)."""
    n = vecs.shape[0]
    rec = np.zeros(n, dtype=REC_DTYPE)
    rec['nfields'] = 2; rec['id_len'] = 8
    rec['vec_len'] = 4 + DIM * 4; rec['vec_dim'] = DIM; rec['vec_pad'] = 0
    rec['id_val'] = ids.astype('>i8')
    rec['vec_data'] = vecs.byteswap().view('>f4')
    fp.write(rec.tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=1_000_000)
    ap.add_argument("--held", type=int, default=1000)
    ap.add_argument("--db", default="bench_wiki")
    ap.add_argument("--port", default="28815")
    ap.add_argument("--host", default="/scratch/pg_turbovec-bench")
    ap.add_argument("--out-queries", default="/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy")
    ap.add_argument("--out-ids",     default="/scratch/pg_turbovec-bench/cohere-wiki/q1000_ids.npy")
    args = ap.parse_args()

    need = args.corpus + args.held
    shards = sorted(glob.glob(os.path.join(LOCAL, "*.parquet")))
    print(f"found {len(shards)} parquet shards; need {need:,} rows", flush=True)

    subprocess.run(
        ["psql", "-h", args.host, "-p", args.port, "-d", args.db,
         "-c", "TRUNCATE TABLE public.docs", "-c", "SET synchronous_commit = off"],
        check=True, stdout=subprocess.DEVNULL)

    p = subprocess.Popen(
        ["psql", "-h", args.host, "-p", args.port, "-d", args.db,
         "-v", "ON_ERROR_STOP=1", "-c", "SET synchronous_commit = off",
         "-c", "\\copy public.docs (id, emb) FROM STDIN WITH (FORMAT BINARY)"],
        stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr, bufsize=0)
    fp = p.stdin
    fp.write(COPY_HEADER)

    rid = 0
    held_vecs = []
    t0 = last = time.time()
    for shard in shards:
        if rid >= need:
            break
        tbl = pq.read_table(shard, columns=["emb"])
        arr = tbl.column("emb").combine_chunks().values.to_numpy(zero_copy_only=True).reshape(-1, DIM).copy()
        normalize_inplace(arr)
        for b in range(0, arr.shape[0], BATCH):
            if rid >= need:
                break
            chunk = arr[b:b+BATCH]
            take = min(chunk.shape[0], need - rid)
            chunk = chunk[:take]
            ids = np.arange(rid, rid + take, dtype=np.int64)
            corpus_mask = ids < args.corpus
            if corpus_mask.any():
                write_rows(fp, ids[corpus_mask], chunk[corpus_mask])
            held_mask = ~corpus_mask
            if held_mask.any():
                held_vecs.append(chunk[held_mask].copy())
            rid += take
            now = time.time()
            if now - last > 10:
                el = now - t0
                print(f"  rid={rid:>10,}/{need:,} {el:5.0f}s {rid/el:7.0f} r/s", flush=True)
                last = now

    fp.write(COPY_TRAILER); fp.close()
    rc = p.wait()
    print(f"psql exited rc={rc}", flush=True)

    held = np.concatenate(held_vecs, axis=0)[:args.held]
    held_ids = np.arange(args.corpus, args.corpus + held.shape[0], dtype=np.int64)
    np.save(args.out_queries, held)
    np.save(args.out_ids, held_ids)
    print(f"saved {held.shape[0]} held-out queries (ids {held_ids[0]}..{held_ids[-1]})", flush=True)
    print(f"done: corpus={args.corpus:,} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(rc)

if __name__ == "__main__":
    main()
