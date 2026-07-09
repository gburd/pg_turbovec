#!/usr/bin/env python3
"""Brute-force exact cosine top-10 ground truth for the Tier-1 query set.

Streams the corpus from PG in chunks (bounds client RAM on a zero-swap box),
keeps a running top-10 per query via argpartition. Vectors are already
L2-normalized at load, so cosine distance ranking == -dot ranking; we verify
norms are ~1 on the first chunk.
"""
import argparse, time
import numpy as np
import psycopg


def fetch_chunk(cur):
    rows = cur.fetchmany(20000)
    if not rows:
        return None, None
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    # emb comes back as text '[a, b, ...]'; parse to float32
    mat = np.array([np.fromstring(r[1][1:-1], sep=",", dtype=np.float32) for r in rows])
    return ids, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--query-npy", default="/tmp/tier1-queries.npy")
    ap.add_argument("--gt-npy", default="/tmp/tier1-gt-top10.npy")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    Q = np.load(args.query_npy).astype(np.float32)   # (nq, d), normalized
    nq = Q.shape[0]
    K = args.k
    # running best: cosine distance = 1 - dot (since normalized). track distances + ids.
    best_d = np.full((nq, K), np.inf, dtype=np.float32)
    best_id = np.full((nq, K), -1, dtype=np.int64)

    t0 = time.perf_counter()
    seen = 0
    with psycopg.connect(args.dsn) as conn, conn.cursor(name="gt_cur") as cur:
        cur.itersize = 20000
        cur.execute("SELECT id, emb::text FROM public.docs ORDER BY id")
        first = True
        while True:
            ids, mat = fetch_chunk(cur)
            if ids is None:
                break
            if first:
                norms = np.linalg.norm(mat, axis=1)
                assert np.allclose(norms, 1.0, atol=1e-3), f"not normalized: {norms[:5]}"
                first = False
            # cosine distance to each query: 1 - Q.mat^T
            dist = 1.0 - Q @ mat.T          # (nq, chunk)
            # merge running top-K with this chunk
            # candidate set: concat running best + chunk; cheap merge via partition
            chunk_ids = np.broadcast_to(ids, dist.shape)
            alld = np.concatenate([best_d, dist], axis=1)
            allids = np.concatenate([best_id, chunk_ids], axis=1)
            part = np.argpartition(alld, K, axis=1)[:, :K]
            best_d = np.take_along_axis(alld, part, axis=1)
            best_id = np.take_along_axis(allids, part, axis=1)
            seen += len(ids)
            if seen % 100000 == 0:
                print(f"  gt scanned {seen} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # final sort each query's top-K by distance ascending
    order = np.argsort(best_d, axis=1)
    gt_ids = np.take_along_axis(best_id, order, axis=1)
    np.save(args.gt_npy, gt_ids)
    print(f"wrote GT {gt_ids.shape} -> {args.gt_npy} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"sample gt[0] = {gt_ids[0].tolist()}", flush=True)


if __name__ == "__main__":
    main()
