#!/usr/bin/env python3
"""Tier-1 search_k sweep: generate + load a deterministic clustered corpus.

Synthetic clustered, L2-normalized vectors. Clustered so IVF cells are
meaningful and recall is non-trivial; normalized for cosine. This measures
the LATENCY-FLOOR mechanism (reorder-recheck candidate count), not absolute
recall quality -- recall@10 vs brute-force GT is still reported so we can
watch it hold as search_k drops.

Streams COPY in chunks to keep client RAM bounded (the host has zero swap).
"""
import argparse, sys, time
import numpy as np
import psycopg


def gen_chunk(rng, centroids, n, noise):
    """n rows: pick a random centroid, add gaussian noise, L2-normalize."""
    c, d = centroids.shape
    idx = rng.integers(0, c, size=n)
    base = centroids[idx]
    pts = base + rng.normal(0.0, noise, size=(n, d)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (pts / norms).astype(np.float32)


def to_copy_rows(start_id, arr):
    """Yield 'id\\t[v1,v2,...]\\n' text rows for COPY."""
    for i, row in enumerate(arr):
        # compact text vector literal; 6 sig figs is plenty for 4-bit codes
        body = ",".join(f"{x:.6g}" for x in row)
        yield f"{start_id + i}\t[{body}]\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--clusters", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--query-npy", default="/tmp/tier1-queries.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # centroids: unit-norm directions so clusters are spread on the sphere
    centroids = rng.normal(0, 1, size=(args.clusters, args.dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    with psycopg.connect(args.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS public.docs CASCADE")
        cur.execute("CREATE TABLE public.docs (id bigint PRIMARY KEY, emb turbovec.vector NOT NULL)")

        t0 = time.perf_counter()
        loaded = 0
        with cur.copy("COPY public.docs (id, emb) FROM STDIN") as cp:
            while loaded < args.rows:
                n = min(args.chunk, args.rows - loaded)
                arr = gen_chunk(rng, centroids, n, args.noise)
                for line in to_copy_rows(loaded, arr):
                    cp.write(line)
                loaded += n
                if loaded % 100_000 == 0 or loaded == args.rows:
                    print(f"  loaded {loaded}/{args.rows} "
                          f"({time.perf_counter()-t0:.1f}s)", flush=True)
        cur.execute("SELECT count(*) FROM public.docs")
        print(f"docs rows = {cur.fetchone()[0]}", flush=True)

    # held-out query set, same distribution
    q = gen_chunk(rng, centroids, args.queries, args.noise)
    np.save(args.query_npy, q)
    print(f"wrote {args.queries} queries -> {args.query_npy} {q.shape}", flush=True)


if __name__ == "__main__":
    main()
