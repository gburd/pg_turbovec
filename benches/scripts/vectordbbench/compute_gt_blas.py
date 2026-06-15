#!/usr/bin/env python3
"""Brute-force exact top-10 ground truth via BLAS, reconstructing the corpus
exactly as load_wiki_1m.py wrote it (first CORPUS rows of the sorted parquet
shards, L2-normalized). Cosine distance = 1 - cos_sim; for unit vectors the
top-10 by max dot == top-10 by min cosine distance.

Writes gt_top10.npy (CORPUS-relative ids, shape (NQ,10)) and
gt_top10_dist.npy. Cross-checks against q1000.npy queries.
"""
import glob, time, numpy as np, pyarrow.parquet as pq

LOCAL = "/scratch/pg_turbovec-bench/cohere-wiki/en"
DIM, CORPUS = 1024, 1_000_000
QPATH = "/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy"
OUT_IDS = "/scratch/pg_turbovec-bench/cohere-wiki/gt_top10.npy"
OUT_DIST = "/scratch/pg_turbovec-bench/cohere-wiki/gt_top10_dist.npy"

def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)

t0 = time.time()
shards = sorted(glob.glob(f"{LOCAL}/*.parquet"))
parts, n = [], 0
for s in shards:
    if n >= CORPUS:
        break
    a = pq.read_table(s, columns=["emb"]).column("emb").combine_chunks().values \
          .to_numpy(zero_copy_only=True).reshape(-1, DIM)
    take = min(a.shape[0], CORPUS - n)
    parts.append(a[:take].astype(np.float32, copy=True)); n += take
corpus = unit(np.concatenate(parts, axis=0))
print(f"corpus {corpus.shape} loaded in {time.time()-t0:.0f}s", flush=True)

Q = unit(np.load(QPATH).astype(np.float32))
print(f"queries {Q.shape}", flush=True)

# Chunked: sims = Q @ corpus.T  (NQ x CORPUS). NQ=1000, CORPUS=1M -> 1000*1M*4 = 4GB.
# Process in column chunks to bound memory, keep running top-10.
NQ = Q.shape[0]
best_sim = np.full((NQ, 10), -2.0, dtype=np.float32)
best_idx = np.full((NQ, 10), -1, dtype=np.int64)
CHUNK = 100_000
t1 = time.time()
for c0 in range(0, CORPUS, CHUNK):
    c1 = min(c0 + CHUNK, CORPUS)
    sims = Q @ corpus[c0:c1].T                       # NQ x chunk
    # merge chunk's top-10 with running best
    k = min(10, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    part_sim = np.take_along_axis(sims, part, axis=1)
    part_idx = part + c0
    cat_sim = np.concatenate([best_sim, part_sim], axis=1)
    cat_idx = np.concatenate([best_idx, part_idx], axis=1)
    order = np.argsort(-cat_sim, axis=1)[:, :10]
    best_sim = np.take_along_axis(cat_sim, order, axis=1)
    best_idx = np.take_along_axis(cat_idx, order, axis=1)
print(f"top-10 done in {time.time()-t1:.0f}s", flush=True)

gt_dist = (1.0 - best_sim).astype(np.float64)        # cosine distance
np.save(OUT_IDS, best_idx)
np.save(OUT_DIST, gt_dist)
print(f"saved {OUT_IDS} {best_idx.shape}", flush=True)
print("q0 top-10 ids:", best_idx[0].tolist())
print("q0 top-10 dist:", gt_dist[0].round(4).tolist())
# sanity: each query's top-10 distinct
assert all(len(set(best_idx[i])) == 10 for i in range(NQ)), "non-distinct gt"
print("all gt rows distinct: OK")
