#!/usr/bin/env python3
"""Generate a 5M x 768 CLUSTERED synthetic corpus (deterministic) + 1000
held-out queries + exact top-100 L2 GT via chunked numpy brute force.

Cluster structure so IVF is meaningful (uniform random caps recall). Each
point = a random cluster center + gaussian jitter, then optionally left raw
(NOT normalized: GIST/SIFT are raw L2; we keep raw to match euclidean GT).

Writes:
  /mnt/nvme/data/syn5m_train.npy  (5M x 768 float32)
  /mnt/nvme/data/syn5m_test.npy   (1000 x 768 float32)
  /mnt/nvme/data/syn5m_gt.npy     (1000 x 100 int64)
"""
import numpy as np, time, sys

N = 5_000_000
DIM = 768
NQ = 1000
NCLUST = 5000          # ~1000 pts/cluster -> real IVF-friendly structure
K = 100
SEED = 1234
OUT = "/mnt/nvme/data"

def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    # cluster centers spread out; jitter smaller than inter-center distance
    centers = rng.standard_normal((NCLUST, DIM)).astype(np.float32) * 6.0
    print(f"centers {centers.shape} {time.time()-t0:.0f}s", flush=True)

    train = np.empty((N, DIM), dtype=np.float32)
    assign = rng.integers(0, NCLUST, size=N)
    BATCH = 500_000
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        jitter = rng.standard_normal((e - s, DIM)).astype(np.float32) * 1.0
        train[s:e] = centers[assign[s:e]] + jitter
        print(f"  train {e}/{N} {time.time()-t0:.0f}s", flush=True)
    np.save(f"{OUT}/syn5m_train.npy", train)
    print(f"train saved {time.time()-t0:.0f}s", flush=True)

    # queries: near clusters too (realistic), held out (fresh jitter)
    qc = rng.integers(0, NCLUST, size=NQ)
    test = (centers[qc] + rng.standard_normal((NQ, DIM)).astype(np.float32) * 1.0).astype(np.float32)
    np.save(f"{OUT}/syn5m_test.npy", test)

    # exact top-K L2 GT by chunked brute force
    # dist^2 = |q|^2 - 2 q.x + |x|^2 ; rank by q.x adjusted with |x|^2
    xnorm = np.einsum("ij,ij->i", train, train)  # |x|^2, (N,)
    gt = np.empty((NQ, K), dtype=np.int64)
    QB = 200
    CB = 1_000_000
    for qs in range(0, NQ, QB):
        qe = min(qs + QB, NQ)
        q = test[qs:qe]                            # (qb, DIM)
        # keep a running top-K via partial sort over chunks
        best_d = np.full((qe - qs, K), np.inf, dtype=np.float32)
        best_i = np.full((qe - qs, K), -1, dtype=np.int64)
        for cs in range(0, N, CB):
            ce = min(cs + CB, N)
            # d^2 = xnorm[c] - 2 q.x   (drop |q|^2, constant per query)
            dots = q @ train[cs:ce].T               # (qb, cb)
            d = xnorm[cs:ce][None, :] - 2.0 * dots   # (qb, cb) monotone in true d^2
            # merge with current best
            cand_d = np.concatenate([best_d, d], axis=1)
            cand_i = np.concatenate([best_i, np.broadcast_to(np.arange(cs, ce), (qe-qs, ce-cs))], axis=1)
            part = np.argpartition(cand_d, K, axis=1)[:, :K]
            best_d = np.take_along_axis(cand_d, part, axis=1)
            best_i = np.take_along_axis(cand_i, part, axis=1)
        # final sort of the K
        order = np.argsort(best_d, axis=1)
        gt[qs:qe] = np.take_along_axis(best_i, order, axis=1)
        print(f"  GT {qe}/{NQ} {time.time()-t0:.0f}s", flush=True)
    np.save(f"{OUT}/syn5m_gt.npy", gt)
    print(f"SYN5M_DONE {time.time()-t0:.0f}s train={train.shape} test={test.shape} gt={gt.shape}", flush=True)

if __name__ == "__main__":
    main()
