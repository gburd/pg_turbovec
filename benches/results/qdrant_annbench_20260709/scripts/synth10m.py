#!/usr/bin/env python3
"""Synthesize a semi-synthetic 10M x 960 corpus from GIST-1M by tiling
10x with small per-copy Gaussian perturbation, then compute EXACT top-10
L2 ground truth for the 1000 GIST test queries against the full 10M.

Clearly LABELLED semi-synthetic: copy 0 is the verbatim GIST-1M train;
copies 1..9 add N(0, sigma^2) noise with sigma = 0.01 * median-per-coord-std,
small enough to keep vectors realistic (near their originals) but distinct.

Writes:
  /mnt/nvme/data/gist10m.npy       (memmap-able float32 10M x 960)
  /mnt/nvme/data/gist10m_test.npy  (1000 x 960, == GIST test queries)
  /mnt/nvme/data/gist10m_gt.npy    (1000 x 10 int64 exact NN ids in [0,10M))
"""
import time
import numpy as np
import h5py

SRC = "/mnt/nvme/data/gist-960-euclidean.hdf5"
OUT = "/mnt/nvme/data/gist10m.npy"
NCOPY = 10
QCAP = 1000
K = 10

t0 = time.time()
with h5py.File(SRC, "r") as h:
    train = np.asarray(h["train"][:], dtype=np.float32)   # 1M x 960
    test = np.asarray(h["test"][:QCAP], dtype=np.float32)  # 1000 x 960
n, dim = train.shape
N = n * NCOPY
print(f"base {n}x{dim} -> {N}x{dim}", flush=True)

# At full 1M base density GIST's true 1-NN gap is ~1.15, so a small
# per-copy displacement makes all 10 copies of the true NN closer than
# any DIFFERENT base -> degenerate GT (top-10 = 10 copies of one point).
# Displacement ~1.5 (sigma = 1.5/sqrt(dim)) makes each copy a genuinely
# distinct manifold point whose 10-NN spans 10 different base identities
# (verified empirically), so recall measures real ANN behaviour, not
# duplicate retrieval. This is a DENSER re-sampling of the GIST manifold,
# clearly semi-synthetic.
sigma = 1.5 / (dim ** 0.5)
print(f"perturbation sigma={sigma:.5f} (per-vec displacement ~1.5 vs 1-NN ~1.15)", flush=True)

# write the big corpus to a memmap so we never hold >38GB + GT buffers at once
big = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(N, dim))
rng = np.random.default_rng(1234)
for c in range(NCOPY):
    if c == 0:
        big[0:n] = train
    else:
        noise = rng.normal(0.0, sigma, size=train.shape).astype(np.float32)
        big[c*n:(c+1)*n] = train + noise
    if c % 2 == 0:
        print(f"  tiled copy {c} ({time.time()-t0:.0f}s)", flush=True)
big.flush()
print(f"corpus written ({time.time()-t0:.0f}s)", flush=True)

# exact top-10 GT: blocked over the corpus to bound memory.
# dist^2(q, x) = |q|^2 + |x|^2 - 2 q.x ; rank by (|x|^2 - 2 q.x) per query.
np.save("/mnt/nvme/data/gist10m_test.npy", test)
qn = test.astype(np.float32)                       # 1000 x 960
best_d = np.full((QCAP, K), np.inf, dtype=np.float32)
best_i = np.full((QCAP, K), -1, dtype=np.int64)
BLK = 500_000
big = np.load(OUT, mmap_mode="r")
for start in range(0, N, BLK):
    end = min(start + BLK, N)
    xb = np.asarray(big[start:end], dtype=np.float32)   # BLK x 960
    xsq = np.einsum("ij,ij->i", xb, xb)                 # BLK
    # cross term: qn (1000x960) @ xb.T (960xBLK) -> 1000 x BLK
    cross = qn @ xb.T
    d = xsq[None, :] - 2.0 * cross                       # 1000 x BLK (rank-equiv to full L2^2)
    # merge this block's top-K into running best
    for qi in range(QCAP):
        row = d[qi]
        # candidate top-K within block
        if row.shape[0] > K:
            part = np.argpartition(row, K)[:K]
        else:
            part = np.arange(row.shape[0])
        cand_d = row[part]; cand_i = part + start
        # merge with running
        md = np.concatenate([best_d[qi], cand_d])
        mi = np.concatenate([best_i[qi], cand_i])
        order = np.argpartition(md, K)[:K]
        so = order[np.argsort(md[order])]
        best_d[qi] = md[so]; best_i[qi] = mi[so]
    print(f"  GT block {end}/{N} ({time.time()-t0:.0f}s)", flush=True)

np.save("/mnt/nvme/data/gist10m_gt.npy", best_i)
print(f"GT written. sample row0 ids={best_i[0][:5]}", flush=True)
print("SYNTH_10M_DONE", flush=True)
