#!/usr/bin/env python3
"""Synthetic corpus for the storage bench: N unit-norm Gaussian vectors per dim
(deterministic seed 20261006+dim) + 200 held-out queries. Storage behaviour
depends only on value size and compressibility, not on embedding content.
Writes /work/fixc/data_<d>.tsv (COPY text: id \t [v,...]) and q_<d>.npy.
usage: gen.py <dim> [N]"""
import sys, numpy as np
d = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000; H = 200
rng = np.random.default_rng(20261006 + d)
A = rng.standard_normal((N + H, d), dtype=np.float32)
A /= np.linalg.norm(A, axis=1, keepdims=True)
np.save(f"/work/fixc/q_{d}.npy", A[N:])
fmt = ",".join(["%.9g"] * d)
with open(f"/work/fixc/data_{d}.tsv", "w") as f:
    for i in range(N):
        f.write(f"{i}\t[" + fmt % tuple(A[i].tolist()) + "]\n")
print("wrote", d, N)
