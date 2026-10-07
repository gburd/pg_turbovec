#!/usr/bin/env python3
"""Step-5 A/B corpus: first 500k Cohere wiki-en multilingual-v3 (1024-d) rows +
200 held-out queries; exact top-10 by dot (unit vectors). Writes
/work/corpus/{base.f32,q.npy,gt.npy}. Same corpus family as v2.11.0 runs."""
import os, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
N, H = 500_000, 200; os.makedirs("/work/corpus", exist_ok=True)
rows = []; got = 0; i = 0
while got < N + H:
    p = hf_hub_download("CohereLabs/wikipedia-2023-11-embed-multilingual-v3", f"en/{i:04d}.parquet", repo_type="dataset", local_dir="/work/hf")
    a = pq.read_table(p, columns=["emb"]).column("emb").combine_chunks().values.to_numpy().reshape(-1, 1024).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30); rows.append(a); got += len(a); i += 1
A = np.concatenate(rows)[:N + H]; Q = A[N:]; A = A[:N]
A.tofile("/work/corpus/base.f32"); np.save("/work/corpus/q.npy", Q)
G = np.empty((H, 10), np.int64)
for s in range(0, H, 50):
    G[s:s+50] = np.argsort(-(Q[s:s+50] @ A.T), axis=1)[:, :10]
np.save("/work/corpus/gt.npy", G); print("corpus", A.shape, Q.shape)
