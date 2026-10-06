#!/usr/bin/env python3
"""Load Cohere wiki-en (multilingual-v3, 1024-d) into public.docs(id bigint,
tv turbovec.vector) via text COPY; hold out H queries; exact cosine top-100 GT
computed in numpy (vectors L2-normalized so cosine == dot). Same corpus as
benches/results/rebench_20260925 (1M x 1024-d)."""
import os, sys, time, subprocess, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
N = int(sys.argv[1]); H = int(sys.argv[2]); DB = sys.argv[3]
REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"; DIM = 1024
OUT = f"/mnt/nvme/corpus_{N}"; os.makedirs(OUT, exist_ok=True)
need = N + H; shards = []; rows = 0; i = 0
while rows < need:
    p = hf_hub_download(REPO, f"en/{i:04d}.parquet", repo_type="dataset", local_dir="/mnt/nvme/hf")
    shards.append(p); rows += pq.ParquetFile(p).metadata.num_rows; i += 1
print(f"{len(shards)} shards, {rows} rows", flush=True)
psql = subprocess.Popen(["psql", "-d", DB, "-v", "ON_ERROR_STOP=1", "-c",
    "\\copy public.docs (id, tv) FROM STDIN"], stdin=subprocess.PIPE, text=True, bufsize=1 << 20)
corpus = np.empty((N, DIM), dtype=np.float32); held = []; rid = 0; t0 = time.time()
for p in shards:
    a = pq.read_table(p, columns=["emb"]).column("emb").combine_chunks().values.to_numpy().reshape(-1, DIM).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)
    for v in a:
        if rid < N:
            corpus[rid] = v
            psql.stdin.write(f"{rid}\t[{','.join(f'{x:.6f}' for x in v)}]\n")
        elif rid < need:
            held.append(v)
        rid += 1
        if rid >= need: break
    print(f"  {rid}/{need} {time.time()-t0:.0f}s", flush=True)
    if rid >= need: break
psql.stdin.close(); assert psql.wait() == 0
Q = np.stack(held)
np.save(f"{OUT}/q.npy", Q)
# Exact top-100 by dot (== cosine on unit vectors), blocked over the corpus.
K = 100; best_s = np.full((H, K), -np.inf, np.float32); best_i = np.zeros((H, K), np.int64)
for s in range(0, N, 100_000):
    sc = Q @ corpus[s:s+100_000].T
    cs = np.concatenate([best_s, sc], 1); ci = np.concatenate([best_i, np.arange(s, s+sc.shape[1])[None, :].repeat(H, 0)], 1)
    part = np.argpartition(-cs, K, 1)[:, :K]
    best_s = np.take_along_axis(cs, part, 1); best_i = np.take_along_axis(ci, part, 1)
o = np.argsort(-best_s, 1); best_i = np.take_along_axis(best_i, o, 1)
np.save(f"{OUT}/gt100.npy", best_i)
print(f"done N={N} H={H} {time.time()-t0:.0f}s", flush=True)
