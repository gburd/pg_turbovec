#!/usr/bin/env python3
"""CO1024 prep (VS_BENCH_PLAN § C): first N unique, non-null, non-zero-norm
L2-normalised fp32 vectors of CohereLabs/wikipedia-2023-11-embed-multilingual-v3
`en/` (in shard/row order) = corpus, ids 0..N-1; the next H distinct valid rows
= held-out queries (qid 0..H-1). Exact GT: fp32 BLAS top-200 then fp64 re-score,
top-100 kept (ties broken by id). Writes .npy + md5, datasets.lock, and
16-way text COPY files with '%.9g' (exact float32 round-trip) shared by BOTH systems.
usage: prep_co1024.py N H OUTDIR"""
import os, sys, time, json, hashlib, numpy as np, pyarrow.parquet as pq
import multiprocessing as mp
from huggingface_hub import hf_hub_download
N = int(sys.argv[1]); H = int(sys.argv[2]); OUT = sys.argv[3]
REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"; DIM = 1024; PARTS = 16
os.makedirs(f"{OUT}/copy", exist_ok=True)
def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
t0 = time.time()
corpus = np.empty((N, DIM), np.float32); src = np.empty((N, 2), np.int32)
Q = []; qsrc = []; seen = set(); qseen = set(); n = 0
stats = dict(null=0, badlen=0, zero=0, corpus_dup_dropped=0, query_dup_dropped=0, query_eq_corpus=0, rows_read=0)
lock = []; i = 0; qdupc = []
while len(Q) < H:
    fn = f"en/{i:04d}.parquet"
    p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir="/data/hf")
    lock.append(dict(repo=REPO, file=fn, bytes=os.path.getsize(p), md5=md5f(p)))
    col = pq.read_table(p, columns=["emb"]).column("emb").combine_chunks()
    valid = col.is_valid().to_numpy(zero_copy_only=False)
    lens = col.value_lengths().to_numpy(zero_copy_only=False)
    vals = col.flatten().to_numpy().astype(np.float32)
    off = 0
    for r in range(len(col)):
        stats["rows_read"] += 1
        if not valid[r]: stats["null"] += 1; continue
        L = int(lens[r]); v = vals[off:off + L]; off += L
        if L != DIM: stats["badlen"] += 1; continue
        nrm = np.linalg.norm(v)
        if not nrm > 1e-30: stats["zero"] += 1; continue
        u = (v / nrm).astype(np.float32)
        key = hashlib.blake2b(u.tobytes(), digest_size=16).digest()
        if n < N:
            if key in seen: stats["corpus_dup_dropped"] += 1; continue
            seen.add(key); corpus[n] = u; src[n] = (i, r); n += 1
        else:
            if key in qseen: stats["query_dup_dropped"] += 1; continue
            qseen.add(key)
            if key in seen: stats["query_eq_corpus"] += 1; qdupc.append(len(Q))
            Q.append(u); qsrc.append((i, r))
            if len(Q) >= H: break
    print(f"  shard {i}: corpus {n}/{N} queries {len(Q)}/{H} {time.time()-t0:.0f}s", flush=True)
    i += 1
Q = np.stack(Q).astype(np.float32); qsrc = np.array(qsrc, np.int32)
stats["max_abs_norm_dev_corpus"] = float(np.max(np.abs(np.linalg.norm(corpus.astype(np.float64), axis=1) - 1)))
stats["max_abs_norm_dev_queries"] = float(np.max(np.abs(np.linalg.norm(Q.astype(np.float64), axis=1) - 1)))
stats["query_eq_corpus_qids"] = qdupc
np.save(f"{OUT}/corpus.npy", corpus); np.save(f"{OUT}/queries.npy", Q)
np.save(f"{OUT}/corpus_src.npy", src); np.save(f"{OUT}/queries_src.npy", qsrc)
print(f"vectors saved {time.time()-t0:.0f}s", flush=True)
# GT: fp32 sgemm top-200, then fp64 re-score
KC = 200; bs = np.full((H, KC), -np.inf, np.float32); bi = np.zeros((H, KC), np.int64)
for s in range(0, N, 100_000):
    sc = Q @ corpus[s:s + 100_000].T
    cs = np.concatenate([bs, sc], 1)
    ci = np.concatenate([bi, np.broadcast_to(np.arange(s, s + sc.shape[1]), (H, sc.shape[1]))], 1)
    part = np.argpartition(-cs, KC, 1)[:, :KC]
    bs = np.take_along_axis(cs, part, 1); bi = np.take_along_axis(ci, part, 1)
gt_i = np.empty((H, KC), np.int64); gt_d = np.empty((H, KC), np.float64)
Q64 = Q.astype(np.float64)
for a in range(0, H, 250):
    ids = bi[a:a + 250]
    d = 1.0 - np.einsum("qd,qkd->qk", Q64[a:a + 250], corpus[ids].astype(np.float64))
    o = np.lexsort((ids, d), axis=1)
    gt_i[a:a + 250] = np.take_along_axis(ids, o, 1); gt_d[a:a + 250] = np.take_along_axis(d, o, 1)
np.save(f"{OUT}/gt_ids.npy", gt_i[:, :100]); np.save(f"{OUT}/gt_dist.npy", gt_d[:, :100])
gap10 = gt_d[:, 10] - gt_d[:, 9]; gap100 = gt_d[:, 100] - gt_d[:, 99]
np.save(f"{OUT}/gt_gap10.npy", gap10); np.save(f"{OUT}/gt_gap100.npy", gap100)
stats["tie_ambiguous_k10"] = int((gap10 < 1e-6).sum()); stats["tie_ambiguous_k100"] = int((gap100 < 1e-6).sum())
stats["queries_with_dist0_top1"] = int((gt_d[:, 0] < 1e-6).sum())
print(f"GT done {time.time()-t0:.0f}s", flush=True)
# COPY text files, identical bytes for both systems
def fmt(rows):
    return "".join(f"{rid}\t[{','.join('%.9g' % x for x in vec.tolist())}]\n" for rid, vec in rows)
def write_part(k):
    lo, hi = k * N // PARTS, (k + 1) * N // PARTS
    with open(f"{OUT}/copy/docs_{k:02d}.tsv", "w") as f:
        for b in range(lo, hi, 5000):
            f.write(fmt(zip(range(b, min(b + 5000, hi)), corpus[b:min(b + 5000, hi)])))
    return k
with mp.get_context("fork").Pool(PARTS) as pool: list(pool.map(write_part, range(PARTS)))
with open(f"{OUT}/copy/queries.tsv", "w") as f: f.write(fmt(enumerate(Q)))
print(f"copy files done {time.time()-t0:.0f}s", flush=True)
man = {f: md5f(f"{OUT}/{f}") for f in ["corpus.npy", "queries.npy", "gt_ids.npy", "gt_dist.npy", "corpus_src.npy", "queries_src.npy"]}
json.dump(dict(dataset="CO1024", N=N, H=H, shards=lock, npy_md5=man, stats=stats), open(f"{OUT}/datasets.lock.json", "w"), indent=1)
print(json.dumps(stats)[:2000]); print(f"done {time.time()-t0:.0f}s", flush=True)
