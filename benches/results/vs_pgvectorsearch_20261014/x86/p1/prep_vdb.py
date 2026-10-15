#!/usr/bin/env python3
"""VDBBench prep -> same .npy + COPY layout as prep_co1024.py so bench.py is unchanged.
Downloads train shards (parallel), builds corpus = first N unique non-zero L2-normalised fp32,
queries = the dataset's own test.parquet (held out), our OWN exact fp64-rescored GT (ignore the
dataset's neighbors.parquet — different norm/metric). Writes 16 COPY tsv files shared by both systems.
usage: prep_vdb.py <dataset cohere_medium_1m|openai_large_5m|cohere_large_10m> <N> <OUTDIR> [DIM]"""
import os, sys, time, json, hashlib, glob, subprocess, numpy as np, pyarrow.parquet as pq
import multiprocessing as mp
DS = sys.argv[1]; N = int(sys.argv[2]); OUT = sys.argv[3]
DIM = int(sys.argv[4]) if len(sys.argv) > 4 else {"cohere_medium_1m":768,"openai_large_5m":1536,"cohere_large_10m":768}[DS]
B = "https://assets.zilliz.com/benchmark"; PARTS = 16
os.makedirs(f"{OUT}/copy", exist_ok=True); RAW = f"/data/vdb/{DS}"; os.makedirs(RAW, exist_ok=True)
def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
t0 = time.time()
# --- discover + download shards in parallel (zilliz ~6 MB/s single stream) ---
def head_ok(url):
    r = subprocess.run(["curl","-sI","--max-time","30",url], capture_output=True, text=True).stdout
    return "200" in r.split("\n")[0]
shards = []
if head_ok(f"{B}/{DS}/train.parquet"):
    shards = ["train.parquet"]
else:
    for i in range(100):
        f = f"train-{i:02d}-of-{'10' if i<10 else '99'}.parquet"
        # try common VDBBench naming train-00-of-10.parquet
        f = f"train-{i:02d}-of-10.parquet"
        if head_ok(f"{B}/{DS}/{f}"): shards.append(f)
        elif i >= 1: break
need = ["test.parquet"] + shards
def dl(f):
    p = f"{RAW}/{f}"
    for _ in range(5):
        subprocess.run(["curl","-sS","-C","-","--retry","5","-o",p,f"{B}/{DS}/{f}"])
        if os.path.exists(p) and os.path.getsize(p) > 0: return f, os.path.getsize(p)
    return f, -1
with mp.get_context("fork").Pool(min(len(need), 12)) as pool:
    dls = pool.map(dl, need)
print(f"downloaded {len(dls)} files {time.time()-t0:.0f}s: {dls}", flush=True)
lock = [dict(dataset=DS, file=f, bytes=os.path.getsize(f"{RAW}/{f}"), md5=md5f(f"{RAW}/{f}")) for f in need]
# --- load corpus from shards in order, dedup, normalise ---
def colname(t): return [c for c in t.column_names if c in ("emb","vector","emb_vector")][0]
corpus = np.empty((N, DIM), np.float32); seen = set(); n = 0
stats = dict(rows_read=0, zero=0, badlen=0, corpus_dup_dropped=0)
for f in shards:
    t = pq.read_table(f"{RAW}/{f}"); col = t.column(colname(t)).combine_chunks()
    lens = col.value_lengths().to_numpy(zero_copy_only=False)
    vals = col.flatten().to_numpy().astype(np.float32); off = 0
    for r in range(len(col)):
        stats["rows_read"] += 1; L = int(lens[r]); v = vals[off:off+L]; off += L
        if L != DIM: stats["badlen"] += 1; continue
        nrm = np.linalg.norm(v)
        if not nrm > 1e-30: stats["zero"] += 1; continue
        u = (v/nrm).astype(np.float32); key = hashlib.blake2b(u.tobytes(), digest_size=16).digest()
        if n < N:
            if key in seen: stats["corpus_dup_dropped"] += 1; continue
            seen.add(key); corpus[n] = u; n += 1
        else: break
    print(f"  {f}: corpus {n}/{N} {time.time()-t0:.0f}s", flush=True)
    if n >= N: break
assert n == N, f"only {n} of {N} corpus rows (need more shards)"
# --- queries from test.parquet (held out), normalise ---
tq = pq.read_table(f"{RAW}/test.parquet"); qcol = tq.column(colname(tq)).combine_chunks()
qlen = qcol.value_lengths().to_numpy(zero_copy_only=False)
qv = qcol.flatten().to_numpy().astype(np.float32); Q = []; off = 0
for r in range(len(qcol)):
    L = int(qlen[r]); v = qv[off:off+L]; off += L
    nrm = np.linalg.norm(v); Q.append((v/nrm).astype(np.float32) if nrm > 1e-30 else v.astype(np.float32))
Q = np.stack(Q).astype(np.float32); H = len(Q)
qseen = {hashlib.blake2b(u.tobytes(), digest_size=16).digest() for u in Q}
stats["query_eq_corpus"] = sum(1 for k in qseen if k in seen)
stats["max_abs_norm_dev_corpus"] = float(np.max(np.abs(np.linalg.norm(corpus.astype(np.float64),axis=1)-1)))
stats["max_abs_norm_dev_queries"] = float(np.max(np.abs(np.linalg.norm(Q.astype(np.float64),axis=1)-1)))
np.save(f"{OUT}/corpus.npy", corpus); np.save(f"{OUT}/queries.npy", Q)
print(f"vectors saved {time.time()-t0:.0f}s corpus{corpus.shape} q{Q.shape}", flush=True)
# --- GT: fp32 sgemm top-200 then fp64 rescore top-100 ---
KC = 200; bs = np.full((H, KC), -np.inf, np.float32); bi = np.zeros((H, KC), np.int64)
for s in range(0, N, 100_000):
    sc = Q @ corpus[s:s+100_000].T
    cs = np.concatenate([bs, sc], 1); ci = np.concatenate([bi, np.broadcast_to(np.arange(s, s+sc.shape[1]), (H, sc.shape[1]))], 1)
    part = np.argpartition(-cs, KC, 1)[:, :KC]; bs = np.take_along_axis(cs, part, 1); bi = np.take_along_axis(ci, part, 1)
gt_i = np.empty((H, KC), np.int64); gt_d = np.empty((H, KC), np.float64); Q64 = Q.astype(np.float64)
for a in range(0, H, 250):
    ids = bi[a:a+250]; d = 1.0 - np.einsum("qd,qkd->qk", Q64[a:a+250], corpus[ids].astype(np.float64))
    o = np.lexsort((ids, d), axis=1); gt_i[a:a+250] = np.take_along_axis(ids, o, 1); gt_d[a:a+250] = np.take_along_axis(d, o, 1)
np.save(f"{OUT}/gt_ids.npy", gt_i[:, :100]); np.save(f"{OUT}/gt_dist.npy", gt_d[:, :100])
gap10 = gt_d[:, 10] - gt_d[:, 9]
stats["tie_ambiguous_k10"] = int((gap10 < 1e-6).sum()); stats["queries_with_dist0_top1"] = int((gt_d[:, 0] < 1e-6).sum())
print(f"GT done {time.time()-t0:.0f}s", flush=True)
# --- COPY tsv, identical bytes both systems ---
def fmt(rows): return "".join(f"{rid}\t[{','.join('%.9g' % x for x in vec.tolist())}]\n" for rid, vec in rows)
def write_part(k):
    lo, hi = k*N//PARTS, (k+1)*N//PARTS
    with open(f"{OUT}/copy/docs_{k:02d}.tsv", "w") as f:
        for b in range(lo, hi, 5000): f.write(fmt(zip(range(b, min(b+5000, hi)), corpus[b:min(b+5000, hi)])))
    return k
with mp.get_context("fork").Pool(PARTS) as pool: list(pool.map(write_part, range(PARTS)))
with open(f"{OUT}/copy/queries.tsv", "w") as f: f.write(fmt(enumerate(Q)))
man = {f: md5f(f"{OUT}/{f}") for f in ["corpus.npy","queries.npy","gt_ids.npy","gt_dist.npy"]}
json.dump(dict(dataset=DS, N=N, H=H, dim=DIM, shards=lock, npy_md5=man, stats=stats), open(f"{OUT}/datasets.lock.json","w"), indent=1)
print(json.dumps(stats)); print(f"done {time.time()-t0:.0f}s", flush=True)
