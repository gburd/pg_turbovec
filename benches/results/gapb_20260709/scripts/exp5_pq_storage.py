#!/usr/bin/env python3
"""
Gap B experiment 5: PQ at STORAGE-COMPETITIVE m (the an internal design note
gate-0 question). m=32 (36 B/vec) and m=64 (68 B/vec) are the m's that would
beat DiskANN's 585 B/vec. Does PQ at that byte budget preserve enough in-cell
recall to be worth building? Compare vs SQ4 (480 B) and SQ8 (960 B) + rerank.
Shares the same L2 coarse quantizer (lists=1000).
"""
import time, json
import numpy as np, h5py, faiss
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)
def recall(found, gt, k=10):
    return sum(len(set(found[i,:k].tolist()) & set(gt[i,:k].tolist()))/k for i in range(found.shape[0]))/found.shape[0]

f = h5py.File("gist-960-euclidean.hdf5","r")
xb = np.asarray(f["train"][:1000000], dtype=np.float32)
xq = np.asarray(f["test"][:1000], dtype=np.float32)
gt = np.asarray(f["neighbors"][:1000,:10], dtype=np.int64)
K, dim, lists = 10, 960, 1000
faiss.omp_set_num_threads(8)
log(f"loaded xb={xb.shape}")

coarse = faiss.IndexFlatL2(dim)
cq = faiss.IndexIVFFlat(coarse, dim, lists, faiss.METRIC_L2)
cq.train(xb); log("coarse trained")

res = {"meta":{"corpus":1000000,"n_query":1000,"K":K,"dim":dim,"lists":lists,
               "gt":"published-exact-L2-1M","faiss":faiss.__version__,
               "note":"PQ at storage-competitive m (36/68 B/vec) + exact-L2 rerank; "
                      "bytes/vec = m for PQ-256 (1 byte/subq)."},
       "indexes":{}}

def rerank_search(idx, xb, xq, gt, probes, sk, K):
    idx.nprobe = probes
    ts = time.time(); _, cand = idx.search(xq, sk)
    out = np.full((xq.shape[0], K), -1, np.int64)
    for i in range(xq.shape[0]):
        c = cand[i][cand[i] >= 0]
        if c.size == 0: continue
        d = ((xb[c]-xq[i])**2).sum(1); o = c[np.argsort(d)][:K]; out[i,:o.size] = o
    return recall(out, gt, K), 1000*(time.time()-ts)/xq.shape[0]

for m, bpv in [(32,36),(64,68)]:
    pq = faiss.IndexIVFPQ(coarse, dim, lists, m, 8, faiss.METRIC_L2)
    tb=time.time(); pq.train(xb); pq.add(xb); log(f"IVFPQ m={m} ({bpv} B/vec incl scale) built in {time.time()-tb:.1f}s")
    raw={}; rr={}
    for p in [64,128]:
        pq.nprobe=p; ts=time.time(); _,ids=pq.search(xq,K); raw[p]={"recall":round(recall(ids,gt,K),4),"ms":round(1000*(time.time()-ts)/1000,3)}
        log(f"  PQ{m} raw p={p}: R@10={raw[p]['recall']}")
    for p in [128]:
        for sk in [200,800]:
            r,ms=rerank_search(pq,xb,xq,gt,p,sk,K); rr[f"p{p}_sk{sk}"]={"recall":round(r,4),"ms":round(ms,3)}
            log(f"  PQ{m}+rerank p={p} sk={sk}: R@10={r:.4f}")
    res["indexes"][f"IVFPQ{m}"]={"bytes_per_vec_approx":bpv,"raw":raw,"exact_rerank":rr}

json.dump(res, open("exp5_pq_storage_competitive.json","w"), indent=2)
log("wrote exp5_pq_storage_competitive.json")
