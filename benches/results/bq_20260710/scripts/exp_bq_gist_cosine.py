#!/usr/bin/env python3
"""
GIST-960 under pg_turbovec's ACTUAL metric: unit-normalize + cosine GT, not
the published L2 GT. Checks whether BQ (sign/Hamming ~ angular) does better on
GIST when the target metric is cosine (as pg_turbovec scores it) rather than
L2. Smaller corpus (200k) for speed — the point is the metric axis, not scale.
"""
import sys, time, json
import numpy as np, h5py, faiss
t0=time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)
def rk(found, gt, k):
    return sum(len(set(found[i,:k].tolist())&set(gt[i,:k].tolist()))/k for i in range(found.shape[0]))/found.shape[0]
def pack(x, th): return np.packbits((x>th).astype(np.uint8), axis=1)

n=int(sys.argv[1]) if len(sys.argv)>1 else 200000
nq=int(sys.argv[2]) if len(sys.argv)>2 else 1000
dim=960; faiss.omp_set_num_threads(8)
f=h5py.File("gist-960-euclidean.hdf5","r")
xb=np.asarray(f["train"][:n],dtype=np.float32); xq=np.asarray(f["test"][:nq],dtype=np.float32)
xb/= (np.linalg.norm(xb,axis=1,keepdims=True)+1e-30); xq/=(np.linalg.norm(xq,axis=1,keepdims=True)+1e-30)
log(f"unit-normalized GIST corpus={xb.shape}")
gi=faiss.IndexFlatIP(dim); gi.add(xb); _,gt=gi.search(xq,100); gt=gt.astype(np.int64)
log("cosine top-100 GT computed")
mean=xb.mean(0)
res={"meta":{"corpus":n,"n_query":nq,"dim":dim,"metric":"cosine (unit-normalized, pg_turbovec's metric)",
             "gt":"self-computed exact cosine top100","bytes_per_vec_bq":dim//8,"faiss":faiss.__version__},
     "binarizations":{}}
for mode,th in [("rawsign",0.0),("meancenter",mean)]:
    cb=pack(xb,th); qb=pack(xq,th)
    of=float(np.unpackbits(cb,axis=1)[:,:dim].mean())
    bf=faiss.IndexBinaryFlat(dim); bf.add(cb)
    _,ids=bf.search(qb,10); r10=round(rk(ids,gt,10),4)
    _,ids=bf.search(qb,100); r100=round(rk(ids,gt,100),4)
    rr={}
    for sk in (200,800,2000):
        _,cand=bf.search(qb,sk)
        o10=np.full((nq,10),-1,np.int64); o100=np.full((nq,100),-1,np.int64)
        for i in range(nq):
            c=cand[i][cand[i]>=0]
            if c.size==0: continue
            s=xb[c]@xq[i]; o=c[np.argsort(-s)]
            o10[i,:min(10,o.size)]=o[:10]; o100[i,:min(100,o.size)]=o[:100]
        rr[f"sk{sk}"]={"R@10":round(rk(o10,gt,10),4),"R@100":round(rk(o100,gt,100),4)}
    log(f"  {mode} ones={of:.3f} rawR@10={r10} rawR@100={r100} rr800R@10={rr['sk800']['R@10']} rr800R@100={rr['sk800']['R@100']}")
    res["binarizations"][mode]={"ones_fraction":round(of,4),"binary_flat_raw":{"R@10":r10,"R@100":r100},"binary_flat_rerank":rr}
json.dump(res,open(f"exp_bq_gist_cosine_{n}.json","w"),indent=2); log("wrote json")
