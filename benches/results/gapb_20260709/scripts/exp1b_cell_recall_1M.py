#!/usr/bin/env python3
"""
Gap B experiment 1b: cell recall at FULL 1M scale using the published
exact-L2 GT (valid over the full corpus). Memory-careful: never holds the
angular and L2 corpora simultaneously.

Same definition as exp1: cell recall = fraction of true L2 top-10 whose
row lands in the top-`probes` cells the coarse quantizer picks.
ANGULAR = pg_turbovec's partition (unit-normalize then cluster).
L2 = plain FAISS/pgvector/vchord IVF partition (cluster raw vectors).
"""
import sys, time, json, gc
import numpy as np
import h5py

RNG = np.random.default_rng(42)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def train_kmeans(sample, lists, seed=42, iters=15):
    X = sample
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], lists, replace=False)].copy()
    for _ in range(iters):
        cn = (C*C).sum(1)
        asg = np.empty(X.shape[0], np.int32)
        for ci in range(0, X.shape[0], 10000):
            xb = X[ci:ci+10000]
            d = (xb*xb).sum(1)[:,None] + cn[None,:] - 2.0*(xb @ C.T)
            asg[ci:ci+10000] = d.argmin(1)
        newC = np.zeros_like(C); cnt = np.zeros(lists, np.int64)
        np.add.at(newC, asg, X); np.add.at(cnt, asg, 1)
        nz = cnt > 0
        newC[nz] /= cnt[nz][:,None]
        empty = np.where(~nz)[0]
        if len(empty):
            newC[empty] = X[rng.choice(X.shape[0], len(empty), replace=False)]
        C = newC.astype(np.float32)
    return C

def assign_all(corpus, centroids, chunk=8000):
    n = corpus.shape[0]
    cn = (centroids*centroids).sum(1)
    asg = np.empty(n, np.int32)
    for ci in range(0, n, chunk):
        cb = corpus[ci:ci+chunk]
        d = (cb*cb).sum(1)[:,None] + cn[None,:] - 2.0*(cb @ centroids.T)
        asg[ci:ci+chunk] = d.argmin(1)
    return asg

def query_cell_order(queries, centroids):
    qn = (queries*queries).sum(1)[:,None]
    cn = (centroids*centroids).sum(1)[None,:]
    d = qn + cn - 2.0*(queries @ centroids.T)
    return np.argsort(d, axis=1)

def cell_recall(gt_topk, row_cell, q_cell_order, probes_list):
    nq, k = gt_topk.shape
    lists = q_cell_order.shape[1]
    gt_cells = row_cell[gt_topk]
    cell_rank = np.empty((nq, lists), np.int32)
    rows = np.arange(nq)[:, None]
    cell_rank[rows, q_cell_order] = np.arange(lists)[None, :]
    nn_rank = np.take_along_axis(cell_rank, gt_cells, axis=1)
    return {p: float((nn_rank < p).mean()) for p in probes_list}

def unit_inplace(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n==0]=1
    x /= n
    return x

def main():
    path = sys.argv[1] if len(sys.argv)>1 else "gist-960-euclidean.hdf5"
    K = 10
    n_query = 1000
    sample_cap = 50000
    lists_list = [1000, 4000]
    probes_list = [1,2,4,8,16,32,64,128,256]

    f = h5py.File(path, "r")
    queries = np.asarray(f["test"][:n_query], dtype=np.float32)
    gt = np.asarray(f["neighbors"][:n_query, :K], dtype=np.int64)  # published exact-L2 top-10 over full 1M
    log(f"queries={queries.shape} gt={gt.shape} (published exact-L2)")
    # sample indices (bounded, like turbovec reservoir)
    n_full = f["train"].shape[0]
    idx = np.sort(RNG.choice(n_full, sample_cap, replace=False))

    results = {"meta":{"scale":"FULL_1M","K":K,"n_query":n_query,"sample_cap":sample_cap,
                       "gt":"published exact-L2 neighbors over full 1M corpus"}, "runs":[]}

    # ---------- ANGULAR (turbovec) phase ----------
    log("loading corpus (angular phase)")
    corpus = np.asarray(f["train"], dtype=np.float32)   # 3.84 GB
    unit_inplace(corpus)                                 # in place -> angular space
    qa = unit_inplace(queries.copy())
    sample = corpus[idx].copy()
    ang = {}
    for lists in lists_list:
        C = train_kmeans(sample, lists); log(f"angular kmeans lists={lists}")
        rc = assign_all(corpus, C)
        qo = query_cell_order(qa, C)
        ang[lists] = cell_recall(gt, rc, qo, probes_list)
        log(f"angular cell-recall lists={lists}: p64={ang[lists][64]:.4f} p128={ang[lists][128]:.4f}")
    del corpus, sample, qa; gc.collect()

    # ---------- L2 (vchord/faiss) phase ----------
    log("loading corpus (L2 phase)")
    corpus = np.asarray(f["train"], dtype=np.float32)
    sample = corpus[idx].copy()
    l2 = {}
    for lists in lists_list:
        C = train_kmeans(sample, lists); log(f"L2 kmeans lists={lists}")
        rc = assign_all(corpus, C)
        qo = query_cell_order(queries, C)
        l2[lists] = cell_recall(gt, rc, qo, probes_list)
        log(f"L2 cell-recall lists={lists}: p64={l2[lists][64]:.4f} p128={l2[lists][128]:.4f}")
    del corpus, sample; gc.collect()

    for lists in lists_list:
        results["runs"].append({"lists":lists,"angular_unit_norm":ang[lists],"l2":l2[lists]})
        print(f"\n=== lists={lists} (FULL 1M) ===")
        print(f"{'probes':>7} {'cellR@10 ANGULAR(tv)':>22} {'cellR@10 L2(vchord)':>22}")
        for p in probes_list:
            print(f"{p:>7} {ang[lists][p]:>22.4f} {l2[lists][p]:>22.4f}")

    json.dump(results, open("exp1b_cell_recall_1M.json","w"), indent=2)
    log("wrote exp1b_cell_recall_1M.json")

if __name__ == "__main__":
    main()
