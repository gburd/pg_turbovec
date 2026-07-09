#!/usr/bin/env python3
"""
Gap B experiment 1: cell recall vs lists/probes on GIST-1M, and the
angular-partition (what pg_turbovec does) vs L2-partition (plain FAISS
IVF, what VectorChord/pgvector IVFFlat do) comparison.

CELL RECALL = of the true L2 top-10 neighbors of a query, what fraction
land in the top-`probes` cells the coarse quantizer selects for that
query. This is the RETRIEVAL UPPER BOUND: no fine-quantization fidelity,
no rerank/oversample, no search_k widening can recover a true NN whose
row lives in an unprobed cell. If cell recall caps below the target
recall band, the gap is retrieval-bound at the coarse quantizer.

pg_turbovec builds its IVF partition on L2-NORMALIZED (unit) then rotated
vectors (src/index/build.rs::ivf_reservoir_push -> normalise_to_vec).
Rotation is orthogonal (distance-preserving) so it is irrelevant to which
cell a vector lands in; the L2-normalization is NOT distance-preserving
for a euclidean metric on non-unit-norm data. So turbovec's partition ==
an ANGULAR (cosine) k-means partition. We reproduce exactly that and
compare it to an L2 k-means partition at matched `lists`.

Subset: to keep k-means tractable offline we cluster on a subsample and
assign the full corpus; cell recall is measured over the full 1M corpus
(every true-NN row is assigned to its nearest cell exactly as the build
does). This matches turbovec's own reservoir-sampled k-means (it trains
on a bounded sample, then assigns all rows).
"""
import sys, time, json
import numpy as np
import h5py
from sklearn.cluster import KMeans, MiniBatchKMeans

RNG = np.random.default_rng(42)

def load(path, n_corpus=None, n_query=1000):
    f = h5py.File(path, "r")
    train = np.asarray(f["train"], dtype=np.float32)
    test  = np.asarray(f["test"][:n_query], dtype=np.float32)
    gt    = np.asarray(f["neighbors"][:n_query], dtype=np.int64)  # top-100 exact L2
    if n_corpus is not None and n_corpus < train.shape[0]:
        train = train[:n_corpus]
        # recompute GT against the subset (published GT is over full 1M)
        gt = None
    return train, test, gt

def exact_topk_l2(corpus, queries, k):
    """Brute-force exact L2 top-k. Chunked to bound memory."""
    n = corpus.shape[0]
    cn = (corpus * corpus).sum(1)  # ||c||^2
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    CH = 4000
    for qi in range(0, queries.shape[0], 256):
        qb = queries[qi:qi+256]
        qn = (qb * qb).sum(1)[:, None]
        best_d = np.full((qb.shape[0], k), np.inf, np.float32)
        best_i = np.zeros((qb.shape[0], k), np.int64)
        for ci in range(0, n, CH):
            cb = corpus[ci:ci+CH]
            # d^2 = ||q||^2 + ||c||^2 - 2 q.c
            d = qn + cn[ci:ci+CH][None, :] - 2.0 * (qb @ cb.T)
            m = d.shape[1]
            # merge with running best
            alld = np.concatenate([best_d, d], axis=1)
            alli = np.concatenate([best_i, np.arange(ci, ci+m)[None,:].repeat(qb.shape[0],0)], axis=1)
            idx = np.argpartition(alld, k, axis=1)[:, :k]
            best_d = np.take_along_axis(alld, idx, 1)
            best_i = np.take_along_axis(alli, idx, 1)
        # final sort of the k
        order = np.argsort(best_d, axis=1)
        out[qi:qi+qb.shape[0]] = np.take_along_axis(best_i, order, 1)
    return out

def train_kmeans(sample, lists, seed=42, iters=15):
    # Own vectorized-GEMM Lloyd k-means. sklearn's MiniBatchKMeans/KMeans
    # thrash BLAS threads on this contended box; this is deterministic,
    # fast, and the partition GEOMETRY (angular vs L2) is all we isolate.
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
        np.add.at(newC, asg, X)
        np.add.at(cnt, asg, 1)
        nz = cnt > 0
        newC[nz] /= cnt[nz][:,None]
        empty = np.where(~nz)[0]
        if len(empty):
            newC[empty] = X[rng.choice(X.shape[0], len(empty), replace=False)]
        C = newC.astype(np.float32)
    return C

def assign_all(corpus, centroids, chunk=8000):
    """Nearest-centroid assignment (argmin L2) for every row."""
    n = corpus.shape[0]
    cn = (centroids * centroids).sum(1)
    asg = np.empty(n, np.int32)
    for ci in range(0, n, chunk):
        cb = corpus[ci:ci+chunk]
        d = (cb*cb).sum(1)[:,None] + cn[None,:] - 2.0*(cb @ centroids.T)
        asg[ci:ci+chunk] = d.argmin(1)
    return asg

def query_cell_order(queries, centroids):
    """For each query, cells sorted ascending by L2 distance to centroid."""
    qn = (queries*queries).sum(1)[:,None]
    cn = (centroids*centroids).sum(1)[None,:]
    d = qn + cn - 2.0*(queries @ centroids.T)
    return np.argsort(d, axis=1)  # (nq, lists)

def cell_recall(gt_topk, row_cell, q_cell_order, probes_list):
    """
    gt_topk: (nq, k) row ids of true top-k
    row_cell: (n,) cell id of each corpus row
    q_cell_order: (nq, lists) cells sorted nearest-first per query
    Returns dict probes -> mean fraction of true top-k whose cell is in top-probes.

    Vectorized: for each query build rank-of-cell (position in probe order),
    then a true NN is retrieved at `probes` iff rank(its cell) < probes.
    """
    nq, k = gt_topk.shape
    lists = q_cell_order.shape[1]
    gt_cells = row_cell[gt_topk]                       # (nq, k)
    # cell_rank[qi, c] = position of cell c in query qi's probe order
    cell_rank = np.empty((nq, lists), np.int32)
    rows = np.arange(nq)[:, None]
    cell_rank[rows, q_cell_order] = np.arange(lists)[None, :]
    nn_rank = np.take_along_axis(cell_rank, gt_cells, axis=1)  # (nq,k) probe-rank of each NN's cell
    out = {}
    for p in probes_list:
        out[p] = float((nn_rank < p).mean())
    return out

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gist-960-euclidean.hdf5"
    n_corpus = int(sys.argv[2]) if len(sys.argv) > 2 else 200000  # subset for tractable offline kmeans
    n_query = 500
    sample_cap = 50000
    K = 10
    lists_list = [1000, 4000]
    probes_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    t0=time.time()
    corpus, queries, _ = load(path, n_corpus=n_corpus, n_query=n_query)
    print(f"[{time.time()-t0:.1f}s] corpus={corpus.shape} queries={queries.shape}", flush=True)

    # Exact L2 GT over the subset corpus (published GT is over full 1M, invalid on a subset)
    gt = exact_topk_l2(corpus, queries, K)
    print(f"[{time.time()-t0:.1f}s] exact L2 GT computed", flush=True)

    # unit-normalized copies (turbovec's clustering space)
    def unit(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n==0]=1
        return x / n
    corpus_u = unit(corpus)
    queries_u = unit(queries)

    # training samples (bounded, like turbovec's reservoir)
    idx = RNG.choice(corpus.shape[0], min(sample_cap, corpus.shape[0]), replace=False)
    idx.sort()

    results = {"meta": {"n_corpus": int(n_corpus), "n_query": int(n_query),
                        "K": K, "sample_cap": int(sample_cap),
                        "note": "cell recall = frac of true L2 top-K whose row's cell is in top-probes"},
               "runs": []}

    for lists in lists_list:
        # ---- turbovec's partition: ANGULAR (unit-normalized) k-means ----
        cent_ang = train_kmeans(corpus_u[idx], lists)
        print(f"[{time.time()-t0:.1f}s] lists={lists} angular kmeans done", flush=True)
        row_cell_ang = assign_all(corpus_u, cent_ang)         # assign unit rows
        q_order_ang = query_cell_order(queries_u, cent_ang)   # probe with unit query
        cr_ang = cell_recall(gt, row_cell_ang, q_order_ang, probes_list)
        print(f"[{time.time()-t0:.1f}s] lists={lists} angular cell recall done", flush=True)

        # ---- plain L2 partition (FAISS IVFFlat / pgvector IVFFlat / vchord coarse) ----
        cent_l2 = train_kmeans(corpus[idx], lists)
        print(f"[{time.time()-t0:.1f}s] lists={lists} L2 kmeans done", flush=True)
        row_cell_l2 = assign_all(corpus, cent_l2)
        q_order_l2 = query_cell_order(queries, cent_l2)
        cr_l2 = cell_recall(gt, row_cell_l2, q_order_l2, probes_list)
        print(f"[{time.time()-t0:.1f}s] lists={lists} L2 cell recall done", flush=True)

        row = {"lists": lists, "angular_unit_norm": cr_ang, "l2": cr_l2}
        results["runs"].append(row)
        print(f"\n=== lists={lists} ===", flush=True)
        print(f"{'probes':>7} {'cell_recall@10 ANGULAR(tv)':>28} {'cell_recall@10 L2(vchord)':>28}")
        for p in probes_list:
            print(f"{p:>7} {cr_ang[p]:>28.4f} {cr_l2[p]:>28.4f}")

    json.dump(results, open("exp1_cell_recall_results.json","w"), indent=2)
    print(f"\n[{time.time()-t0:.1f}s] wrote exp1_cell_recall_results.json", flush=True)

if __name__ == "__main__":
    main()
