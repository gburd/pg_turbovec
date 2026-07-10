#!/usr/bin/env python3
"""
BQ feasibility, GIST-960 (high-dim proxy). Measures binary quantization
(1-bit sign codes, Hamming distance) recall vs the SQ4/SQ8/PQ numbers the
Gap-B investigation already has, and the exact-L2-rerank window BQ needs.

FAISS binary indexes are the trusted analog (as Gap-B used SQ4 for TurboQuant):
  IndexBinaryFlat  = exact Hamming over sign bits  (raw-BQ ceiling)
  IndexBinaryIVF   = coarse-cell BQ scan            (the IVF+BQ scan)
  IndexBinaryHNSW  = graph over Hamming             (HNSW+BQ, req 9)
plus BQ-coarse + exact-L2 rerank of the top-`sk` survivors (the pattern
Qdrant / pgvector-BQ / DiskANN-SBQ all use, and Gap-B proved SQ4 needs).

Binarization: bit = 1 iff coord > threshold. GIST features are NON-NEGATIVE,
so raw sign(coord) is degenerate (nearly all bits 1). We therefore test BOTH:
  - "meancenter": subtract the per-dimension mean, then sign  (the standard
    BQ preprocessing; pgvector-BQ and Qdrant center/normalize first).
  - "rawsign": sign(coord) directly (pg_turbovec's binary_quantize as written
    — bit=1 iff coord>0 — WITHOUT centering; shows why centering matters).
Ground truth is the published exact-L2 top-100.
"""
import sys, time, json
import numpy as np, h5py, faiss

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def recall_at_k(found, gt, k):
    n = found.shape[0]; s = 0.0
    for i in range(n):
        s += len(set(found[i, :k].tolist()) & set(gt[i, :k].tolist())) / k
    return s / n

def pack_bits(x, thresh):
    """Sign-binarize (bit=1 iff x>thresh) and pack to uint8, MSB-first per byte
    (faiss binary layout). thresh is a per-dim vector or scalar 0."""
    bits = (x > thresh).astype(np.uint8)          # (n, dim)
    return np.packbits(bits, axis=1)              # (n, ceil(dim/8))

def main():
    path = "gist-960-euclidean.hdf5"
    n_corpus = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    n_query = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dim, lists = 960, 1000
    faiss.omp_set_num_threads(8)

    f = h5py.File(path, "r")
    xb = np.asarray(f["train"][:n_corpus], dtype=np.float32)
    xq = np.asarray(f["test"][:n_query], dtype=np.float32)
    gt = np.asarray(f["neighbors"][:n_query, :100], dtype=np.int64)
    log(f"loaded xb={xb.shape} xq={xq.shape} gt=published-exact-L2-top100")

    mean = xb.mean(axis=0)
    results = {"meta": {"corpus": n_corpus, "n_query": n_query, "dim": dim,
                        "lists": lists, "gt": "published-exact-L2-1M",
                        "faiss": faiss.__version__,
                        "bytes_per_vec_bq": dim // 8,
                        "note": "BQ=1-bit sign codes, Hamming. rerank=exact-L2 of top-sk."},
               "binarizations": {}}

    for mode, thresh in [("meancenter", mean), ("rawsign", 0.0)]:
        log(f"=== binarization={mode} ===")
        cb = pack_bits(xb, thresh)   # (n, dim/8) uint8
        qb = pack_bits(xq, thresh)
        ones_frac = float(np.unpackbits(cb, axis=1)[:, :dim].mean())
        log(f"  mean fraction of 1-bits in corpus = {ones_frac:.3f} "
            f"(0.5 is ideal; near 1.0 = degenerate)")
        mres = {"ones_fraction": round(ones_frac, 4)}

        # ---- IndexBinaryFlat: exact Hamming (raw-BQ ceiling) ----
        bf = faiss.IndexBinaryFlat(dim)
        bf.add(cb)
        raw = {}
        for K in (10, 100):
            ts = time.time(); _, ids = bf.search(qb, K)
            raw[f"R@{K}"] = round(recall_at_k(ids, gt, K), 4)
            raw[f"ms@{K}"] = round(1000*(time.time()-ts)/n_query, 3)
        log(f"  BinaryFlat (raw exact Hamming): R@10={raw['R@10']} R@100={raw['R@100']}")
        mres["binary_flat_raw"] = raw

        # ---- BinaryFlat + exact-L2 rerank, as fn of rerank window sk ----
        rr = {}
        for sk in (100, 200, 400, 800, 2000):
            ts = time.time(); _, cand = bf.search(qb, sk)
            out10 = np.full((n_query, 10), -1, np.int64)
            out100 = np.full((n_query, 100), -1, np.int64)
            for i in range(n_query):
                c = cand[i][cand[i] >= 0]
                if c.size == 0: continue
                d = ((xb[c] - xq[i])**2).sum(1); o = c[np.argsort(d)]
                out10[i, :min(10, o.size)] = o[:10]
                out100[i, :min(100, o.size)] = o[:100]
            rr[f"sk{sk}"] = {
                "R@10": round(recall_at_k(out10, gt, 10), 4),
                "R@100": round(recall_at_k(out100, gt, 100), 4),
                "ms": round(1000*(time.time()-ts)/n_query, 3),
            }
            log(f"  BinaryFlat+exactL2rerank sk={sk}: "
                f"R@10={rr[f'sk{sk}']['R@10']} R@100={rr[f'sk{sk}']['R@100']}")
        mres["binary_flat_rerank"] = rr

        # ---- IndexBinaryIVF (coarse-cell BQ scan) raw + rerank ----
        try:
            quant = faiss.IndexBinaryFlat(dim)
            biv = faiss.IndexBinaryIVF(quant, dim, lists)
            biv.train(cb); biv.add(cb)
            ivf = {}
            for p in (32, 64, 128):
                biv.nprobe = p
                # raw
                _, ids = biv.search(qb, 100)
                ivf[f"p{p}_raw_R@10"] = round(recall_at_k(ids, gt, 10), 4)
                ivf[f"p{p}_raw_R@100"] = round(recall_at_k(ids, gt, 100), 4)
                # +rerank sk=800
                _, cand = biv.search(qb, 800)
                out10 = np.full((n_query, 10), -1, np.int64)
                out100 = np.full((n_query, 100), -1, np.int64)
                for i in range(n_query):
                    c = cand[i][cand[i] >= 0]
                    if c.size == 0: continue
                    d = ((xb[c] - xq[i])**2).sum(1); o = c[np.argsort(d)]
                    out10[i, :min(10, o.size)] = o[:10]
                    out100[i, :min(100, o.size)] = o[:100]
                ivf[f"p{p}_rr800_R@10"] = round(recall_at_k(out10, gt, 10), 4)
                ivf[f"p{p}_rr800_R@100"] = round(recall_at_k(out100, gt, 100), 4)
                log(f"  BinaryIVF p={p}: raw R@10={ivf[f'p{p}_raw_R@10']} "
                    f"| +rr800 R@10={ivf[f'p{p}_rr800_R@10']} R@100={ivf[f'p{p}_rr800_R@100']}")
            mres["binary_ivf"] = ivf
        except Exception as e:
            log(f"  SKIP BinaryIVF: {e}"); mres["binary_ivf"] = {"error": str(e)}

        # ---- IndexBinaryHNSW (HNSW+BQ, req 9) raw + rerank ----
        try:
            bh = faiss.IndexBinaryHNSW(dim, 32)
            bh.hnsw.efConstruction = 128
            tb = time.time(); bh.add(cb); log(f"  built BinaryHNSW in {time.time()-tb:.1f}s")
            hn = {}
            for ef in (64, 128, 256, 512):
                bh.hnsw.efSearch = ef
                _, ids = bh.search(qb, 100)
                hn[f"ef{ef}_raw_R@10"] = round(recall_at_k(ids, gt, 10), 4)
                hn[f"ef{ef}_raw_R@100"] = round(recall_at_k(ids, gt, 100), 4)
                # rerank: pull sk=800 from the graph then exact-L2
                _, cand = bh.search(qb, 800)
                out10 = np.full((n_query, 10), -1, np.int64)
                out100 = np.full((n_query, 100), -1, np.int64)
                for i in range(n_query):
                    c = cand[i][cand[i] >= 0]
                    if c.size == 0: continue
                    d = ((xb[c] - xq[i])**2).sum(1); o = c[np.argsort(d)]
                    out10[i, :min(10, o.size)] = o[:10]
                    out100[i, :min(100, o.size)] = o[:100]
                hn[f"ef{ef}_rr800_R@10"] = round(recall_at_k(out10, gt, 10), 4)
                hn[f"ef{ef}_rr800_R@100"] = round(recall_at_k(out100, gt, 100), 4)
                log(f"  BinaryHNSW ef={ef}: raw R@10={hn[f'ef{ef}_raw_R@10']} "
                    f"| +rr800 R@10={hn[f'ef{ef}_rr800_R@10']} R@100={hn[f'ef{ef}_rr800_R@100']}")
            mres["binary_hnsw"] = hn
        except Exception as e:
            log(f"  SKIP BinaryHNSW: {e}"); mres["binary_hnsw"] = {"error": str(e)}

        results["binarizations"][mode] = mres

    out = f"exp_bq_gist_{n_corpus}.json"
    json.dump(results, open(out, "w"), indent=2); log(f"wrote {out}")

if __name__ == "__main__":
    main()
