#!/usr/bin/env python3
"""
Gap B experiment 4 (DECISIVE), full-1M edition. Trains ONE coarse L2 quantizer
(lists=1000) and reuses it for every IVF in-cell scorer, so cell recall is
byte-identical across scorers and build cost is paid once.

In-cell scorers: Flat (=ceiling), SQ4 (~TurboQuant), SQ8 (more-bits),
PQ480 (matched-bytes PQ). Plus HNSW M32 (graph proxy).
See exp4_faiss_incell.py for the full rationale.
"""
import sys, time, json
import numpy as np, h5py, faiss

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def recall_at_k(found, gt, k=10):
    n = found.shape[0]; s = 0.0
    for i in range(n):
        s += len(set(found[i, :k].tolist()) & set(gt[i, :k].tolist())) / k
    return s / n

def main():
    path = "gist-960-euclidean.hdf5"
    n_corpus = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    n_query = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    K, dim, lists = 10, 960, 1000
    probes_list = [16, 32, 64, 128]
    faiss.omp_set_num_threads(8)

    f = h5py.File(path, "r")
    xb = np.asarray(f["train"][:n_corpus], dtype=np.float32)
    xq = np.asarray(f["test"][:n_query], dtype=np.float32)
    gt = np.asarray(f["neighbors"][:n_query, :K], dtype=np.int64)
    log(f"loaded xb={xb.shape} gt=published-exact-L2-1M")

    # ---- train ONE coarse quantizer, reuse everywhere ----
    coarse = faiss.IndexFlatL2(dim)
    cq = faiss.IndexIVFFlat(coarse, dim, lists, faiss.METRIC_L2)
    tb = time.time(); cq.train(xb); log(f"trained shared coarse quantizer in {time.time()-tb:.1f}s")

    results = {"meta": {"corpus": n_corpus, "n_query": n_query, "K": K, "dim": dim,
                        "lists": lists, "probes": probes_list, "gt": "published-exact-L2-1M",
                        "faiss": faiss.__version__, "shared_coarse_quantizer": True},
               "indexes": {}}

    def run_ivf(name, idx):
        tb = time.time(); idx.add(xb); log(f"added {name} in {time.time()-tb:.1f}s")
        pp = {}
        for p in probes_list:
            idx.nprobe = p; ts = time.time(); _, ids = idx.search(xq, K)
            r = recall_at_k(ids, gt, K)
            pp[p] = {"recall": round(r, 4), "ms_per_query": round(1000*(time.time()-ts)/n_query, 3)}
            log(f"  {name} p={p}: R@10={r:.4f}")
        results["indexes"][name] = {"per_probe": pp}
        return idx

    # IVFFlat (ceiling) -- reuse cq directly (already trained)
    run_ivf("IVFFlat", cq)

    # SQ4 / SQ8 share the same coarse quantizer object
    for name, qt in [("IVFSQ4", faiss.ScalarQuantizer.QT_4bit),
                     ("IVFSQ8", faiss.ScalarQuantizer.QT_8bit)]:
        idx = faiss.IndexIVFScalarQuantizer(coarse, dim, lists, qt, faiss.METRIC_L2)
        idx.by_residual = False
        tb = time.time(); idx.train(xb); log(f"trained {name} in {time.time()-tb:.1f}s")
        run_ivf(name, idx)

    # PQ480 = 480 bytes/vec = SQ4 bytes (2 dims/subq, 8-bit each)
    try:
        pq = faiss.IndexIVFPQ(coarse, dim, lists, 480, 8, faiss.METRIC_L2)
        tb = time.time(); pq.train(xb); log(f"trained IVFPQ480 in {time.time()-tb:.1f}s")
        run_ivf("IVFPQ480", pq)
    except Exception as e:
        log(f"SKIP IVFPQ480: {e}"); results["indexes"]["IVFPQ480"] = {"error": str(e)}

    # SQ4 + exact rerank (turbovec's xs_recheckorderby analog)
    sq = faiss.IndexIVFScalarQuantizer(coarse, dim, lists, faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_L2)
    sq.by_residual = False; sq.train(xb); sq.add(xb)
    rr = {}
    for p in [64, 128]:
        sq.nprobe = p
        for sk in [64, 200, 800]:
            ts = time.time(); _, cand = sq.search(xq, sk)
            out = np.full((n_query, K), -1, np.int64)
            for i in range(n_query):
                c = cand[i][cand[i] >= 0]
                if c.size == 0: continue
                d = ((xb[c] - xq[i])**2).sum(1); o = c[np.argsort(d)][:K]
                out[i, :o.size] = o
            r = recall_at_k(out, gt, K)
            rr[f"p{p}_sk{sk}"] = {"recall": round(r, 4), "ms_per_query": round(1000*(time.time()-ts)/n_query, 3)}
            log(f"  IVFSQ4+rerank p={p} sk={sk}: R@10={r:.4f}")
    results["indexes"]["IVFSQ4_exact_rerank"] = {"per_config": rr}

    # HNSW M32 (graph proxy, no IVF cells)
    try:
        h = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2); h.hnsw.efConstruction = 128
        tb = time.time(); h.add(xb); log(f"built HNSW M32 in {time.time()-tb:.1f}s")
        pe = {}
        for ef in [40, 80, 120, 200, 400]:
            h.hnsw.efSearch = ef; ts = time.time(); _, ids = h.search(xq, K)
            r = recall_at_k(ids, gt, K)
            pe[ef] = {"recall": round(r, 4), "ms_per_query": round(1000*(time.time()-ts)/n_query, 3)}
            log(f"  HNSW M32 ef={ef}: R@10={r:.4f}")
        results["indexes"]["HNSW_M32"] = {"per_ef": pe}
    except Exception as e:
        log(f"SKIP HNSW: {e}")

    out = f"exp4_faiss_incell_{n_corpus}.json"
    json.dump(results, open(out, "w"), indent=2); log(f"wrote {out}")

if __name__ == "__main__":
    main()
