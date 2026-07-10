#!/usr/bin/env python3
"""
BQ feasibility, 1536-d OpenAI text-embedding-3-large (Qdrant dbpedia HF set,
2 parquet shards ~= 77k rows). This is the PRODUCTION distribution the 40GB/
100M/1536d requirement targets — real cosine-metric text embeddings, the
regime BQ is DESIGNED for (unlike GIST's non-negative L2 image descriptors).

pg_turbovec unit-normalizes on insert and scores cosine/inner-product, so:
  - vectors are L2-normalized here (matching turbovec.normalize_on_insert),
  - ground truth is exact COSINE top-100 (== L2 top-100 on unit vectors),
  - BQ = sign bits (bit=1 iff coord>0). text-embedding-3 is ~zero-centered so
    rawsign is already balanced; we also report meancenter for completeness.

Same measurement shape as exp_bq_gist.py: BinaryFlat (raw Hamming ceiling),
BinaryFlat+exact rerank vs sk, BinaryIVF, BinaryHNSW (HNSW+BQ), R@10 & R@100.
"""
import sys, time, json, glob
import numpy as np, faiss
import pyarrow.parquet as pq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def recall_at_k(found, gt, k):
    n = found.shape[0]; s = 0.0
    for i in range(n):
        s += len(set(found[i, :k].tolist()) & set(gt[i, :k].tolist())) / k
    return s / n

def pack_bits(x, thresh):
    bits = (x > thresh).astype(np.uint8)
    return np.packbits(bits, axis=1)

def load_embeddings(n_total):
    col = "text-embedding-3-large-1536-embedding"
    files = sorted(glob.glob("/tmp/bq-investigation/train-*.parquet"))
    out = []
    got = 0
    for fp in files:
        t = pq.read_table(fp, columns=[col])
        arr = np.array(t.column(col).to_pylist(), dtype=np.float32)
        out.append(arr); got += arr.shape[0]
        if got >= n_total: break
    x = np.concatenate(out, axis=0)[:n_total]
    return x

def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 76000
    n_query = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dim, lists = 1536, 256
    faiss.omp_set_num_threads(8)

    x = load_embeddings(n_total)
    log(f"loaded embeddings {x.shape}")
    # L2-normalize (turbovec.normalize_on_insert=true default).
    x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    # Split: last n_query rows are queries, rest corpus.
    xq = x[-n_query:].copy()
    xb = x[:-n_query].copy()
    n_corpus = xb.shape[0]
    log(f"corpus={xb.shape} query={xq.shape} dim={dim}")

    # Exact cosine top-100 GT (== L2 top-100 on unit vectors). Brute force.
    tb = time.time()
    gtidx = faiss.IndexFlatIP(dim); gtidx.add(xb)
    _, gt = gtidx.search(xq, 100)
    gt = gt.astype(np.int64)
    log(f"computed exact cosine top-100 GT in {time.time()-tb:.1f}s")

    mean = xb.mean(axis=0)
    results = {"meta": {"corpus": n_corpus, "n_query": n_query, "dim": dim,
                        "lists": lists, "gt": "self-computed exact cosine/IP top100",
                        "dataset": "OpenAI text-embedding-3-large-1536 (Qdrant dbpedia)",
                        "faiss": faiss.__version__,
                        "bytes_per_vec_bq": dim // 8,
                        "note": "unit-normalized (turbovec metric); BQ=sign bits, Hamming."},
               "binarizations": {}}

    for mode, thresh in [("rawsign", 0.0), ("meancenter", mean)]:
        log(f"=== binarization={mode} ===")
        cb = pack_bits(xb, thresh); qb = pack_bits(xq, thresh)
        ones_frac = float(np.unpackbits(cb, axis=1)[:, :dim].mean())
        log(f"  mean fraction of 1-bits = {ones_frac:.3f}")
        mres = {"ones_fraction": round(ones_frac, 4)}

        bf = faiss.IndexBinaryFlat(dim); bf.add(cb)
        raw = {}
        for K in (10, 100):
            _, ids = bf.search(qb, K)
            raw[f"R@{K}"] = round(recall_at_k(ids, gt, K), 4)
        log(f"  BinaryFlat raw: R@10={raw['R@10']} R@100={raw['R@100']}")
        mres["binary_flat_raw"] = raw

        rr = {}
        for sk in (100, 200, 400, 800, 2000):
            _, cand = bf.search(qb, sk)
            out10 = np.full((n_query, 10), -1, np.int64)
            out100 = np.full((n_query, 100), -1, np.int64)
            for i in range(n_query):
                c = cand[i][cand[i] >= 0]
                if c.size == 0: continue
                # exact cosine rerank == max IP on unit vectors == min L2
                sims = xb[c] @ xq[i]; o = c[np.argsort(-sims)]
                out10[i, :min(10, o.size)] = o[:10]
                out100[i, :min(100, o.size)] = o[:100]
            rr[f"sk{sk}"] = {"R@10": round(recall_at_k(out10, gt, 10), 4),
                             "R@100": round(recall_at_k(out100, gt, 100), 4)}
            log(f"  BinaryFlat+rerank sk={sk}: R@10={rr[f'sk{sk}']['R@10']} "
                f"R@100={rr[f'sk{sk}']['R@100']}")
        mres["binary_flat_rerank"] = rr

        # BinaryHNSW (HNSW+BQ, req 9)
        try:
            bh = faiss.IndexBinaryHNSW(dim, 32); bh.hnsw.efConstruction = 128
            tb = time.time(); bh.add(cb); log(f"  built BinaryHNSW in {time.time()-tb:.1f}s")
            hn = {}
            for ef in (64, 128, 256, 512):
                bh.hnsw.efSearch = ef
                _, ids = bh.search(qb, 100)
                hn[f"ef{ef}_raw_R@10"] = round(recall_at_k(ids, gt, 10), 4)
                hn[f"ef{ef}_raw_R@100"] = round(recall_at_k(ids, gt, 100), 4)
                _, cand = bh.search(qb, 800)
                out10 = np.full((n_query, 10), -1, np.int64)
                out100 = np.full((n_query, 100), -1, np.int64)
                for i in range(n_query):
                    c = cand[i][cand[i] >= 0]
                    if c.size == 0: continue
                    sims = xb[c] @ xq[i]; o = c[np.argsort(-sims)]
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

    # Reference: SQ4/SQ8 + rerank on the SAME 1536d corpus (compare to BQ).
    try:
        log("=== reference: SQ4 / SQ8 + exact rerank (IP metric) ===")
        ref = {}
        for name, qt, bpv in [("SQ4", faiss.ScalarQuantizer.QT_4bit, dim//2),
                               ("SQ8", faiss.ScalarQuantizer.QT_8bit, dim)]:
            sq = faiss.IndexScalarQuantizer(dim, qt, faiss.METRIC_INNER_PRODUCT)
            sq.train(xb); sq.add(xb)
            _, ids = sq.search(xq, 100)
            r = {"bytes_per_vec": bpv,
                 "raw_R@10": round(recall_at_k(ids, gt, 10), 4),
                 "raw_R@100": round(recall_at_k(ids, gt, 100), 4)}
            _, cand = sq.search(xq, 800)
            out10 = np.full((n_query, 10), -1, np.int64)
            out100 = np.full((n_query, 100), -1, np.int64)
            for i in range(n_query):
                c = cand[i][cand[i] >= 0]
                if c.size == 0: continue
                sims = xb[c] @ xq[i]; o = c[np.argsort(-sims)]
                out10[i, :min(10, o.size)] = o[:10]
                out100[i, :min(100, o.size)] = o[:100]
            r["rr800_R@10"] = round(recall_at_k(out10, gt, 10), 4)
            r["rr800_R@100"] = round(recall_at_k(out100, gt, 100), 4)
            ref[name] = r
            log(f"  {name} ({bpv} B/vec): raw R@10={r['raw_R@10']} "
                f"| +rr800 R@10={r['rr800_R@10']} R@100={r['rr800_R@100']}")
        results["reference_sq"] = ref
    except Exception as e:
        log(f"  SKIP SQ ref: {e}"); results["reference_sq"] = {"error": str(e)}

    out = f"exp_bq_1536_{n_corpus}.json"
    json.dump(results, open(out, "w"), indent=2); log(f"wrote {out}")

if __name__ == "__main__":
    main()
