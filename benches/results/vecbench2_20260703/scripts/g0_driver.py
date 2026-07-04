#!/usr/bin/env python3
"""Phase G-0 driver: IVF levers sweep vs pgvector HNSW, at 1M and 5M.

Uses persistent connections (bench_lib.measure) for low-noise warm latency.
Writes intermediate JSON to /mnt/nvme/results as each config completes so a
dropped SSH loses nothing.

Usage:
  g0_driver.py hnsw   <corpus>   # build HNSW variants, sweep ef_search
  g0_driver.py ivf    <corpus>   # build IVF lever variants, sweep probes/parallelism
corpus in {gist1m, sift1m, syn5m}
"""
import os, sys, time, json, statistics
import numpy as np
import psycopg2

sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit, build_index

RESULTS = "/mnt/nvme/results"


def measure(tab, col, op, k, test, gt, setup_sql, repeats=3, cast=""):
    """Table-aware warm-latency + recall measure over a persistent conn.
    Warms once, then times `repeats` full passes; keeps the best (min mean)
    pass. Recall computed on the first pass vs exact GT."""
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    for s in setup_sql:
        cur.execute(s)
    opq = f"{op} %s{cast}"
    q = f"SELECT id FROM {tab} ORDER BY {col} {opq} LIMIT {k}"
    for i in range(len(test)):          # warm
        cur.execute(q, (vlit(test[i]),)); cur.fetchall()
    recall = None; best = None
    for r in range(repeats):
        hits = 0; total = 0; lats = []
        for i in range(len(test)):
            lit = vlit(test[i])
            t0 = time.perf_counter()
            cur.execute(q, (lit,))
            res = [row[0] for row in cur.fetchall()]
            lats.append((time.perf_counter() - t0) * 1000.0)
            if r == 0:
                truth = set(int(x) for x in gt[i][:k])
                hits += len(truth & set(res)); total += k
        if r == 0:
            recall = hits / total
        lats.sort()
        m = statistics.mean(lats)
        cand = {"p50": lats[len(lats)//2], "p95": lats[int(len(lats)*0.95)],
                "mean": m, "qps_1conn": 1000.0/m}
        if best is None or cand["mean"] < best["mean"]:
            best = cand
    conn.close()
    best["recall"] = round(recall, 4)
    for kk in ("p50", "p95", "mean", "qps_1conn"):
        best[kk] = round(best[kk], 3)
    return best

CORPORA = {
    "gist1m": {"table": "gist_corpus",  "dim": 960, "n": 1_000_000, "sqrtn": 1000},
    "sift1m": {"table": "sift_corpus",  "dim": 128, "n": 1_000_000, "sqrtn": 1000},
    "syn5m":  {"table": "syn_corpus",   "dim": 768, "n": 5_000_000, "sqrtn": 2236},
}

# ---- ground truth loading (HDF5 for gist/sift; .npy for syn) ----
def load_gt(corpus, qcap=1000):
    import h5py
    if corpus == "gist1m":
        h = h5py.File("/mnt/nvme/data/gist-960-euclidean.hdf5", "r")
        return np.asarray(h["test"][:qcap], np.float32), np.asarray(h["neighbors"][:qcap], np.int64)
    if corpus == "sift1m":
        h = h5py.File("/mnt/nvme/data/sift-128-euclidean.hdf5", "r")
        return np.asarray(h["test"][:qcap], np.float32), np.asarray(h["neighbors"][:qcap], np.int64)
    if corpus == "syn5m":
        test = np.load("/mnt/nvme/data/syn5m_test.npy")[:qcap]
        gt = np.load("/mnt/nvme/data/syn5m_gt.npy")[:qcap]
        return test, gt
    raise ValueError(corpus)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return None


def dump(corpus, engine, rows):
    path = f"{RESULTS}/g0_{engine}_{corpus}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def run_hnsw(corpus):
    c = CORPORA[corpus]
    tab, dim = c["table"], c["dim"]
    test, gt = load_gt(corpus)
    print(f"[hnsw {corpus}] {len(test)} queries dim={dim} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    variants = [("m16_efc64", 16, 64), ("m32_efc128", 32, 128)]
    for label, m, efc in variants:
        idx = f"{tab}_hnsw_{label}"
        create = (f"DROP INDEX IF EXISTS {idx}; "
                  f"CREATE INDEX {idx} ON {tab} USING hnsw (emb vector_l2_ops) "
                  f"WITH (m={m}, ef_construction={efc})")
        print(f"[hnsw] building {label} ...", flush=True)
        b = build_index(create, idx)
        print(f"[hnsw] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for ef in [40, 80, 120, 200, 400]:
            setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
            r = measure(tab, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            row = {"engine": "hnsw", "variant": label, "m": m, "efc": efc,
                   "ef_search": ef, **b, **r, "load": loadavg()}
            rows.append(row)
            print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms p95={r['p95']}ms "
                  f"qps1={r['qps_1conn']} load={row['load']}", flush=True)
            dump(corpus, "hnsw", rows)
    return rows


def run_ivf(corpus):
    c = CORPORA[corpus]
    tab, dim, sqrtn = c["table"], c["dim"], c["sqrtn"]
    test, gt = load_gt(corpus)
    print(f"[ivf {corpus}] {len(test)} queries dim={dim} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    # lists sweep: sqrt(n), 4x (tiny cells, >4096 engages sublinear at 4x/8x),
    #              8x (only 1M, gist heavy so skip 8x for gist to save build time)
    list_mults = [1, 4]
    if corpus in ("sift1m",):
        list_mults = [1, 4, 8]
    if corpus == "gist1m":
        list_mults = [1, 4, 8]
    bit_widths = [4, 2]
    for bw in bit_widths:
        for mult in list_mults:
            lists = sqrtn * mult
            idx = f"{tab}_ivf_bw{bw}_L{lists}"
            create = (f"DROP INDEX IF EXISTS {idx}; "
                      f"SET turbovec.bit_width_default={bw}; "
                      f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) "
                      f"WITH (lists={lists})")
            print(f"[ivf] building bw{bw} lists={lists} (sublinear={'YES' if lists>4096 else 'no'}) "
                  f"memGB={mem_avail_gb():.0f} ...", flush=True)
            t0 = time.time()
            try:
                b = build_index(create, idx)
            except Exception as e:
                print(f"[ivf] BUILD FAILED bw{bw} lists={lists}: {e}", flush=True)
                rows.append({"engine": "ivf", "bw": bw, "lists": lists,
                             "sublinear": lists > 4096, "build_FAILED": str(e)[:200]})
                dump(corpus, "ivf", rows)
                continue
            print(f"[ivf] bw{bw} lists={lists} built build_s={b['build_s']} "
                  f"idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
            # probes x scan_parallelism sweep
            for probes in [8, 16, 32, 64, 128]:
                for sp in [1, 0, 8]:  # 1=serial, 0=auto, 8=8-way
                    setup = ["SET enable_seqscan=off",
                             f"SET turbovec.probes={probes}",
                             f"SET turbovec.scan_parallelism={sp}",
                             "SET turbovec.search_k=32"]
                    r = measure(tab, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                    row = {"engine": "ivf", "bw": bw, "lists": lists, "mult": mult,
                           "sublinear": lists > 4096, "probes": probes, "scan_parallelism": sp,
                           "search_k": 32, **b, **r, "load": loadavg()}
                    rows.append(row)
                    print(f"  bw{bw} L{lists} p={probes} sp={sp}: R@10={r['recall']} "
                          f"p50={r['p50']}ms p95={r['p95']}ms qps1={r['qps_1conn']} "
                          f"load={row['load']}", flush=True)
                    dump(corpus, "ivf", rows)
            # search_k effect at one representative config (best-ish recall probes)
            for sk in [32, 64, 100]:
                setup = ["SET enable_seqscan=off", f"SET turbovec.probes=64",
                         "SET turbovec.scan_parallelism=0", f"SET turbovec.search_k={sk}"]
                r = measure(tab, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                row = {"engine": "ivf_sk", "bw": bw, "lists": lists, "probes": 64,
                       "scan_parallelism": 0, "search_k": sk, **b, **r, "load": loadavg()}
                rows.append(row)
                print(f"  [sk] bw{bw} L{lists} p64 sk={sk}: R@10={r['recall']} "
                      f"p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
                dump(corpus, "ivf", rows)
    return rows


if __name__ == "__main__":
    engine, corpus = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if engine == "hnsw":
        run_hnsw(corpus)
    elif engine == "ivf":
        run_ivf(corpus)
    print(f"DONE {engine} {corpus} in {time.time()-t0:.0f}s", flush=True)
