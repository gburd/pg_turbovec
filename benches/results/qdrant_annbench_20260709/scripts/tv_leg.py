#!/usr/bin/env python3
"""Leg 1/2 driver: recall-matched frontier for pg_turbovec (with the
hi_dim_rerank off-vs-auto comparison FRONT AND CENTRE), pgvector HNSW,
on a shared heap. Adds parallel-8 QPS to every measured config.

Reuses g0_driver.measure (table-aware warm p50/recall) and bench_lib.build_index.

Usage:
  tv_leg.py hnsw     <corpus>
  tv_leg.py turbovec <corpus> <lists_csv>   # sweeps probes/search_k x rerank{off,auto}
corpus in {sift1m, gist1m, gist10m}
"""
import sys, time, json, statistics, threading
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit, build_index
from g0_driver import measure, loadavg, mem_avail_gb

RESULTS = "/mnt/nvme/results"
TABS = {"sift1m": "sift_corpus", "gist1m": "gist_corpus", "gist10m": "gist10m_corpus"}
DIMS = {"sift1m": 128, "gist1m": 960, "gist10m": 960}


def load_gt(corpus, qcap=1000):
    import h5py
    if corpus == "sift1m":
        h = h5py.File("/mnt/nvme/data/sift-128-euclidean.hdf5", "r")
        return np.asarray(h["test"][:qcap], np.float32), np.asarray(h["neighbors"][:qcap], np.int64)
    if corpus == "gist1m":
        h = h5py.File("/mnt/nvme/data/gist-960-euclidean.hdf5", "r")
        return np.asarray(h["test"][:qcap], np.float32), np.asarray(h["neighbors"][:qcap], np.int64)
    if corpus == "gist10m":
        test = np.load("/mnt/nvme/data/gist10m_test.npy")[:qcap]
        gt = np.load("/mnt/nvme/data/gist10m_gt.npy")[:qcap]
        return test.astype(np.float32), gt.astype(np.int64)
    raise ValueError(corpus)


def measure_qps(tab, col, op, k, test, setup_sql, nconn, cast="", duration=8.0):
    opq = f"{op} %s{cast}"
    q = f"SELECT id FROM {tab} ORDER BY {col} {opq} LIMIT {k}"
    stop = time.time() + duration
    counts = [0] * nconn
    def worker(wid):
        conn = connect(); conn.autocommit = True; cur = conn.cursor()
        for s in setup_sql:
            cur.execute(s)
        idx = wid; c = 0
        while time.time() < stop:
            cur.execute(q, (vlit(test[idx % len(test)]),)); cur.fetchall()
            idx += 1; c += 1
        counts[wid] = c
        conn.close()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(nconn)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    return round(sum(counts) / (time.time() - t0), 1)


def drop_all_on(tab):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    for (nm,) in cur.fetchall():
        cur.execute(f"DROP INDEX IF EXISTS {nm} CASCADE")
    conn.close()


def dump(name, rows):
    with open(f"{RESULTS}/{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {RESULTS}/{name}.json ({len(rows)} rows)", flush=True)


def run_hnsw(corpus):
    tab = TABS[corpus]; test, gt = load_gt(corpus)
    print(f"[hnsw {corpus}] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    variants = [("m32_efc256", 32, 256)]  # Qdrant-matched build params
    for label, m, efc in variants:
        drop_all_on(tab)
        idx = f"{tab}_hnsw_{label}"
        create = f"CREATE INDEX {idx} ON {tab} USING hnsw (emb vector_l2_ops) WITH (m={m}, ef_construction={efc})"
        print(f"[hnsw] building {label} ...", flush=True)
        b = build_index(create, idx)
        print(f"[hnsw] {label} build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for ef in [40, 80, 120, 200, 400, 800]:
            setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
            r = measure(tab, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            qps8 = measure_qps(tab, "emb", "<->", 10, test, setup, 8, cast="::vector")
            row = {"engine": "hnsw", "variant": label, "m": m, "efc": efc, "ef_search": ef,
                   **b, **r, "qps_8conn": qps8}
            rows.append(row)
            print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
            dump(f"hnsw_{corpus}", rows)
    return rows


def run_turbovec(corpus, lists_list):
    tab = TABS[corpus]; col = "embt"; test, gt = load_gt(corpus)
    print(f"[turbovec {corpus}] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for lists in lists_list:
        drop_all_on(tab)
        idx = f"{tab}_tv_L{lists}"
        create = (f"SET turbovec.bit_width_default=4; "
                  f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})")
        print(f"[turbovec] building lists={lists} ...", flush=True)
        b = build_index(create, idx)
        print(f"[turbovec] lists={lists} build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        # HEADLINE sweep: hi_dim_rerank off vs auto, across probes/search_k.
        # In-RAM regime (out_of_core=off) -- the fair, designed-for regime
        # (247GiB box, 1M/10M x 960 fits in RAM). OOC re-reads relfile and
        # is 4-5x slower here; not the comparison we want vs in-RAM Qdrant/HNSW.
        for rerank in ["off", "auto"]:
            for probes in [8, 16, 32, 64]:
                for sk in [32, 100]:
                    setup = ["SET enable_seqscan=off",
                             f"SET turbovec.probes={probes}",
                             f"SET turbovec.search_k={sk}",
                             f"SET turbovec.hi_dim_rerank={rerank}",
                             "SET turbovec.scan_parallelism=0",
                             "SET turbovec.out_of_core=off"]
                    if lists >= 4096:
                        setup.append("SET turbovec.coarse_graph=on")
                    r = measure(tab, col, "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                    qps8 = measure_qps(tab, col, "<->", 10, test, setup, 8, cast="::turbovec.vector")
                    row = {"engine": "turbovec", "rerank": rerank, "lists": lists,
                           "probes": probes, "search_k": sk, **b, **r, "qps_8conn": qps8}
                    rows.append(row)
                    print(f"  rr={rerank} L{lists} p{probes} sk{sk}: R@10={r['recall']} "
                          f"p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
                    dump(f"turbovec_{corpus}", rows)
    return rows


if __name__ == "__main__":
    cmd, corpus = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if cmd == "hnsw":
        run_hnsw(corpus)
    elif cmd == "turbovec":
        lists = [int(x) for x in sys.argv[3].split(",")]
        run_turbovec(corpus, lists)
    print(f"DONE {cmd} {corpus} in {time.time()-t0:.0f}s", flush=True)
