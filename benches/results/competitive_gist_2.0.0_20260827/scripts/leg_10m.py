#!/usr/bin/env python3
"""10M GIST (semi-synthetic) build+bench with a 2h build budget per engine.
Records DNF as an honest datapoint. engine in {turbovec, hnsw, vchord, ivfflat}.
GT = exact BLAS top-10 (gist10m_gt.npy). Writes /mnt/nvme/results/<engine>_gist10m.json.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit
from g0_driver import measure, mem_avail_gb
from tv_leg import load_gt, measure_qps, drop_all_on

RESULTS = "/mnt/nvme/results"
TAB = "gist10m_corpus"
BUDGET_S = 2 * 3600  # 2 hours


def build_budget(create_sql, idxname):
    drop_all_on(TAB)
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SET max_parallel_maintenance_workers = 32")
    cur.execute("SET maintenance_work_mem = '32GB'")
    cur.execute(f"SET statement_timeout = '{BUDGET_S * 1000}'")
    t0 = time.time()
    try:
        cur.execute(create_sql)
    except Exception as e:
        conn.close()
        return {"build_FAILED": str(e)[:300], "build_s": round(time.time() - t0, 1)}
    build_s = time.time() - t0
    cur.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s)", (idxname, idxname))
    rel, tot = cur.fetchone()
    conn.close()
    return {"build_s": round(build_s, 2), "idx_bytes": rel, "idx_total_bytes": tot}


def save(name, rows):
    json.dump(rows, open(f"{RESULTS}/{name}.json", "w"), indent=2)
    print(f"  wrote {name}.json ({len(rows)} rows)", flush=True)


def run_turbovec():
    test, gt = load_gt("gist10m")
    print(f"[turbovec 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    lists = 4000
    idx = f"{TAB}_tv_L{lists}"
    create = f"SET turbovec.bit_width_default=4; CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})"
    print(f"[turbovec] building lists={lists} (budget 2h) ...", flush=True)
    b = build_budget(create, idx)
    print(f"[turbovec] {b}", flush=True)
    rows = []
    if "build_FAILED" in b:
        save("turbovec_gist10m", [{"engine": "turbovec", "lists": lists, **b}]); return
    for rerank in ["off", "auto"]:
        for probes in [16, 32, 64, 128]:
            setup = ["SET enable_seqscan=off", f"SET turbovec.probes={probes}",
                     "SET turbovec.search_k=100", f"SET turbovec.hi_dim_rerank={rerank}",
                     "SET turbovec.scan_parallelism=0", "SET turbovec.out_of_core=off",
                     "SET turbovec.coarse_graph=on"]
            r = measure(TAB, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
            qps8 = measure_qps(TAB, "embt", "<->", 10, test, setup, 8, cast="::turbovec.vector")
            rows.append({"engine": "turbovec", "rerank": rerank, "lists": lists, "probes": probes, "search_k": 100, **b, **r, "qps_8conn": qps8})
            print(f"  rr={rerank} p{probes}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
            save("turbovec_gist10m", rows)


def run_hnsw():
    test, gt = load_gt("gist10m")
    print(f"[hnsw 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    idx = f"{TAB}_hnsw"
    create = f"CREATE INDEX {idx} ON {TAB} USING hnsw (emb vector_l2_ops) WITH (m=32, ef_construction=256)"
    print("[hnsw] building m32/efc256 (budget 2h) ...", flush=True)
    b = build_budget(create, idx)
    print(f"[hnsw] {b}", flush=True)
    if "build_FAILED" in b:
        save("hnsw_gist10m", [{"engine": "hnsw", **b}]); return
    rows = []
    for ef in [80, 120, 200, 400, 800]:
        setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
        r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
        qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
        rows.append({"engine": "hnsw", "ef_search": ef, **b, **r, "qps_8conn": qps8})
        print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
        save("hnsw_gist10m", rows)


def run_vchord():
    test, gt = load_gt("gist10m")
    print(f"[vchord 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    nlists = 4000
    idx = f"{TAB}_vchord_L{nlists}"
    options = f"[build.internal]\nlists = [{nlists}]\nspherical_centroids = false\nbuild_threads = 32\n"
    create = f"CREATE INDEX {idx} ON {TAB} USING vchordrq (emb vector_l2_ops) WITH (options = $$\n{options}$$)"
    print(f"[vchord] building lists={nlists} (budget 2h) ...", flush=True)
    b = build_budget(create, idx)
    print(f"[vchord] {b}", flush=True)
    if "build_FAILED" in b:
        save("vchord_gist10m", [{"engine": "vchord", "index": "vchordrq", "lists": nlists, **b}]); return
    rows = []
    for probes in [30, 100, 300, 500]:
        for eps in [1.0, 1.9]:
            setup = ["SET enable_seqscan=off", f"SET vchordrq.probes = {probes}", f"SET vchordrq.epsilon = {eps}"]
            r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
            rows.append({"engine": "vchord", "index": "vchordrq", "lists": nlists, "probes": probes, "epsilon": eps, **b, **r, "qps_8conn": qps8})
            print(f"  probes={probes} eps={eps}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
            save("vchord_gist10m", rows)


def run_ivfflat():
    test, gt = load_gt("gist10m")
    print(f"[ivfflat 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    lists = 4000
    idx = f"{TAB}_ivfflat_L{lists}"
    create = f"CREATE INDEX {idx} ON {TAB} USING ivfflat (emb vector_l2_ops) WITH (lists={lists})"
    print(f"[ivfflat] building lists={lists} (budget 2h) ...", flush=True)
    b = build_budget(create, idx)
    print(f"[ivfflat] {b}", flush=True)
    if "build_FAILED" in b:
        save("ivfflat_gist10m", [{"engine": "ivfflat", "lists": lists, **b}]); return
    rows = []
    for probes in [32, 64, 128]:
        setup = ["SET enable_seqscan=off", f"SET ivfflat.probes={probes}"]
        r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
        qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
        rows.append({"engine": "ivfflat", "lists": lists, "probes": probes, **b, **r, "qps_8conn": qps8})
        print(f"  probes={probes}: R@10={r['recall']} p50={r['p50']}ms qps8={qps8}", flush=True)
        save("ivfflat_gist10m", rows)


if __name__ == "__main__":
    eng = sys.argv[1]; t0 = time.time()
    {"turbovec": run_turbovec, "hnsw": run_hnsw, "vchord": run_vchord, "ivfflat": run_ivfflat}[eng]()
    print(f"DONE {eng} 10m in {time.time()-t0:.0f}s", flush=True)
