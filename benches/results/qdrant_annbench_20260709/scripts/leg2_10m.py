#!/usr/bin/env python3
"""Leg 2: 10M GIST (semi-synthetic) build + bench with a build-time budget.
Each engine builds with statement_timeout so a runaway build self-cancels
and we record "build did not complete in budget" as an honest datapoint.

Usage: leg2_10m.py <engine>   engine in {hnsw, turbovec, qdrant}
"""
import sys, time, json, statistics, threading
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit
from g0_driver import measure, mem_avail_gb
from tv_leg import load_gt, measure_qps, drop_all_on

RESULTS = "/mnt/nvme/results"
TAB = "gist10m_corpus"
BUILD_BUDGET_S = 90 * 60  # 90 min per build


def build_with_budget(create_sql, idxname, budget_s=BUILD_BUDGET_S):
    drop_all_on(TAB)
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SET max_parallel_maintenance_workers = 32")
    cur.execute("SET maintenance_work_mem = '32GB'")
    cur.execute(f"SET statement_timeout = '{budget_s * 1000}'")
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


def run_hnsw():
    test, gt = load_gt("gist10m")
    print(f"[hnsw 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    idx = f"{TAB}_hnsw"
    create = f"CREATE INDEX {idx} ON {TAB} USING hnsw (emb vector_l2_ops) WITH (m=32, ef_construction=256)"
    print("[hnsw] building m32/efc256 (budget 90m) ...", flush=True)
    b = build_with_budget(create, idx)
    print(f"[hnsw] {b}", flush=True)
    rows = []
    if "build_FAILED" not in b:
        for ef in [80, 120, 200, 400, 800]:
            setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
            r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
            rows.append({"engine": "hnsw", "ef_search": ef, **b, **r, "qps_8conn": qps8})
            print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
            json.dump(rows, open(f"{RESULTS}/hnsw_gist10m.json", "w"), indent=2)
    else:
        json.dump([{"engine": "hnsw", **b}], open(f"{RESULTS}/hnsw_gist10m.json", "w"), indent=2)


def run_turbovec():
    test, gt = load_gt("gist10m")
    print(f"[turbovec 10m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    lists = 4000
    idx = f"{TAB}_tv_L{lists}"
    create = (f"SET turbovec.bit_width_default=4; "
              f"CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})")
    print(f"[turbovec] building lists={lists} (budget 90m) ...", flush=True)
    b = build_with_budget(create, idx)
    print(f"[turbovec] {b}", flush=True)
    rows = []
    if "build_FAILED" not in b:
        for rerank in ["off", "auto"]:
            for probes in [16, 32, 64, 128]:
                setup = ["SET enable_seqscan=off", f"SET turbovec.probes={probes}",
                         "SET turbovec.search_k=100", f"SET turbovec.hi_dim_rerank={rerank}",
                         "SET turbovec.scan_parallelism=0", "SET turbovec.out_of_core=off",
                         "SET turbovec.coarse_graph=on"]
                r = measure(TAB, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                qps8 = measure_qps(TAB, "embt", "<->", 10, test, setup, 8, cast="::turbovec.vector")
                rows.append({"engine": "turbovec", "rerank": rerank, "lists": lists,
                             "probes": probes, "search_k": 100, **b, **r, "qps_8conn": qps8})
                print(f"  rr={rerank} p{probes}: R@10={r['recall']} p50={r['p50']}ms qps8={qps8}", flush=True)
                json.dump(rows, open(f"{RESULTS}/turbovec_gist10m.json", "w"), indent=2)
    else:
        json.dump([{"engine": "turbovec", "lists": lists, **b}], open(f"{RESULTS}/turbovec_gist10m.json", "w"), indent=2)


if __name__ == "__main__":
    eng = sys.argv[1]
    t0 = time.time()
    if eng == "hnsw":
        run_hnsw()
    elif eng == "turbovec":
        run_turbovec()
    print(f"DONE {eng} 10m in {time.time()-t0:.0f}s", flush=True)
