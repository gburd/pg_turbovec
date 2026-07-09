#!/usr/bin/env python3
"""Phase G-2d gate driver: pg_turbovec graph kind vs pgvector HNSW vs
VectorChord (vchordrq) vs pg_turbovec IVF baseline, at 5M / 768-dim on
the syn5m CLUSTERED synthetic corpus. Exact top-10 L2 GT for 1000
queries.

Reuses g0_driver.measure / load_gt / loadavg / mem_avail_gb and
bench_lib.connect / build_index. Writes per-config JSON to
/mnt/nvme/results as each config finishes.

The graph kind has NO dedicated scan-time ef/beam GUC. Its beam width
is ef = max(k*4, 64), and k = ceil(LIMIT * turbovec.oversample). So
turbovec.oversample IS the graph's recall/latency sweep knob:
  oversample=1.0 -> k=10  -> ef=64   (the shipped fixed floor)
  oversample=4.0 -> k=40  -> ef=160
  oversample=8.0 -> k=80  -> ef=320
  oversample=16  -> k=160 -> ef=640
We sweep it to trace the frontier the way hnsw.ef_search is swept.

Usage: g2d_driver.py {graph|hnsw|vchord|ivf} syn5m
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, build_index
from g0_driver import measure, load_gt, loadavg, mem_avail_gb

RESULTS = "/mnt/nvme/results"
TAB = "syn_corpus"


def drop_all_on(tab):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    names = [r[0] for r in cur.fetchall()]
    for nm in names:
        cur.execute(f"DROP INDEX IF EXISTS {nm} CASCADE")
    conn.close()
    if names:
        print(f"  dropped {names}", flush=True)


def confirm_one(tab):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    names = [r[0] for r in cur.fetchall()]
    conn.close()
    assert len(names) == 1, f"expected 1 index on {tab}, got {names}"


def dump(engine, rows):
    path = f"{RESULTS}/g2d_{engine}_syn5m.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


# ---------- pg_turbovec GRAPH kind (Vamana, WITH (graph=true)) ----------
def run_graph():
    test, gt = load_gt("syn5m")
    print(f"[graph syn5m] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    drop_all_on(TAB)
    idx = f"{TAB}_tv_graph"
    create = (f"SET turbovec.bit_width_default=4; "
              f"CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) "
              f"WITH (graph = true)")
    print("[graph] building WITH (graph=true) ...", flush=True)
    b = build_index(create, idx)
    confirm_one(TAB)
    print(f"[graph] built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
    # oversample sweep = the beam/ef sweep for the graph kind.
    for ovs in [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
        setup = ["SET enable_seqscan=off",
                 "SET turbovec.iterative_scan=off",
                 f"SET turbovec.oversample={ovs}"]
        r = measure(TAB, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
        row = {"engine": "graph", "oversample": ovs, "ef_approx": max(int(10*ovs)*4, 64),
               **b, **r, "load": loadavg()}
        rows.append(row)
        print(f"  ovs={ovs} (ef~{row['ef_approx']}): R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
        dump("graph", rows)
    return rows


# ---------- pgvector HNSW ----------
def run_hnsw():
    test, gt = load_gt("syn5m")
    print(f"[hnsw syn5m] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for label, m, efc in [("m16_efc64", 16, 64), ("m32_efc128", 32, 128)]:
        drop_all_on(TAB)
        idx = f"{TAB}_hnsw_{label}"
        create = f"CREATE INDEX {idx} ON {TAB} USING hnsw (emb vector_l2_ops) WITH (m={m}, ef_construction={efc})"
        print(f"[hnsw] building {label} ...", flush=True)
        b = build_index(create, idx)
        confirm_one(TAB)
        print(f"[hnsw] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for ef in [40, 80, 120, 200, 400]:
            setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
            r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            row = {"engine": "hnsw", "variant": label, "m": m, "efc": efc, "ef_search": ef,
                   **b, **r, "load": loadavg()}
            rows.append(row)
            print(f"  {label} ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
            dump("hnsw", rows)
    return rows


# ---------- VectorChord ----------
def run_vchord():
    test, gt = load_gt("syn5m")
    print(f"[vchord syn5m] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for label, nlists in [("L2236", 2236), ("L8944", 8944)]:
        drop_all_on(TAB)
        idx = f"{TAB}_vchord_{label}"
        options = ("[build.internal]\n"
                   f"lists = [{nlists}]\n"
                   "spherical_centroids = false\n"
                   "build_threads = 32\n")
        create = (f"CREATE INDEX {idx} ON {TAB} USING vchordrq (emb vector_l2_ops) "
                  f"WITH (options = $$\n{options}$$)")
        print(f"[vchord] building {label} (lists={nlists}) ...", flush=True)
        try:
            b = build_index(create, idx)
        except Exception as e:
            print(f"[vchord] BUILD FAILED {label}: {e}", flush=True)
            rows.append({"engine": "vchord", "variant": label, "build_FAILED": str(e)[:300]})
            dump("vchord", rows); continue
        confirm_one(TAB)
        print(f"[vchord] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for probes in [10, 30, 100, 300]:
            for eps in [1.0, 1.9]:
                setup = ["SET enable_seqscan=off",
                         f"SET vchordrq.probes = {probes}",
                         f"SET vchordrq.epsilon = {eps}"]
                try:
                    r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
                except Exception as e:
                    print(f"  probes={probes} eps={eps}: FAILED {e}", flush=True); continue
                row = {"engine": "vchord", "variant": label, "lists": nlists,
                       "probes": probes, "epsilon": eps, **b, **r, "load": loadavg()}
                rows.append(row)
                print(f"  {label} p={probes} eps={eps}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
                dump("vchord", rows)
    return rows


# ---------- pg_turbovec IVF (the "keep IVF" baseline) ----------
def run_ivf():
    test, gt = load_gt("syn5m")
    print(f"[ivf syn5m] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for lists in [2236, 8944]:
        drop_all_on(TAB)
        idx = f"{TAB}_tv_ivf_L{lists}"
        create = (f"SET turbovec.bit_width_default=4; "
                  f"CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})")
        print(f"[ivf] building lists={lists} ...", flush=True)
        b = build_index(create, idx)
        confirm_one(TAB)
        print(f"[ivf] lists={lists} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for probes in [16, 32, 64, 128]:
            for sk in [32, 64, 100]:
                setup = ["SET enable_seqscan=off",
                         "SET turbovec.iterative_scan=off",
                         f"SET turbovec.probes={probes}",
                         f"SET turbovec.search_k={sk}",
                         "SET turbovec.scan_parallelism=0"]
                if lists >= 4096:
                    setup.append("SET turbovec.out_of_core=on")
                    setup.append("SET turbovec.coarse_graph=on")
                r = measure(TAB, "embt", "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                row = {"engine": "ivf", "lists": lists, "probes": probes, "search_k": sk,
                       **b, **r, "load": loadavg()}
                rows.append(row)
                print(f"  L{lists} p{probes} sk{sk}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
                dump("ivf", rows)
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.time()
    {"graph": run_graph, "hnsw": run_hnsw, "vchord": run_vchord, "ivf": run_ivf}[cmd]()
    print(f"DONE {cmd} in {time.time()-t0:.0f}s", flush=True)
