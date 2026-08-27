#!/usr/bin/env python3
"""GIST-1M competitor legs with qps@8 (matches tv_leg/leg2 schema):
vchord, diskann, ivfflat. HNSW uses tv_leg.py. turbovec uses tv_leg.py.
Reuses g0_driver.measure (warm p50/recall) + tv_leg.measure_qps (8-conn qps).
Writes /mnt/nvme/results/<engine>_gist1m.json incrementally.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit, build_index
from g0_driver import measure, mem_avail_gb
from tv_leg import load_gt, measure_qps, drop_all_on

RESULTS = "/mnt/nvme/results"
TAB = "gist_corpus"


def dump(name, rows):
    json.dump(rows, open(f"{RESULTS}/{name}.json", "w"), indent=2)
    print(f"  wrote {RESULTS}/{name}.json ({len(rows)} rows)", flush=True)


def run_ivfflat():
    test, gt = load_gt("gist1m")
    print(f"[ivfflat gist1m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for lists in [1000]:
        drop_all_on(TAB)
        idx = f"{TAB}_ivfflat_L{lists}"
        create = f"CREATE INDEX {idx} ON {TAB} USING ivfflat (emb vector_l2_ops) WITH (lists={lists})"
        print(f"[ivfflat] building lists={lists} ...", flush=True)
        b = build_index(create, idx)
        print(f"[ivfflat] built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for probes in [16, 32, 64, 128]:
            setup = ["SET enable_seqscan=off", f"SET ivfflat.probes={probes}"]
            r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
            rows.append({"engine": "ivfflat", "lists": lists, "probes": probes, **b, **r, "qps_8conn": qps8})
            print(f"  probes={probes}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
            dump("ivfflat_gist1m", rows)


def run_vchord():
    test, gt = load_gt("gist1m")
    print(f"[vchord gist1m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for label, nlists in [("L1000", 1000), ("L4000", 4000)]:
        drop_all_on(TAB)
        idx = f"{TAB}_vchord_{label}"
        options = f"[build.internal]\nlists = [{nlists}]\nspherical_centroids = false\nbuild_threads = 32\n"
        create = f"CREATE INDEX {idx} ON {TAB} USING vchordrq (emb vector_l2_ops) WITH (options = $$\n{options}$$)"
        print(f"[vchord] building {label} (lists={nlists}) ...", flush=True)
        try:
            b = build_index(create, idx)
        except Exception as e:
            print(f"[vchord] BUILD FAILED {label}: {e}", flush=True)
            rows.append({"engine": "vchord", "variant": label, "build_FAILED": str(e)[:300]})
            dump("vchord_gist1m", rows); continue
        print(f"[vchord] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for probes in [10, 30, 100, 300]:
            for eps in [1.0, 1.9, 3.0]:
                setup = ["SET enable_seqscan=off", f"SET vchordrq.probes = {probes}", f"SET vchordrq.epsilon = {eps}"]
                try:
                    r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
                    qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
                except Exception as e:
                    print(f"  probes={probes} eps={eps}: FAILED {e}", flush=True); continue
                rows.append({"engine": "vchord", "index": "vchordrq", "variant": label, "lists": nlists,
                             "probes": probes, "epsilon": eps, **b, **r, "qps_8conn": qps8})
                print(f"  probes={probes:3d} eps={eps}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
                dump("vchord_gist1m", rows)


def run_diskann():
    test, gt = load_gt("gist1m")
    print(f"[diskann gist1m] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    variants = [("default", None), ("wide_nn100_sl300", "num_neighbors=100, search_list_size=300"),
                ("plain_no_sbq", "storage_layout=plain")]
    for label, opts in variants:
        drop_all_on(TAB)
        idx = f"{TAB}_diskann_{label}"
        withclause = f" WITH ({opts})" if opts else ""
        create = f"CREATE INDEX {idx} ON {TAB} USING diskann (emb vector_l2_ops){withclause}"
        print(f"[diskann] building {label} ...", flush=True)
        try:
            b = build_index(create, idx)
        except Exception as e:
            print(f"[diskann] BUILD FAILED {label}: {e}", flush=True)
            rows.append({"engine": "diskann", "variant": label, "build_FAILED": str(e)[:300]})
            dump("diskann_gist1m", rows); continue
        print(f"[diskann] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for sl in [100, 200, 400]:
            for rs in [50, 100, 200, 400, 800, 1200, 1600]:
                setup = ["SET enable_seqscan=off", f"SET diskann.query_search_list_size={sl}", f"SET diskann.query_rescore={rs}"]
                try:
                    r = measure(TAB, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
                    qps8 = measure_qps(TAB, "emb", "<->", 10, test, setup, 8, cast="::vector")
                except Exception as e:
                    print(f"  sl={sl} rescore={rs}: FAILED {e}", flush=True); continue
                rows.append({"engine": "diskann", "variant": label, "search_list_size": sl,
                             "query_rescore": rs, **b, **r, "qps_8conn": qps8})
                print(f"  sl={sl:4d} rescore={rs:4d}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
                dump("diskann_gist1m", rows)


if __name__ == "__main__":
    eng = sys.argv[1]; t0 = time.time()
    {"ivfflat": run_ivfflat, "vchord": run_vchord, "diskann": run_diskann}[eng]()
    print(f"DONE {eng} gist1m in {time.time()-t0:.0f}s", flush=True)
