#!/usr/bin/env python3
"""Lean turbovec GIST-1M sweep. Flat (lists=0) measured ONCE per rerank
(recall is scan-invariant to probes/search_k), then full IVF sweep.
Preserves the tv_leg schema; writes /mnt/nvme/results/turbovec_gist1m.json.
"""
import sys, time, json
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import build_index
from g0_driver import measure, mem_avail_gb
from tv_leg import load_gt, measure_qps, drop_all_on

RESULTS = "/mnt/nvme/results"
TAB = "gist_corpus"; COL = "embt"


def dump(rows):
    json.dump(rows, open(f"{RESULTS}/turbovec_gist1m.json", "w"), indent=2)
    print(f"  wrote turbovec_gist1m.json ({len(rows)} rows)", flush=True)


def bench(rows, b, lists, rerank, probes, sk):
    setup = ["SET enable_seqscan=off", f"SET turbovec.probes={probes}",
             f"SET turbovec.search_k={sk}", f"SET turbovec.hi_dim_rerank={rerank}",
             "SET turbovec.scan_parallelism=0", "SET turbovec.out_of_core=off"]
    if lists >= 4096:
        setup.append("SET turbovec.coarse_graph=on")
    r = measure(TAB, COL, "<->", 10, load_gt.test, load_gt.gt, setup, repeats=3, cast="::turbovec.vector")
    qps8 = measure_qps(TAB, COL, "<->", 10, load_gt.test, setup, 8, cast="::turbovec.vector")
    row = {"engine": "turbovec", "rerank": rerank, "lists": lists,
           "probes": probes, "search_k": sk, **b, **r, "qps_8conn": qps8}
    rows.append(row)
    print(f"  rr={rerank} L{lists} p{probes} sk{sk}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
    dump(rows)


def main():
    test, gt = load_gt("gist1m")
    load_gt.test = test; load_gt.gt = gt   # stash for bench()
    print(f"[turbovec gist1m LEAN] q={len(test)} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    # ---- flat: build once, measure off + auto (2 sk each; scan-invariant to probes) ----
    drop_all_on(TAB)
    idx = f"{TAB}_tv_L0"
    create = f"SET turbovec.bit_width_default=4; CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists=0)"
    print("[turbovec] building flat lists=0 ...", flush=True)
    b = build_index(create, idx)
    print(f"[turbovec] flat build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
    for rerank in ["off", "auto"]:
        for sk in [32, 100]:
            bench(rows, b, 0, rerank, 8, sk)
    # ---- IVF L1000 + L4000: full probes x sk x rerank ----
    for lists in [1000, 4000]:
        drop_all_on(TAB)
        idx = f"{TAB}_tv_L{lists}"
        create = f"SET turbovec.bit_width_default=4; CREATE INDEX {idx} ON {TAB} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})"
        print(f"[turbovec] building IVF lists={lists} ...", flush=True)
        b = build_index(create, idx)
        print(f"[turbovec] L{lists} build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for rerank in ["off", "auto"]:
            for probes in [8, 16, 32, 64, 128]:
                for sk in [32, 100]:
                    bench(rows, b, lists, rerank, probes, sk)


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"DONE turbovec gist1m in {time.time()-t0:.0f}s", flush=True)
