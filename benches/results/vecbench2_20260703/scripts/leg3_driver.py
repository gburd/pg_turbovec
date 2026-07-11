#!/usr/bin/env python3
"""Leg 3 driver: refreshed competitive comparison (pgvector HNSW vs
pgvectorscale/DiskANN vs pg_turbovec v1.21.0) on sift_corpus / gist_corpus
(and syn_corpus if time allows).

IMPORTANT: bench_lib.drop_all_indexes() only targets a table literally named
'corpus' (a no-op for sift_corpus/gist_corpus/syn_corpus) -- this driver
always explicitly drops ALL indexes on the target table before building a
new one, so the planner can never silently pick a stale index left over
from a previous engine's build.
"""
import sys, time, json, statistics
import numpy as np
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit, build_index
from g0_driver import measure, load_gt, loadavg, mem_avail_gb

RESULTS = "/mnt/nvme/results"
TABS = {"sift1m": "sift_corpus", "gist1m": "gist_corpus", "syn5m": "syn_corpus"}
DIMS = {"sift1m": 128, "gist1m": 960, "syn5m": 768}


def drop_all_on(tab):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    names = [r[0] for r in cur.fetchall()]
    for nm in names:
        cur.execute(f"DROP INDEX IF EXISTS {nm} CASCADE")
    conn.close()
    if names:
        print(f"  dropped {names}", flush=True)


def dump(name, rows):
    path = f"{RESULTS}/leg3_{name}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def confirm_one_index(tab):
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    names = [r[0] for r in cur.fetchall()]
    conn.close()
    assert len(names) == 1, f"expected exactly 1 index on {tab}, got {names}"
    return names[0]


# ---------- pgvector HNSW ----------
def run_hnsw(corpus):
    tab = TABS[corpus]
    test, gt = load_gt(corpus)
    print(f"[hnsw {corpus}] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for label, m, efc in [("m16_efc64", 16, 64), ("m32_efc128", 32, 128)]:
        drop_all_on(tab)
        idx = f"{tab}_hnsw_{label}"
        create = f"CREATE INDEX {idx} ON {tab} USING hnsw (emb vector_l2_ops) WITH (m={m}, ef_construction={efc})"
        print(f"[hnsw] building {label} ...", flush=True)
        b = build_index(create, idx)
        confirm_one_index(tab)
        print(f"[hnsw] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        for ef in [40, 80, 120, 200, 400]:
            setup = ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"]
            r = measure(tab, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
            row = {"engine": "hnsw", "variant": label, "m": m, "efc": efc, "ef_search": ef, **b, **r, "load": loadavg()}
            rows.append(row)
            print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
            dump(f"hnsw_{corpus}", rows)
    return rows


# ---------- pgvectorscale / DiskANN ----------
def run_diskann(corpus, variants=None):
    tab = TABS[corpus]
    test, gt = load_gt(corpus)
    print(f"[diskann {corpus}] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    # build-time variants: default, and a wider num_neighbors/search_list_size
    if variants is None:
        variants = [
            ("default", None),
            ("wide_nn100_sl300", "num_neighbors=100, search_list_size=300"),
            ("plain_no_sbq", "storage_layout=plain"),
        ]
    for label, opts in variants:
        drop_all_on(tab)
        idx = f"{tab}_diskann_{label}"
        withclause = f" WITH ({opts})" if opts else ""
        create = f"CREATE INDEX {idx} ON {tab} USING diskann (emb vector_l2_ops){withclause}"
        print(f"[diskann] building {label} ...", flush=True)
        t0 = time.time()
        try:
            b = build_index(create, idx)
        except Exception as e:
            print(f"[diskann] BUILD FAILED {label}: {e}", flush=True)
            rows.append({"engine": "diskann", "variant": label, "build_FAILED": str(e)[:300]})
            dump(f"diskann_{corpus}", rows)
            continue
        confirm_one_index(tab)
        print(f"[diskann] {label} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f} elapsed={time.time()-t0:.0f}s", flush=True)
        for sl in [100, 200, 400]:
            for rs in [50, 100, 200, 400, 800, 1200, 1600]:
                setup = ["SET enable_seqscan=off",
                         f"SET diskann.query_search_list_size={sl}",
                         f"SET diskann.query_rescore={rs}"]
                try:
                    r = measure(tab, "emb", "<->", 10, test, gt, setup, repeats=3, cast="::vector")
                except Exception as e:
                    print(f"  sl={sl} rescore={rs}: FAILED {e}", flush=True)
                    continue
                row = {"engine": "diskann", "variant": label, "search_list_size": sl,
                       "query_rescore": rs, **b, **r, "load": loadavg()}
                rows.append(row)
                print(f"  sl={sl:4d} rescore={rs:4d}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
                dump(f"diskann_{corpus}", rows)
    return rows


# ---------- pg_turbovec: default tier + tuned tier ----------
def run_turbovec(corpus, lists_list, tier):
    """tier in {'default', 'tuned'}. default = shipped GUCs untouched
    (iterative_scan=off, coarse_graph=auto, search_k=32, probes=8).
    tuned = widen probes/search_k + force coarse_graph=on if lists
    crosses GRAPH_MIN_LISTS, per Leg 1's finding that graph=on wins."""
    tab = TABS[corpus]
    col = "embt"
    test, gt = load_gt(corpus)
    print(f"[turbovec {tier} {corpus}] q={len(test)} load={loadavg()} memGB={mem_avail_gb():.0f}", flush=True)
    rows = []
    for lists in lists_list:
        drop_all_on(tab)
        idx = f"{tab}_tv_{tier}_L{lists}"
        create = (f"SET turbovec.bit_width_default=4; "
                  f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists})")
        print(f"[turbovec] building lists={lists} ...", flush=True)
        b = build_index(create, idx)
        confirm_one_index(tab)
        print(f"[turbovec] lists={lists} built build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f}", flush=True)
        if tier == "default":
            probes_list = [8]  # shipped default
            sk_list = [32]     # shipped default
        else:
            probes_list = [8, 16, 32, 64, 128]
            sk_list = [32, 64, 100]
        for probes in probes_list:
            for sk in sk_list:
                setup = ["SET enable_seqscan=off", f"SET turbovec.probes={probes}",
                         f"SET turbovec.search_k={sk}", "SET turbovec.scan_parallelism=0"]
                if tier == "tuned":
                    # The best-tuned tier means an informed operator: force
                    # out_of_core=on so the CentroidGraph actually gets built
                    # (turbovec.out_of_core=auto's size threshold is NOT
                    # crossed at these corpus sizes --
                    # -- so leaving it on auto silently falls back to the
                    # slower whole-load path, which never builds the graph
                    # at all). This was caught mid-run: an earlier pass of
                    # this driver left out_of_core on auto and measured the
                    # whole-load path by mistake (2x slower than OOC even
                    # with the graph off, 8x slower than OOC+graph-on).
                    setup.append("SET turbovec.out_of_core=on")
                    if lists >= 4096:
                        setup.append("SET turbovec.coarse_graph=on")
                r = measure(tab, col, "<->", 10, test, gt, setup, repeats=3, cast="::turbovec.vector")
                row = {"engine": f"turbovec_{tier}", "lists": lists, "probes": probes,
                       "search_k": sk, **b, **r, "load": loadavg()}
                rows.append(row)
                print(f"  L{lists} p{probes} sk{sk}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']}", flush=True)
                dump(f"turbovec_{tier}_{corpus}", rows)
    return rows


if __name__ == "__main__":
    cmd, corpus = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if cmd == "hnsw":
        run_hnsw(corpus)
    elif cmd == "diskann":
        variants = None
        if len(sys.argv) > 3 and sys.argv[3] == "default_only":
            variants = [("default", None)]
        run_diskann(corpus, variants)
    elif cmd == "turbovec_default":
        lists_list = [int(x) for x in sys.argv[3].split(",")]
        run_turbovec(corpus, lists_list, "default")
    elif cmd == "turbovec_tuned":
        lists_list = [int(x) for x in sys.argv[3].split(",")]
        run_turbovec(corpus, lists_list, "tuned")
    else:
        print("unknown cmd", file=sys.stderr); sys.exit(2)
    print(f"DONE {cmd} {corpus} in {time.time()-t0:.0f}s", flush=True)
