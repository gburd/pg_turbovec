#!/usr/bin/env python3
"""Phase G-1 validation driver: does turbovec.coarse_graph=on measurably cut
IVF coarse-cell-selection latency vs off, at lists >= GRAPH_MIN_LISTS (4096)?

IMPORTANT: the CentroidGraph only lives inside OocIvfIndex (the
out-of-core cell-scoped scan path) -- see cache.rs. turbovec.out_of_core=auto
does NOT engage for our corpus sizes at the box's cache_size_mb=8192 (codes
are 64MB-1.9GB vs a 4.3GB auto-threshold), so we force out_of_core=on to
actually exercise the graph. This is noted explicitly in the report.

Usage:
  leg1_driver.py build  <corpus> <lists>     # build IVF index at bit_width=4
  leg1_driver.py sweep  <corpus> <lists> <label>
  leg1_driver.py correctness <corpus> <lists>
"""
import sys, os, time, json
sys.path.insert(0, "/mnt/nvme/src")
from bench_lib import connect, vlit, build_index
from g0_driver import measure, load_gt, loadavg, mem_avail_gb

RESULTS = "/mnt/nvme/results"

TABS = {"sift1m": "sift_corpus", "gist1m": "gist_corpus"}
COLS = {"sift1m": "embt", "gist1m": "embt"}


def dump(name, rows):
    path = f"{RESULTS}/g1_{name}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def do_build(corpus, lists):
    tab = TABS[corpus]
    idx = f"{tab}_ivf_bw4_L{lists}_g1"
    create = (f"DROP INDEX IF EXISTS {idx}; "
              f"SET turbovec.bit_width_default=4; "
              f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) "
              f"WITH (lists={lists})")
    print(f"[build] {corpus} lists={lists} idx={idx} memGB={mem_avail_gb():.1f} load={loadavg()}", flush=True)
    t0 = time.time()
    b = build_index(create, idx)
    print(f"[build] DONE {corpus} lists={lists} build_s={b['build_s']} "
          f"idx_MB={b['idx_bytes']/1e6:.0f} elapsed={time.time()-t0:.0f}s", flush=True)
    path = f"{RESULTS}/g1_build_{corpus}_L{lists}.json"
    json.dump({"corpus": corpus, "lists": lists, "idx": idx, **b}, open(path, "w"), indent=2)
    print(f"  wrote {path}", flush=True)
    return idx, b


def _ids_for_mode(tab, col, mode, probes, test):
    """FRESH connection per mode: the OocIvfIndex (and its embedded
    CentroidGraph) is built ONCE per backend cache-install and is NOT
    rebuilt if turbovec.coarse_graph changes mid-connection (the cache
    key is rel_oid/attnum/bit_width/dim + relfile_node + version, not
    the GUC value). Flipping the GUC on a persistent connection after
    the first scan is a no-op and silently compares the graph against
    itself -- a fresh connection per mode is required for this test to
    mean anything."""
    conn = connect(); conn.autocommit = True; cur = conn.cursor()
    cur.execute("SET enable_seqscan=off")
    cur.execute("SET turbovec.out_of_core=on")
    cur.execute(f"SET turbovec.coarse_graph={mode}")
    cur.execute(f"SET turbovec.probes={probes}")
    cur.execute("SET turbovec.scan_parallelism=0")
    cur.execute("SET turbovec.search_k=32")
    out = []
    for i in range(len(test)):
        q = vlit(test[i])
        cur.execute(f"SELECT id FROM {tab} ORDER BY {col} <-> %s::turbovec.vector LIMIT 10", (q,))
        out.append(set(r[0] for r in cur.fetchall()))
    conn.close()
    return out


def correctness_check(corpus, lists, probes_list=(8, 32, 64), nq=25):
    tab, col = TABS[corpus], COLS[corpus]
    test, gt = load_gt(corpus, qcap=nq)
    mismatches = []
    for probes in probes_list:
        off_ids = _ids_for_mode(tab, col, "off", probes, test)
        on_ids = _ids_for_mode(tab, col, "on", probes, test)
        for i in range(len(test)):
            if off_ids[i] != on_ids[i]:
                mismatches.append({"probes": probes, "query": i,
                                    "off": sorted(off_ids[i]), "on": sorted(on_ids[i])})
    ok = len(mismatches) == 0
    result = {"corpus": corpus, "lists": lists, "probes_tested": list(probes_list),
              "n_queries": len(test), "ok": ok, "mismatches": mismatches[:20],
              "n_mismatches": len(mismatches)}
    path = f"{RESULTS}/g1_correctness_{corpus}_L{lists}.json"
    json.dump(result, open(path, "w"), indent=2)
    print(f"[correctness] {corpus} L{lists}: ok={ok} n_mismatches={len(mismatches)} -> {path}", flush=True)
    return ok, result


def sweep(corpus, lists, label, probes_list=(8, 32, 64)):
    tab, col = TABS[corpus], COLS[corpus]
    test, gt = load_gt(corpus)
    print(f"[sweep {label}] {corpus} lists={lists} q={len(test)} "
          f"load={loadavg()} memGB={mem_avail_gb():.1f}", flush=True)
    rows = []
    for mode in ["off", "on"]:
        for probes in probes_list:
            setup = ["SET enable_seqscan=off",
                     "SET turbovec.out_of_core=on",
                     f"SET turbovec.coarse_graph={mode}",
                     f"SET turbovec.probes={probes}",
                     "SET turbovec.scan_parallelism=0",
                     "SET turbovec.search_k=32"]
            r = measure(tab, col, "<->", 10, test, gt, setup, repeats=5, cast="::turbovec.vector")
            row = {"corpus": corpus, "lists": lists, "coarse_graph": mode,
                   "probes": probes, **r, "load": loadavg()}
            rows.append(row)
            print(f"  graph={mode:3s} probes={probes:3d}: R@10={r['recall']} "
                  f"p50={r['p50']}ms p95={r['p95']}ms qps1={r['qps_1conn']} load={row['load']}",
                  flush=True)
            dump(label, rows)
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        do_build(sys.argv[2], int(sys.argv[3]))
    elif cmd == "correctness":
        ok, res = correctness_check(sys.argv[2], int(sys.argv[3]))
        sys.exit(0 if ok else 1)
    elif cmd == "sweep":
        sweep(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        print(f"unknown cmd {cmd}", file=sys.stderr)
        sys.exit(2)
