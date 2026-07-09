#!/usr/bin/env python3
"""Tier-1 search_k sweep on floki (local AVX2).

Answers: does lowering turbovec.search_k cut the warm-latency floor
proportionally while recall@10 holds >= 0.96? (Tier-1 plan section 3, item 1a.)

Three phases:
  1. probe-find:  sweep probes at fixed search_k, find the probes that hits
                  R@10 ~= 0.96.
  2. searchk:     fix that probes, sweep search_k in {10,25,50,100,200,400}.
                  record warm p50/p95 + R@10.
  3. probe-indep: at fixed search_k, probes in {1,64} to reconfirm the
                  probes-independence the plan claims.

Latency = server-side `Execution Time` from EXPLAIN (ANALYZE) -- excludes
client RTT. Recall@10 vs brute-force GT. Contention-gated + outlier-filtered.
"""
import argparse, json, os, platform, statistics, time
import numpy as np
import psycopg

TV_Q = ("SELECT id FROM public.docs ORDER BY emb "
        "OPERATOR (turbovec.<=>) %s::turbovec.vector LIMIT 10")


def vec_literal(arr):
    return "[" + ",".join(f"{x:.6g}" for x in arr) + "]"


# ---- contention sampling (from sweep_latency_isolated.py) ----------------

def _cpu_busy_idle():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    v = [int(x) for x in parts[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = (v + [0] * 8)[:8]
    return user + nice + system + irq + softirq, idle, iowait, steal


def sample_contention():
    la1, la5, la15 = (float(x) for x in open("/proc/loadavg").read().split()[:3])
    busy, idle, iowait, steal = _cpu_busy_idle()
    mem = {}
    for line in open("/proc/meminfo"):
        k, *rest = line.split()
        if k.rstrip(":") in ("MemFree", "MemAvailable"):
            mem[k.rstrip(":")] = int(rest[0]) // 1024
    return {"loadavg_1m": la1, "loadavg_5m": la5, "loadavg_15m": la15,
            "_busy": busy, "_idle": idle, "_iowait": iowait, "_steal": steal,
            "mem_free_mib": mem.get("MemFree"), "mem_avail_mib": mem.get("MemAvailable")}


def delta_cpu(b, a):
    db, di = a["_busy"] - b["_busy"], a["_idle"] - b["_idle"]
    dio, dst = a["_iowait"] - b["_iowait"], a["_steal"] - b["_steal"]
    tot = db + di + dio + dst
    if tot <= 0:
        return {"cpu_busy_pct": None, "cpu_iowait_pct": None, "cpu_steal_pct": None}
    return {"cpu_busy_pct": round(100 * db / tot, 2),
            "cpu_iowait_pct": round(100 * dio / tot, 2),
            "cpu_steal_pct": round(100 * dst / tot, 2)}


# ---- stats ---------------------------------------------------------------

def pctl(s, p):
    if not s:
        return None
    s = sorted(s)
    k = (len(s) - 1) * p
    lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo + 1] - s[lo])


def trimmed_mean(s, frac=0.05):
    if not s:
        return None
    s = sorted(s)
    n = len(s)
    k = int(n * frac)
    core = s[k:n - k] if n - 2 * k > 0 else s
    return statistics.mean(core)


def split_outliers(times, factor=3.0):
    med = statistics.median(times)
    thr = factor * med
    kept = [t for t in times if t <= thr]
    nout = len(times) - len(kept)
    return kept, nout


# ---- query exec ----------------------------------------------------------

def exec_ms_and_ids(cur, vec_lit):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS off, COSTS off, TIMING on) " + TV_Q, (vec_lit,))
    ems = None
    for (line,) in cur.fetchall():
        if line.startswith("Execution Time:"):
            ems = float(line.split()[2])
            break
    cur.execute(TV_Q, (vec_lit,))
    return ems, [r[0] for r in cur.fetchall()]


def recall_at_10(tops, GT):
    rec = [len(set(int(x) for x in GT[i]) & set(int(x) for x in pred)) / 10.0
           for i, pred in enumerate(tops)]
    return float(np.mean(rec))


def run_one(cur, label, probes, search_k, oversample, lits, GT, n_warm, load_gate):
    cur.execute("SET enable_seqscan=off")
    cur.execute(f"SET turbovec.probes={probes}")
    cur.execute(f"SET turbovec.search_k={search_k}")
    cur.execute(f"SET turbovec.oversample={oversample}")
    # warm
    for i in range(n_warm):
        cur.execute(TV_Q, (lits[i % len(lits)],))
        cur.fetchall()
    before = sample_contention()
    t0 = time.perf_counter()
    times, tops = [], []
    for i in range(len(lits)):
        ems, ids = exec_ms_and_ids(cur, lits[i])
        times.append(ems)
        tops.append(ids)
    wall_s = time.perf_counter() - t0
    after = sample_contention()
    kept, nout = split_outliers(times)
    cpu = delta_cpu(before, after)
    obs = max(before["loadavg_1m"], after["loadavg_1m"])
    rec = {
        "label": label, "probes": probes, "search_k": search_k, "oversample": oversample,
        "n_queries": len(times), "recall_at_10": round(recall_at_10(tops, GT), 4),
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(pctl(times, .5), 3), "p95_ms": round(pctl(times, .95), 3),
        "p99_ms": round(pctl(times, .99), 3),
        "min_ms": round(min(times), 3), "max_ms": round(max(times), 3),
        "trimmed_mean_ms": round(trimmed_mean(times), 3),
        "n_outliers": nout,
        "filt_p50_ms": round(pctl(kept, .5), 3) if kept else None,
        "filt_p95_ms": round(pctl(kept, .95), 3) if kept else None,
        "contention": {
            "loadavg_1m_before": before["loadavg_1m"], "loadavg_1m_after": after["loadavg_1m"],
            "loadavg_1m_observed": round(obs, 2),
            "mem_avail_mib_before": before["mem_avail_mib"],
            "cpu_busy_pct": cpu["cpu_busy_pct"], "cpu_iowait_pct": cpu["cpu_iowait_pct"],
            "cpu_steal_pct": cpu["cpu_steal_pct"], "wall_s": round(wall_s, 2),
            "contended_flag": bool(obs > load_gate), "load_gate": load_gate,
        },
    }
    print(f"  {label}: probes={probes} search_k={search_k} os={oversample} "
          f"R@10={rec['recall_at_10']:.4f} p50={rec['p50_ms']}ms "
          f"filt_p50={rec['filt_p50_ms']}ms p95={rec['p95_ms']}ms "
          f"load={obs:.2f} contended={rec['contention']['contended_flag']} "
          f"out={nout}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--query-npy", default="/tmp/tier1-queries.npy")
    ap.add_argument("--gt-npy", default="/tmp/tier1-gt-top10.npy")
    ap.add_argument("--n-warm", type=int, default=30)
    ap.add_argument("--load-gate", type=float, default=1.5)
    ap.add_argument("--probe-find-searchk", type=int, default=200)
    ap.add_argument("--probe-find", default="8,16,32,64,128")
    ap.add_argument("--searchk-sweep", default="10,25,50,100,200,400")
    ap.add_argument("--fixed-probes", type=int, default=0,
                    help="if 0, auto-pick from probe-find at R@10>=target")
    ap.add_argument("--recall-target", type=float, default=0.96)
    ap.add_argument("--probe-indep-searchk", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Q = np.load(args.query_npy)
    GT = np.load(args.gt_npy)
    lits = [vec_literal(Q[i]) for i in range(Q.shape[0])]
    print(f"queries={Q.shape} gt={GT.shape} n={len(lits)} warm={args.n_warm} "
          f"load_gate={args.load_gate}", flush=True)

    meta = {"host": platform.node(), "uname": platform.platform(),
            "n_queries": len(lits), "n_warm": args.n_warm, "load_gate": args.load_gate,
            "latency_basis": "server_exec_explain_analyze",
            "corpus": {"rows": 500000, "dim": 768, "lists": 707, "bit_width": 4,
                       "synthetic_clustered": True, "normalized": True},
            "start_loadavg": open("/proc/loadavg").read().strip(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    results = {"meta": meta, "probe_find": [], "searchk_sweep": [], "probe_indep": []}

    with psycopg.connect(args.dsn, autocommit=True) as conn, conn.cursor() as cur:
        # phase 1: find probes
        print("== phase 1: probe-find ==", flush=True)
        for p in [int(x) for x in args.probe_find.split(",")]:
            r = run_one(cur, f"pf_p{p}", p, args.probe_find_searchk, 1.0,
                        lits, GT, args.n_warm, args.load_gate)
            results["probe_find"].append(r)
            json.dump(results, open(args.out, "w"), indent=2)

        # pick fixed probes
        if args.fixed_probes > 0:
            fixed_probes = args.fixed_probes
        else:
            ok = [r for r in results["probe_find"] if r["recall_at_10"] >= args.recall_target]
            fixed_probes = min(r["probes"] for r in ok) if ok else \
                max(results["probe_find"], key=lambda r: r["recall_at_10"])["probes"]
        meta["fixed_probes"] = fixed_probes
        print(f"== fixed_probes = {fixed_probes} (target R@10>={args.recall_target}) ==", flush=True)

        # phase 2: search_k sweep at fixed probes
        print("== phase 2: search_k sweep ==", flush=True)
        for k in [int(x) for x in args.searchk_sweep.split(",")]:
            r = run_one(cur, f"sk_{k}", fixed_probes, k, 1.0,
                        lits, GT, args.n_warm, args.load_gate)
            results["searchk_sweep"].append(r)
            json.dump(results, open(args.out, "w"), indent=2)

        # phase 3: probes independence at fixed search_k
        print("== phase 3: probes-independence ==", flush=True)
        for p in (1, 64):
            r = run_one(cur, f"pi_p{p}", p, args.probe_indep_searchk, 1.0,
                        lits, GT, args.n_warm, args.load_gate)
            results["probe_indep"].append(r)
            json.dump(results, open(args.out, "w"), indent=2)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
