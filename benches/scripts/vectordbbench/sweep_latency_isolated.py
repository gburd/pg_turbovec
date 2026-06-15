#!/usr/bin/env python3
"""Isolation-aware recall-vs-latency sweep for AVX2 hosts (arnold).

Designed for a BUSY shared workstation: the bench backend is CPU-pinned
(the postmaster is started under `taskset -c <cores>`, all backends inherit
the mask) and this driver pins itself too. The point is defensible warm
latency on a contended box.

For every timed batch we sample contention (/proc/loadavg, /proc/stat
steal+iowait, free RAM) before and after, flag per-query >3x-median
outliers, and report BOTH raw and outlier-filtered percentiles plus a
trimmed mean (drop top/bottom 5%). A batch run at high load is reported with
its observed load so the reader can judge trust.

Latency = server-side `Execution Time` from EXPLAIN (ANALYZE) for BOTH
engines (the fair engine-to-engine number; excludes client RTT). Recall@10
vs precomputed brute-force GT. Single-connection QPS = N / sum(latencies).

Configs (the v1.9.0 AVX2 frontier):
  pgvector HNSW ef_search   in {40,100,200,400}
  pg_turbovec 4-bit search_k in {100,200,500,1000}
  pg_turbovec 2-bit search_k in {100,200,500}
  pg_turbovec 4-bit oversample in {1,2,4} at a fixed search_k (default 200)
"""
import argparse, json, os, platform, statistics, time
import numpy as np
import psycopg

DSN = "host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_wiki"

PGV_Q = "SELECT id FROM public.docs ORDER BY emb <=> %s::real[]::vector LIMIT 10"
TV_Q = ("SELECT id FROM public.docs "
        "ORDER BY (emb::real[]::turbovec.vector) "
        "OPERATOR (turbovec.<=>) (%s::real[]::turbovec.vector) LIMIT 10")


def pgv(ef):
    return {"label": f"hnsw_ef{ef}", "engine": "pgvector_hnsw",
            "param": {"ef_search": ef},
            "setup": ["SET enable_seqscan=off", f"SET hnsw.ef_search={ef}"],
            "query": PGV_Q}


def tv(bits, k, oversample=1.0):
    p = {"bit_width": bits, "search_k": k, "oversample": oversample}
    lbl = f"tv_{bits}b_k{k}" + (f"_os{oversample:g}" if oversample != 1.0 else "")
    return {"label": lbl, "engine": f"pg_turbovec_{bits}bit",
            "param": p,
            "setup": ["SET search_path=turbovec,public", "SET enable_seqscan=off",
                      f"SET turbovec.search_k={k}",
                      f"SET turbovec.oversample={oversample}",
                      f"SELECT set_config('turbovec.bit_width_default','{bits}',false)"],
            "query": TV_Q}


def build_configs(which, os_k, os_vals, os_bits=4):
    c = []
    if which in ("all", "pgv"):
        for ef in (40, 100, 200, 400):
            c.append(pgv(ef))
    if which in ("all", "tv4"):
        for k in (100, 200, 500, 1000):
            c.append(tv(4, k))
    if which in ("all", "tv2"):
        for k in (100, 200, 500):
            c.append(tv(2, k))
    if which in ("all", "os"):
        # oversample frontier at a fixed search_k on os_bits-bit codes.
        for ov in os_vals:
            c.append(tv(os_bits, os_k, oversample=float(ov)))
    return c


# ---- contention sampling -------------------------------------------------

def _cpu_busy_idle():
    """Aggregate jiffies (busy_excl_steal, idle, iowait, steal) from /proc/stat."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    v = [int(x) for x in parts[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = (v + [0] * 8)[:8]
    busy = user + nice + system + irq + softirq
    return busy, idle, iowait, steal


def sample_contention():
    la1, la5, la15 = (float(x) for x in open("/proc/loadavg").read().split()[:3])
    busy, idle, iowait, steal = _cpu_busy_idle()
    mem = {}
    for line in open("/proc/meminfo"):
        k, *rest = line.split()
        if k.rstrip(":") in ("MemFree", "MemAvailable"):
            mem[k.rstrip(":")] = int(rest[0]) // 1024  # MiB
    return {"loadavg_1m": la1, "loadavg_5m": la5, "loadavg_15m": la15,
            "_busy": busy, "_idle": idle, "_iowait": iowait, "_steal": steal,
            "mem_free_mib": mem.get("MemFree"), "mem_avail_mib": mem.get("MemAvailable")}


def delta_cpu(before, after):
    db = after["_busy"] - before["_busy"]
    di = after["_idle"] - before["_idle"]
    dio = after["_iowait"] - before["_iowait"]
    dst = after["_steal"] - before["_steal"]
    tot = db + di + dio + dst
    if tot <= 0:
        return {"cpu_busy_pct": None, "cpu_iowait_pct": None, "cpu_steal_pct": None}
    return {"cpu_busy_pct": round(100.0 * db / tot, 2),
            "cpu_iowait_pct": round(100.0 * dio / tot, 2),
            "cpu_steal_pct": round(100.0 * dst / tot, 2)}


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
    """Return (kept, outlier_idx). An outlier is > factor * batch median."""
    med = statistics.median(times)
    thr = factor * med
    kept, outliers = [], []
    for i, t in enumerate(times):
        (outliers if t > thr else kept).append((i, t))
    return [t for _, t in kept], [i for i, _ in outliers]


# ---- query exec ----------------------------------------------------------

def exec_ms_and_ids(cur, sql, vec):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS off, COSTS off, TIMING on) " + sql, (vec,))
    ems = None
    for (line,) in cur.fetchall():
        if line.startswith("Execution Time:"):
            ems = float(line.split()[2])
            break
    cur.execute(sql, (vec,))
    return ems, [r[0] for r in cur.fetchall()]


def run_config(cfg, Q, n_timed, n_warm, dsn):
    print(f"=== {cfg['label']}", flush=True)
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for s in cfg["setup"]:
            cur.execute(s)
        # Warm the per-backend Arc cache + OS page cache.
        for i in range(n_warm):
            cur.execute(cfg["query"], (Q[i % Q.shape[0]].tolist(),))
            cur.fetchall()
        before = sample_contention()
        t_wall0 = time.perf_counter()
        times, tops = [], []
        for i in range(n_timed):
            ems, ids = exec_ms_and_ids(cur, cfg["query"], Q[i % Q.shape[0]].tolist())
            times.append(ems)
            tops.append(ids)
        wall_s = time.perf_counter() - t_wall0
        after = sample_contention()
    return times, tops, before, after, wall_s


def recall_at_10(tops, GT):
    rec = [len(set(int(x) for x in GT[i % len(GT)]) & set(int(x) for x in pred)) / 10.0
           for i, pred in enumerate(tops)]
    return float(np.mean(rec))


def summarize(cfg, times, tops, GT, before, after, wall_s, load_gate):
    kept, out_idx = split_outliers(times, factor=3.0)
    total_s = sum(times) / 1000.0
    cpu = delta_cpu(before, after)
    obs_load = max(before["loadavg_1m"], after["loadavg_1m"])
    rec = {
        "label": cfg["label"], "engine": cfg["engine"], "param": cfg["param"],
        "n_queries": len(times), "recall_at_10": round(recall_at_10(tops, GT), 4),
        "latency_basis": "server_exec_explain_analyze",
        # raw (all timed queries)
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(pctl(times, .5), 3),
        "p95_ms": round(pctl(times, .95), 3),
        "p99_ms": round(pctl(times, .99), 3),
        "min_ms": round(min(times), 3), "max_ms": round(max(times), 3),
        "trimmed_mean_ms": round(trimmed_mean(times, 0.05), 3),
        "qps_single_conn": round(len(times) / total_s, 2),
        # outlier-filtered (>3x median dropped)
        "n_outliers": len(out_idx),
        "filt_p50_ms": round(pctl(kept, .5), 3) if kept else None,
        "filt_p95_ms": round(pctl(kept, .95), 3) if kept else None,
        "filt_p99_ms": round(pctl(kept, .99), 3) if kept else None,
        # contention metadata for THIS batch
        "contention": {
            "loadavg_1m_before": before["loadavg_1m"],
            "loadavg_1m_after": after["loadavg_1m"],
            "loadavg_1m_observed": round(obs_load, 2),
            "mem_free_mib_before": before["mem_free_mib"],
            "mem_avail_mib_before": before["mem_avail_mib"],
            "cpu_busy_pct": cpu["cpu_busy_pct"],
            "cpu_iowait_pct": cpu["cpu_iowait_pct"],
            "cpu_steal_pct": cpu["cpu_steal_pct"],
            "wall_s": round(wall_s, 2),
            # trustworthy iff observed 1m load stayed under the gate
            "contended_flag": bool(obs_load > load_gate),
            "load_gate": load_gate,
        },
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("BENCH_DSN", DSN))
    ap.add_argument("--queries", default="/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy")
    ap.add_argument("--gt", default="/scratch/pg_turbovec-bench/cohere-wiki/gt_top10.npy")
    ap.add_argument("--n-timed", type=int, default=400,
                    help="timed queries for fast (pgvector) configs")
    ap.add_argument("--n-timed-tv", type=int, default=40,
                    help="timed queries for slow turbovec full-scan configs")
    ap.add_argument("--n-warm", type=int, default=20)
    ap.add_argument("--which", default="all", choices=["all", "pgv", "tv4", "tv2", "os"])
    ap.add_argument("--os-k", type=int, default=200, help="fixed search_k for the oversample sweep")
    ap.add_argument("--os-bits", type=int, default=4, help="bit width for the oversample sweep")
    ap.add_argument("--os-vals", default="1,2,4")
    ap.add_argument("--load-gate", type=float, default=1.5,
                    help="observed 1m load above this flags the batch as contended")
    ap.add_argument("--reps", type=int, default=1, help="repeat each config N times (keep best=lowest-load run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Q = np.load(args.queries)
    GT = np.load(args.gt)
    os_vals = [float(x) for x in args.os_vals.split(",")]
    print(f"queries {Q.shape}  gt {GT.shape}  n_timed={args.n_timed}  "
          f"which={args.which}  load_gate={args.load_gate}  reps={args.reps}", flush=True)

    meta = {"host": platform.node(), "uname": platform.platform(),
            "n_timed": args.n_timed, "n_warm": args.n_warm,
            "load_gate": args.load_gate, "latency_basis": "server_exec_explain_analyze",
            "start_loadavg": open("/proc/loadavg").read().strip(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    out = []
    for cfg in build_configs(args.which, args.os_k, os_vals, args.os_bits):
        # turbovec is a full O(n*d) scan (~seconds/query at 1M*1024d even on
        # AVX2); use a smaller-but-still-stable timed count for it. pgvector
        # HNSW is ~tens of ms, so it gets the full count.
        n_timed = args.n_timed if cfg["engine"] == "pgvector_hnsw" else args.n_timed_tv
        best = None
        for rep in range(args.reps):
            times, tops, before, after, wall_s = run_config(
                cfg, Q, n_timed, args.n_warm, args.dsn)
            rec = summarize(cfg, times, tops, GT, before, after, wall_s, args.load_gate)
            obs = rec["contention"]["loadavg_1m_observed"]
            print(f"  rep{rep}: R@10={rec['recall_at_10']:.4f} "
                  f"p50={rec['p50_ms']}ms filt_p50={rec['filt_p50_ms']}ms "
                  f"p95={rec['p95_ms']}ms p99={rec['p99_ms']}ms "
                  f"trimmed={rec['trimmed_mean_ms']}ms qps={rec['qps_single_conn']} "
                  f"load={obs} contended={rec['contention']['contended_flag']} "
                  f"outliers={rec['n_outliers']}", flush=True)
            # keep the rep observed at the LOWEST load (most trustworthy)
            if best is None or obs < best["contention"]["loadavg_1m_observed"]:
                best = rec
        out.append(best)
        json.dump({"meta": meta, "configs": out}, open(args.out, "w"), indent=2)

    json.dump({"meta": meta, "configs": out}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
