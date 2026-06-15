#!/usr/bin/env python3
"""VectorDBBench-style recall-vs-latency sweep: pgvector HNSW vs pg_turbovec.

Per config: 2 warmup queries (untimed), then >=N timed queries. Latency is
server-side execution time from EXPLAIN (ANALYZE) (excludes client RTT, the
fair engine-to-engine number). Top-10 ids captured per query for recall@10
against precomputed brute-force ground truth.

Single-connection QPS = N / sum(latencies). Ground truth lives in table
gt_top10(qid, hit_id). Held-out query vectors come from q1000.npy.
"""
import argparse, json, os, statistics, time
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

def tv(bits, k):
    return {"label": f"tv_{bits}b_k{k}", "engine": f"pg_turbovec_{bits}bit",
            "param": {"bit_width": bits, "search_k": k},
            "setup": ["SET search_path=turbovec,public", "SET enable_seqscan=off",
                      f"SET turbovec.search_k={k}",
                      f"SELECT set_config('turbovec.bit_width_default','{bits}',false)"],
            "query": TV_Q}

def build_configs(which):
    c = []
    if which in ("all", "pgv"):
        for ef in (40, 100, 200, 400):
            c.append(pgv(ef))
    if which in ("all", "tv4"):
        for k in (100, 200, 500, 1000):
            c.append(tv(4, k))
    if which in ("all", "tv2"):
        for k in (100, 200, 500, 1000):
            c.append(tv(2, k))
    return c

def exec_ms_and_ids(cur, sql, vec):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS off, COSTS off, TIMING on) " + sql, (vec,))
    ems = None
    for (line,) in cur.fetchall():
        if line.startswith("Execution Time:"):
            ems = float(line.split()[2]); break
    cur.execute(sql, (vec,))
    return ems, [r[0] for r in cur.fetchall()]

def run_config(cfg, Q, n_timed, dsn):
    print(f"=== {cfg['label']}", flush=True)
    times, tops = [], []
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for s in cfg["setup"]:
            cur.execute(s)
        for i in range(2):
            cur.execute(cfg["query"], (Q[i].tolist(),)); cur.fetchall()
        for i in range(n_timed):
            ems, ids = exec_ms_and_ids(cur, cfg["query"], Q[i % Q.shape[0]].tolist())
            times.append(ems); tops.append(ids)
    return times, tops

def recall_at_10(tops, GT):
    rec = []
    for i, pred in enumerate(tops):
        gt = set(int(x) for x in GT[i % len(GT)])
        rec.append(len(gt & set(int(x) for x in pred)) / 10.0)
    return float(np.mean(rec)), rec

def pctl(s, p):
    s = sorted(s); k = (len(s) - 1) * p; lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo+1] - s[lo])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("BENCH_DSN", DSN))
    ap.add_argument("--queries", default="/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy")
    ap.add_argument("--gt", default="/scratch/pg_turbovec-bench/cohere-wiki/gt_top10.npy")
    ap.add_argument("--n-timed", type=int, default=200)
    ap.add_argument("--which", default="all", choices=["all", "pgv", "tv4", "tv2"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Q = np.load(args.queries); GT = np.load(args.gt)
    print(f"queries {Q.shape}  gt {GT.shape}  n_timed={args.n_timed}", flush=True)

    out = []
    for cfg in build_configs(args.which):
        times, tops = run_config(cfg, Q, args.n_timed, args.dsn)
        r, _ = recall_at_10(tops, GT)
        mean = statistics.mean(times); total_s = sum(times) / 1000.0
        rec = {"label": cfg["label"], "engine": cfg["engine"], "param": cfg["param"],
               "n_queries": len(times), "recall_at_10": round(r, 4),
               "mean_ms": round(mean, 3), "p50_ms": round(pctl(times, .5), 3),
               "p95_ms": round(pctl(times, .95), 3), "p99_ms": round(pctl(times, .99), 3),
               "min_ms": round(min(times), 3), "max_ms": round(max(times), 3),
               "qps_single_conn": round(len(times) / total_s, 2)}
        print(f"  R@10={r:.4f} p50={rec['p50_ms']}ms p95={rec['p95_ms']}ms "
              f"p99={rec['p99_ms']}ms qps={rec['qps_single_conn']}", flush=True)
        out.append(rec)

    with open(args.out, "w") as f:
        json.dump({"configs": out}, f, indent=2)
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
