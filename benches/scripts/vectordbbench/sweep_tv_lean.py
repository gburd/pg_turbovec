#!/usr/bin/env python3
"""Lean turbovec sweep for the slow pre-AVX2 full-scan reality.

Single pass per search_k: NQ plain queries timed with clock_timestamp on the
client (psycopg perf_counter). No EXPLAIN double-run (halves wall-clock).
Captures top-10 ids for recall@10. Latency is ~flat across search_k (full
quantized scan dominates), so we report each k's measured latency anyway.

Client-side timing here includes the query-vector cast + tiny RTT over a
unix socket (sub-ms); the dominant term is the in-engine scan, so the
numbers are comparable to the server-exec numbers used for HNSW. We note
this in the artifact (latency_basis).
"""
import argparse, json, statistics, time, numpy as np, psycopg

DSN = "host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_wiki"
TV_Q = ("SELECT id FROM public.docs ORDER BY (emb::real[]::turbovec.vector) "
        "OPERATOR (turbovec.<=>) (%s::real[]::turbovec.vector) LIMIT 10")

def pctl(s, p):
    s = sorted(s); k = (len(s) - 1) * p; lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo+1] - s[lo])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--ks", default="100,500,1000")
    ap.add_argument("--nq", type=int, default=12)
    ap.add_argument("--queries", default="/scratch/pg_turbovec-bench/cohere-wiki/q1000.npy")
    ap.add_argument("--gt", default="/scratch/pg_turbovec-bench/cohere-wiki/gt_top10.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Q = np.load(args.queries); GT = np.load(args.gt)
    ks = [int(x) for x in args.ks.split(",")]
    print(f"queries {Q.shape}  bits={args.bits}  ks={ks}  nq={args.nq}", flush=True)

    out = []
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SET search_path=turbovec,public")
        cur.execute("SET enable_seqscan=off")
        cur.execute("SELECT indexrelid::regclass::text FROM pg_index "
                    "WHERE indrelid='public.docs'::regclass AND indexrelid::regclass::text LIKE '%tv%'")
        print("tv indexes present:", [r[0] for r in cur.fetchall()], flush=True)
        cur.execute(f"SET turbovec.search_k={ks[0]}")
        cur.execute(TV_Q, (Q[0].tolist(),)); cur.fetchall()   # 1 warmup
        for k in ks:
            cur.execute(f"SET turbovec.search_k={k}")
            times, tops = [], []
            for i in range(args.nq):
                v = Q[i % Q.shape[0]].tolist()
                t0 = time.perf_counter()
                cur.execute(TV_Q, (v,))
                ids = [r[0] for r in cur.fetchall()]
                times.append((time.perf_counter() - t0) * 1000.0)
                tops.append(ids)
            rec = float(np.mean([len(set(int(x) for x in GT[i]) & set(tops[i])) / 10.0
                                 for i in range(len(tops))]))
            total_s = sum(times) / 1000.0
            r = {"label": f"tv_{args.bits}b_k{k}", "engine": f"pg_turbovec_{args.bits}bit",
                 "param": {"bit_width": args.bits, "search_k": k}, "n_queries": len(times),
                 "latency_basis": "client_wall_unix_socket", "recall_at_10": round(rec, 4),
                 "mean_ms": round(statistics.mean(times), 1), "p50_ms": round(pctl(times, .5), 1),
                 "p95_ms": round(pctl(times, .95), 1), "p99_ms": round(pctl(times, .99), 1),
                 "min_ms": round(min(times), 1), "max_ms": round(max(times), 1),
                 "qps_single_conn": round(len(times) / total_s, 4)}
            print(f"  {r['label']}: R@10={rec:.4f} p50={r['p50_ms']}ms qps={r['qps_single_conn']}", flush=True)
            out.append(r)
            json.dump({"configs": out}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
