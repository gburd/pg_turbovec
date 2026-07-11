#!/usr/bin/env python3
"""Q-1 recall+storage sweep for a loaded corpus table.
Builds flat + IVF (given lists/bit_width), measures build time and
pg_relation_size, then sweeps probes/search_k x hi_dim_rerank{off,auto}
measuring recall@10 AND recall@100 against the exact GT .npy.

Usage:
  q1_sweep.py <table> <dim> <flat|ivf> [lists] [bit_width]

Reads <table>_queries.npy / <table>_gt.npy from /mnt/nvme/data.
Writes JSON to /mnt/nvme/results/<tag>.json
"""
import sys, time, json, statistics
import numpy as np
import psycopg2

SOCK = "/mnt/nvme/pg"; DB = "vecbench"
DATA = "/mnt/nvme/data"; RES = "/mnt/nvme/results"


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor(); cur.execute("SET search_path=public,turbovec"); c.commit(); cur.close()
    return c


def vlit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def drop_all(tab):
    c = connect(); c.autocommit = True; cur = c.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s", (tab,))
    for (nm,) in cur.fetchall():
        cur.execute(f"DROP INDEX IF EXISTS {nm} CASCADE")
    c.close()


def build(tab, kind, lists, bw):
    drop_all(tab)
    idx = f"{tab}_{kind}"
    c = connect(); c.autocommit = True; cur = c.cursor()
    cur.execute("SET max_parallel_maintenance_workers=32")
    cur.execute("SET maintenance_work_mem='32GB'")
    cur.execute(f"SET turbovec.bit_width_default={bw}")
    if kind == "flat":
        create = f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) WITH (bit_width={bw})"
    else:
        create = f"CREATE INDEX {idx} ON {tab} USING turbovec (embt turbovec.vec_l2_ops) WITH (lists={lists}, bit_width={bw})"
    t0 = time.time(); cur.execute(create); build_s = time.time() - t0
    cur.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s), count(*) FROM " + tab, (idx, idx))
    rel, tot, n = cur.fetchone()
    c.close()
    return idx, {"kind": kind, "lists": lists, "bit_width": bw, "build_s": round(build_s, 1),
                 "idx_bytes": rel, "idx_total_bytes": tot, "n": n}


def recall_at(res, truth, k):
    t = set(int(x) for x in truth[:k] if x >= 0)
    if not t:
        return None
    return len(t & set(res[:k])) / len(t)


def measure(tab, setup_sql, test, gt, kmax=100):
    """One pass: recall@10, recall@100, p50 (best-of-3 mean latency)."""
    c = connect(); c.autocommit = True; cur = c.cursor()
    for s in setup_sql:
        cur.execute(s)
    q = f"SELECT id FROM {tab} ORDER BY embt <-> %s::turbovec.vector LIMIT {kmax}"
    # warm
    for i in range(len(test)):
        cur.execute(q, (vlit(test[i]),)); cur.fetchall()
    r10 = r100 = 0.0; n10 = n100 = 0; lats = []
    for i in range(len(test)):
        t0 = time.perf_counter()
        cur.execute(q, (vlit(test[i]),))
        res = [row[0] for row in cur.fetchall()]
        lats.append((time.perf_counter() - t0) * 1000.0)
        a = recall_at(res, gt[i], 10)
        b = recall_at(res, gt[i], 100)
        if a is not None: r10 += a; n10 += 1
        if b is not None: r100 += b; n100 += 1
    c.close()
    lats.sort()
    return {"recall10": round(r10 / n10, 4), "recall100": round(r100 / n100, 4),
            "p50": round(lats[len(lats)//2], 2), "mean": round(statistics.mean(lats), 2),
            "qps_1conn": round(1000.0 / statistics.mean(lats), 1)}


def main():
    tab, dim, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lists = int(sys.argv[4]) if len(sys.argv) > 4 and kind == "ivf" else 0
    bw = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    test = np.load(f"{DATA}/{tab}_queries.npy").astype(np.float32)
    gt = np.load(f"{DATA}/{tab}_gt.npy").astype(np.int64)
    tag = f"{tab}_{kind}" + (f"_L{lists}" if kind == "ivf" else "") + f"_b{bw}"
    print(f"[{tag}] q={len(test)} dim={dim}", flush=True)
    idx, b = build(tab, kind, lists, bw)
    print(f"[{tag}] build_s={b['build_s']} idx_MB={b['idx_bytes']/1e6:.0f} "
          f"per_vec_B={b['idx_bytes']/b['n']:.1f} n={b['n']}", flush=True)
    # CRITICAL (v1.25.0 semantics): hi_dim_rerank=auto only auto-widens the
    # exact-recheck window to the dim-floor when search_k is LEFT AT DEFAULT
    # (32). An explicit search_k override disables auto. So:
    #   off  rows -> set search_k explicitly (manual recheck window)
    #   auto rows -> DO NOT set search_k (leave default 32 so the floor engages)
    # This mirrors benches/.../tv_leg.py, the run that established the fix works.
    rows = []
    if kind == "flat":
        combos = [("off", None, 100), ("off", None, 500), ("auto", None, None)]
    else:
        combos = [("off", p, sk) for p in [8, 16, 32, 64, 128] for sk in [100, 500]] + \
                 [("auto", p, None) for p in [8, 16, 32, 64, 128]]
    for rr, p, sk in combos:
        setup = ["SET enable_seqscan=off", "SET turbovec.out_of_core=off",
                 f"SET turbovec.hi_dim_rerank={rr}"]
        if sk is not None:
            setup.append(f"SET turbovec.search_k={sk}")
        if p is not None:
            setup.append(f"SET turbovec.probes={p}")
        m = measure(tab, setup, test, gt)
        row = {**b, "rerank": rr, "probes": p, "search_k": sk, **m}
        rows.append(row)
        print(f"  rerank={rr} p={p} sk={sk}: R@10={m['recall10']} R@100={m['recall100']} "
              f"p50={m['p50']}ms", flush=True)
        with open(f"{RES}/{tag}.json", "w") as f:
            json.dump(rows, f, indent=2)
    print(f"  wrote {RES}/{tag}.json", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
