#!/usr/bin/env python3
"""2.13.0 qualification copy of the P0 single-client harness (bench.py). Additions, all OUTSIDE the timed
region (outside each query's perf_counter window): warm_first_ms (first warm-up query = per-backend cache cold-open), threads_max (live task count
sampled after every timed query), thread_names (comm histogram at the end), slow_gt60/slow_gt100 counts.
P0 single-client harness. One long-lived connection per invocation.
usage: bench.py --db DB --sys {tv,pv} --label L --set "guc=val;guc=val" --mode {ids,explain} [--warm 200] [--n 5000] [--k 10]
ids:     prepared SELECT id ... ORDER BY emb <=> $1::T LIMIT k, client wall per query (perf_counter_ns), returned ids,
         recall@k vs GT, fp64 order check (D6). CPU of the backend: /proc/<pid>/stat (thread group, includes exited
         threads) AND sum over /proc/<pid>/task/*/stat (live threads only) before/after the timed block.
explain: EXPLAIN (ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON) with the literal inlined -> server Execution Time.
Output: <out>/<label>.<mode>.jsonl (per query) + <label>.<mode>.summary.json"""
import argparse, json, os, time, glob, numpy as np, psycopg
ap = argparse.ArgumentParser()
ap.add_argument("--db"); ap.add_argument("--sys"); ap.add_argument("--label"); ap.add_argument("--set", default="")
ap.add_argument("--mode", default="ids"); ap.add_argument("--warm", type=int, default=200); ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--k", type=int, default=10); ap.add_argument("--out", default=os.path.expanduser("~/results/p0/warm"))
ap.add_argument("--data", default="/data/co1024_1m"); ap.add_argument("--table", default="docs")
ap.add_argument("--expect-index", default=""); ap.add_argument("--qoff", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T = "vector" if a.sys == "pv" else "turbovec.vector"
Q = np.load(f"{a.data}/queries.npy"); GT = np.load(f"{a.data}/gt_ids.npy")
qtext = lambda i: "[" + ",".join("%.9g" % x for x in Q[i].tolist()) + "]"
conn = psycopg.connect(f"host=/tmp dbname={a.db}", autocommit=True, prepare_threshold=0)
cur = conn.cursor()
if a.sys == "pv":
    cur.execute("RESET prism.query_limit")
for kv in [s for s in a.set.split(";") if s.strip()]:
    g, v = kv.split("="); cur.execute(f"SET {g.strip()} = {v.strip()}")
bpid = cur.execute("SELECT pg_backend_pid()").fetchone()[0]
def cpu():
    with open(f"/proc/{bpid}/stat") as f: s = f.read().rsplit(")", 1)[1].split()
    tg = int(s[11]) + int(s[12])  # utime+stime fields 14,15 (thread group incl. exited threads)
    tk = 0; nt = 0
    for p in glob.glob(f"/proc/{bpid}/task/*/stat"):
        try:
            with open(p) as f: t = f.read().rsplit(")", 1)[1].split()
            tk += int(t[11]) + int(t[12]); nt += 1
        except FileNotFoundError: pass
    return tg, tk, nt
sql = f"SELECT id FROM {a.table} ORDER BY emb <=> %s::{T} LIMIT {a.k}"
ex = f"EXPLAIN (ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON) SELECT id FROM {a.table} ORDER BY emb <=> '{{}}'::{T} LIMIT {a.k}"
plan0 = cur.execute(f"EXPLAIN (VERBOSE, COSTS OFF) SELECT id FROM {a.table} ORDER BY emb <=> '{qtext(0)}'::{T} LIMIT {a.k}").fetchall()
plan0 = "\n".join(r[0] for r in plan0)
if a.expect_index: assert a.expect_index in plan0, plan0
settings = {r[0]: r[1] for r in cur.execute("SELECT name, setting FROM pg_settings WHERE name ~ '^(prism|turbovec|hnsw)\\.' OR name IN ('work_mem','shared_buffers','io_method','effective_io_concurrency','enable_seqscan')")}
if "prism" in plan0:
    assert settings.get("prism.query_limit") == "0", settings.get("prism.query_limit")
    assert settings.get("prism.leaf_refine_threshold") == "0"; assert settings.get("prism.recent_buffers") == "on"
qids = list(range(a.qoff, a.qoff + a.warm + a.n)); rows = []
def run(i):
    if a.mode == "ids":
        t0 = time.perf_counter_ns(); ids = [r[0] for r in cur.execute(sql, (qtext(i),))]; t1 = time.perf_counter_ns()
        return dict(qid=i, wall_us=(t1 - t0) / 1e3, ids=ids)
    j = cur.execute(ex.format(qtext(i))).fetchone()[0][0]
    p = j["Plan"]
    return dict(qid=i, exec_ms=j["Execution Time"], plan_ms=j["Planning Time"], sh_hit=p.get("Shared Hit Blocks"), sh_read=p.get("Shared Read Blocks"), rows=p.get("Actual Rows"))
wf = [run(i) for i in qids[:a.warm]]
warm_first_ms = wf[0]["wall_us"] / 1e3 if wf and "wall_us" in wf[0] else None
ntask = lambda: len(os.listdir(f"/proc/{bpid}/task"))
tmax = ntask()
la0 = open("/proc/loadavg").read().split()[0]; c0 = cpu(); w0 = time.time()
for i in qids[a.warm:]:
    rows.append(run(i)); tmax = max(tmax, ntask())
w1 = time.time(); c1 = cpu(); la1 = open("/proc/loadavg").read().split()[0]
smaps = {}
for line in open(f"/proc/{bpid}/smaps_rollup"):
    k, *v = line.split()
    if k.rstrip(":") in ("Rss", "Pss", "Pss_Anon", "Pss_Shmem", "Pss_File", "Anonymous"): smaps[k.rstrip(":")] = int(v[0])
tn = {}
for p in glob.glob(f"/proc/{bpid}/task/*/comm"):
    try: nm = open(p).read().strip(); tn[nm] = tn.get(nm, 0) + 1
    except FileNotFoundError: pass
hz = os.sysconf("SC_CLK_TCK"); n = len(rows)
summ = dict(label=a.label, mode=a.mode, sys=a.sys, db=a.db, set=a.set, n=n, warm=a.warm, k=a.k, backend_pid=bpid,
            batch_wall_s=w1 - w0, cpu_ms_per_q_threadgroup=(c1[0] - c0[0]) * 1000 / hz / n,
            cpu_ms_per_q_tasksum=(c1[1] - c0[1]) * 1000 / hz / n, threads_before=c0[2], threads_after=c1[2], threads_max=tmax, thread_names=tn, warm_first_ms=warm_first_ms,
            loadavg=[la0, la1], smaps_rollup_kb=smaps, plan=plan0, settings=settings)
if a.mode == "ids":
    lat = np.array([r["wall_us"] for r in rows]) / 1e3
    rec = np.array([len(set(r["ids"][:a.k]) & set(GT[r["qid"], :a.k].tolist())) / a.k for r in rows])
    C = np.load(f"{a.data}/corpus.npy", mmap_mode="r"); bad = 0; short = 0
    for r in rows:
        if len(r["ids"]) < a.k: short += 1
        d = 1 - C[r["ids"]].astype(np.float64) @ Q[r["qid"]].astype(np.float64)
        if np.any(np.diff(d) < -1e-5): bad += 1
    summ.update(recall=float(rec.mean()), wall_p50_ms=float(np.percentile(lat, 50)), wall_p95_ms=float(np.percentile(lat, 95)),
                wall_mean_ms=float(lat.mean()), wall_p99_ms=float(np.percentile(lat, 99)), wall_max_ms=float(lat.max()),
                slow_gt60=int((lat > 60).sum()), slow_gt100=int((lat > 100).sum()), order_violations=bad, short_results=short,
                lat_first10_ms=lat[:10].round(3).tolist(), lat_p50_first_half=float(np.median(lat[:n // 2])), lat_p50_second_half=float(np.median(lat[n // 2:])))
else:
    lat = np.array([r["exec_ms"] for r in rows])
    summ.update(exec_p50_ms=float(np.percentile(lat, 50)), exec_p95_ms=float(np.percentile(lat, 95)), exec_mean_ms=float(lat.mean()),
                plan_p50_ms=float(np.median([r["plan_ms"] for r in rows])), sh_read_total=int(sum(r["sh_read"] or 0 for r in rows)),
                sh_hit_mean=float(np.mean([r["sh_hit"] or 0 for r in rows])), rows_ne_k=int(sum(1 for r in rows if r["rows"] != a.k)))
with open(f"{a.out}/{a.label}.{a.mode}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump(summ, open(f"{a.out}/{a.label}.{a.mode}.summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k not in ("settings", "plan")}))
