#!/usr/bin/env python3
"""Cold-cache cost per storage variant. Before EVERY measurement: pg_ctl stop
-m fast, sync + drop OS page cache, pg_ctl start (so shared_buffers is empty
too). Variants alternate within each rep.
  seq : SELECT count(*), sum(id) FROM t  (a scan that never reads tv), forced
        to a Seq Scan (index/IOS/bitmap off) -- stands in for any query that
        must read the heap but not the vector column; then re-run warm
  knn : first kNN query of a fresh backend, search_k=1024 (includes loading
        the turbovec index into the backend cache, identical for all variants)
usage: cold.py <dim> <reps> <variant>...   -> JSON on stdout"""
import sys, json, subprocess, time, numpy as np, psycopg
d, R, VS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
B = "/work/pg16rel_c/bin"; D = "/work/fixc/data"
q = np.load(f"/work/fixc/q_{d}.npy")[7]; lit = "[" + ",".join("%.9g" % x for x in q) + "]"
def cold_restart():
    subprocess.run([f"{B}/pg_ctl", "-D", D, "-m", "fast", "-w", "stop"], check=True, capture_output=True)
    subprocess.run("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null", shell=True, check=True)
    subprocess.run([f"{B}/pg_ctl", "-D", D, "-l", "/work/fixc/pg.log", "-w", "start"], check=True, capture_output=True)
def measure(v, mode):
    cold_restart()
    c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
    c.execute("SET jit = off; SET track_io_timing = on")
    if mode == "seq":
        c.execute("SET enable_indexscan = off; SET enable_indexonlyscan = off; SET enable_bitmapscan = off")
        sql = f"SELECT count(*), sum(id) FROM t{d}_{v}"
    else:
        c.execute("SET enable_seqscan = off; SET turbovec.oversample = 1.0; SET turbovec.hi_dim_rerank = off; SET turbovec.search_k = 1024")
        sql = f"SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '{lit}'::turbovec.vector LIMIT 10"
    t0 = time.perf_counter()
    p = c.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) {sql}").fetchone()[0][0]
    wall = (time.perf_counter() - t0) * 1e3
    warm = c.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) {sql}").fetchone()[0][0]
    c.close()
    pl = p["Plan"]; node = pl["Plans"][0]["Node Type"]
    assert node == ("Seq Scan" if mode == "seq" else "Index Scan"), node
    return {"exec_ms": p["Execution Time"], "wall_ms": wall, "node": node,
            "shared_read_blocks": pl.get("Shared Read Blocks"), "io_read_ms": pl.get("I/O Read Time"),
            "warm_exec_ms": warm["Execution Time"], "warm_shared_hit": warm["Plan"].get("Shared Hit Blocks")}
out = {f"{m}/{v}": [] for m in ("seq", "knn") for v in VS}
for r in range(R):
    order = VS[r % len(VS):] + VS[:r % len(VS)]
    for m in ("seq", "knn"):
        for v in order:
            out[f"{m}/{v}"].append(measure(v, m)); print(m, v, out[f"{m}/{v}"][-1], file=sys.stderr, flush=True)
print(json.dumps({"dim": d, "reps": R, "variants": VS, "runs": out}))
