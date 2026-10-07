#!/bin/sh
# Flat self-time profile of the backend running 200 warm kNN queries
# (search_k=1024, oversample 1, hi_dim_rerank off). usage: prof.sh <table> <dim>
T=$1; D=$2; OUT=/work/fixc/prof_$T
/work/venv/bin/python - "$T" "$D" "$OUT" <<'PY'
import sys, subprocess, time, signal, numpy as np, psycopg
t, d, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
c.execute("SET enable_seqscan=off; SET jit=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k=1024")
Q = np.load(f"/work/fixc/q_{d}.npy")
sql = [f"SELECT id FROM {t} ORDER BY tv OPERATOR(turbovec.<=>) '[" + ",".join("%.9g" % x for x in q) + "]'::turbovec.vector LIMIT 10" for q in Q]
for s in sql: c.execute(s).fetchall()
pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
p = subprocess.Popen(["sudo", "perf", "record", "-q", "-e", "task-clock", "-c", "100000", "-t", str(pid), "-o", out + ".data"], stderr=subprocess.DEVNULL)
time.sleep(0.5); t0 = time.perf_counter()
for s in sql: c.execute(s).fetchall()
wall = time.perf_counter() - t0
subprocess.run(["sudo", "kill", "-INT", str(p.pid)]); p.wait()
print(f"{t}: {len(sql)} queries, wall {wall*1e3/len(sql):.2f} ms/query")
PY
sudo perf report -i $OUT.data --no-children --sort dso,sym --stdio -q 2>/dev/null | head -25
