#!/usr/bin/env python3
"""Count mmap/munmap/brk syscalls of a warm backend over 50 kNN queries
(search_k=1024), to explain the minor-fault 'slow state' seen under MAIN/PLAIN.
usage: strace_alloc.py <dim> <variant>"""
import sys, subprocess, time, numpy as np, psycopg
d, v = int(sys.argv[1]), sys.argv[2]
Q = np.load(f"/work/fixc/q_{d}.npy")[:50]
sel = [f"SELECT id FROM t{d}_{v} ORDER BY tv OPERATOR(turbovec.<=>) '[" + ",".join("%.9g" % x for x in q) + "]'::turbovec.vector LIMIT 10" for q in Q]
c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
c.execute("SET enable_seqscan=off; SET jit=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k=1024")
pid = c.execute("SELECT pg_backend_pid()").fetchone()[0]
for s in sel: c.execute(s).fetchall()
p = subprocess.Popen(["sudo", "strace", "-f", "-qq", "-e", "trace=mmap,munmap,brk", "-p", str(pid), "-o", f"/work/fixc/strace_{v}.txt"])
time.sleep(1.0)
for s in sel: c.execute(s).fetchall()
time.sleep(0.5); subprocess.run(["sudo", "kill", "-INT", str(p.pid)]); p.wait()
lines = open(f"/work/fixc/strace_{v}.txt").read().splitlines()
import collections, re
cnt = collections.Counter(re.sub(r"^\d+\s+", "", l).split("(")[0] for l in lines)
big = [l for l in lines if "mmap(NULL" in l]
sizes = collections.Counter(int(l.split(",")[1]) for l in big)
print(v, "per query:", {k: round(n / len(sel), 1) for k, n in cnt.items()}, "mmap sizes:", sizes.most_common(4))
