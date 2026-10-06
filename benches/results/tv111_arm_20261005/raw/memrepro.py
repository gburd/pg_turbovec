#!/usr/bin/env python3
"""Does ONE long-lived backend's RSS grow without bound when it repeatedly scans
an index that other backends keep inserting into (each commit bumps
am_version -> the scanner's cache entry is stale -> rebuild)? Run per arm.
Reports the scanner backend's RSS (anon) after each of N rounds."""
import os, sys, time, random, psycopg, numpy as np
ARM = sys.argv[1]; ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
C = np.load("/mnt/nvme/corpus_1000000/q.npy")
def vec(i):
    v = C[i % len(C)] + 0.05 * np.random.default_rng(i).standard_normal(1024).astype(np.float32)
    v /= np.linalg.norm(v); return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
def conn():
    c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True); c.execute("SET search_path=turbovec,public"); return c
a = conn()
a.execute("DROP TABLE IF EXISTS mr"); a.execute("CREATE TABLE mr(id bigint, tv turbovec.vector)")
with a.cursor().copy("COPY mr(id, tv) FROM STDIN") as cp:
    for i in range(200_000): cp.write_row((i, vec(i)))
a.execute("CREATE INDEX mr_idx ON mr USING turbovec (tv vec_cosine_ops) WITH (bit_width = 4)")
scanner = conn(); spid = scanner.execute("SELECT pg_backend_pid()").fetchone()[0]
w = conn(); nid = 200_000
def rss(pid):
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("RssAnon"): return int(l.split()[1]) // 1024
out = []
for r in range(ROUNDS):
    with w.transaction():
        with w.cursor().copy("COPY mr(id, tv) FROM STDIN") as cp:
            for i in range(nid, nid + 128): cp.write_row((i, vec(i)))
    nid += 128
    scanner.execute("SELECT id FROM mr ORDER BY tv <=> (SELECT tv FROM mr WHERE id = 7) LIMIT 10").fetchall()
    out.append(rss(spid))
print(ARM, "scanner RssAnon MB per round:", out, flush=True)
print(ARM, f"growth first->last: {out[0]} -> {out[-1]} MB", flush=True)
