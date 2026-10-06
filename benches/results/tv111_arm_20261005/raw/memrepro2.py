#!/usr/bin/env python3
"""Discriminate: (a) scans only, no concurrent writes; (b) scans + concurrent
commits. If (a) is flat and (b) grows by ~index-size per round, the leak is
'stale cache entry evicted while a leaked scan Arc still pins it'."""
import sys, psycopg
def conn():
    c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True); c.execute("SET search_path=turbovec,public"); return c
def rss(pid):
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("RssAnon"): return int(l.split()[1]) // 1024
s = conn(); pid = s.execute("SELECT pg_backend_pid()").fetchone()[0]
q = "SELECT id FROM mr ORDER BY tv <=> (SELECT tv FROM mr WHERE id = 7) LIMIT 10"
a = []
for i in range(20): s.execute(q).fetchall(); a.append(rss(pid))
print(sys.argv[1], "(a) scans only:", a[0], "->", a[-1], "MB")
w = conn(); b = []
for i in range(10):
    w.execute("INSERT INTO mr SELECT id + 10000000, tv FROM mr WHERE id = %s", (i,))
    s.execute(q).fetchall(); b.append(rss(pid))
print(sys.argv[1], "(b) scan after each commit:", b)
