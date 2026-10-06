#!/usr/bin/env python3
"""sha256 of an index's meta page + codes/scales/ids chains (payload bytes only,
skipping PG page headers, whose LSN differs run to run)."""
import sys, struct, hashlib, psycopg
c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True)
c.execute("CREATE EXTENSION IF NOT EXISTS pageinspect")
rel = sys.argv[1]
def pg(b): return c.execute("SELECT get_raw_page(%s,%s)", (rel, b)).fetchone()[0]
m = pg(0)[24:]
n, = struct.unpack_from("<Q", m, 12)
cf, _, sf, _, idf, _, rpc, rps, rpi, stride = struct.unpack_from("<IIIIIIIIII", m, 20)
h = hashlib.sha256()
for first, st, rpp in ((cf, stride, rpc), (sf, 4, rps), (idf, 8, rpi)):
    left = n * st; b = first
    while left:
        t = min(rpp * st, left); h.update(pg(b)[24:24 + t]); left -= t; b += 1
# meta minus am_version (offset 60..64) which is a counter
print(rel, "n=%d stride=%d" % (n, stride), "chains_sha256=" + h.hexdigest()[:16],
      "meta_sha256=" + hashlib.sha256(m[:60] + m[64:512]).hexdigest()[:16])
