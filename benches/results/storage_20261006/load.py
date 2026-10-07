#!/usr/bin/env python3
"""Load one table per (dim, storage variant). SET STORAGE is applied BEFORE
the COPY, so rows are stored the way the variant says. VACUUM ANALYZE +
CHECKPOINT afterwards so later cold scans don't pay hint-bit writes.
usage: load.py <dim> <variant>... [--index]   variant in ext|extl|main|plain|ttt
  ext = default (EXTENDED), extl = EXTERNAL,
  ttt = default column storage + ALTER TABLE ... SET (toast_tuple_target = 8160)"""
import sys, time, psycopg
d = int(sys.argv[1]); idx = "--index" in sys.argv
vs = [a for a in sys.argv[2:] if not a.startswith("--")]
ST = {"ext": None, "extl": "EXTERNAL", "main": "MAIN", "plain": "PLAIN", "ttt": None}
c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
for v in vs:
    t = f"t{d}_{v}"
    c.execute(f"DROP TABLE IF EXISTS {t}")
    c.execute(f"CREATE TABLE {t} (id bigint PRIMARY KEY, tv turbovec.vector)")
    if ST[v]:
        c.execute(f"ALTER TABLE {t} ALTER COLUMN tv SET STORAGE {ST[v]}")
    if v == "ttt":
        c.execute(f"ALTER TABLE {t} SET (toast_tuple_target = 8160)")
    t0 = time.perf_counter()
    try:
        c.execute(f"COPY {t} (id, tv) FROM '/work/fixc/data_{d}.tsv'")
        res = f"copy {time.perf_counter()-t0:.1f}s"
    except psycopg.Error as e:
        res = f"COPY FAILED after {time.perf_counter()-t0:.1f}s: {str(e).splitlines()[0]}"
    c.execute(f"VACUUM ANALYZE {t}")
    if idx and "FAILED" not in res:
        t0 = time.perf_counter()
        c.execute(f"CREATE INDEX {t}_tv ON {t} USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4)")
        res += f", index {time.perf_counter()-t0:.1f}s"
    c.execute("CHECKPOINT")
    print(t, res, flush=True)
