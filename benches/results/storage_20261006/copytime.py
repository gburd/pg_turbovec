#!/usr/bin/env python3
"""COPY wall time per storage variant (does EXTENDED's compression attempt cost
anything at insert?). Fresh table per run, variants rotated per round.
usage: copytime.py <dim> <rounds> <variant>...   -> JSON on stdout"""
import sys, json, time, statistics as st, psycopg
d, R, VS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
ST = {"ext": None, "extl": "EXTERNAL", "main": "MAIN", "plain": "PLAIN"}
c = psycopg.connect("host=/work/fixc port=55433 dbname=bench", autocommit=True)
out = {v: [] for v in VS}
for r in range(R):
    for v in VS[r % len(VS):] + VS[:r % len(VS)]:
        c.execute("DROP TABLE IF EXISTS copy_t"); c.execute("CREATE TABLE copy_t (id bigint, tv turbovec.vector)")
        if ST[v]: c.execute(f"ALTER TABLE copy_t ALTER COLUMN tv SET STORAGE {ST[v]}")
        c.execute("CHECKPOINT")
        t0 = time.perf_counter(); c.execute(f"COPY copy_t FROM '/work/fixc/data_{d}.tsv'"); out[v].append(time.perf_counter() - t0)
c.execute("DROP TABLE copy_t")
print(json.dumps({"dim": d, "rounds": R, "copy_s": out, "median_s": {v: st.median(x) for v, x in out.items()}}))
