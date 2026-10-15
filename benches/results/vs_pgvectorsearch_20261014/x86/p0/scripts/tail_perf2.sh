#!/bin/bash
set -uo pipefail
export PATH=/opt/pg18/bin:$PATH
R=~/results/tailprobe; mkdir -p $R; cd /tmp
for SP in 8 1; do
  rm -f /tmp/tp_go /tmp/tp_pid /tmp/tp_lat
  python3 - "$SP" > /tmp/tp_lat 2>&1 <<'PY' &
import psycopg, time, sys
conn=psycopg.connect("host=/tmp dbname=db_tv", autocommit=True); cur=conn.cursor()
for s in ("turbovec.cache_size_mb=256","turbovec.out_of_core=on","turbovec.probes=16",f"turbovec.scan_parallelism={sys.argv[1]}"):
    g,v=s.split("="); cur.execute(f"SET {g}={v}")
open('/tmp/tp_pid','w').write(str(cur.execute("select pg_backend_pid()").fetchone()[0]))
vecs=[r[0] for r in cur.execute("SELECT emb::text FROM docs LIMIT 120")]
for v in vecs[:20]: cur.execute("SELECT id FROM docs ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(v,)).fetchall()
open('/tmp/tp_go','w').write('1')
lat=[]; t0=time.time()
while time.time()-t0 < 12:
    for v in vecs[20:]:
        t=time.perf_counter(); cur.execute("SELECT id FROM docs ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(v,)).fetchall(); lat.append((time.perf_counter()-t)*1000)
lat.sort(); print(f"n={len(lat)} p50={lat[len(lat)//2]:.1f} p95={lat[int(.95*len(lat))]:.1f}")
PY
  while [ ! -f /tmp/tp_go ]; do sleep 0.1; done
  PID=$(cat /tmp/tp_pid)
  sudo perf record -q -g -F 999 -p $PID -o /tmp/perf_sp$SP.data -- sleep 8 > /dev/null 2>&1
  wait
  echo "== scan_parallelism=$SP $(cat /tmp/tp_lat)"
  sudo perf report -q -i /tmp/perf_sp$SP.data --stdio --no-children --sort sym 2>/dev/null | grep -E '^ +[0-9]+\.[0-9]+%' | head -12
done
echo PERF2_DONE
