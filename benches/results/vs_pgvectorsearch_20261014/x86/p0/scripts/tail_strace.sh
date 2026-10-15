#!/bin/bash
set -uo pipefail
export PATH=/opt/pg18/bin:$PATH
R=~/results/tailprobe; mkdir -p $R
# one backend: warm, then strace -f -T it while it runs 60 OOC p16 sp4 queries
python3 - <<'PY' &
import psycopg, numpy as np, time, os
Q=np.load('/data/co1024_1m/queries.npy') if os.path.exists('/data/co1024_1m/queries.npy') else None
conn=psycopg.connect("host=/tmp dbname=db_tv", autocommit=True); cur=conn.cursor()
for s in ("turbovec.cache_size_mb=256","turbovec.out_of_core=on","turbovec.probes=16","turbovec.scan_parallelism=4"):
    g,v=s.split("="); cur.execute(f"SET {g}={v}")
pid=cur.execute("select pg_backend_pid()").fetchone()[0]; open('/tmp/tail_pid','w').write(str(pid))
vecs=[r[0] for r in cur.execute("SELECT emb::text FROM docs LIMIT 80")]
for v in vecs[:20]: cur.execute("SELECT id FROM docs ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(v,)).fetchall()
open('/tmp/tail_go','w').write('1'); time.sleep(3)
lat=[]
for v in vecs[20:80]:
    t=time.perf_counter(); cur.execute("SELECT id FROM docs ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(v,)).fetchall(); lat.append((time.perf_counter()-t)*1000)
open(os.path.expanduser('~/results/tailprobe/strace_lat.txt'),'w').write(" ".join(f"{x:.1f}" for x in lat))
PY
while [ ! -f /tmp/tail_go ]; do sleep 0.2; done
PID=$(cat /tmp/tail_pid)
sudo timeout 25 strace -f -T -e trace=futex,clock_nanosleep,nanosleep,sched_yield,epoll_wait,poll,select,pread64 -p $PID -o $R/strace.txt 2>/dev/null
wait
rm -f /tmp/tail_go /tmp/tail_pid
echo "latencies: $(cat $R/strace_lat.txt)"
echo "--- syscalls taking > 50 ms:"
grep -E '<0\.(0[5-9]|[1-9])[0-9]*>|<[1-9][0-9]*\.[0-9]+>' $R/strace.txt | sed -E 's/^[0-9]+ +//' | awk '{print $1}' | sort | uniq -c | sort -rn | head
grep -E '<0\.(0[5-9]|[1-9])[0-9]*>' $R/strace.txt | head -8
echo STRACE_DONE
