#!/usr/bin/env bash
# Q4: /proc CPU accounting vs perf stat on one batch (turbovec flat, multi-threaded); Q5: pgbench \gset script +
# whole-core pinning; plus the RAYON_NUM_THREADS=1 secondary row (postmaster env).
set -x
export PATH=/opt/pg18/bin:$PATH; R=~/results/p0/q45; mkdir -p $R
P="psql -h /tmp -X -v ON_ERROR_STOP=1"
SIB=""; for c in 44 45 46 47; do SIB="$SIB,$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"; done; CLI=${SIB#,}
SRV=0-43,48-91; echo "client cpus=$CLI server cpus=$SRV" > $R/pinning.txt
start() { pg_ctl -D /data/pgdata -m fast -w stop; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  env "$@" taskset -c $SRV pg_ctl -D /data/pgdata -l ~/pg.log -w start
  PM=$(head -1 /data/pgdata/postmaster.pid); echo "postmaster $PM affinity: $(taskset -pc $PM)"; tr '\0' '\n' < /proc/$PM/environ | grep -E 'RAYON|GLIBC' || echo "no RAYON/GLIBC env"
  for db_r in db_tv:docs db_tv:docs_tv_flat db_tv:queries db_pv:docs db_pv:docs_prism db_pv:queries; do $P -d ${db_r%%:*} -Atc "SELECT pg_prewarm('${db_r#*:}')"; done; }
start >> $R/pinning.txt 2>&1
# ---- Q4: run a 500-query flat batch; attach perf stat to the backend for the whole batch
python3 - > $R/q4_perf.txt 2>&1 <<'PY'
import psycopg, numpy as np, subprocess, time, os, glob
Q = np.load("/data/co1024_1m/queries.npy"); qt = lambda i: "[" + ",".join("%.9g" % x for x in Q[i].tolist()) + "]"
for db, T, sql_t in (("db_tv", "turbovec.vector", "docs"), ("db_pv", "vector", "docs")):
    c = psycopg.connect(f"host=/tmp dbname={db}", autocommit=True, prepare_threshold=0); cur = c.cursor()
    pid = cur.execute("SELECT pg_backend_pid()").fetchone()[0]
    sql = f"SELECT id FROM {sql_t} ORDER BY emb <=> %s::{T} LIMIT 10"
    for i in range(50): cur.execute(sql, (qt(i),)).fetchall()
    def cpu():
        s = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); tg = int(s[11]) + int(s[12]); tk = 0
        for p in glob.glob(f"/proc/{pid}/task/*/stat"):
            t = open(p).read().rsplit(")", 1)[1].split(); tk += int(t[11]) + int(t[12])
        return tg, tk
    perf = subprocess.Popen(["sudo", "perf", "stat", "-e", "task-clock,context-switches,cpu-migrations", "-p", str(pid), "-o", f"/home/fedora/results/p0/q45/q4_perfstat_{db}.txt"])
    time.sleep(1.0); c0 = cpu(); w0 = time.time()
    for i in range(200, 700): cur.execute(sql, (qt(i),)).fetchall()
    w1 = time.time(); c1 = cpu(); time.sleep(0.2)
    subprocess.run(["sudo", "kill", "-INT", str(perf.pid)]); perf.wait(); time.sleep(0.5)
    hz = os.sysconf("SC_CLK_TCK")
    print(db, f"wall={w1-w0:.3f}s /proc thread-group cpu={(c1[0]-c0[0])/hz:.3f}s task-sum cpu={(c1[1]-c0[1])/hz:.3f}s per-query tg={(c1[0]-c0[0])*1000/hz/500:.2f}ms nthreads={len(glob.glob(f'/proc/{pid}/task/*'))}")
    print(open(f"/home/fedora/results/p0/q45/q4_perfstat_{db}.txt").read())
PY
# ---- Q5: pgbench \gset scripts (plan form uses psql-only :'qv' quoting; test it, then the :qv form)
for sys in tv pv; do
  if [ $sys = tv ]; then DB=db_tv; T=turbovec.vector; else DB=db_pv; T=vector; fi
  printf '\\set qid random(0, 5199)\nSELECT q::text AS qv FROM queries WHERE qid = :qid \\gset\nSELECT id FROM docs ORDER BY emb <=> :'"'"'qv'"'"'::%s LIMIT 10;\n' $T > $R/pgb_plan_form_$sys.sql
  printf '\\set qid random(0, 5199)\nSELECT q::text AS qv FROM queries WHERE qid = :qid \\gset\nSELECT id FROM docs ORDER BY emb <=> :qv::%s LIMIT 10;\n' $T > $R/pgb_$sys.sql
  printf '\\set qid random(0, 5199)\nSELECT q::text AS qv FROM queries WHERE qid = :qid \\gset\nINSERT INTO pgb_out SELECT :qid, array_agg(id) FROM (SELECT id FROM docs ORDER BY emb <=> :qv::%s LIMIT 10) s;\n' $T > $R/pgb_capture_$sys.sql
  $P -d $DB -c "DROP TABLE IF EXISTS pgb_out" -c "CREATE TABLE pgb_out(qid int, ids bigint[])"
  echo "== plan form ($sys)"; taskset -c $CLI pgbench -h /tmp -n -M prepared -t 5 -c 1 -f $R/pgb_plan_form_$sys.sql $DB 2>&1 | tail -5
  echo "== :qv form capture ($sys)"; taskset -c $CLI pgbench -h /tmp -n -M prepared -t 100 -c 1 -f $R/pgb_capture_$sys.sql $DB 2>&1 | grep -E "latency|tps|failed|error"
  $P -d $DB -Atc "EXPLAIN (COSTS OFF) INSERT INTO pgb_out SELECT 1, array_agg(id) FROM (SELECT id FROM docs ORDER BY emb <=> (SELECT q FROM queries WHERE qid=1) LIMIT 10) s" | head -8
  $P -d $DB -Atc "COPY (SELECT qid, array_to_string(ids, ',') FROM pgb_out) TO STDOUT" > $R/pgb_capture_$sys.tsv
  echo "== :qv form 30 s 4 clients pinned ($sys)"; taskset -c $CLI pgbench -h /tmp -n -M prepared -T 30 -c 4 -j 4 -P 10 -l --log-prefix=$R/pgb_log_$sys -f $R/pgb_$sys.sql $DB 2>&1 | grep -E "progress|latency|tps|failed|error" 
  PGB=$(pgrep -x pgbench | head -1); [ -n "$PGB" ] && taskset -pc $PGB
done > $R/q5_pgbench.txt 2>&1
# literal-form ids for the same qids, compare
python3 - > $R/q5_compare.txt 2>&1 <<'PY'
import psycopg, numpy as np
Q = np.load("/data/co1024_1m/queries.npy"); qt = lambda i: "[" + ",".join("%.9g" % x for x in Q[i].tolist()) + "]"
for sys, db, T in (("tv", "db_tv", "turbovec.vector"), ("pv", "db_pv", "vector")):
    cur = psycopg.connect(f"host=/tmp dbname={db}", autocommit=True).cursor()
    rows = [l.split("\t") for l in open(f"/home/fedora/results/p0/q45/pgb_capture_{sys}.tsv").read().split("\n") if l]
    same = 0; same_set = 0
    for qid, ids in rows:
        a = [int(x) for x in ids.split(",")]
        b = [r[0] for r in cur.execute(f"SELECT id FROM docs ORDER BY emb <=> '{qt(int(qid))}'::{T} LIMIT 10")]
        same += a == b; same_set += set(a) == set(b)
    print(sys, f"captured={len(rows)} identical_order={same} identical_set={same_set}")
PY
# ---- equal-CPU rows (lead directive): RAYON_NUM_THREADS in the POSTMASTER env, 1 and 4, vs default (unset)
for RT in 1 4 default; do
  if [ $RT = default ]; then start > $R/rayon_${RT}_start.txt 2>&1; else start RAYON_NUM_THREADS=$RT > $R/rayon_${RT}_start.txt 2>&1; fi
  $P -d db_tv -Atc "SELECT pg_prewarm('docs_ivf'), pg_prewarm('docs_ivf_tv')" >> $R/rayon_${RT}_start.txt
  python3 ~/bench.py --db db_tv --sys tv --label rayon${RT}_flat_default --mode ids --warm 50 --n 1000 --out $R --expect-index docs_tv_flat >> $R/rayon.txt 2>&1
  for sk in 16 32 64 128 256 512 1024; do
    python3 ~/bench.py --db db_tv --sys tv --label rayon${RT}_flat_hdroff_sk$sk --set "turbovec.hi_dim_rerank=off;turbovec.search_k=$sk" --mode ids --warm 50 --n 1000 --out $R --expect-index docs_tv_flat >> $R/rayon.txt 2>&1
  done
  python3 ~/bench.py --db db_tv --sys tv --label rayon${RT}_ivf_p16_whole --table docs_ivf --set "turbovec.out_of_core=off;turbovec.probes=16" --mode ids --warm 50 --n 1000 --out $R --expect-index docs_ivf_tv >> $R/rayon.txt 2>&1
  python3 ~/bench.py --db db_tv --sys tv --label rayon${RT}_ivf_p16_ooc --table docs_ivf --set "turbovec.out_of_core=on;turbovec.probes=16" --mode ids --warm 50 --n 1000 --out $R --expect-index docs_ivf_tv >> $R/rayon.txt 2>&1
done
echo Q45_DONE
