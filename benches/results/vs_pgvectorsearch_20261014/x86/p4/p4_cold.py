#!/usr/bin/env python3
"""P4 cold + new-connection cost (plan D4.2). One (tag,kind,sys) per call.
 (a) cold: pg_ctl stop; drop_caches; start; time-to-first-result (1 query) + first-100 p50/p95.
 (b) new-connection: OS/shared warm, open a FRESH connection, time 200 queries (both AMs build per-backend state).
usage: p4_cold.py --tag co1m --data /data/co1024_1m --kind flat --sys tv --knob 'turbovec.search_k=16;turbovec.hi_dim_rerank=off' --out ~/results/p4"""
import argparse,json,os,subprocess,time,numpy as np,psycopg
ap=argparse.ArgumentParser()
ap.add_argument("--tag");ap.add_argument("--data");ap.add_argument("--kind");ap.add_argument("--sys");ap.add_argument("--knob",default="");ap.add_argument("--out")
a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
PGDATA="/data/pgdata";PGBIN="/opt/pg18/bin";ENV=dict(os.environ,PATH=f"{PGBIN}:"+os.environ["PATH"],LD_LIBRARY_PATH="/opt/pg18/lib")
Q=np.load(f"{a.data}/queries.npy");GT=np.load(f"{a.data}/gt_ids.npy")
qtext=lambda i:"["+",".join("%.9g"%x for x in Q[i].tolist())+"]"
db="db_tv" if a.sys=="tv" else "db_pv";VT="turbovec.vector" if a.sys=="tv" else "vector"
if a.kind=="flat": tbl=f"dtv_flat_{a.tag}" if a.sys=="tv" else f"dpv_prism_{a.tag}"
else: tbl=f"dtv_ivf_{a.tag}" if a.sys=="tv" else f"dpv_prism_{a.tag}"
def sql(cur,i): return [r[0] for r in cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::{VT} LIMIT 10",(qtext(i),))]
def setk(cur):
    if a.sys=="pv": cur.execute("RESET prism.query_limit")
    for kv in [s for s in a.knob.split(";") if s.strip()]:
        g,v=kv.split("=");cur.execute(f"SET {g.strip()} = {v.strip()}")
# (a) cold
subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-m","fast","-w","stop"],env=ENV,check=True)
subprocess.run("sync",shell=True);subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null",shell=True)
subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-l",os.path.expanduser("~/pg.log"),"-w","start"],env=ENV,check=True)
conn=psycopg.connect(f"host=/tmp dbname={db}",autocommit=True,prepare_threshold=0);cur=conn.cursor();setk(cur)
t0=time.perf_counter_ns();sql(cur,0);ttfr=(time.perf_counter_ns()-t0)/1e6
cold=[]
for i in range(1,101):
    t0=time.perf_counter_ns();sql(cur,i);cold.append((time.perf_counter_ns()-t0)/1e6)
conn.close()
cold=np.array(cold)
# (b) new-connection cost (OS+shared warm now): fresh conn, 200 queries, record first-query vs rest
conn=psycopg.connect(f"host=/tmp dbname={db}",autocommit=True,prepare_threshold=0);cur=conn.cursor();setk(cur)
nc=[]
for i in range(200):
    t0=time.perf_counter_ns();sql(cur,i);nc.append((time.perf_counter_ns()-t0)/1e6)
conn.close();nc=np.array(nc)
out=dict(tag=a.tag,kind=a.kind,sys=a.sys,knob=a.knob,table=tbl,
    cold_ttfr_ms=float(ttfr),cold_first100_p50=float(np.percentile(cold,50)),cold_first100_p95=float(np.percentile(cold,95)),cold_first100_max=float(cold.max()),
    newconn_q0_ms=float(nc[0]),newconn_q1_ms=float(nc[1]),newconn_rest_p50=float(np.percentile(nc[1:],50)),newconn_rest_p95=float(np.percentile(nc[1:],95)))
json.dump(out,open(f"{a.out}/p4_{a.tag}_{a.kind}_{a.sys}.json","w"),indent=1)
print(json.dumps(out))
