#!/usr/bin/env python3
"""P3 warm single-client frontier for ONE (dataset,scale,host).
Steps, all in one invocation (keeps one restart budget):
 1. calibration grid per system (recall@10 only, 1 conn/point) -> operating_points.json (bracket 4/target).
 2. round 0 (discarded) + ABBA x2 counted, both systems, 'ids' pass over --n timed queries,
    at the UNION of each system's selected knob values. restart+drop_caches+prewarm before every arm-pass.
 3. one 'explain' pass per system per knob (server Execution Time + buffers), round 0 only (untimed validity).
Writes <out>/calib_<sys>.jsonl, operating_points.json, round<r>_<arm>_<sys>_<knob>.ids.summary.json.
Latency aggregation (median-of-rounds per query, bootstrap CI) is done offline by p3_agg.py.
usage: p3_frontier.py --tag co1m --data /data/co1024_1m --scale 1000000 --kind flat|ivf --n 5000 --out ~/results/p3/co1m"""
import argparse, json, os, subprocess, time, glob, numpy as np, psycopg
ap = argparse.ArgumentParser()
ap.add_argument("--tag"); ap.add_argument("--data"); ap.add_argument("--scale", type=int)
ap.add_argument("--kind", choices=["flat","ivf"]); ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--warm", type=int, default=200); ap.add_argument("--out"); ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--calib-n", type=int, default=0, help="queries for calibration recall (0=same as --n)")
ap.add_argument("--tvivf-max", type=int, default=256); ap.add_argument("--prism-max", type=int, default=1024)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
PGDATA="/data/pgdata"; PGBIN="/opt/pg18/bin"; ENV=dict(os.environ, PATH=f"{PGBIN}:"+os.environ["PATH"], LD_LIBRARY_PATH="/opt/pg18/lib")
GT = np.load(f"{a.data}/gt_ids.npy"); Q = np.load(f"{a.data}/queries.npy")
C = np.load(f"{a.data}/corpus.npy", mmap_mode="r")
qtext = lambda i: "[" + ",".join("%.9g" % x for x in Q[i].tolist()) + "]"
# per-system config: (sys, db, table, index, knob, extra, grid)
PRISM_GRID=[4,6,8,10,14,20,28,40,56,80,112,160,224,320,448,640,1024]
TVFLAT_GRID=[10,16,24,32,48,64,96,128,192,256,384,512,768,1024]
TVIVF_GRID=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256]
if a.kind == "flat":
    SYS = [("tv","db_tv",f"dtv_flat_{a.tag}",f"ixtvflat_{a.tag}","turbovec.search_k",";turbovec.hi_dim_rerank=off",TVFLAT_GRID),
           ("pv","db_pv",f"dpv_prism_{a.tag}",f"ixprism_{a.tag}","prism.nprobe","",[v for v in PRISM_GRID if v<=a.prism_max])]
else:
    SYS = [("tv","db_tv",f"dtv_ivf_{a.tag}",f"ixtvivf_{a.tag}","turbovec.probes","",[v for v in TVIVF_GRID if v<=a.tvivf_max]),
           ("pv","db_pv",f"dpv_prism_{a.tag}",f"ixprism_{a.tag}","prism.nprobe","",[v for v in PRISM_GRID if v<=a.prism_max])]
CALN = a.calib_n or a.n
def pg(cmd): subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-l",os.path.expanduser("~/pg.log"),"-w",cmd], env=ENV, check=True)
def restart_prewarm(db, rels):
    subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-m","fast","-w","stop"], env=ENV, check=True)
    subprocess.run("sync", shell=True); subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null", shell=True)
    subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-l",os.path.expanduser("~/pg.log"),"-w","start"], env=ENV, check=True)
    c = psycopg.connect(f"host=/tmp dbname={db}", autocommit=True)
    for r in rels: c.execute(f"SELECT pg_prewarm('{r}')")
    c.close()
def connect(db, sets):
    c = psycopg.connect(f"host=/tmp dbname={db}", autocommit=True, prepare_threshold=0); cur = c.cursor()
    if db == "db_pv": cur.execute("RESET prism.query_limit")
    for kv in [s for s in sets.split(";") if s.strip()]:
        g,v = kv.split("="); cur.execute(f"SET {g.strip()} = {v.strip()}")
    return c, cur
def backend_cpu(cur, pid):
    tg=tk=nt=0
    with open(f"/proc/{pid}/stat") as f: s=f.read().rsplit(')',1)[1].split(); tg=int(s[11])+int(s[12])
    for p in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            with open(p) as f: t=f.read().rsplit(')',1)[1].split(); tk+=int(t[11])+int(t[12]); nt+=1
        except FileNotFoundError: pass
    return tg,tk,nt
HZ=os.sysconf("SC_CLK_TCK")
# ---- 1. calibration (recall only) ----
sel={}
for sys,db,tbl,idx,knob,extra,grid in SYS:
    restart_prewarm(db,[tbl,idx,f"queries_{a.tag}"])
    conn,cur = connect(db, extra)
    plan=cur.execute(f"EXPLAIN (COSTS OFF) SELECT id FROM {tbl} ORDER BY emb <=> '{qtext(0)}'::{'turbovec.vector' if sys=='tv' else 'vector'} LIMIT 10").fetchall()
    ptxt="\n".join(r[0] for r in plan)
    assert "Index Scan" in ptxt and tbl in ptxt and "Seq Scan" not in ptxt, ptxt[:400]
    recs={}
    with open(f"{a.out}/calib_{sys}_{a.tag}.jsonl","w") as fh:
        for v in grid:
            cur.execute(f"SET {knob} = {v}")
            r=[]
            for i in range(a.warm, a.warm+CALN):
                ids=[x[0] for x in cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::{'turbovec.vector' if sys=='tv' else 'vector'} LIMIT 10",(qtext(i),))]
                r.append(len(set(ids)&set(GT[i,:10].tolist()))/10.0)
            rec=float(np.mean(r)); recs[v]=rec; fh.write(json.dumps(dict(knob=v,recall=rec))+"\n"); fh.flush()
    conn.close()
    # bracket 4/target
    pts=set()
    for T in (0.90,0.95,0.98,0.99):
        below=[v for v in grid if recs[v]<T]; above=[v for v in grid if recs[v]>=T]
        pts.update(sorted(below)[-2:]); pts.update(sorted(above)[:2])
        if above: pts.add(min(above))   # headline = smallest knob >= T
    sel[sys]=dict(selected=sorted(pts), recalls=recs,
                  headline={str(T):(min([v for v in grid if recs[v]>=T]) if any(recs[v]>=T for v in grid) else None) for T in (0.90,0.95,0.98,0.99)})
json.dump(sel, open(f"{a.out}/operating_points_{a.tag}.json","w"), indent=1)
print("CALIB_DONE", json.dumps({s:sel[s]['headline'] for s in sel}))
# ---- 2. timed rounds: round 0 (discarded) + ABBA x rounds ----
# arm order: round0 = [tv,pv]; then ABBA: r1=[tv,pv], r2=[pv,tv], (r3=[tv,pv], r4=[pv,tv]) truncated to --rounds counted + round0
order=[("r0",["tv","pv"])]
abba=[["tv","pv"],["pv","tv"],["tv","pv"],["pv","tv"]]
for r in range(a.rounds): order.append((f"r{r+1}", abba[r]))
SYSMAP={s[0]:s for s in SYS}
for rname,arms in order:
    for sysname in arms:
        sys,db,tbl,idx,knob,extra,grid = SYSMAP[sysname]
        restart_prewarm(db,[tbl,idx,f"queries_{a.tag}"])
        conn,cur = connect(db, extra); pid=cur.execute("SELECT pg_backend_pid()").fetchone()[0]
        for i in range(a.warm): cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::{'turbovec.vector' if sys=='tv' else 'vector'} LIMIT 10",(qtext(i),)).fetchall()
        T="turbovec.vector" if sys=="tv" else "vector"
        for v in sel[sys]["selected"]:
            cur.execute(f"SET {knob} = {v}")
            la0=open('/proc/loadavg').read().split()[0]; c0=backend_cpu(cur,pid)
            rows=[]
            for i in range(a.warm, a.warm+a.n):
                t0=time.perf_counter_ns(); ids=[x[0] for x in cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::{T} LIMIT 10",(qtext(i),))]; t1=time.perf_counter_ns()
                rows.append((i,(t1-t0)/1e3,ids))
            c1=backend_cpu(cur,pid); la1=open('/proc/loadavg').read().split()[0]
            lat=np.array([r[1] for r in rows])/1e3
            rec=np.mean([len(set(r[2])&set(GT[r[0],:10].tolist()))/10.0 for r in rows])
            bad=0
            for r in rows:
                d=1-C[r[2]].astype(np.float64)@Q[r[0]].astype(np.float64)
                if np.any(np.diff(d)<-1e-5): bad+=1
            sm={}
            for line in open(f"/proc/{pid}/smaps_rollup"):
                k,*vv=line.split()
                if k.rstrip(':') in ("Pss_Anon","Rss","Pss"): sm[k.rstrip(':')]=int(vv[0])
            summ=dict(round=rname,arm=sysname,sys=sys,knob_name=knob,knob=v,n=len(rows),recall=float(rec),
                      wall_p50_ms=float(np.percentile(lat,50)),wall_p95_ms=float(np.percentile(lat,95)),wall_mean_ms=float(lat.mean()),
                      cpu_ms_per_q_tasksum=(c1[1]-c0[1])*1000/HZ/len(rows),cpu_ms_per_q_tg=(c1[0]-c0[0])*1000/HZ/len(rows),
                      threads=c1[2],pss_anon_kb=sm.get("Pss_Anon"),order_violations=bad,loadavg=[la0,la1])
            json.dump(summ, open(f"{a.out}/{rname}_{sysname}_{v}.ids.summary.json","w"))
            with open(f"{a.out}/{rname}_{sysname}_{v}.ids.jsonl","w") as fh:
                for r in rows: fh.write(json.dumps(dict(qid=r[0],wall_ms=r[1]/1e3))+"\n")
            print(rname,sysname,knob,v,"R@10=%.4f"%rec,"p50=%.3f"%summ["wall_p50_ms"],"p95=%.3f"%summ["wall_p95_ms"],"cpu=%.1f"%summ["cpu_ms_per_q_tasksum"],"th=%d"%summ["threads"],flush=True)
        conn.close()
print(f"P3_FRONTIER_DONE_{a.tag}")
