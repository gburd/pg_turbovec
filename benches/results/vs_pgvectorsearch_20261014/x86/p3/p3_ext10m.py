#!/usr/bin/env python3
"""P3 extension: pin turbovec's 10M IVF 0.98/0.99 (extra probes) + a 10M FLAT exact point.
Recommended GUCs only (IVF: all defaults = oversample 1.0, search_k 32, hi_dim_rerank auto;
FLAT: defaults). round 0 discarded + ABBA x2 at the extra IVF probes; flat timed once (--flat-n).
Builds the 10M flat index on db_tv if missing (dtv_flat_co10m). Writes r{0,1,2}_tv_<knob>.ids.summary.json
(extending the co10m dir) + flat_tv.ids.summary.json.
usage: p3_ext10m.py --out ~/results/p3/co10m --data /data/co1024_10m --flat-n 300"""
import argparse, json, os, subprocess, time, glob, numpy as np, psycopg
ap=argparse.ArgumentParser()
ap.add_argument("--out"); ap.add_argument("--data",default="/data/co1024_10m")
ap.add_argument("--n",type=int,default=5000); ap.add_argument("--warm",type=int,default=200)
ap.add_argument("--flat-n",type=int,default=300); ap.add_argument("--rounds",type=int,default=2)
ap.add_argument("--extra-probes",default="128,192,256")
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
PGDATA="/data/pgdata";PGBIN="/opt/pg18/bin";ENV=dict(os.environ,PATH=f"{PGBIN}:"+os.environ["PATH"],LD_LIBRARY_PATH="/opt/pg18/lib")
GT=np.load(f"{a.data}/gt_ids.npy");Q=np.load(f"{a.data}/queries.npy");C=np.load(f"{a.data}/corpus.npy",mmap_mode="r")
qtext=lambda i:"["+",".join("%.9g"%x for x in Q[i].tolist())+"]"
HZ=os.sysconf("SC_CLK_TCK")
def restart_prewarm(db,rels):
    subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-m","fast","-w","stop"],env=ENV,check=True)
    subprocess.run("sync",shell=True);subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null",shell=True)
    subprocess.run([f"{PGBIN}/pg_ctl","-D",PGDATA,"-l",os.path.expanduser("~/pg.log"),"-w","start"],env=ENV,check=True)
    c=psycopg.connect("host=/tmp dbname=db_tv",autocommit=True)
    for r in rels: c.execute(f"SELECT pg_prewarm('{r}')")
    c.close()
def cpu(pid):
    tg=tk=nt=0
    with open(f"/proc/{pid}/stat") as f: s=f.read().rsplit(')',1)[1].split(); tg=int(s[11])+int(s[12])
    for p in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            with open(p) as f: t=f.read().rsplit(')',1)[1].split(); tk+=int(t[11])+int(t[12]); nt+=1
        except FileNotFoundError: pass
    return tg,tk,nt
def measure(cur,pid,tbl,knob,v,nq,label):
    if knob: cur.execute(f"SET {knob} = {v}")
    gucs={r[0]:r[1] for r in cur.execute("SELECT name,setting FROM pg_settings WHERE name ~ '^turbovec\\.'")}
    for i in range(a.warm): cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(qtext(i),)).fetchall()
    c0=cpu(pid);rows=[]
    for i in range(a.warm,a.warm+nq):
        t0=time.perf_counter_ns();ids=[x[0] for x in cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(qtext(i),))];t1=time.perf_counter_ns()
        rows.append((i,(t1-t0)/1e3,ids))
    c1=cpu(pid);lat=np.array([r[1] for r in rows])/1e3
    rec=float(np.mean([len(set(r[2])&set(GT[r[0],:10].tolist()))/10.0 for r in rows]))
    bad=sum(1 for r in rows if np.any(np.diff(1-C[r[2]].astype(np.float64)@Q[r[0]].astype(np.float64))<-1e-5))
    sm={}
    for line in open(f"/proc/{pid}/smaps_rollup"):
        k,*vv=line.split()
        if k.rstrip(':')=="Pss_Anon": sm['Pss_Anon']=int(vv[0])
    return dict(label=label,knob=v,n=len(rows),recall=rec,wall_p50_ms=float(np.percentile(lat,50)),wall_p95_ms=float(np.percentile(lat,95)),
                wall_mean_ms=float(lat.mean()),cpu_ms_per_q_tasksum=(c1[1]-c0[1])*1000/HZ/len(rows),threads=c1[2],
                pss_anon_kb=sm.get('Pss_Anon'),order_violations=bad,gucs={k:gucs[k] for k in ('turbovec.search_k','turbovec.oversample','turbovec.hi_dim_rerank','turbovec.probes','turbovec.out_of_core','turbovec.scan_parallelism') if k in gucs})
# build 10M flat if missing
PS=["psql","-h","/tmp","-X","-d","db_tv","-v","ON_ERROR_STOP=1","-Atc"]
has=subprocess.run(PS+["SELECT 1 FROM pg_class WHERE relname='ixtvflat_co10m'"],env=ENV,capture_output=True,text=True).stdout.strip()
if has!="1":
    print("building 10M flat index...",flush=True)
    subprocess.run(["psql","-h","/tmp","-X","-d","db_tv","-v","ON_ERROR_STOP=1",
        "-c","DROP TABLE IF EXISTS dtv_flat_co10m",
        "-c","CREATE TABLE dtv_flat_co10m (LIKE docs_co10m INCLUDING ALL)",
        "-c","INSERT INTO dtv_flat_co10m SELECT * FROM docs_co10m",
        "-c","VACUUM (FREEZE, ANALYZE) dtv_flat_co10m",
        "-c","SET turbovec.build_parallelism=32",
        "-c","CREATE INDEX ixtvflat_co10m ON dtv_flat_co10m USING turbovec (emb vec_cosine_ops)"],env=ENV,check=True)
# extra IVF probes: round0 + ABBA x2
probes=[int(x) for x in a.extra_probes.split(",") if x.strip()]
order=([("r0",probes)]+[(f"r{r+1}",probes) for r in range(a.rounds)]) if probes else []
for rname,ps in order:
    restart_prewarm("db_tv",["dtv_ivf_co10m","ixtvivf_co10m","queries_co10m"])
    conn=psycopg.connect("host=/tmp dbname=db_tv",autocommit=True,prepare_threshold=0);cur=conn.cursor()
    pid=cur.execute("SELECT pg_backend_pid()").fetchone()[0]
    plan="\n".join(r[0] for r in cur.execute(f"EXPLAIN (COSTS OFF) SELECT id FROM dtv_ivf_co10m ORDER BY emb <=> '{qtext(0)}'::turbovec.vector LIMIT 10").fetchall())
    assert "Index Scan" in plan and "dtv_ivf_co10m" in plan, plan[:300]
    for v in ps:
        s=measure(cur,pid,"dtv_ivf_co10m","turbovec.probes",v,a.n,f"ivf_p{v}")
        s.update(round=rname,sys="tv",knob_name="turbovec.probes")
        json.dump(s,open(f"{a.out}/{rname}_tv_{v}.ids.summary.json","w"))
        with open(f"{a.out}/{rname}_tv_{v}.ids.jsonl","w") as fh:
            pass  # keep summaries only for the extension to save space
        print(rname,"ivf",v,"R@10=%.4f"%s["recall"],"p50=%.1f"%s["wall_p50_ms"],"cpu=%.0f"%s["cpu_ms_per_q_tasksum"],"th=%d"%s["threads"],flush=True)
    conn.close()
# flat exact point (defaults), timed once
restart_prewarm("db_tv",["dtv_flat_co10m","ixtvflat_co10m","queries_co10m"])
conn=psycopg.connect("host=/tmp dbname=db_tv",autocommit=True,prepare_threshold=0);cur=conn.cursor()
pid=cur.execute("SELECT pg_backend_pid()").fetchone()[0]
plan="\n".join(r[0] for r in cur.execute(f"EXPLAIN (COSTS OFF) SELECT id FROM dtv_flat_co10m ORDER BY emb <=> '{qtext(0)}'::turbovec.vector LIMIT 10").fetchall())
assert "Index Scan" in plan and "dtv_flat_co10m" in plan, plan[:300]
s=measure(cur,pid,"dtv_flat_co10m","",0,a.flat_n,"flat_defaults")
s.update(sys="tv",note=f"10M flat 4-bit, all defaults, n={a.flat_n} (flat scan is O(n); smaller n)")
json.dump(s,open(f"{a.out}/flat_tv.ids.summary.json","w"))
print("flat_defaults","R@10=%.4f"%s["recall"],"p50=%.1f"%s["wall_p50_ms"],"cpu=%.0f"%s["cpu_ms_per_q_tasksum"],"th=%d"%s["threads"],"n=%d"%s["n"],flush=True)
conn.close()
print("P3_EXT10M_DONE")
