#!/usr/bin/env python3
"""Aggregate P5: QPS, QPS/core (server CPU-seconds from mpstat %idle), p50/p99 from pgbench -l per-txn logs.
usage: p5_agg.py <p5dir> <host>"""
import sys,glob,os,re,numpy as np,json
D,HOST=sys.argv[1],sys.argv[2]
ncpu=os.cpu_count()
def tps(f):
    t=open(f).read();m=re.search(r"tps = ([0-9.]+)",t);return float(m.group(1)) if m else None
def avglat(f):
    m=re.search(r"latency average = ([0-9.]+)",open(f).read());return float(m.group(1)) if m else None
def server_busy_cores(mp):
    # mpstat 'all' rows: cores_busy = ncpu*(100-%idle)/100 averaged over steady window
    idles=[]
    for line in open(mp):
        p=line.split()
        if len(p)>=12 and p[-12:][0]=="all" or (len(p)>=3 and p[2]=="all"):
            try: idles.append(float(p[-1]))
            except: pass
    if not idles: return None
    idles=idles[len(idles)//4:]  # drop first quarter
    return ncpu*(100-np.mean(idles))/100
def plog_pcts(prefix):
    lats=[]
    for lf in glob.glob(prefix+".*"):
        if lf.endswith(('.pgbench','.mpstat')): continue
        for line in open(lf):
            p=line.split()
            if len(p)>=3:
                try: lats.append(int(p[2])/1000.0)  # schedule_lag? field 3 = latency(us) in pgbench -l
                except: pass
    if not lats: return None,None,None
    a=np.array(lats);return float(np.percentile(a,50)),float(np.percentile(a,95)),float(np.percentile(a,99))
rows=[]
for pb in sorted(glob.glob(f"{D}/*_c*.pgbench")):
    b=os.path.basename(pb)[:-8]  # strip .pgbench
    m=re.match(r"(co\w+?)_(tv1|tv|pv)_c(\d+)",b)
    if not m: continue
    tag,sysn,c=m.group(1),m.group(2),int(m.group(3))
    t=tps(pb);mp=f"{D}/{b}.mpstat";busy=server_busy_cores(mp) if os.path.exists(mp) else None
    p50,p95,p99=plog_pcts(f"{D}/{b}")
    rows.append(dict(tag=tag,sys=sysn,clients=c,tps=t,avg_lat_ms=avglat(pb),
        server_busy_cores=busy,qps_per_core=(t/busy if (t and busy) else None),
        p50_ms=p50,p95_ms=p95,p99_ms=p99))
json.dump(dict(host=HOST,ncpu=ncpu,rows=rows),open(f"{D}/p5_agg.json","w"),indent=1)
# md
L=[f"### {HOST} — P5 closed-loop QPS (20s warm + 120s timed, pinned client)",""]
L.append("| dataset | system | clients | TPS | QPS/core | avg lat ms | p99 ms (pgbench log) |")
L.append("|---|---|---|---|---|---|---|")
sysname={"tv":"pg_turbovec (sp=4)","tv1":"pg_turbovec (sp=1)","pv":"pg_vectorsearch"}
for r in sorted(rows,key=lambda r:(r["tag"],r["sys"],r["clients"])):
    qpc="%.1f"%r["qps_per_core"] if r["qps_per_core"] else "—"
    p99="%.1f"%r["p99_ms"] if r["p99_ms"] else "—"
    L.append(f"| {r['tag']} | {sysname.get(r['sys'],r['sys'])} | {r['clients']} | {r['tps']:.0f} | {qpc} | {r['avg_lat_ms']:.2f} | {p99} |")
open(f"{D}/p5_agg.md","w").write("\n".join(L)+"\n");print("\n".join(L))
