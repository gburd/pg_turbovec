#!/usr/bin/env python3
"""P7 filtered search (plan D4.6) at 1M, both systems. Adds a uniform cat column (selectivity 50/10/1%),
WHERE cat < k ORDER BY emb <=> q LIMIT 10. Metrics: recall@10 vs FILTERED exact GT (computed here),
short-result rate (<10 rows), p50/p95. Each system's documented mechanism (prism default;
turbovec iterative_scan=relaxed_order). Writes p7_filtered.json.
usage: p7_filtered.py --data /data/co1024_1m --out ~/results/p7"""
import argparse,json,os,time,numpy as np,psycopg
ap=argparse.ArgumentParser();ap.add_argument("--data");ap.add_argument("--out");ap.add_argument("--n",type=int,default=1000)
a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
ENV_LDP="/opt/pg18/lib"
Q=np.load(f"{a.data}/queries.npy");C=np.load(f"{a.data}/corpus.npy",mmap_mode="r");N=C.shape[0]
qtext=lambda i:"["+",".join("%.9g"%x for x in Q[i].tolist())+"]"
# deterministic uniform cat in [0,100): cat = id % 100. selectivity s% => WHERE cat < s
rng=np.random.default_rng(7); cat=(np.arange(N)%100).astype(np.int32)
# filtered exact GT per selectivity: top-10 among rows with cat<s, by fp64 dot
def filtered_gt(s, qi):
    mask=cat< s
    idx=np.nonzero(mask)[0]
    sub=C[idx].astype(np.float32)
    out={}
    for i in qi:
        d=1.0-sub@Q[i].astype(np.float32)
        top=idx[np.argpartition(d,10)[:10]]
        # fp64 refine
        dd=1.0-C[top].astype(np.float64)@Q[i].astype(np.float64)
        out[i]=set(top[np.argsort(dd)][:10].tolist())
    return out
res={"data":a.data,"N":N,"cells":[]}
qi=list(range(200,200+a.n))
for sysn,db,tbl,vt,setup in [("tv","db_tv","dtv_flat_co1m","turbovec.vector","SET turbovec.iterative_scan='relaxed_order'; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k=64"),
                             ("pv","db_pv","dpv_prism_co1m","vector","SET prism.nprobe=56")]:
    conn=psycopg.connect(f"host=/tmp dbname={db}",autocommit=True,prepare_threshold=0);cur=conn.cursor()
    if sysn=="pv": cur.execute("RESET prism.query_limit")
    # add cat column + btree, once
    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS cat int")
    cur.execute(f"UPDATE {tbl} SET cat = mod(id, 100) WHERE cat IS NULL")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {tbl}_cat ON {tbl}(cat)")
    cur.execute(f"VACUUM (ANALYZE) {tbl}")
    for stmt in setup.split(";"):
        if stmt.strip(): cur.execute(stmt)
    for s in (50,10,1):
        gt=filtered_gt(s,qi)
        lat=[];rec=[];short=0
        for i in qi:
            t0=time.perf_counter_ns()
            ids=[r[0] for r in cur.execute(f"SELECT id FROM {tbl} WHERE cat < {s} ORDER BY emb <=> %s::{vt} LIMIT 10",(qtext(i),))]
            t1=time.perf_counter_ns()
            lat.append((t1-t0)/1e3)
            if len(ids)<10: short+=1
            rec.append(len(set(ids)&gt[i])/10.0)
        lat=np.array(lat)/1e3
        res["cells"].append(dict(sys=sysn,selectivity_pct=s,n=a.n,recall=float(np.mean(rec)),
            short_result_rate=short/a.n,p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95))))
        print(sysn,"sel=%d%%"%s,"R@10=%.4f"%np.mean(rec),"short=%.3f"%(short/a.n),"p50=%.2f"%np.percentile(lat,50),flush=True)
    conn.close()
json.dump(res,open(f"{a.out}/p7_filtered.json","w"),indent=1)
print("P7_FILTERED_DONE")
