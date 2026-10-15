#!/usr/bin/env python3
"""P6 churn (plan D4.5), turbovec IVF focus. Build is on docs_<tag> (N rows). We treat the existing
built index as N0 = N; then INSERT +5% and +11.1% MORE rows (held-out range reused as synthetic new
rows by re-embedding queries won't scale, so we duplicate-with-jitter from the tail of corpus? No —
we insert REAL additional corpus rows). For CO1024 the corpus is exactly N; to get +X% we re-insert
perturbed copies with NEW ids. Simpler + honest: insert copies of random existing rows with new ids
(same vectors) — this grows the index past ivf_max_delta_pct and is what the degrade test needs
(the DEGRADE trigger is row COUNT delta, not novelty). Records index_is_degraded() after each step,
recall@10 + p50 at the 0.95 point (no re-tuning), rows/s, WAL bytes.
usage: p6_churn.py --tag co10m --data /data/co1024_10m --n0 10000000 --knob-probes 32 --out ~/results/p6"""
import argparse,json,os,subprocess,time,numpy as np,psycopg
ap=argparse.ArgumentParser()
ap.add_argument("--tag");ap.add_argument("--data");ap.add_argument("--n0",type=int);ap.add_argument("--knob-probes",type=int,default=32);ap.add_argument("--out")
ap.add_argument("--timed",type=int,default=2000)
a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
ENV=dict(os.environ,PATH="/opt/pg18/bin:"+os.environ["PATH"],LD_LIBRARY_PATH="/opt/pg18/lib")
Q=np.load(f"{a.data}/queries.npy");GT=np.load(f"{a.data}/gt_ids.npy")
qtext=lambda i:"["+",".join("%.9g"%x for x in Q[i].tolist())+"]"
tbl=f"dtv_ivf_{a.tag}";c=psycopg.connect("host=/tmp dbname=db_tv",autocommit=True,prepare_threshold=0);cur=c.cursor()
def wal(): return cur.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
def waldiff(a0,a1): return cur.execute("SELECT pg_wal_lsn_diff(%s,%s)",(a1,a0)).fetchone()[0]
def degraded():
    try: return cur.execute("SELECT turbovec.index_is_degraded(%s::regclass)",(f"ixtvivf_{a.tag}",)).fetchone()[0]
    except Exception as e:
        try: return cur.execute(f"SELECT turbovec.index_is_degraded('ixtvivf_{a.tag}')").fetchone()[0]
        except Exception as e2: return f"err:{e2}"
def measure(probes,n):
    cur.execute(f"SET turbovec.probes = {probes}");lat=[];rec=[]
    for i in range(200): cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(qtext(i),)).fetchall()
    for i in range(200,200+n):
        t0=time.perf_counter_ns();ids=[r[0] for r in cur.execute(f"SELECT id FROM {tbl} ORDER BY emb <=> %s::turbovec.vector LIMIT 10",(qtext(i),))];t1=time.perf_counter_ns()
        lat.append((t1-t0)/1e3);rec.append(len(set(ids)&set(GT[i,:10].tolist()))/10.0)
    lat=np.array(lat)/1e3
    return dict(recall=float(np.mean(rec)),p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)))
def insert_pct(pct,start_id):
    nnew=int(a.n0*pct/100.0);t0=time.time();w0=wal()
    # insert copies of the first nnew rows with new ids (vectors identical; grows row count past threshold)
    cur.execute(f"INSERT INTO {tbl} (id, emb) SELECT id + {start_id}, emb FROM {tbl} WHERE id < {nnew} ORDER BY id")
    dt=time.time()-t0;wb=waldiff(w0,wal())
    return nnew,dt,int(wb)
res={"tag":a.tag,"n0":a.n0,"probes":a.knob_probes,"steps":[]}
res["baseline"]=dict(degraded=degraded(),**measure(a.knob_probes,a.timed))
# +5%
nnew,dt,wb=insert_pct(5.0, 100_000_000)
cur.execute(f"VACUUM (ANALYZE) {tbl}")
res["steps"].append(dict(step="+5%",rows_inserted=nnew,insert_s=dt,rows_per_s=nnew/dt,wal_bytes=wb,degraded=degraded(),**measure(a.knob_probes,a.timed)))
# +11.1% total -> insert another 6.1% (ids offset so no PK clash)
nnew2,dt2,wb2=insert_pct(6.1, 200_000_000)
cur.execute(f"VACUUM (ANALYZE) {tbl}")
res["steps"].append(dict(step="+11.1%",rows_inserted=nnew2,insert_s=dt2,rows_per_s=nnew2/dt2,wal_bytes=wb2,degraded=degraded(),**measure(a.knob_probes,a.timed)))
# correctness: turbovec_check + no dup ids in a sample result
try: res["turbovec_check"]=cur.execute(f"SELECT is_corrupt, reason, n_vectors, duplicate_id FROM turbovec.turbovec_check('ixtvivf_{a.tag}'::regclass)").fetchone()
except Exception as e: res["turbovec_check"]=f"err:{e}"
json.dump(res,open(f"{a.out}/p6_churn_{a.tag}.json","w"),indent=1)
print(json.dumps(res,indent=1));c.close()
print("P6_CHURN_DONE")
