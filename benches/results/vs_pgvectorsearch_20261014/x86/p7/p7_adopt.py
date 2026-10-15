#!/usr/bin/env python3
"""P7 adoption path (plan D4.9), x86 1M. pg_vectorsearch indexes an existing pgvector column in place;
pg_turbovec needs EITHER a turbovec.vector column (measured: column copy time + heap) OR an expression
index over the ::real[]::turbovec.vector cast (no copy, but a per-candidate cast cost at query time).
Measures the expression-index route vs the native-column route at the 0.95 point. Uses the pgvector
column already present in db_pv.docs_co1m (vector type) as the "existing" column.
usage: p7_adopt.py --data /data/co1024_1m --out ~/results/p7"""
import argparse,json,os,time,subprocess,numpy as np,psycopg
ap=argparse.ArgumentParser();ap.add_argument("--data");ap.add_argument("--out");ap.add_argument("--n",type=int,default=1000)
a=ap.parse_args();os.makedirs(a.out,exist_ok=True)
ENV=dict(os.environ,PATH="/opt/pg18/bin:"+os.environ["PATH"],LD_LIBRARY_PATH="/opt/pg18/lib")
Q=np.load(f"{a.data}/queries.npy");GT=np.load(f"{a.data}/gt_ids.npy")
qtext=lambda i:"["+",".join("%.9g"%x for x in Q[i].tolist())+"]"
res={"data":a.data,"routes":{}}
conn=psycopg.connect("host=/tmp dbname=db_tv",autocommit=True,prepare_threshold=0);cur=conn.cursor()
# starting point: a pgvector-typed column in db_tv (simulate "existing pgvector column"): build from docs_co1m
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("DROP TABLE IF EXISTS adopt_src")
cur.execute("CREATE TABLE adopt_src (id bigint PRIMARY KEY, emb vector(1024))")
cur.execute("ALTER TABLE adopt_src ALTER COLUMN emb SET STORAGE PLAIN")
# populate from the turbovec column via real[] bridge
t0=time.time();cur.execute("INSERT INTO adopt_src SELECT id, (emb::real[])::vector FROM docs_co1m");pop=time.time()-t0
cur.execute("VACUUM (FREEZE, ANALYZE) adopt_src")
# Route (i): column copy to turbovec.vector + heap growth
w0=cur.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
t0=time.time()
cur.execute("ALTER TABLE adopt_src ADD COLUMN emb_tv turbovec.vector")
cur.execute("UPDATE adopt_src SET emb_tv = (emb::real[])::turbovec.vector")
copy_s=time.time()-t0
w1=cur.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
copy_wal=cur.execute("SELECT pg_wal_lsn_diff(%s,%s)",(w1,w0)).fetchone()[0]
cur.execute("VACUUM (ANALYZE) adopt_src")
heap=cur.execute("SELECT pg_total_relation_size('adopt_src')").fetchone()[0]
res["routes"]["column_copy"]=dict(update_s=copy_s,wal_bytes=int(copy_wal),heap_bytes_after=int(heap))
# build index on the native turbovec column, time warm p50 at sk=16
t0=time.time();cur.execute("SET turbovec.build_parallelism=32");cur.execute("CREATE INDEX adopt_native ON adopt_src USING turbovec (emb_tv vec_cosine_ops)");res["routes"]["column_copy"]["build_s"]=time.time()-t0
# Route (ii): expression index over the cast (no copy)
t0=time.time();cur.execute("CREATE INDEX adopt_expr ON adopt_src USING turbovec ((emb::real[]::turbovec.vector) vec_cosine_ops)");res["routes"]["expr_index"]=dict(build_s=time.time()-t0)
def warm_p50(orderexpr, vt, label):
    cur.execute("SET turbovec.search_k=16");cur.execute("SET turbovec.hi_dim_rerank=off")
    for i in range(200): cur.execute(f"SELECT id FROM adopt_src ORDER BY {orderexpr} <=> %s::{vt} LIMIT 10",(qtext(i),)).fetchall()
    lat=[];rec=[]
    for i in range(200,200+a.n):
        t0=time.perf_counter_ns();ids=[r[0] for r in cur.execute(f"SELECT id FROM adopt_src ORDER BY {orderexpr} <=> %s::{vt} LIMIT 10",(qtext(i),))];t1=time.perf_counter_ns()
        lat.append((t1-t0)/1e3);rec.append(len(set(ids)&set(GT[i,:10].tolist()))/10.0)
    lat=np.array(lat)/1e3
    plan="\n".join(r[0] for r in cur.execute(f"EXPLAIN (COSTS OFF) SELECT id FROM adopt_src ORDER BY {orderexpr} <=> '{qtext(0)}'::{vt} LIMIT 10").fetchall())
    return dict(p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)),recall=float(np.mean(rec)),uses_index=("Index Scan" in plan))
# native column query
res["routes"]["column_copy"]["query"]=warm_p50("emb_tv","turbovec.vector","native")
# expression query (per-candidate cast): drop native index so planner picks expr index
cur.execute("DROP INDEX adopt_native")
res["routes"]["expr_index"]["query"]=warm_p50("(emb::real[]::turbovec.vector)","turbovec.vector","expr")
res["note"]=f"populate adopt_src from docs_co1m via real[] bridge took {pop:.1f}s (setup, not a route cost)"
json.dump(res,open(f"{a.out}/p7_adopt.json","w"),indent=1)
print(json.dumps(res,indent=1));conn.close();print("P7_ADOPT_DONE")
