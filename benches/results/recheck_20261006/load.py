import sys, numpy as np, pyarrow.parquet as pq, psycopg
N=200_000; H=200
c=psycopg.connect("host=/tmp/rck port=55432 dbname=bench", autocommit=True)
c.execute("DROP TABLE IF EXISTS docs"); c.execute("CREATE TABLE docs(id bigint primary key, tv turbovec.vector, fa real[])")
rows=[]; held=[]
for p in ("/tmp/rck/s0000.parquet","/tmp/rck/s0001.parquet"):
    a=pq.read_table(p,columns=["emb"]).column("emb").combine_chunks().values.to_numpy().reshape(-1,1024).astype(np.float32)
    a/=np.linalg.norm(a,axis=1,keepdims=True); rows.append(a)
A=np.concatenate(rows)[:N+H]; Q=A[N:]; A=A[:N]
with c.cursor().copy("COPY docs(id,tv,fa) FROM STDIN") as cp:
    for i in range(N):
        s=",".join(f"{x:.6f}" for x in A[i]); cp.write_row((i,"["+s+"]","{"+s+"}"))
np.save("/tmp/rck/q.npy",Q)
G=np.argsort(-(Q@A.T),axis=1)[:,:10]; np.save("/tmp/rck/gt.npy",G)
c.execute("VACUUM ANALYZE docs")
print("loaded", N, c.execute("SELECT pg_size_pretty(pg_table_size('docs'))").fetchone())
