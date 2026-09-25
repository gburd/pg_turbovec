#!/usr/bin/env bash
# Cold-scan A/B: turbovec cold-backend p50 (the regime parallel repack attacks)
# vs warm, on 1M x 1024d 4-bit flat. Run once per build; TAG labels output.
set -euo pipefail
export DB=coldbench
TAG=${TAG:-new}
PY=/home/ubuntu/venv/bin/python3
echo "=== [$TAG] load 1M x 1024d synthetic + 20 queries (idempotent) ==="
$PY - <<'PY'
import numpy as np, psycopg
n,dim=1_000_000,1024
con=psycopg.connect("dbname=coldbench"); con.autocommit=True; cur=con.cursor()
cur.execute("SELECT to_regclass('public.docs')")
if cur.fetchone()[0] is None:
    cur.execute("CREATE TABLE public.docs(id bigint primary key, tv turbovec.vector)")
    rng=np.random.default_rng(7); B=20000
    with cur.copy("COPY public.docs(id,tv) FROM STDIN") as cp:
        for s in range(0,n,B):
            m=min(B,n-s); v=rng.standard_normal((m,dim)).astype(np.float32); v/=np.linalg.norm(v,axis=1,keepdims=True)
            for i in range(m): cp.write_row((s+i,"["+",".join(f"{x:.5f}" for x in v[i])+"]"))
    cur.execute("CREATE TABLE public.q(qid int primary key, tv turbovec.vector)")
    qs=rng.standard_normal((20,dim)).astype(np.float32); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    with cur.copy("COPY public.q(qid,tv) FROM STDIN") as cp:
        for i in range(20): cp.write_row((i,"["+",".join(f"{x:.5f}" for x in qs[i])+"]"))
    print("loaded")
else: print("already loaded")
PY
echo "=== [$TAG] (re)build flat bw4 ==="
psql -d coldbench -qc "SET maintenance_work_mem='4GB'; SET max_parallel_maintenance_workers=8;
DROP INDEX IF EXISTS docs_tv; CREATE INDEX docs_tv ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width=4);"
psql -d coldbench -c "SELECT pg_size_pretty(pg_relation_size('docs_tv'))"
psql -d coldbench -qAt -c "SELECT tv::text FROM public.q ORDER BY qid" > /tmp/cs_q.txt
mapfile -t QV < /tmp/cs_q.txt
Q(){ echo "SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.search_k=100;
EXPLAIN (ANALYZE, FORMAT JSON, TIMING ON) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '$1'::turbovec.vector LIMIT 10;"; }
et(){ $PY -c "import sys,json; print(json.load(sys.stdin)[0]['Execution Time'])"; }
: > /tmp/cs_warm_$TAG.txt; : > /tmp/cs_cold_$TAG.txt
echo "=== [$TAG] R3 warm-backend (ONE psql session: warm-ups + timed, marked) ==="
{ for i in $(seq 0 14); do Q "${QV[$((i%20))]}"; done
  for i in $(seq 0 19); do echo "\echo @@ROW"; Q "${QV[$i]}"; done; } \
  | psql -d coldbench -qAt 2>/dev/null \
  | $PY -c "import sys,json; out=sys.stdin.read().split('@@ROW')[1:]; [print(json.loads(b.strip())[0]['Execution Time']) for b in out if b.strip()]" \
  > /tmp/cs_warm_$TAG.txt
echo "=== [$TAG] R2 cold-backend + warm-OS (FRESH psql per query = cold per-backend cache) ==="
for i in $(seq 0 19); do Q "${QV[$i]}" | psql -d coldbench -qAtf - | et >> /tmp/cs_cold_$TAG.txt; done
$PY - "$TAG" <<'PY'
import sys,statistics
tag=sys.argv[1]
def load(p): return [float(x) for x in open(p) if x.strip()]
def s(a): return f"p50={statistics.median(a):.1f} p95={sorted(a)[max(int(0.95*len(a))-1,0)]:.1f} min={min(a):.1f} n={len(a)}"
w=load(f"/tmp/cs_warm_{tag}.txt"); c=load(f"/tmp/cs_cold_{tag}.txt")
print(f"RESULT[{tag}] R3_warm_backend:       {s(w)}")
print(f"RESULT[{tag}] R2_cold_backend_warmOS: {s(c)}")
PY
