#!/usr/bin/env bash
# Cold-scan A/B for prefetch: TAG labels the build. Measures cold-backend p50 in
# two I/O regimes: warm-OS (page cache hot, PG restart only) and true-cold
# (drop_caches + PG restart, so read_chain actually hits disk — where prefetch
# should help most).
set -euo pipefail
export DB=coldbench
TAG=${TAG:-prefetch}
PY=/home/ubuntu/venv/bin/python3
echo "=== [$TAG] load 1M x 1024d + 20 queries (idempotent) ==="
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
psql -d coldbench -qAt -c "SELECT tv::text FROM public.q ORDER BY qid" > /tmp/pf_q.txt
mapfile -t QV < /tmp/pf_q.txt
et(){ $PY -c "import sys,json; print(json.load(sys.stdin)[0]['Execution Time'])"; }
Q(){ echo "SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.search_k=100;
EXPLAIN (ANALYZE,FORMAT JSON,TIMING ON) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '$1'::turbovec.vector LIMIT 10;"; }
# WARM-OS cold-backend: PG restart (drops per-backend cache, keeps OS page cache)
: > /tmp/pf_warmos_$TAG.txt
for i in $(seq 0 9); do
  sudo systemctl restart postgresql@16-main; sleep 3
  Q "${QV[$i]}" | psql -d coldbench -qAtf - | et >> /tmp/pf_warmos_$TAG.txt
done
# TRUE-COLD: drop_caches + PG restart per trial (read_chain hits disk; prefetch matters)
: > /tmp/pf_cold_$TAG.txt
for i in $(seq 0 9); do
  sudo systemctl stop postgresql@16-main; sleep 1
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sudo systemctl start postgresql@16-main; sleep 3
  Q "${QV[$i]}" | psql -d coldbench -qAtf - | et >> /tmp/pf_cold_$TAG.txt
done
$PY - "$TAG" <<'PY'
import sys,statistics
t=sys.argv[1]
def L(p): return [float(x) for x in open(p) if x.strip()]
def s(a): return f"p50={statistics.median(a):.1f} p95={max(a):.1f} min={min(a):.1f} n={len(a)}"
print(f"RESULT[{t}] cold_backend_warmOS: {s(L(f'/tmp/pf_warmos_{t}.txt'))}")
print(f"RESULT[{t}] true_cold_disk:      {s(L(f'/tmp/pf_cold_{t}.txt'))}")
PY
echo "PF_DONE_$TAG"
