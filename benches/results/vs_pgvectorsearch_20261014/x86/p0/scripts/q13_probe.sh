#!/usr/bin/env bash
# Q1 (flat cached whole at cache_size_mb=256?), Q3 (residency route: buffer manager vs mmap),
# Q2 (IVF path per cache_size_mb), Q4 (CPU accounting).
set -x
export PATH=/opt/pg18/bin:$PATH; R=~/results/p0/q13; mkdir -p $R
P="psql -h /tmp -X -v ON_ERROR_STOP=1"
restart_cold() {
  pg_ctl -D /data/pgdata -m fast -w stop; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  env | grep -i GLIBC > $R/glibc_env_$1.txt; pg_ctl -D /data/pgdata -l ~/pg.log -w start
}
idx_files() { local db=$1 idx=$2; local base=/data/pgdata/$($P -d $db -Atc "SELECT pg_relation_filepath('$idx')"); ls ${base} ${base}.* 2>/dev/null; }
bufcount() { $P -d $1 -Atc "SELECT count(*) FROM pg_buffercache WHERE relfilenode = pg_relation_filenode('$2') AND reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())"; }
# ---- Q3: cold query on flat index, strace the backend, fincore + pg_buffercache before/after
restart_cold q3
$P -d db_tv -Atc "SELECT pg_relation_size('docs_tv_flat')/8192 AS idx_blocks" > $R/q3_flat.txt
echo "fincore before:" >> $R/q3_flat.txt; fincore $(idx_files db_tv docs_tv_flat) >> $R/q3_flat.txt 2>&1
echo "bufcache before: $(bufcount db_tv docs_tv_flat)" >> $R/q3_flat.txt
Q=$($P -d db_tv -Atc "SELECT q::text FROM queries WHERE qid=0")
# a session that stops itself, so strace can attach before the query
( $P -d db_tv -Atc "SELECT pg_backend_pid()" -c "SELECT pg_sleep(3)" -c "SET turbovec.cache_size_mb=256" -c "EXPLAIN (ANALYZE, TIMING OFF, BUFFERS) SELECT id FROM docs ORDER BY emb <=> '$Q'::turbovec.vector LIMIT 10" > $R/q3_cold_query.txt 2>&1 ) &
sleep 1.5; BP=$(head -1 $R/q3_cold_query.txt)
sudo strace -f -tt -p $BP -e trace=mmap,munmap,pread64,preadv,read,openat,io_uring_enter,madvise -o $R/q3_strace.txt &
SPID=$!; wait %1 2>/dev/null; sleep 2; sudo kill $SPID 2>/dev/null; sleep 1
echo "fincore after cold query:" >> $R/q3_flat.txt; fincore $(idx_files db_tv docs_tv_flat) >> $R/q3_flat.txt 2>&1
echo "bufcache after cold query: $(bufcount db_tv docs_tv_flat)" >> $R/q3_flat.txt
REL=$($P -d db_tv -Atc "SELECT pg_relation_filenode('docs_tv_flat')")
{ echo "strace backend=$BP relfilenode=$REL"; echo "openat of index relfile:"; grep -c "/$REL" $R/q3_strace.txt;
  echo "mmap calls (all):"; grep -c " mmap(" $R/q3_strace.txt; echo "mmap with fd (not MAP_ANONYMOUS):"; grep " mmap(" $R/q3_strace.txt | grep -vc MAP_ANONYMOUS;
  echo "pread64 calls:"; grep -c "pread64(" $R/q3_strace.txt; echo "preadv calls:"; grep -c "preadv(" $R/q3_strace.txt; echo "io_uring_enter:"; grep -c io_uring_enter $R/q3_strace.txt;
  echo "big anon mmaps (>=64MB):"; grep " mmap(NULL, [0-9]\{8,\}" $R/q3_strace.txt | head -20; } >> $R/q3_flat.txt
# prewarm route: pg_prewarm(index) then fincore/bufcache
restart_cold q3b
$P -d db_tv -Atc "SELECT pg_prewarm('docs_tv_flat')" >> $R/q3_flat.txt
echo "after pg_prewarm(index): bufcache $(bufcount db_tv docs_tv_flat)" >> $R/q3_flat.txt; fincore $(idx_files db_tv docs_tv_flat) >> $R/q3_flat.txt 2>&1
# ---- Q1: flat index cached whole per backend at default 256 MB?
$P -d db_tv -Atc "SELECT pg_prewarm('docs'), pg_prewarm('queries'), pg_prewarm('docs_tv_flat')" >> $R/q3_flat.txt
for cap in 256 2048; do
  python3 ~/bench.py --db db_tv --sys tv --label q1_flat_cap${cap}_sk32 --set "turbovec.cache_size_mb=$cap" --mode ids --warm 1 --n 500 --out $R --expect-index docs_tv_flat > $R/q1_cap${cap}.txt 2>&1
done
# per-query timing trace + PSS growth in ONE backend (first 10 queries individually, PSS after each)
python3 - > $R/q1_pss_trace.txt 2>&1 <<'PY'
import psycopg, numpy as np, time
Q = np.load("/data/co1024_1m/queries.npy"); qt = lambda i: "[" + ",".join("%.9g" % x for x in Q[i].tolist()) + "]"
c = psycopg.connect("host=/tmp dbname=db_tv", autocommit=True); cur = c.cursor()
pid = cur.execute("SELECT pg_backend_pid()").fetchone()[0]
def pss():
    d = {}
    for l in open(f"/proc/{pid}/smaps_rollup"):
        k, *v = l.split()
        if k in ("Rss:", "Pss:", "Pss_Anon:", "Pss_Shmem:", "Anonymous:"): d[k[:-1]] = int(v[0])
    return d
print("before any query", pss())
for i in range(12):
    t0 = time.perf_counter(); cur.execute("SELECT id FROM docs ORDER BY emb <=> %s::turbovec.vector LIMIT 10", (qt(1000+i),)).fetchall(); t1 = time.perf_counter()
    print(f"q{i} {1000*(t1-t0):.2f} ms", pss())
PY
# ---- Q2: IVF path selection per cache_size_mb (codes_bytes = n*dim*bw/8 = 512,000,000 B; auto threshold = 0.5*cap)
$P -d db_tv -Atc "SELECT pg_prewarm('docs_ivf'), pg_prewarm('docs_ivf_tv')" > $R/q2_ivf.txt
for cap in 256 1024 2048; do
  for ooc in auto off on; do
    python3 ~/bench.py --db db_tv --sys tv --label q2_ivf_cap${cap}_ooc${ooc} --table docs_ivf --set "turbovec.cache_size_mb=$cap;turbovec.out_of_core=$ooc;turbovec.probes=16" --mode ids --warm 20 --n 500 --out $R --expect-index docs_ivf_tv >> $R/q2_ivf.txt 2>&1
  done
done
echo Q13_DONE
