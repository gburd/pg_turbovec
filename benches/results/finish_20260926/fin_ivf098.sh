#!/usr/bin/env bash
# Extend the IVF sweep to high probes to pin the R@10>=0.98 crossover and show
# it converges to flat's cost. One family at a time (no planner ambiguity).
set -euo pipefail
export DB=bench
PY=/home/ubuntu/venv/bin/python3
mkidx(){ psql -d bench -qc "SET maintenance_work_mem='8GB'; SET max_parallel_maintenance_workers=8; $1"; }
dropx(){ for x in docs_tv_flat_bw4 docs_tv_flat_bw1 docs_tv_ivf_bw4 docs_tv_ivf_bw1; do psql -d bench -qc "DROP INDEX IF EXISTS $x"; done; }
# query literals + GT
mapfile -t QIDS < <(psql -d bench -qAt -c "SELECT qid FROM public.query_set ORDER BY qid")
$PY - <<'PY'
import subprocess,json
qids=[int(x) for x in subprocess.run(["psql","-d","bench","-qAt","-c","SELECT qid FROM public.query_set ORDER BY qid"],capture_output=True,text=True).stdout.split()]
gt={}
for line in subprocess.run(["psql","-d","bench","-qAt","-F","|","-c","SELECT qid,hit_id FROM public.gt_top10"],capture_output=True,text=True).stdout.splitlines():
    if "|" in line: q,h=line.split("|"); gt.setdefault(int(q),[]).append(int(h))
json.dump({"qids":qids,"gt":{str(k):v for k,v in gt.items()}}, open("/tmp/gt.json","w"))
print("gt cached", len(qids), "queries")
PY
sweep_family(){ # $1=famtag  $2=index  $3=setup-extra (probes handled per arm)
  local fam=$1 idx=$2 opt=$3
  dropx
  mkidx "CREATE INDEX $idx ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH ($opt);"
  psql -d bench -c "SELECT pg_size_pretty(pg_relation_size('$idx'))"
  $PY - "$fam" <<'PY'
import subprocess,json,statistics,sys
fam=sys.argv[1]; d=json.load(open("/tmp/gt.json")); qids=d["qids"]; gt={int(k):v for k,v in d["gt"].items()}
def lit(q): return subprocess.run(["psql","-d","bench","-qAt","-c",f"SELECT tv::text FROM public.query_set WHERE qid={q}"],capture_output=True,text=True).stdout.strip()
lits={q:lit(q) for q in qids}
def run(probes,k):
    times=[]; recs=[]
    # warm 3
    for i in range(3):
        subprocess.run(["psql","-d","bench","-qAt","-c",f"SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.probes={probes}; SET turbovec.search_k={k};\nSELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{lits[qids[0]]}'::turbovec.vector LIMIT 10;"],capture_output=True,text=True)
    # one session, all queries, EXPLAIN JSON + ids
    lines=[]
    for q in qids:
        lines.append(r"\echo @@T")
        lines.append(f"SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.probes={probes}; SET turbovec.search_k={k};")
        lines.append(f"EXPLAIN (ANALYZE,FORMAT JSON,TIMING ON,BUFFERS OFF) SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{lits[q]}'::turbovec.vector LIMIT 10;")
        lines.append(r"\echo @@I")
        lines.append(f"SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '{lits[q]}'::turbovec.vector LIMIT 10;")
    out=subprocess.run(["psql","-d","bench","-qAt","-F","|"],input="\n".join(lines)+"\n",capture_output=True,text=True).stdout
    blocks=out.split("@@T\n")[1:]
    for i,b in enumerate(blocks):
        j,rest=b.split("@@I\n",1)
        et=json.loads(j)[0]["Execution Time"]; times.append(et)
        ids=[int(x) for x in rest.strip().splitlines() if x.strip().lstrip("-").isdigit()]
        recs.append(len(set(ids)&set(gt[qids[i]]))/10.0)
    return statistics.median(times), statistics.mean(recs)
for probes in (128,256,512):
    for k in (256,800,2000):
        ms,r=run(probes,k)
        print(f"  {fam} probes={probes} k={k}: R@10={r:.4f} p50={ms:.2f}ms",flush=True)
PY
}
echo "### IVF bw1 high-probe sweep"; sweep_family ivf_bw1 docs_tv_ivf_bw1 "lists=1024, bit_width=1"
echo "### IVF bw4 high-probe sweep"; sweep_family ivf_bw4 docs_tv_ivf_bw4 "lists=1024, bit_width=4"
echo "### flat bw1 reference (probes=lists equivalent = full scan)"
dropx; mkidx "CREATE INDEX docs_tv_flat_bw1 ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width=1);"
psql -d bench -c "SELECT pg_size_pretty(pg_relation_size('docs_tv_flat_bw1'))"
echo "IVF098_DONE"
