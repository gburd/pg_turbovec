#!/usr/bin/env bash
set -euo pipefail
export DB=bench OUT=/home/ubuntu/rb_result.json
rm -f "$OUT"
PY=/home/ubuntu/venv/bin/python3
gate=${LOAD_GATE:-6.0}; nt=${N_TIMED:-40}
mkidx(){ psql -d bench -v ON_ERROR_STOP=1 -c "SET maintenance_work_mem='8GB'; SET max_parallel_maintenance_workers=8; $1"; }
dropx(){ psql -d bench -qc "DROP INDEX IF EXISTS $1"; }
wait_load(){ for i in $(seq 1 40); do la=$(cut -d' ' -f1 /proc/loadavg); awk -v l=$la -v g=$gate 'BEGIN{exit !(l<g*0.5)}' && return; sleep 15; done; }

echo "### build HNSW (kept for all passes)"
dropx docs_hnsw
mkidx "CREATE INDEX docs_hnsw ON public.docs USING hnsw (emb vector_cosine_ops) WITH (m=16, ef_construction=64);"
wait_load
echo "### sweep HNSW"
LOAD_GATE=$gate N_TIMED=$nt ARM_FAMILY=hnsw $PY /home/ubuntu/rb_driver.py

for fam in flat_bw4 flat_bw1 ivf_bw4 ivf_bw1; do
  case $fam in
    flat_bw4) idx=docs_tv_flat_bw4; opt="bit_width=4";;
    flat_bw1) idx=docs_tv_flat_bw1; opt="bit_width=1";;
    ivf_bw4)  idx=docs_tv_ivf_bw4;  opt="lists=1024, bit_width=4";;
    ivf_bw1)  idx=docs_tv_ivf_bw1;  opt="lists=1024, bit_width=1";;
  esac
  echo "### build $idx ($opt) — the ONLY tv index this pass, so no planner ambiguity"
  for x in docs_tv_flat_bw4 docs_tv_flat_bw1 docs_tv_ivf_bw4 docs_tv_ivf_bw1; do dropx $x; done
  mkidx "CREATE INDEX $idx ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH ($opt);"
  wait_load
  echo "### sweep $fam"
  LOAD_GATE=$gate N_TIMED=$nt ARM_FAMILY=$fam $PY /home/ubuntu/rb_driver.py
done
echo "### DONE. sizes:"
psql -d bench -c "SELECT c.relname, pg_size_pretty(pg_relation_size(c.oid)) FROM pg_class c WHERE c.relkind='i' AND c.relname LIKE 'docs%' ORDER BY 1"
