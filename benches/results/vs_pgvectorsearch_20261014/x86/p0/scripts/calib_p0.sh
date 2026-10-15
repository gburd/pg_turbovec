#!/usr/bin/env bash
# P0 calibration grid (recall only; latencies are P0 timing info, NOT headline) on 1,000 of the timed qids
# to time the cell types. Per arm: restart + drop_caches + prewarm (heap+index+queries), then the grid in ONE connection per point.
set -x
export PATH=/opt/pg18/bin:$PATH; R=~/results/p0/calib; mkdir -p $R
P="psql -h /tmp -X -v ON_ERROR_STOP=1"
N=${N:-1000}
restart_prewarm() { # db rels...
  local db=$1; shift
  pg_ctl -D /data/pgdata -m fast -w stop; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  pg_ctl -D /data/pgdata -l ~/pg.log -w start; env | grep -ci GLIBC_TUNABLES
  for r in "$@"; do $P -d $db -Atc "SELECT '$r', pg_prewarm('$r')"; done
}
arm() { # name db table idx sys knob values...
  local name=$1 db=$2 tbl=$3 idx=$4 sys=$5 knob=$6 extra=$7; shift 7
  local t0=$(date +%s)
  restart_prewarm $db $tbl queries $idx
  local t1=$(date +%s)
  for v in "$@"; do
    python3 ~/bench.py --db $db --sys $sys --label ${name}_${v} --table $tbl --set "$knob=$v$extra" --mode ids --warm 50 --n $N --out $R --expect-index $idx
  done
  local t2=$(date +%s); echo "ARM $name restart_prewarm_s=$((t1-t0)) grid_s=$((t2-t1)) points=$#" | tee -a $R/arm_times.txt
}
case ${1:-all} in
 prism|all) arm prism db_pv docs docs_prism pv prism.nprobe "" 4 6 8 10 14 20 28 40 56 80 112 160 224 320 448 640 1024 0 ;;&
 tvflat|all) arm tvflat db_tv docs docs_tv_flat tv turbovec.search_k ";turbovec.hi_dim_rerank=off" 10 16 24 32 48 64 96 128 192 256 384 512 768 1024 ;;&
 tvflatdef|all) arm tvflatdef db_tv docs docs_tv_flat tv turbovec.search_k "" 32 ;;&
 tvivf|all) arm tvivf db_tv docs_ivf docs_ivf_tv tv turbovec.probes "" 1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 ;;&
 hnsw|all) arm hnsw db_pv docs_h docs_h_hnsw pv hnsw.ef_search "" 40 100 200 400 ;;
esac
echo CALIB_DONE
