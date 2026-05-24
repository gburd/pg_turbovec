#!/usr/bin/env bash
# Single-shot: build pg_turbovec 4-bit and 2-bit on bench_dbpedia.docs.
# Aimed at fitting in 31 GiB of RAM with 8 GiB shared_buffers.
set -uo pipefail

export PGHOST=/scratch/pg_turbovec-bench
export PGPORT=28815
export PGUSER=gburd
export PGDATABASE=bench_dbpedia
export LD_LIBRARY_PATH=/lib64
export PATH=/home/gburd/.pgrx/17.9/pgrx-install/bin:$HOME/.local/bin:$PATH
export PSQLRC=/dev/null

OUT=/scratch/pg_turbovec-bench/dbpedia_sweep
mkdir -p "$OUT"
LOG="$OUT/tv_build.log"
: > "$LOG"

ts(){ date -u +'%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

build_one() {
    local bw=$1
    log "build pg_turbovec ${bw}-bit"
    T=$(date +%s)
    psql -X -q -P pager=off -v ON_ERROR_STOP=1 <<SQL >>"$LOG" 2>&1
SET search_path = turbovec, public;
SET maintenance_work_mem = '1GB';
SET max_parallel_maintenance_workers = 0;
DROP INDEX IF EXISTS docs_tv_${bw}bit;
CREATE INDEX docs_tv_${bw}bit ON docs
  USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
  WITH (bit_width = ${bw});
SQL
    rc=$?
    el=$(($(date +%s)-T))
    if [ $rc -eq 0 ]; then
        log "  ${bw}-bit done in ${el}s"
        echo "tv_${bw}bit_build_s=${el}" >> "$OUT/build_times.txt"
    else
        log "  ${bw}-bit FAILED rc=$rc after ${el}s"
        return $rc
    fi
}

build_one 4 || exit 1
build_one 2 || exit 1
log "all turbovec indexes built"
