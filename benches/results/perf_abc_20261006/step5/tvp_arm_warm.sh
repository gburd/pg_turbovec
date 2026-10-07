#!/bin/bash
# Like tvp_arm.sh, but pg_prewarm heap + TOAST + index into shared_buffers
# after the restart, so timed runs see steady-state warm shared_buffers.
set -euo pipefail
ARM=$1; R=$2; B=/work/pg16ab/bin; D=/work/ab/data
$B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true
cp /work/ab/so/$ARM.so /work/pg16ab/lib/postgresql/pg_turbovec.so
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 \
  $B/pg_ctl -D $D -l /work/ab/pg.log -w start >/dev/null
echo "arm=$ARM round=$R md5=$(md5sum /work/pg16ab/lib/postgresql/pg_turbovec.so | cut -c1-8) GLIBC_TUNABLES=pinned"
$B/psql -h /tmp -p 5440 -d bench -qAt -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS pg_prewarm" \
  -c "SELECT 'prewarm', c.relname, pg_prewarm(c.oid) FROM pg_class c WHERE c.relname IN ('docs_ext','docs_main','docs_ext_tv','docs_main_tv') OR c.oid IN (SELECT reltoastrelid FROM pg_class WHERE relname IN ('docs_ext','docs_main') AND reltoastrelid <> 0) OR c.oid IN (SELECT indexrelid FROM pg_index WHERE indrelid IN (SELECT reltoastrelid FROM pg_class WHERE relname='docs_ext'))"
$B/psql -h /tmp -p 5440 -d bench -qAt -c "SELECT 'buffers_used_MB', count(*) * 8 / 1024 FROM pg_buffercache WHERE relfilenode IS NOT NULL" 2>/dev/null || true
OUT=/work/ab/results_warm.jsonl /work/venv/bin/python ~/tvp_ab.py run $ARM $R
