#!/bin/bash
# tvp_arm.sh <arm> <round> : install /work/ab/so/<arm>.so into pg16ab, restart
# postmaster with pinned glibc malloc tunables (Fix C finding: avoids bimodal
# trim/grow page-fault state), warm OS cache, run the A/B pass.
set -euo pipefail
ARM=$1; R=$2; B=/work/pg16ab/bin; D=/work/ab/data
$B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true
cp /work/ab/so/$ARM.so /work/pg16ab/lib/postgresql/pg_turbovec.so
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 \
  $B/pg_ctl -D $D -l /work/ab/pg.log -w start >/dev/null
echo "arm=$ARM round=$R md5=$(md5sum /work/pg16ab/lib/postgresql/pg_turbovec.so | cut -c1-8)"
# warm heap+toast+index into shared_buffers/OS cache
$B/psql -h /tmp -p 5440 -d bench -qAt -c "SELECT count(*) FROM docs_ext" -c "SELECT count(*) FROM docs_main" >/dev/null
/work/venv/bin/python ~/tvp_ab.py run $ARM $R
