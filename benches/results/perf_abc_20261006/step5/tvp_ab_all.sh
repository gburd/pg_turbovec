#!/bin/bash
# Step-5 A/B: arms old (perf/recheck-abc = v2.11.0 code) vs new (A+B integrated),
# alternated; 3 rounds latency+recall (tvp_arm.sh), then one attribution pass
# per arm per table at search_k=1024 and 100. Postmaster restarted per arm with
# pinned glibc malloc tunables.
set -uo pipefail
ROUNDS=${ROUNDS:-3}
: > /work/ab/results.jsonl
for r in $(seq 1 $ROUNDS); do
  for arm in old new; do bash ~/tvp_arm.sh $arm $r; done
done
: > /work/ab/attr.jsonl
for arm in old new; do
  B=/work/pg16ab/bin; D=/work/ab/data
  $B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true
  cp /work/ab/so/$arm.so /work/pg16ab/lib/postgresql/pg_turbovec.so
  GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 $B/pg_ctl -D $D -l /work/ab/pg.log -w start >/dev/null
  for t in docs_ext docs_main; do for k in 1024 100; do /work/venv/bin/python ~/tvp_attr.py $arm $t $k 150; done; done
done
/work/venv/bin/python ~/tvp_ab.py summary
echo AB_ALL_DONE
