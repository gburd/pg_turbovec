#!/bin/bash
# attribution with warm shared_buffers, per arm
set -uo pipefail
B=/work/pg16ab/bin; D=/work/ab/data
: > /work/ab/attr_warm.jsonl
for arm in old new; do
  $B/pg_ctl -D $D -m fast -w stop >/dev/null 2>&1 || true
  cp /work/ab/so/$arm.so /work/pg16ab/lib/postgresql/pg_turbovec.so
  GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 $B/pg_ctl -D $D -l /work/ab/pg.log -w start >/dev/null
  $B/psql -h /tmp -p 5440 -d bench -qAt -c "SELECT pg_prewarm(c.oid) FROM pg_class c WHERE c.relname IN ('docs_ext','docs_main','docs_ext_tv','docs_main_tv') OR c.oid IN (SELECT reltoastrelid FROM pg_class WHERE relname IN ('docs_ext','docs_main') AND reltoastrelid <> 0) OR c.oid IN (SELECT indexrelid FROM pg_index WHERE indrelid IN (SELECT reltoastrelid FROM pg_class WHERE relname='docs_ext'))" >/dev/null
  for t in docs_ext docs_main; do sed 's|/work/ab/attr.jsonl|/work/ab/attr_warm.jsonl|' ~/tvp_attr.py > ~/tvp_attr_w.py; /work/venv/bin/python ~/tvp_attr_w.py $arm $t 1024 150; done
done
