#!/bin/bash
# swap old|new [ENV=VAL...] : install that .so, restart PG with the malloc tunables, prewarm.
set -euo pipefail
W=/work/ab; export PATH=$W/pg16r/bin:$PATH
pg_ctl -D $W/data -m fast -w stop >/dev/null
cp $W/so/$1.so $(pg_config --pkglibdir)/pg_turbovec.so; A=$1; shift
env GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 "$@" pg_ctl -D $W/data -l $W/pg.log -w start >/dev/null
psql -h /tmp -d bench -qAtc "SELECT pg_prewarm('public.docs'), pg_prewarm('public.docs_pkey'), pg_prewarm('public.docs_flat4')" >/dev/null
echo "swapped -> $A $* md5=$(md5sum $(pg_config --pkglibdir)/pg_turbovec.so | cut -c1-8)"
