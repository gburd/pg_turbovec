#!/bin/sh
# Warm A/B under (1) default glibc malloc and (2) pinned malloc thresholds
# (no trim, no per-query mmap). Restarts the cluster for each configuration.
cd /work/fixc; B=/work/pg16rel_c/bin; PY=/work/venv/bin/python
for cfg in default pinned; do
  $B/pg_ctl -D data -m fast -w stop >/dev/null
  if [ $cfg = pinned ]; then
    GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 $B/pg_ctl -D data -l pg.log -w start >/dev/null
  else
    $B/pg_ctl -D data -l pg.log -w start >/dev/null
  fi
  $PY bench.py 1024 5 200 32,256,1024 ext extl main plain > warm_${cfg}_1024.json
  for d in 384 768 1536; do $PY bench.py $d 3 200 32,256,1024 ext main > warm_${cfg}_$d.json; done
done
$B/pg_ctl -D data -m fast -w stop >/dev/null; $B/pg_ctl -D data -l pg.log -w start >/dev/null
