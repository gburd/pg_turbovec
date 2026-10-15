#!/usr/bin/env bash
# idx_sha.sh <label> : CHECKPOINT, then sha256 of every segment of both turbovec index relfiles.
export PATH=/opt/pg18/bin:$PATH; D=/data/pgdata; P="psql -h /tmp -X -d db_tv -Atc"
$P "CHECKPOINT" >/dev/null
for i in docs_tv_flat docs_ivf_tv; do f=$D/$($P "SELECT pg_relation_filepath('$i')")
  for s in $f $f.* ${f}_*; do [ -f $s ] && echo "$1 $i $(stat -c %s $s) $(sha256sum $s)"; done; done >> ~/results/q/index_sha.txt
tail -4 ~/results/q/index_sha.txt
