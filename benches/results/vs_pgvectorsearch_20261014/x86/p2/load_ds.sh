#!/usr/bin/env bash
# P2 load: COPY a prepared dataset into db_pv (pgvector) + db_tv (turbovec), LOGGED, STORAGE PLAIN,
# 16 parallel streams. Suffix-namespaces every object by $TAG so 1M/10M/OA coexist.
# usage: load_ds.sh <datadir> <dim> <tag>     e.g. load_ds.sh /data/co1024_10m 1024 co10m
set -euxo pipefail
export PATH=/opt/pg18/bin:$PATH
DD=$1; DIM=$2; TAG=$3; D=$DD/copy; R=~/results/p2; mkdir -p $R
PS="psql -h /tmp -v ON_ERROR_STOP=1 -X"
DOCS=docs_$TAG; Q=queries_$TAG
$PS -d db_pv -c "DROP TABLE IF EXISTS $DOCS, $Q;
  CREATE TABLE $DOCS(id bigint PRIMARY KEY, emb vector($DIM)); ALTER TABLE $DOCS ALTER COLUMN emb SET STORAGE PLAIN;
  CREATE TABLE $Q(qid int PRIMARY KEY, q vector($DIM)); ALTER TABLE $Q ALTER COLUMN q SET STORAGE PLAIN;"
$PS -d db_tv -c "DROP TABLE IF EXISTS $DOCS, $Q;
  CREATE TABLE $DOCS(id bigint PRIMARY KEY, emb turbovec.vector); ALTER TABLE $DOCS ALTER COLUMN emb SET STORAGE PLAIN;
  CREATE TABLE $Q(qid int PRIMARY KEY, q turbovec.vector); ALTER TABLE $Q ALTER COLUMN q SET STORAGE PLAIN;"
for db in db_pv db_tv; do
  t0=$(date +%s.%N)
  for f in $D/docs_*.tsv; do $PS -d $db -c "\\copy $DOCS FROM '$f'" & done; wait
  $PS -d $db -c "\\copy $Q FROM '$D/queries.tsv'"
  t1=$(date +%s.%N); echo "LOAD $db $TAG $(echo "$t1-$t0"|bc) s" | tee -a $R/load_times.txt
  $PS -d $db -c "VACUUM (FREEZE, ANALYZE) $DOCS" -c "VACUUM (FREEZE, ANALYZE) $Q"
  t2=$(date +%s.%N); echo "VACUUM $db $TAG $(echo "$t2-$t1"|bc) s" | tee -a $R/load_times.txt
done
{
echo "== $TAG checksums (count,sum(id),sum c1,sum cDIM)"
echo "pv: $($PS -d db_pv -Atc "SELECT count(*),sum(id),sum((emb::real[])[1]::float8),sum((emb::real[])[$DIM]::float8) FROM $DOCS")"
echo "tv: $($PS -d db_tv -Atc "SELECT count(*),sum(id),sum((emb::real[])[1]::float8),sum((emb::real[])[$DIM]::float8) FROM $DOCS")"
echo "pv q: $($PS -d db_pv -Atc "SELECT count(*),sum((q::real[])[1]::float8) FROM $Q")"
echo "tv q: $($PS -d db_tv -Atc "SELECT count(*),sum((q::real[])[1]::float8) FROM $Q")"
for db in db_pv db_tv; do echo "== $db $DOCS sizes/persist/storage";
  $PS -d $db -Atc "SELECT relname,relpersistence,pg_relation_size(oid),pg_total_relation_size(oid),relpages,(SELECT attstorage FROM pg_attribute WHERE attrelid=c.oid AND attnum=2) FROM pg_class c WHERE relname='$DOCS'";
  $PS -d $db -Atc "SELECT avg(pg_column_size(emb)) FROM (SELECT emb FROM $DOCS LIMIT 10000) s"; done
} > $R/load_checksums_$TAG.txt 2>&1
cat $R/load_checksums_$TAG.txt
echo "LOAD_DONE_$TAG"
