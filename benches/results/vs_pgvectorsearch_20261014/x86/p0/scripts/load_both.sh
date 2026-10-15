#!/usr/bin/env bash
# Load CO1024 1M into db_pv (pgvector vector(1024)) and db_tv (turbovec.vector), LOGGED, STORAGE PLAIN,
# 16 parallel text COPY streams from the SAME files. Then VACUUM (FREEZE, ANALYZE) + checksums (plan C (b)).
set -euxo pipefail
export PATH=/opt/pg18/bin:$PATH; D=/data/co1024_1m/copy; R=~/results/p0; mkdir -p $R
PS="psql -h /tmp -v ON_ERROR_STOP=1 -X"
$PS -d db_pv -c "DROP TABLE IF EXISTS docs, queries; CREATE TABLE docs(id bigint PRIMARY KEY, emb vector(1024)); ALTER TABLE docs ALTER COLUMN emb SET STORAGE PLAIN;
  CREATE TABLE queries(qid int PRIMARY KEY, q vector(1024)); ALTER TABLE queries ALTER COLUMN q SET STORAGE PLAIN;"
$PS -d db_tv -c "DROP TABLE IF EXISTS docs, queries; CREATE TABLE docs(id bigint PRIMARY KEY, emb turbovec.vector); ALTER TABLE docs ALTER COLUMN emb SET STORAGE PLAIN;
  CREATE TABLE queries(qid int PRIMARY KEY, q turbovec.vector); ALTER TABLE queries ALTER COLUMN q SET STORAGE PLAIN;"
for db in ${DBS:-db_pv db_tv}; do
  if [ "${SKIPLOAD_PV:-0}" = 1 ] && [ $db = db_pv ]; then t1=$(date +%s.%N); else
  t0=$(date +%s.%N)
  for f in $D/docs_*.tsv; do $PS -d $db -c "\\copy docs FROM '$f'" & done; wait
  $PS -d $db -c "\\copy queries FROM '$D/queries.tsv'"
  t1=$(date +%s.%N); echo "LOAD $db $(echo "$t1-$t0" | bc) s" | tee -a $R/load_times.txt; fi
  $PS -d $db -c "VACUUM (FREEZE, ANALYZE) docs" -c "VACUUM (FREEZE, ANALYZE) queries"
  t2=$(date +%s.%N); echo "VACUUM_FREEZE_ANALYZE $db $(echo "$t2-$t1" | bc) s" | tee -a $R/load_times.txt
done
{
echo "== pv"; $PS -d db_pv -Atc "SELECT count(*), sum(id), sum((emb::real[])[1]::float8), sum((emb::real[])[1024]::float8) FROM docs"
echo "== tv"; $PS -d db_tv -Atc "SELECT count(*), sum(id), sum((emb::real[])[1]::float8), sum((emb::real[])[1024]::float8) FROM docs"
echo "== pv queries"; $PS -d db_pv -Atc "SELECT count(*), sum((q::real[])[1]::float8) FROM queries"
echo "== tv queries"; $PS -d db_tv -Atc "SELECT count(*), sum((q::real[])[1]::float8) FROM queries"
for db in db_pv db_tv; do echo "== $db sizes/persistence"; $PS -d $db -Atc "SELECT c.relname, c.relpersistence, pg_relation_size(c.oid), pg_total_relation_size(c.oid), c.relpages, c.reltuples, (SELECT attstorage FROM pg_attribute WHERE attrelid=c.oid AND attnum=2) FROM pg_class c WHERE relname IN ('docs','queries')";
  $PS -d $db -Atc "SELECT pg_relation_size(reltoastrelid) FROM pg_class WHERE relname='docs'";
  $PS -d $db -Atc "SELECT avg(pg_column_size(emb)) FROM (SELECT emb FROM docs LIMIT 10000) s"; done
} > $R/load_checksums.txt 2>&1
cat $R/load_checksums.txt
echo LOAD_DONE
