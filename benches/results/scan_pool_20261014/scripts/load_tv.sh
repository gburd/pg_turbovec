#!/usr/bin/env bash
# Load CO1024 1M into db_tv twice (docs -> flat index, docs_ivf -> IVF index; one vector index per table,
# P0 harness finding), LOGGED, STORAGE PLAIN, 16 parallel text COPY streams from the P0-format files.
set -euxo pipefail
export PATH=/opt/pg18/bin:$PATH; D=/data/co1024_1m/copy; R=~/results/q; mkdir -p $R
PS="psql -h /tmp -v ON_ERROR_STOP=1 -X -d db_tv"
$PS -c "DROP TABLE IF EXISTS docs, docs_ivf, queries" -c "CREATE TABLE queries(qid int PRIMARY KEY, q turbovec.vector)" -c "ALTER TABLE queries ALTER COLUMN q SET STORAGE PLAIN"
for t in docs docs_ivf; do
  $PS -c "CREATE TABLE $t(id bigint PRIMARY KEY, emb turbovec.vector)" -c "ALTER TABLE $t ALTER COLUMN emb SET STORAGE PLAIN"
  t0=$(date +%s.%N); for f in $D/docs_*.tsv; do $PS -c "\\copy $t FROM '$f'" & done; wait
  echo "LOAD $t $(echo "$(date +%s.%N)-$t0" | bc) s" | tee -a $R/load_times.txt
  $PS -c "VACUUM (FREEZE, ANALYZE) $t"
done
$PS -c "\\copy queries FROM '$D/queries.tsv'"; $PS -c "VACUUM (FREEZE, ANALYZE) queries"
{ for t in docs docs_ivf; do echo "== $t"; $PS -Atc "SELECT count(*), sum(id), sum((emb::real[])[1]::float8), sum((emb::real[])[1024]::float8) FROM $t"; done
  echo "== queries"; $PS -Atc "SELECT count(*), sum((q::real[])[1]::float8) FROM queries"
  $PS -Atc "SELECT c.relname, c.relpersistence, pg_relation_size(c.oid), c.relpages, c.reltuples FROM pg_class c WHERE relname IN ('docs','docs_ivf','queries')"
} > $R/load_checksums.txt 2>&1
cat $R/load_checksums.txt
echo LOAD_DONE
