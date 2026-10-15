#!/usr/bin/env bash
# One vector index per table so the planner can't switch indexes between arms.
set -x
export PATH=/opt/pg18/bin:$PATH; B=~/build_index.sh
P="psql -h /tmp -X -v ON_ERROR_STOP=1"
$P -d db_tv -c "DROP INDEX docs_tv_ivf" -c "CREATE TABLE docs_ivf (LIKE docs INCLUDING ALL)" -c "INSERT INTO docs_ivf SELECT * FROM docs" -c "VACUUM (FREEZE, ANALYZE) docs_ivf"
$P -d db_pv -c "DROP INDEX docs_hnsw" -c "CREATE TABLE docs_h (LIKE docs INCLUDING ALL)" -c "INSERT INTO docs_h SELECT * FROM docs" -c "VACUUM (FREEZE, ANALYZE) docs_h"
$P -d db_tv -Atc "SELECT relname, relpersistence, (SELECT attstorage FROM pg_attribute WHERE attrelid=c.oid AND attnum=2), pg_relation_size(oid) FROM pg_class c WHERE relname IN ('docs','docs_ivf')"
$P -d db_pv -Atc "SELECT relname, relpersistence, (SELECT attstorage FROM pg_attribute WHERE attrelid=c.oid AND attnum=2), pg_relation_size(oid) FROM pg_class c WHERE relname IN ('docs','docs_h')"
bash $B db_tv tv_ivf1000_t32_b "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_ivf_tv ON docs_ivf USING turbovec (emb vec_cosine_ops) WITH (lists = 1000)"
bash $B db_pv hnsw_m16_ef64_p32_b "SET max_parallel_maintenance_workers = 31" "CREATE INDEX docs_h_hnsw ON docs_h USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
$P -d db_pv -c "ALTER TABLE docs_h SET (parallel_workers = 31)"
echo SPLIT_DONE
