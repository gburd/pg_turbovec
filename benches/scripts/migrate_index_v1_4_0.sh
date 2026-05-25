#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/lib64
PSQL="$HOME/.pgrx/17.9/pgrx-install/bin/psql -X -P pager=off -h /scratch/pg_turbovec-bench -p 28815 -d bench_dbpedia"

echo "=== ensure extension v1.4.0 ==="
$PSQL <<'SQL'
DROP EXTENSION IF EXISTS pg_turbovec CASCADE;
CREATE EXTENSION pg_turbovec;
SELECT extversion FROM pg_extension WHERE extname = 'pg_turbovec';
SQL

echo "=== build index v1.4.0 (Phase O-3 settings: 512MB / 0 workers) ==="
T0=$(date +%s)
$PSQL <<'SQL'
SET search_path = turbovec, public;
SET maintenance_work_mem = '512MB';
SET max_parallel_maintenance_workers = 0;
\timing on
CREATE INDEX docs_tv_4bit ON docs USING turbovec
    ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 4);
SQL
T1=$(date +%s)
BUILD_S=$(( T1 - T0 ))
echo "BUILD_TIME_S=${BUILD_S}"

echo "=== index size ==="
$PSQL <<'SQL'
SELECT pg_size_pretty(pg_relation_size('docs_tv_4bit')) AS size,
       pg_relation_size('docs_tv_4bit') AS bytes;
SQL
echo "=== done ==="
