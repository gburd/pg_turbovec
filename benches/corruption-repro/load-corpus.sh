#!/bin/bash
# Build a ~1.76M-row 768d flat turbovec corpus of DISTINCT vectors.
#
# CRITICAL: the per-element expression references the outer row id `g`
# (sin(g*.. + s*..) + random()) so the planner CANNOT hoist the array
# subquery into one shared value. `array_agg(random())` alone gets
# hoisted -> every row identical (n_distinct=1), the resume note's trap.
# We verify n_distinct afterward.
set -e
PSQL="/data/.pgrx/18.4/pgrx-install/bin/psql -p 28818 -d postgres -v ON_ERROR_STOP=1"
NROWS=${NROWS:-1760000}
DIM=768
echo "[load] dropping + recreating table (nrows=$NROWS dim=$DIM)"
$PSQL <<SQL
DROP TABLE IF EXISTS corpus CASCADE;
CREATE TABLE corpus (
  id      bigint PRIMARY KEY,
  emb     turbovec.vector
);
SQL

echo "[load] inserting $NROWS rows in batches of 20000 (correlated random)"
BATCH=20000
i=0
while [ $i -lt $NROWS ]; do
  end=$(( i + BATCH ))
  [ $end -gt $NROWS ] && end=$NROWS
  $PSQL -q <<SQL
INSERT INTO corpus (id, emb)
SELECT g,
       turbovec.array_to_vec(
         (SELECT array_agg((sin(g * 0.001 + s * 0.01) + random() * 0.01)::real)
          FROM generate_series(1,$DIM) AS s))
FROM generate_series($i, $end - 1) AS g;
SQL
  i=$end
  if [ $(( i % 200000 )) -eq 0 ]; then echo "[load]  $i rows @ $(date +%H:%M:%S)"; fi
done
echo "[load] row count:"
$PSQL -tAc "SELECT count(*) FROM corpus;"
echo "[load] distinctness sample (first 20000):"
$PSQL -tAc "SELECT count(DISTINCT emb::text) FROM (SELECT emb FROM corpus ORDER BY id LIMIT 20000) t;"
