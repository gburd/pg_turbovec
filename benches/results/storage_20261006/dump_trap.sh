#!/bin/sh
# pg_dump writes a column's SET STORAGE only when it differs from the column
# TYPE's current storage. Dump once with the type at its default and once
# after ALTER TYPE ... (STORAGE = main), then restore each dump into a fresh
# database and read attstorage back.
B=/work/pg16rel_c/bin; P="$B/psql -X -h /work/fixc -p 55433"; DUMP="$B/pg_dump -h /work/fixc -p 55433 -s"
for db in trap trap_before trap_after; do $P -d bench -qc "DROP DATABASE IF EXISTS $db" 2>/dev/null; done
st() {
  $P -d $1 -qAt -c "SELECT 'type turbovec.vector: typstorage = ' || typstorage::text FROM pg_type WHERE oid = 'turbovec.vector'::regtype" \
    -c "SELECT attrelid::regclass || '.tv: attstorage = ' || attstorage::text FROM pg_attribute WHERE attname = 'tv' AND attrelid::regclass::text IN ('per_column', 'via_type') ORDER BY 1"
}
$P -d bench -qc "CREATE DATABASE trap"
$P -d trap -q -c "CREATE EXTENSION pg_turbovec" -c "CREATE TABLE per_column (id int, tv turbovec.vector)" \
  -c "ALTER TABLE per_column ALTER COLUMN tv SET STORAGE MAIN"
echo "=== 1. type at its default; per_column set to MAIN by ALTER TABLE"; st trap
$DUMP trap > /tmp/trap_before.sql
echo "--- pg_dump -s (BEFORE ALTER TYPE): CREATE EXTENSION / CREATE TABLE / SET STORAGE lines"
grep -E "^CREATE EXTENSION|^CREATE TABLE|SET STORAGE" /tmp/trap_before.sql
$P -d trap -q -c "ALTER TYPE turbovec.vector SET (STORAGE = main)" -c "CREATE TABLE via_type (id int, tv turbovec.vector)"
echo; echo "=== 2. after ALTER TYPE turbovec.vector SET (STORAGE = main); via_type created afterwards"; st trap
$DUMP trap > /tmp/trap_after.sql
echo "--- pg_dump -s (AFTER ALTER TYPE): CREATE EXTENSION / CREATE TABLE / SET STORAGE lines"
grep -E "^CREATE EXTENSION|^CREATE TABLE|SET STORAGE" /tmp/trap_after.sql
for w in before after; do
  $P -d bench -qc "CREATE DATABASE trap_$w"
  echo; echo "=== 3. '$w' dump restored into a fresh database (errors shown, if any)"
  $P -d trap_$w -q -f /tmp/trap_$w.sql 2>&1 | grep ERROR
  st trap_$w
done
for db in trap trap_before trap_after; do $P -d bench -qc "DROP DATABASE $db"; done
