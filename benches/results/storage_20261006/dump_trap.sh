#!/bin/sh
# pg_dump keeps a per-column SET STORAGE only when it differs from the type default.
P="/work/pg16rel_c/bin/psql -X -h /work/fixc -p 55433"
$P -d bench -qc "DROP DATABASE IF EXISTS trap" -c "CREATE DATABASE trap"
$P -d trap -q <<'SQL'
CREATE EXTENSION pg_turbovec;
CREATE TABLE per_column (id int, tv turbovec.vector);
ALTER TABLE per_column ALTER COLUMN tv SET STORAGE MAIN;
ALTER TYPE turbovec.vector SET (STORAGE = main);
CREATE TABLE via_type (id int, tv turbovec.vector);
SELECT attrelid::regclass, attstorage FROM pg_attribute WHERE attname = 'tv' AND attrelid IN ('per_column'::regclass, 'via_type'::regclass);
SQL
echo "--- pg_dump -s, storage lines:"
/work/pg16rel_c/bin/pg_dump -h /work/fixc -p 55433 -s trap | grep -i -E "SET STORAGE|^CREATE TABLE"
$P -d bench -qc "DROP DATABASE trap"
