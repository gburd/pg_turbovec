#!/bin/sh
# toast_tuple_target = 8160 as an alternative to SET STORAGE MAIN, on 20k x
# 1024-d loaded with the default storage: the lock it takes, whether existing
# rows move without a rewrite, VACUUM FULL and pg_repack, what LIKE / CTAS
# copy, what pg_dump writes, and HOT for a non-vector UPDATE.
B=/work/pg16rel_c/bin; P="$B/psql -X -h /work/fixc -p 55433 -d bench"
$P -qc "CREATE EXTENSION IF NOT EXISTS pg_repack"
$P -q <<'SQL' 2>&1 | grep -v "NOTICE\|DETAIL\|HINT"
\pset footer off
DROP TABLE IF EXISTS tt, tt2, tt_like, tt_ctas;
CREATE TABLE tt  (id bigint PRIMARY KEY, tv turbovec.vector);
CREATE TABLE tt2 (id bigint PRIMARY KEY, tv turbovec.vector);
COPY tt  FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
COPY tt2 FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX tt_ivf  ON tt  USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 64);
CREATE INDEX tt2_ivf ON tt2 USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 64);
VACUUM ANALYZE tt; VACUUM ANALYZE tt2;
CREATE OR REPLACE VIEW tt_sz AS
SELECT c.relname, c.reloptions, round(pg_relation_size(c.oid) / 1048576.0, 1) AS heap_mb,
       round(pg_relation_size(c.reltoastrelid) / 1048576.0, 1) AS toast_mb,
       (SELECT n_vectors FROM turbovec.turbovec_check(i.indexrelid)) AS ivf_entries,
       turbovec.index_is_degraded(i.indexrelid) AS ivf_degraded
  FROM pg_class c JOIN pg_index i ON i.indrelid = c.oid AND NOT i.indisprimary
 WHERE c.relname IN ('tt', 'tt2') ORDER BY 1;
SELECT '0 loaded, default storage' AS step, * FROM tt_sz;
BEGIN;
ALTER TABLE tt SET (toast_tuple_target = 8160);
SELECT 'lock held by ALTER TABLE tt SET (toast_tuple_target = 8160)' AS what, mode
  FROM pg_locks WHERE relation = 'tt'::regclass AND pid = pg_backend_pid() AND granted;
COMMIT;
ALTER TABLE tt2 SET (toast_tuple_target = 8160);
UPDATE tt SET id = id WHERE id < 1000;
VACUUM tt;
SELECT '1 after ALTER + UPDATE of 1,000 rows (vector unchanged) + VACUUM' AS step, * FROM tt_sz;
VACUUM FULL tt;
SELECT '2 tt after VACUUM FULL' AS step, * FROM tt_sz WHERE relname = 'tt';
CREATE TABLE tt_like (LIKE tt INCLUDING ALL);
CREATE TABLE tt_ctas AS SELECT * FROM tt LIMIT 0;
SELECT relname, reloptions FROM pg_class WHERE relname IN ('tt', 'tt_like', 'tt_ctas') ORDER BY 1;
SQL
echo "--- pg_repack -t tt2"
$B/pg_repack -h /work/fixc -p 55433 -d bench -t tt2 2>&1 | grep -v "built a FLAT\|^DETAIL"
$P -q -c "\pset footer off" -c "SELECT '3 tt2 after pg_repack' AS step, * FROM tt_sz WHERE relname = 'tt2'"
echo "--- pg_dump -s -t tt (CREATE TABLE / WITH lines)"
$B/pg_dump -h /work/fixc -p 55433 -s -t tt bench | grep -E "^CREATE TABLE|^WITH|toast_tuple_target"
$P -q -c "DROP VIEW tt_sz" -c "DROP TABLE tt, tt2, tt_like, tt_ctas"
echo "--- HOT for a non-vector UPDATE (hot.sql shape: 20k rows, fillfactor = 50, 2,000 updates), toast_tuple_target = 8160"
$P -q <<'SQL' 2>&1 | grep -v "NOTICE"
\pset footer off
\pset tuples_only on
DROP TABLE IF EXISTS h_ttt;
CREATE TABLE h_ttt (id bigint PRIMARY KEY, meta int NOT NULL DEFAULT 0, tv turbovec.vector) WITH (fillfactor = 50, toast_tuple_target = 8160);
COPY h_ttt (id, tv) FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX ON h_ttt USING turbovec (tv turbovec.vec_cosine_ops);
VACUUM ANALYZE h_ttt;
UPDATE h_ttt SET meta = meta + 1 WHERE id < 2000;
SELECT pg_stat_force_next_flush();
SELECT pg_sleep(1);
\pset tuples_only off
SELECT relname, n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname = 'h_ttt';
DROP TABLE h_ttt;
SQL
