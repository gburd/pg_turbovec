-- Cost of updating a NON-vector column, default storage vs MAIN.
-- 100k x 1024-d, an int column "meta", flat 4-bit turbovec index.
-- One committed transaction updates meta on 10,000 rows, right after a
-- CHECKPOINT (worst case for full-page images). Reports HOT count, WAL bytes,
-- and heap / TOAST / turbovec-index size before and after.
-- usage: psql -v st=EXTENDED|MAIN [-v ttt=8160] -f update_cost.sql
\pset footer off
DROP TABLE IF EXISTS u_t;
CREATE TABLE u_t (id bigint PRIMARY KEY, meta int NOT NULL DEFAULT 0, tv turbovec.vector);
ALTER TABLE u_t ALTER COLUMN tv SET STORAGE :st;
\if :{?ttt}
ALTER TABLE u_t SET (toast_tuple_target = :ttt);
\else
\set ttt default
\endif
COPY u_t (id, tv) FROM '/work/fixc/data_1024.tsv';
CREATE INDEX u_t_tv ON u_t USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4);
VACUUM ANALYZE u_t;
CREATE TEMP VIEW sz AS SELECT round(pg_relation_size('u_t') / 1048576.0, 1) AS heap_mb,
  round(pg_relation_size((SELECT reltoastrelid FROM pg_class WHERE relname = 'u_t')) / 1048576.0, 1) AS toast_mb,
  round(pg_relation_size('u_t_tv') / 1048576.0, 1) AS tv_index_mb;
SELECT :'st' AS storage, :'ttt' AS toast_tuple_target, 'before' AS at, * FROM sz;
CHECKPOINT;
SELECT pg_current_wal_lsn() AS l0 \gset
\timing on
UPDATE u_t SET meta = meta + 1 WHERE id < 10000;
\timing off
SELECT pg_stat_force_next_flush();
SELECT :'st' AS storage, :'ttt' AS toast_tuple_target, 'after' AS at, *, round((pg_current_wal_lsn() - :'l0'::pg_lsn) / 1048576.0, 1) AS wal_mb FROM sz;
SELECT pg_sleep(1);
SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname = 'u_t';
SELECT n_vectors, slot_count, is_corrupt FROM turbovec.turbovec_check('u_t_tv'::regclass);
DROP VIEW sz; DROP TABLE u_t;
