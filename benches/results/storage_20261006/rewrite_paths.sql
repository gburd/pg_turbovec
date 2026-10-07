-- Which rewrite paths actually move existing out-of-line vectors inline after
-- SET STORAGE MAIN, and which DDL keeps the column's storage setting.
-- Source: 20,000 rows x 1024-d loaded with the default (EXTENDED) storage.
\pset footer off
DROP TABLE IF EXISTS rw_src, rw_ctas, rw_like, rw_upd CASCADE;
CREATE TABLE rw_src (id bigint PRIMARY KEY, tv turbovec.vector);
COPY rw_src FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE OR REPLACE FUNCTION pg_temp.where_(t regclass) RETURNS TABLE(tbl text, attstorage "char", heap_mb numeric, toast_mb numeric)
LANGUAGE sql AS $$
  SELECT t::text, a.attstorage, round(pg_relation_size(t) / 1048576.0, 1),
         round(coalesce(pg_relation_size(nullif(c.reltoastrelid, 0)), 0) / 1048576.0, 1)
    FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'tv' WHERE c.oid = t $$;
ALTER TABLE rw_src ALTER COLUMN tv SET STORAGE MAIN;
-- 1. CREATE TABLE AS: new column gets the TYPE default (EXTENDED), not MAIN.
CREATE TABLE rw_ctas AS SELECT * FROM rw_src;
-- 2. LIKE ... INCLUDING STORAGE copies the column's MAIN; INSERT ... SELECT
--    from the TOASTed source then stores the values inline.
CREATE TABLE rw_like (LIKE rw_src INCLUDING ALL);
INSERT INTO rw_like SELECT * FROM rw_src;
-- 3. In-place UPDATE: "SET tv = tv" keeps the old TOAST pointer (no move);
--    a cast round-trip produces a new datum and does move it.
CREATE TABLE rw_upd (LIKE rw_src INCLUDING ALL);
ALTER TABLE rw_upd ALTER COLUMN tv SET STORAGE EXTENDED;
INSERT INTO rw_upd SELECT * FROM rw_src;
ALTER TABLE rw_upd ALTER COLUMN tv SET STORAGE MAIN;
SELECT * FROM pg_temp.where_('rw_upd') AS before_update;
UPDATE rw_upd SET tv = tv WHERE id < 10000;
VACUUM rw_upd;
SELECT * FROM pg_temp.where_('rw_upd') AS after_set_tv_eq_tv;
UPDATE rw_upd SET tv = tv::real[]::turbovec.vector WHERE id < 10000;
VACUUM rw_upd;
SELECT * FROM pg_temp.where_('rw_upd') AS after_cast_roundtrip_half;
SELECT bool_and(a.tv::text = b.tv::text) AS values_identical FROM rw_upd a JOIN rw_src b USING (id);
SELECT * FROM pg_temp.where_('rw_src') UNION ALL SELECT * FROM pg_temp.where_('rw_ctas') UNION ALL SELECT * FROM pg_temp.where_('rw_like');
