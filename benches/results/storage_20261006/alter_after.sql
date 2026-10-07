-- SET STORAGE on a populated table changes only FUTURE writes. Show it:
-- load 100k x 1024-d with the default (EXTENDED), then SET STORAGE MAIN, then
-- touch rows two ways, then rewrite with VACUUM FULL.
-- "external_rows" = distinct TOAST chunk_ids (after VACUUM, so dead versions are gone).
\timing on
\pset footer off
CREATE OR REPLACE FUNCTION pg_temp.sz(label text) RETURNS TABLE(step text, attstorage "char", heap_mb numeric, toast_mb numeric, external_rows bigint)
LANGUAGE plpgsql AS $$
DECLARE tr regclass; n bigint;
BEGIN
  SELECT reltoastrelid::regclass INTO tr FROM pg_class WHERE oid = 't1024_alt'::regclass;
  EXECUTE format('SELECT count(DISTINCT chunk_id) FROM %s', tr) INTO n;
  RETURN QUERY SELECT label, a.attstorage,
         round(pg_relation_size('t1024_alt') / 1048576.0, 1),
         round(pg_relation_size(tr) / 1048576.0, 1), n
    FROM pg_attribute a WHERE a.attrelid = 't1024_alt'::regclass AND a.attname = 'tv';
END $$;
DROP TABLE IF EXISTS t1024_alt;
CREATE TABLE t1024_alt (id bigint PRIMARY KEY, tv turbovec.vector);
COPY t1024_alt FROM '/work/fixc/data_1024.tsv';
CREATE INDEX t1024_alt_tv ON t1024_alt USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4);
VACUUM ANALYZE t1024_alt;
SELECT * FROM pg_temp.sz('1. loaded with default storage');
ALTER TABLE t1024_alt ALTER COLUMN tv SET STORAGE MAIN;
SELECT * FROM pg_temp.sz('2. after ALTER ... SET STORAGE MAIN');
UPDATE t1024_alt SET id = id + 0 WHERE id < 1000;               -- other column only
UPDATE t1024_alt SET tv = tv WHERE id >= 1000 AND id < 2000;     -- vector "rewritten" to itself
VACUUM t1024_alt;
SELECT * FROM pg_temp.sz('3. after UPDATE SET id=id (1000 rows) + SET tv=tv (1000 rows)');
UPDATE t1024_alt SET tv = turbovec.l2_normalize(tv) WHERE id >= 2000 AND id < 3000;  -- new vector value
VACUUM t1024_alt;
SELECT * FROM pg_temp.sz('4. after UPDATE SET tv = l2_normalize(tv) (1000 rows)');
VACUUM FULL t1024_alt;
SELECT * FROM pg_temp.sz('5. after VACUUM FULL');
SELECT count(*) AS knn_rows FROM (SELECT id FROM t1024_alt ORDER BY tv OPERATOR(turbovec.<=>) (SELECT tv FROM t1024_alt WHERE id = 5) LIMIT 10) s;
SELECT n_vectors, is_corrupt FROM turbovec.turbovec_check('t1024_alt_tv'::regclass);
DROP TABLE t1024_alt;
