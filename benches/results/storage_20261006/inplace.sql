-- Migrating a POPULATED table to MAIN without a full-table rewrite lock:
-- ALTER ... SET STORAGE MAIN, then re-store each vector with an UPDATE in
-- committed batches. What happens to heap/TOAST size and to the turbovec
-- index (flat and IVF)? 20k x 1024-d, default storage at load.
\pset footer off
DROP TABLE IF EXISTS ip_flat, ip_ivf;
CREATE TABLE ip_flat (id bigint PRIMARY KEY, tv turbovec.vector);
CREATE TABLE ip_ivf  (id bigint PRIMARY KEY, tv turbovec.vector);
COPY ip_flat FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
COPY ip_ivf  FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX ip_flat_tv ON ip_flat USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4);
CREATE INDEX ip_ivf_tv  ON ip_ivf  USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 64);
VACUUM ANALYZE ip_flat; VACUUM ANALYZE ip_ivf;
CREATE TEMP VIEW st AS
SELECT t.relname, a.attstorage,
       round(pg_relation_size(t.oid) / 1048576.0, 1) AS heap_mb,
       round(pg_relation_size(t.reltoastrelid) / 1048576.0, 1) AS toast_mb,
       (SELECT n_vectors FROM turbovec.turbovec_check(i.indexrelid)) AS index_entries,
       turbovec.index_is_degraded(i.indexrelid) AS degraded
  FROM pg_class t JOIN pg_attribute a ON a.attrelid = t.oid AND a.attname = 'tv'
  JOIN pg_index i ON i.indrelid = t.oid AND NOT i.indisprimary
 WHERE t.relname IN ('ip_flat', 'ip_ivf') ORDER BY 1;
SELECT '0 loaded' AS step, * FROM st;
ALTER TABLE ip_flat ALTER COLUMN tv SET STORAGE MAIN;
ALTER TABLE ip_ivf  ALTER COLUMN tv SET STORAGE MAIN;
-- batch 1 of 4: a cast round-trip yields a new datum, which is stored per the new setting
UPDATE ip_flat SET tv = tv::real[]::turbovec.vector WHERE id < 5000;
UPDATE ip_ivf  SET tv = tv::real[]::turbovec.vector WHERE id < 5000;
SELECT '1 after UPDATE of 25% (no VACUUM yet)' AS step, * FROM st;
VACUUM ip_flat; VACUUM ip_ivf;
SELECT '2 after VACUUM' AS step, * FROM st;
UPDATE ip_flat SET tv = tv::real[]::turbovec.vector WHERE id >= 5000;
UPDATE ip_ivf  SET tv = tv::real[]::turbovec.vector WHERE id >= 5000;
VACUUM ip_flat; VACUUM ip_ivf;
SELECT '3 after UPDATE of the rest + VACUUM' AS step, * FROM st;
REINDEX INDEX ip_ivf_tv;
SELECT '4 after REINDEX of the IVF index' AS step, * FROM st;
DROP VIEW st; DROP TABLE ip_flat, ip_ivf;
