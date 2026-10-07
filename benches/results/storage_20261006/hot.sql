-- HOT updates of a non-vector column need room for the new tuple version on
-- the same page. Default storage vs MAIN, 20k x 1024-d, both fillfactor = 50.
\pset footer off
DROP TABLE IF EXISTS h_ext, h_main;
CREATE TABLE h_ext  (id bigint PRIMARY KEY, meta int NOT NULL DEFAULT 0, tv turbovec.vector) WITH (fillfactor = 50);
CREATE TABLE h_main (id bigint PRIMARY KEY, meta int NOT NULL DEFAULT 0, tv turbovec.vector) WITH (fillfactor = 50);
ALTER TABLE h_main ALTER COLUMN tv SET STORAGE MAIN;
COPY h_ext  (id, tv) FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
COPY h_main (id, tv) FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX ON h_ext  USING turbovec (tv turbovec.vec_cosine_ops);
CREATE INDEX ON h_main USING turbovec (tv turbovec.vec_cosine_ops);
VACUUM ANALYZE h_ext; VACUUM ANALYZE h_main;
UPDATE h_ext  SET meta = meta + 1 WHERE id < 2000;
UPDATE h_main SET meta = meta + 1 WHERE id < 2000;
SELECT pg_stat_force_next_flush();
SELECT pg_sleep(1);
SELECT relname, n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname IN ('h_ext', 'h_main') ORDER BY 1;
DROP TABLE h_ext, h_main;
