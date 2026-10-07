-- Follow-up to inplace.sql: does VACUUM remove the dead index entries left by
-- an UPDATE of the vector column? VERBOSE output shows whether index
-- vacuuming ran or was bypassed (PG 14+ skips it when < 2% of heap pages have
-- dead items). Run with -v st=EXTENDED (column unchanged) and -v st=MAIN
-- (column switched to MAIN first, so the new versions are inline and the heap
-- grows ~30x, which pushes the dead-page share under the bypass threshold).
\pset footer off
DROP TABLE IF EXISTS iv;
CREATE TABLE iv (id bigint PRIMARY KEY, tv turbovec.vector);
COPY iv FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX iv_tv ON iv USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4);
VACUUM ANALYZE iv;
ALTER TABLE iv ALTER COLUMN tv SET STORAGE :st;
UPDATE iv SET tv = tv::real[]::turbovec.vector WHERE id < 5000;
SELECT :'st' AS storage, 'after update' AS step, n_vectors FROM turbovec.turbovec_check('iv_tv'::regclass);
VACUUM (VERBOSE) iv;
SELECT :'st' AS storage, 'after VACUUM' AS step, n_vectors FROM turbovec.turbovec_check('iv_tv'::regclass);
VACUUM (VERBOSE, INDEX_CLEANUP ON) iv;
SELECT :'st' AS storage, 'after VACUUM (INDEX_CLEANUP ON)' AS step, n_vectors FROM turbovec.turbovec_check('iv_tv'::regclass);
SELECT count(*) AS rows_visible FROM iv;
DROP TABLE iv;
