-- Two IVF checks behind the PRODUCTION.md § Column storage bullets.
-- (a) Phase Z5: an IVF index keeps its cells while the appended tail (rows
--     added by UPDATE/INSERT since the build) stays within
--     turbovec.ivf_max_delta_pct of the indexed rows; past that it degrades.
--     20k x 1024-d, SET STORAGE MAIN, value-preserving UPDATE batches.
-- (b) An IVF index created on an EMPTY table, then loaded: never gets cells,
--     and is not reported as degraded.
-- (c) The same through CREATE TABLE ... (LIKE src INCLUDING ALL), which copies
--     the IVF index definition onto the empty copy.
-- (d) The same through TRUNCATE + reload.
\pset footer off
SET enable_seqscan = off;
SHOW turbovec.ivf_max_delta_pct;
DROP TABLE IF EXISTS dl, em, em_like;
CREATE TABLE dl (id bigint PRIMARY KEY, tv turbovec.vector);
COPY dl FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX dl_ivf ON dl USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 64);
VACUUM ANALYZE dl;
ALTER TABLE dl ALTER COLUMN tv SET STORAGE MAIN;
CREATE TEMP VIEW deg AS
SELECT i.indexrelid::regclass AS index, d.degraded, turbovec.index_is_degraded(i.indexrelid) AS is_degraded,
       d.lists, d.n_vectors, round(d.scan_fraction::numeric, 4) AS scan_fraction
  FROM pg_index i, turbovec.index_degradation(i.indexrelid) d
 WHERE i.indrelid::regclass::text IN ('dl', 'em', 'em_like') AND NOT i.indisprimary ORDER BY 1;
SELECT 'a0 built on 20,000 rows' AS step, * FROM deg;
UPDATE dl SET tv = tv::real[]::turbovec.vector WHERE id < 1000;
SELECT 'a1 UPDATE 1,000 rows (5%)' AS step, * FROM deg;
UPDATE dl SET tv = tv::real[]::turbovec.vector WHERE id >= 1000 AND id < 2000;
SELECT 'a2 +1,000 (10% cumulative)' AS step, * FROM deg;
UPDATE dl SET tv = tv::real[]::turbovec.vector WHERE id >= 2000 AND id < 3000;
SELECT 'a3 +1,000 (15% cumulative)' AS step, * FROM deg;

CREATE TABLE em (id bigint PRIMARY KEY, tv turbovec.vector);
CREATE INDEX em_ivf ON em USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 16);
COPY em FROM PROGRAM 'head -n 3000 /work/fixc/data_1024.tsv';
SELECT 'b1 CREATE INDEX (lists = 16) on empty table, then COPY 3,000 rows' AS step, * FROM deg WHERE index::text = 'em_ivf';
SELECT pg_get_indexdef('em_ivf'::regclass);
-- a kNN scan through it (any WARNING would print here)
SELECT count(*) AS knn_rows FROM (SELECT id FROM em ORDER BY tv OPERATOR(turbovec.<=>) (SELECT tv FROM em WHERE id = 5) LIMIT 10) s;
REINDEX INDEX em_ivf;
SELECT 'b2 after REINDEX' AS step, * FROM deg WHERE index::text = 'em_ivf';

CREATE TABLE em_like (LIKE dl INCLUDING ALL);
INSERT INTO em_like SELECT * FROM dl;
SELECT 'c1 (LIKE dl INCLUDING ALL), then INSERT ... SELECT 20,000 rows' AS step, * FROM deg WHERE index::text LIKE 'em_like%';
SELECT pg_get_indexdef(indexrelid) FROM pg_index WHERE indrelid = 'em_like'::regclass AND NOT indisprimary;
-- (d) TRUNCATE of a table with a healthy IVF index, then reload
TRUNCATE em;
SELECT 'd0 em_ivf after the REINDEX above, then TRUNCATE em' AS step, * FROM deg WHERE index::text = 'em_ivf';
COPY em FROM PROGRAM 'head -n 3000 /work/fixc/data_1024.tsv';
SELECT 'd1 COPY 3,000 rows back in' AS step, * FROM deg WHERE index::text = 'em_ivf';
DROP VIEW deg; DROP TABLE dl, em, em_like;
