-- pg_turbovec — end-to-end demo script.
--
-- Walks through every public feature with realistic-ish data.
-- Run with:
--   cargo pgrx run pg16
--   \i tests/03_full_demo.sql

\echo === pg_turbovec end-to-end demo ===
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = turbovec, public;

\echo --- 1. vector type basics ---
SELECT '[1, 2, 3]'::vector AS literal;
SELECT vector_dims('[1,2,3,4]'::vector) AS dims;
SELECT vector_norm('[3,4]'::vector) AS norm;
SELECT vec_normalize('[3,4]'::vector) AS unit;

\echo --- 2. Distance ops ---
SELECT '[1,0,0]'::vector <-> '[0,1,0]'::vector AS l2_sqrt2;
SELECT '[1,0,0]'::vector <#> '[1,0,0]'::vector AS neg_ip_neg1;
SELECT '[1,0]'::vector  <=> '[0,1]'::vector  AS cosine_1;
SELECT '[0,0]'::vector  <+> '[3,4]'::vector  AS l1_7;

\echo --- 3. Element-wise arithmetic ---
SELECT '[1,2,3]'::vector + '[4,5,6]'::vector AS sum;
SELECT '[10,20,30]'::vector - '[1,2,3]'::vector AS diff;
SELECT '[2,3,4]'::vector * '[5,6,7]'::vector AS hadamard;

\echo --- 4. Casts ---
SELECT (ARRAY[1,2,3]::real[])::vector AS from_real;
SELECT '[1,2,3]'::vector::real[] AS to_real;
SELECT '[1, 2.5, -3]'::vector::jsonb AS to_jsonb;
SELECT '[10.5, 20.5, 30.5]'::jsonb::vector AS from_jsonb;

\echo --- 5. Phase 5 helpers ---
SELECT subvector('[10,20,30,40,50]'::vector, 2, 3) AS slice_2_3;
SELECT vec_zeros(5) AS zeros;
SELECT vec_check_dim('[1,2,3]'::vector, 3) AS dim_ok;

\echo --- 6. Aggregates ---
CREATE TEMP TABLE demo (id bigint PRIMARY KEY, emb vector);
INSERT INTO demo VALUES
  (1, '[1,2,3]'),
  (2, '[3,4,5]'),
  (3, '[5,6,7]');
SELECT avg(emb) AS centroid, sum(emb) AS total FROM demo;

\echo --- 7. turbovec.knn() function-driven ANN ---
DROP TABLE IF EXISTS knn_demo CASCADE;
CREATE TABLE knn_demo (id bigint PRIMARY KEY, emb vector, tag text);
INSERT INTO knn_demo VALUES
  (1, '[1,0,0,0,0,0,0,0]', 'A'),
  (2, '[0.9,0.1,0,0,0,0,0,0]', 'B'),
  (3, '[0,1,0,0,0,0,0,0]', 'A'),
  (4, '[-1,0,0,0,0,0,0,0]', 'B');

\echo Top-3 unfiltered:
SELECT k.id, k.score
FROM   turbovec.knn(
         'knn_demo'::regclass, 'id', 'emb',
         '[1,0,0,0,0,0,0,0]'::vector,
         3
       ) k
ORDER  BY k.score DESC;

\echo Top-3 with allowlist tag = 'A':
SELECT k.id, k.score
FROM   turbovec.knn(
         'knn_demo'::regclass, 'id', 'emb',
         '[1,0,0,0,0,0,0,0]'::vector,
         3, 4,
         ARRAY(SELECT id FROM knn_demo WHERE tag = 'A')::bigint[]
       ) k
ORDER  BY k.score DESC;

\echo --- 8. Index AM lifecycle ---
CREATE INDEX knn_demo_idx
  ON knn_demo USING turbovec (emb vec_cosine_ops)
  WITH (bit_width = 4);
SELECT count(*) AS n_rows FROM knn_demo;

\echo aminsert path:
INSERT INTO knn_demo VALUES (5, '[0.5,0.5,0,0,0,0,0,0]', 'A');
SELECT count(*) AS n_rows FROM knn_demo;

\echo ambulkdelete path:
DELETE FROM knn_demo WHERE id = 4;
VACUUM knn_demo;
SELECT count(*) AS n_rows FROM knn_demo;

\echo REINDEX path:
REINDEX INDEX knn_demo_idx;
SELECT count(*) AS n_rows FROM knn_demo;

\echo --- 9. GUCs ---
SHOW turbovec.bit_width_default;
SHOW turbovec.cache_size_mb;
SHOW turbovec.normalize_on_insert;

\echo --- 10. Diagnostics ---
SELECT turbovec.turbovec_version();
SELECT turbovec.turbovec_self_score(
         turbovec.vec_normalize('[1,0,0,0,0,0,0,0]'::vector),
         4)
       AS self_score;

\echo === Demo complete ===
