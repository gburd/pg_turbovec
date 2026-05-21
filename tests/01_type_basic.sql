-- Test suite for pg_turbovec — Phase 1 (type, operators, functions, aggregates).
-- Run with: cargo pgrx test pg17

-- Setup
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = public, turbovec;

BEGIN;

-- 1. tvector text I/O round-trips.
SELECT '[1, 2, 3]'::tvector;
SELECT '[ 1.5 , -2.0 , 3 ]'::tvector;
SELECT vector_dims('[1,2,3,4]'::tvector);     -- expect 4

-- 2. Distance operators on equal-dim vectors.
SELECT '[1,2,3]'::tvector <-> '[4,6,3]'::tvector;          -- L2:   5
SELECT '[1,2,3]'::tvector <+> '[4,6,3]'::tvector;          -- L1:   7
SELECT '[1,0,0]'::tvector <#> '[1,0,0]'::tvector;          -- -IP: -1
SELECT '[1,0]'::tvector   <=> '[0,1]'::tvector;            -- cos:  1.0

-- 3. Named functions.
SELECT l2_distance('[0,0]'::tvector, '[3,4]'::tvector);    -- 5
SELECT inner_product('[1,2]'::tvector, '[3,4]'::tvector);  -- 11
SELECT cosine_distance('[1,0]'::tvector, '[1,0]'::tvector);-- 0
SELECT l1_distance('[0,0]'::tvector, '[3,4]'::tvector);    -- 7
SELECT vector_norm('[3,4]'::tvector);                      -- 5

-- 4. Element-wise arithmetic.
SELECT ('[1,2,3]'::tvector + '[4,5,6]'::tvector);          -- [5,7,9]
SELECT ('[10,20,30]'::tvector - '[1,2,3]'::tvector);       -- [9,18,27]
SELECT ('[2,3,4]'::tvector * '[5,6,7]'::tvector);          -- [10,18,28]

-- 5. Aggregates.
CREATE TEMP TABLE pgtv_test (v tvector);
INSERT INTO pgtv_test VALUES
  ('[1,2,3]'::tvector),
  ('[3,4,5]'::tvector),
  ('[5,6,7]'::tvector);
SELECT avg(v) FROM pgtv_test;                  -- [3,4,5]
SELECT sum(v) FROM pgtv_test;                  -- [9,12,15]

-- 6. Ordered nearest-neighbour exact scan via cosine.
CREATE TEMP TABLE pgtv_items (id int PRIMARY KEY, emb tvector);
INSERT INTO pgtv_items VALUES
  (1, '[1, 0, 0]'),
  (2, '[0.9, 0.1, 0]'),
  (3, '[0, 1, 0]'),
  (4, '[-1, 0, 0]');
SELECT id
FROM   pgtv_items
ORDER  BY emb <=> '[1, 0, 0]'::tvector
LIMIT  3;
-- expected order: 1, 2, 3 (cosine), 4 last

-- 7. GUC visible.
SELECT current_setting('turbovec.bit_width_default');

ROLLBACK;

-- ===================================================================
-- Phase 2: turbovec.knn() function-driven ANN
-- ===================================================================

BEGIN;
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = public, turbovec;

CREATE TEMP TABLE knn_items (
    id  bigint PRIMARY KEY,
    emb tvector
);
INSERT INTO knn_items VALUES
  (1, '[1, 0, 0, 0, 0, 0, 0, 0]'::tvector),
  (2, '[0.9, 0.1, 0, 0, 0, 0, 0, 0]'::tvector),
  (3, '[0, 1, 0, 0, 0, 0, 0, 0]'::tvector),
  (4, '[-1, 0, 0, 0, 0, 0, 0, 0]'::tvector);

-- Top-3 most similar to e1.
SELECT id, score
FROM   turbovec.knn(
         'knn_items'::regclass,
         'id', 'emb',
         '[1, 0, 0, 0, 0, 0, 0, 0]'::tvector,
         3)
ORDER  BY score DESC;
-- expected: id=1 first (self), then 2 and 3 in either order (depending
-- on TurboQuant residual)

ROLLBACK;
