-- Test suite for pg_turbovec — Phase 1 (type, operators, functions, aggregates).
-- Run with: cargo pgrx test pg17

-- Setup
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = public, turbovec;

BEGIN;

-- 1. vector text I/O round-trips.
SELECT '[1, 2, 3]'::vector;
SELECT '[ 1.5 , -2.0 , 3 ]'::vector;
SELECT vector_dims('[1,2,3,4]'::vector);     -- expect 4

-- 2. Distance operators on equal-dim vectors.
SELECT '[1,2,3]'::vector <-> '[4,6,3]'::vector;          -- L2:   5
SELECT '[1,2,3]'::vector <+> '[4,6,3]'::vector;          -- L1:   7
SELECT '[1,0,0]'::vector <#> '[1,0,0]'::vector;          -- -IP: -1
SELECT '[1,0]'::vector   <=> '[0,1]'::vector;            -- cos:  1.0

-- 3. Named functions.
SELECT l2_distance('[0,0]'::vector, '[3,4]'::vector);    -- 5
SELECT inner_product('[1,2]'::vector, '[3,4]'::vector);  -- 11
SELECT cosine_distance('[1,0]'::vector, '[1,0]'::vector);-- 0
SELECT l1_distance('[0,0]'::vector, '[3,4]'::vector);    -- 7
SELECT vector_norm('[3,4]'::vector);                      -- 5

-- 4. Element-wise arithmetic.
SELECT ('[1,2,3]'::vector + '[4,5,6]'::vector);          -- [5,7,9]
SELECT ('[10,20,30]'::vector - '[1,2,3]'::vector);       -- [9,18,27]
SELECT ('[2,3,4]'::vector * '[5,6,7]'::vector);          -- [10,18,28]

-- 5. Aggregates.
CREATE TEMP TABLE pgtv_test (v vector);
INSERT INTO pgtv_test VALUES
  ('[1,2,3]'::vector),
  ('[3,4,5]'::vector),
  ('[5,6,7]'::vector);
SELECT avg(v) FROM pgtv_test;                  -- [3,4,5]
SELECT sum(v) FROM pgtv_test;                  -- [9,12,15]

-- 6. Ordered nearest-neighbour exact scan via cosine.
CREATE TEMP TABLE pgtv_items (id int PRIMARY KEY, emb vector);
INSERT INTO pgtv_items VALUES
  (1, '[1, 0, 0]'),
  (2, '[0.9, 0.1, 0]'),
  (3, '[0, 1, 0]'),
  (4, '[-1, 0, 0]');
SELECT id
FROM   pgtv_items
ORDER  BY emb <=> '[1, 0, 0]'::vector
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
    emb vector
);
INSERT INTO knn_items VALUES
  (1, '[1, 0, 0, 0, 0, 0, 0, 0]'::vector),
  (2, '[0.9, 0.1, 0, 0, 0, 0, 0, 0]'::vector),
  (3, '[0, 1, 0, 0, 0, 0, 0, 0]'::vector),
  (4, '[-1, 0, 0, 0, 0, 0, 0, 0]'::vector);

-- Top-3 most similar to e1.
SELECT id, score
FROM   turbovec.knn(
         'knn_items'::regclass,
         'id', 'emb',
         '[1, 0, 0, 0, 0, 0, 0, 0]'::vector,
         3)
ORDER  BY score DESC;
-- expected: id=1 first (self), then 2 and 3 in either order (depending
-- on TurboQuant residual)

ROLLBACK;

-- ===================================================================
-- Phase 5: subvector / jsonb casts / dim assertion / zeros
-- ===================================================================

BEGIN;
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = public, turbovec;

-- subvector
SELECT turbovec.subvector('[10, 20, 30, 40, 50]'::vector, 2, 3);  -- [20, 30, 40]

-- jsonb round trip
SELECT '[1, 2.5, -3]'::vector::jsonb;                             -- [1, 2.5, -3]
SELECT '[1, 2.5, -3]'::jsonb::vector;                             -- [1, 2.5, -3]

-- dim assertion
SELECT turbovec.vec_check_dim('[1, 2, 3]'::vector, 3);        -- pass
DO $$
BEGIN
    PERFORM turbovec.vec_check_dim('[1, 2, 3]'::vector, 4);
    RAISE EXCEPTION 'should have errored';
EXCEPTION WHEN OTHERS THEN
    -- expected
    NULL;
END$$;

-- zeros + norm
SELECT turbovec.vector_norm(turbovec.vec_zeros(8));            -- 0

ROLLBACK;
