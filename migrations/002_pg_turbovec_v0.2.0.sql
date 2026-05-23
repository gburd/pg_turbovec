-- pg_turbovec v0.2.0 — Phase 2: function-driven ANN search.
--
-- This file is a *reference mirror* of the SQL surface that pgrx
-- generates. Authoritative SQL: sql/pg_turbovec--0.2.0.sql produced
-- by `cargo pgrx schema`.

SET search_path = turbovec, public;

-- ===================================================================
-- ANN search function
-- ===================================================================

-- turbovec.knn(
--     rel       regclass,           -- table to search
--     id_col    text,               -- bigint primary-key column name
--     vec_col   text,               -- vector column name
--     query     vector,            -- query point
--     k         integer,            -- number of neighbours
--     bit_width integer DEFAULT 4   -- 2 | 3 | 4 (TurboQuant constraint)
-- ) RETURNS TABLE (
--     id    bigint,
--     score double precision        -- inner product on unit vectors;
--                                   --  higher = more similar
-- )
--
-- STABLE PARALLEL SAFE.
--
-- Constraints:
--   * dim must be a multiple of 8 (turbovec kernel constraint)
--   * bit_width ∈ {2, 3, 4}
--   * k > 0
--
-- Behaviour: rebuilds an in-memory `turbovec::IdMapIndex` on every
-- call.  Vectors are unit-normalised when
-- `turbovec.normalize_on_insert = true` (the default).  Returned
-- scores are inner products in `[-1, 1]` — `ORDER BY score DESC` to
-- get most-similar-first.
--
-- Phase 3 will add a backend-local cache invalidated by the relcache
-- callback registered in `_PG_init`, removing the rebuild cost.

-- ===================================================================
-- Phase 3 (planned, NOT in this migration)
-- ===================================================================

-- CREATE OPERATOR CLASS vec_ip_ops
--   DEFAULT FOR TYPE vector USING turbovec AS
--     OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
--     FUNCTION 1 negative_inner_product(vector, vector);
--
-- CREATE OPERATOR CLASS vec_cosine_ops
--   FOR TYPE vector USING turbovec AS
--     OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
--     FUNCTION 1 cosine_distance(vector, vector);
--
-- CREATE ACCESS METHOD turbovec
--   TYPE INDEX
--   HANDLER turbovec_index_handler;
