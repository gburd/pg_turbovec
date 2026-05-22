-- pg_turbovec v0.4.0 — Phase 4: experimental `turbovec` index AM.
--
-- This file is a *reference mirror*. It is only emitted into the
-- generated `sql/pg_turbovec--0.4.0.sql` when pg_turbovec is built
-- with `--features experimental_index_am`. Default builds skip
-- everything below.
--
-- DO NOT EDIT BY HAND — regenerate with:
--   cargo pgrx schema --features experimental_index_am > sql/pg_turbovec--0.4.0.sql

SET search_path = turbovec, public;

-- ===================================================================
-- Side table backing the AM (SPI-managed in v0.4)
-- ===================================================================

-- CREATE TABLE turbovec.am_storage (
--     indexrelid  oid PRIMARY KEY,
--     bit_width   int4 NOT NULL,
--     dim         int4 NOT NULL,
--     n_vectors   int8 NOT NULL,
--     payload     bytea NOT NULL,         -- IdMapIndex::write output
--     version     int4 NOT NULL,
--     updated_at  timestamptz NOT NULL DEFAULT now()
-- );
-- ALTER TABLE turbovec.am_storage ALTER COLUMN payload SET STORAGE EXTERNAL;

-- ===================================================================
-- Handler + access method registration
-- ===================================================================

-- CREATE FUNCTION turbovec_index_handler(internal) RETURNS index_am_handler
--     AS '$libdir/pg_turbovec', 'turbovec_index_handler_wrapper'
--     LANGUAGE c;
--
-- CREATE ACCESS METHOD turbovec
--     TYPE INDEX HANDLER turbovec_index_handler;

-- ===================================================================
-- Operator classes
-- ===================================================================

-- CREATE OPERATOR CLASS tvector_ip_ops
--     DEFAULT FOR TYPE tvector USING turbovec AS
--         OPERATOR 1 <#> (tvector, tvector) FOR ORDER BY float_ops,
--         FUNCTION 1 negative_inner_product(tvector, tvector);
--
-- CREATE OPERATOR CLASS tvector_cosine_ops
--     FOR TYPE tvector USING turbovec AS
--         OPERATOR 1 <=> (tvector, tvector) FOR ORDER BY float_ops,
--         FUNCTION 1 cosine_distance(tvector, tvector);

-- ===================================================================
-- Reloptions
-- ===================================================================

-- bit_width — int, default turbovec.bit_width_default GUC, range 2..=4
-- dim       — int, default 0 (auto-detect on first build)
--
-- Examples:
--   CREATE INDEX docs_emb_cosine_idx
--       ON docs USING turbovec (embedding tvector_cosine_ops)
--       WITH (bit_width = 4);
--   CREATE INDEX docs_emb_ip_idx
--       ON docs USING turbovec (embedding tvector_ip_ops)
--       WITH (bit_width = 2, dim = 1536);
