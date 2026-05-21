-- pg_turbovec v0.1.0 — Phase 1: type, operators, functions, aggregates.
--
-- This file is a *reference mirror* of the SQL surface that pgrx
-- generates from src/. The authoritative SQL lives in
-- sql/pg_turbovec--0.1.0.sql produced by `cargo pgrx schema`. We
-- check this file in to make API changes show up in code review even
-- when the generated SQL is out of tree.
--
-- DO NOT EDIT BY HAND — regenerate with:
--   cargo pgrx schema --target-dir target/pgrx-schema > migrations/001_pg_turbovec_v0.1.0.sql

-- ===================================================================
-- Schema
-- ===================================================================

CREATE SCHEMA IF NOT EXISTS turbovec;
SET search_path = turbovec, public;

-- ===================================================================
-- Type
-- ===================================================================

-- The `tvector` type is a CBOR-serialised varlena (Phase 1). Phase 2
-- swaps in a binary layout compatible with pgvector's `vector`.
--
-- CREATE TYPE tvector;            -- shell (pgrx)
-- CREATE FUNCTION tvector_in(cstring) RETURNS tvector;
-- CREATE FUNCTION tvector_out(tvector) RETURNS cstring;
-- CREATE FUNCTION tvector_send(tvector) RETURNS bytea;
-- CREATE FUNCTION tvector_recv(internal) RETURNS tvector;
-- CREATE TYPE tvector (
--     INPUT          = tvector_in,
--     OUTPUT         = tvector_out,
--     RECEIVE        = tvector_recv,
--     SEND           = tvector_send,
--     INTERNALLENGTH = VARIABLE,
--     STORAGE        = EXTENDED,
--     ALIGNMENT      = double
-- );

-- ===================================================================
-- Distance functions (immutable, parallel safe)
-- ===================================================================

-- l2_distance(tvector, tvector)            -> double precision
-- l2_squared_distance(tvector, tvector)    -> double precision
-- inner_product(tvector, tvector)          -> double precision
-- negative_inner_product(tvector, tvector) -> double precision
-- cosine_distance(tvector, tvector)        -> double precision
-- l1_distance(tvector, tvector)            -> double precision
-- vector_dims(tvector)                     -> integer
-- vector_norm(tvector)                     -> double precision

-- ===================================================================
-- Element-wise arithmetic
-- ===================================================================

-- tvector_add(tvector, tvector) -> tvector
-- tvector_sub(tvector, tvector) -> tvector
-- tvector_mul(tvector, tvector) -> tvector

-- ===================================================================
-- Operators
-- ===================================================================

-- a <-> b   = l2_distance(a, b)
-- a <#> b   = negative_inner_product(a, b)        -- so ASC = most-similar-first
-- a <=> b   = cosine_distance(a, b)
-- a <+> b   = l1_distance(a, b)
-- a +   b   = tvector_add(a, b)
-- a -   b   = tvector_sub(a, b)
-- a *   b   = tvector_mul(a, b)                   -- Hadamard product

-- ===================================================================
-- Aggregates
-- ===================================================================

-- avg(tvector) -> tvector            -- element-wise mean
-- sum(tvector) -> tvector            -- element-wise sum
--
-- Internal state TvectorAccum { sum: float8[dim], count: int8 } is a
-- CBOR varlena. Both aggregates are PARALLEL SAFE; combinefn merges
-- partial states.

-- ===================================================================
-- Configuration (GUC)
-- ===================================================================

-- turbovec.bit_width_default     int   default 4   range 2..=4
-- turbovec.cache_size_mb         int   default 256 range 0..=65536
-- turbovec.warn_on_rebuild       bool  default true
-- turbovec.search_concurrency    int   default 1   range 1..=128
-- turbovec.normalize_on_insert   bool  default true

-- ===================================================================
-- Phase 2 (planned, NOT in this migration)
-- ===================================================================

-- CREATE OPERATOR CLASS tvector_ip_ops
--   DEFAULT FOR TYPE tvector USING turbovec AS
--     OPERATOR 1 <#> (tvector, tvector) FOR ORDER BY float_ops,
--     FUNCTION 1 negative_inner_product(tvector, tvector);
--
-- CREATE OPERATOR CLASS tvector_cosine_ops
--   FOR TYPE tvector USING turbovec AS
--     OPERATOR 1 <=> (tvector, tvector) FOR ORDER BY float_ops,
--     FUNCTION 1 cosine_distance(tvector, tvector);
--
-- CREATE ACCESS METHOD turbovec
--   TYPE INDEX
--   HANDLER turbovec_index_handler;
