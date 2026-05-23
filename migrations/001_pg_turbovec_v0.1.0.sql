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

-- The `vector` type is a CBOR-serialised varlena (Phase 1). Phase 2
-- swaps in a binary layout compatible with pgvector's `vector`.
--
-- CREATE TYPE vector;            -- shell (pgrx)
-- CREATE FUNCTION vec_in(cstring) RETURNS vector;
-- CREATE FUNCTION vec_out(vector) RETURNS cstring;
-- CREATE FUNCTION vec_send(vector) RETURNS bytea;
-- CREATE FUNCTION vec_recv(internal) RETURNS vector;
-- CREATE TYPE vector (
--     INPUT          = vec_in,
--     OUTPUT         = vec_out,
--     RECEIVE        = vec_recv,
--     SEND           = vec_send,
--     INTERNALLENGTH = VARIABLE,
--     STORAGE        = EXTENDED,
--     ALIGNMENT      = double
-- );

-- ===================================================================
-- Distance functions (immutable, parallel safe)
-- ===================================================================

-- l2_distance(vector, vector)            -> double precision
-- l2_squared_distance(vector, vector)    -> double precision
-- inner_product(vector, vector)          -> double precision
-- negative_inner_product(vector, vector) -> double precision
-- cosine_distance(vector, vector)        -> double precision
-- l1_distance(vector, vector)            -> double precision
-- vector_dims(vector)                     -> integer
-- vector_norm(vector)                     -> double precision

-- ===================================================================
-- Element-wise arithmetic
-- ===================================================================

-- vec_add(vector, vector) -> vector
-- vec_sub(vector, vector) -> vector
-- vec_mul(vector, vector) -> vector

-- ===================================================================
-- Operators
-- ===================================================================

-- a <-> b   = l2_distance(a, b)
-- a <#> b   = negative_inner_product(a, b)        -- so ASC = most-similar-first
-- a <=> b   = cosine_distance(a, b)
-- a <+> b   = l1_distance(a, b)
-- a +   b   = vec_add(a, b)
-- a -   b   = vec_sub(a, b)
-- a *   b   = vec_mul(a, b)                   -- Hadamard product

-- ===================================================================
-- Aggregates
-- ===================================================================

-- avg(vector) -> vector            -- element-wise mean
-- sum(vector) -> vector            -- element-wise sum
--
-- Internal state VecAccum { sum: float8[dim], count: int8 } is a
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
