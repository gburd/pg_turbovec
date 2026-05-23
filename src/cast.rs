//! Casts between `vector` and SQL array types.
//!
//! All casts are explicit (no implicit promotion). pgvector's `vector`
//! type is intentionally *not* listed here — Phase 1 stores vecs
//! in a CBOR varlena that is not byte-compatible with pgvector. A
//! `vec_to_pgvector` helper will appear in Phase 2 once we
//! switch to the binary-compatible layout.

use pgrx::prelude::*;

use crate::vec::Vector;

/// `real[]` -> `vector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn array_to_vec(arr: Vec<Option<f32>>) -> Vector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| error!("vector cannot contain NULL element at index {}", i))
        })
        .collect();
    Vector::from_vec(data)
}

/// `double precision[]` -> `vector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn float8_array_to_vec(arr: Vec<Option<f64>>) -> Vector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let v =
                v.unwrap_or_else(|| error!("vector cannot contain NULL element at index {}", i));
            if !v.is_finite() {
                error!("vector value at index {} is not finite ({})", i, v);
            }
            v as f32
        })
        .collect();
    Vector::from_vec(data)
}

/// `integer[]` -> `vector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn int4_array_to_vec(arr: Vec<Option<i32>>) -> Vector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let v =
                v.unwrap_or_else(|| error!("vector cannot contain NULL element at index {}", i));
            v as f32
        })
        .collect();
    Vector::from_vec(data)
}

/// `vector` -> `real[]` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn vec_to_array(v: Vector) -> Vec<f32> {
    v.data
}

/// `to_vec(text) -> vector` — parse a `'[1, 2, 3]'`-formatted
/// text literal into a `vector`. Equivalent to `text::vector`,
/// provided as a named function to mirror pgvector's `to_vector`.
#[pg_extern(immutable, parallel_safe)]
fn to_vec_text(s: &str) -> Vector {
    match crate::vec::parse_vec(s) {
        Ok(v) => Vector::from_vec(v),
        Err(msg) => error!("to_vec: invalid input '{}': {}", s, msg),
    }
}

/// pgvector-compat alias: `to_vector(text) -> vector`.
#[pg_extern(name = "to_vector", immutable, parallel_safe)]
fn to_vector_text(s: &str) -> Vector {
    to_vec_text(s)
}

/// `to_vec(text, integer, boolean) -> vector` — parse and
/// optionally enforce a dimension. Matches pgvector's signature
/// for drop-in compatibility.
///
/// * `dim` — if non-zero, raise an ERROR when the parsed vector
///   doesn't have exactly this many dimensions.
/// * `_transpose` — accepted for pgvector compatibility but a
///   no-op for `vector` (we have no row/column distinction; a
///   vector is always a 1-D sequence).
#[pg_extern(immutable, parallel_safe)]
fn to_vec_text_dim(s: &str, dim: i32, _transpose: bool) -> Vector {
    let v = to_vec_text(s);
    if dim != 0 && v.dim() != dim as usize {
        error!(
            "to_vec: expected dim {}, got {}",
            dim,
            v.dim()
        );
    }
    v
}

/// pgvector-compat alias: `to_vector(text, integer, boolean)`.
#[pg_extern(name = "to_vector", immutable, parallel_safe)]
fn to_vector_text_dim(s: &str, dim: i32, transpose: bool) -> Vector {
    to_vec_text_dim(s, dim, transpose)
}

/// `vector_to_float4(vector, integer, boolean) -> real[]` —
/// pgvector-compat. The `dim` arg gates a runtime dim check (0
/// disables); `transpose` is a no-op for our 1-D vectors.
#[pg_extern(immutable, parallel_safe)]
fn vector_to_float4(v: Vector, dim: i32, _transpose: bool) -> Vec<f32> {
    if dim != 0 && v.dim() != dim as usize {
        error!(
            "vector_to_float4: expected dim {}, got {}",
            dim,
            v.dim()
        );
    }
    v.data
}

/// `array_to_vec(real[], integer, boolean) -> vector` —
/// pgvector-compatible array-to-vector conversion with explicit
/// dim check. The two-argument form (without dim/transpose) is
/// the cast `(real[] AS vector)`.
#[pg_extern(immutable, parallel_safe)]
fn array_to_vec_dim(arr: Vec<Option<f32>>, dim: i32, _transpose: bool) -> Vector {
    let v = array_to_vec(arr);
    if dim != 0 && v.dim() != dim as usize {
        error!(
            "array_to_vec: expected dim {}, got {}",
            dim,
            v.dim()
        );
    }
    v
}

/// pgvector-compat alias: `array_to_vector(real[], integer, boolean)`.
#[pg_extern(name = "array_to_vector", immutable, parallel_safe)]
fn array_to_vector_dim(arr: Vec<Option<f32>>, dim: i32, transpose: bool) -> Vector {
    array_to_vec_dim(arr, dim, transpose)
}

extension_sql!(
    r"
    CREATE CAST (real[]             AS vector) WITH FUNCTION array_to_vec(real[]);
    CREATE CAST (double precision[] AS vector) WITH FUNCTION float8_array_to_vec(double precision[]);
    CREATE CAST (integer[]          AS vector) WITH FUNCTION int4_array_to_vec(integer[]);
    CREATE CAST (vector AS real[])             WITH FUNCTION vec_to_array(vector);

    -- pgvector-style aliases. The single-argument forms are
    -- equivalent to '...'::vector / array::vector; the
    -- three-argument forms add an explicit dim check.
    CREATE FUNCTION to_vec(text)                       RETURNS vector
        AS 'MODULE_PATHNAME', 'to_vec_text_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    CREATE FUNCTION to_vec(text, integer, boolean)     RETURNS vector
        AS 'MODULE_PATHNAME', 'to_vec_text_dim_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    CREATE FUNCTION array_to_vec(real[], integer, boolean) RETURNS vector
        AS 'MODULE_PATHNAME', 'array_to_vec_dim_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    ",
    name = "vec_casts",
    requires = [
        Vector,
        array_to_vec,
        float8_array_to_vec,
        int4_array_to_vec,
        vec_to_array,
        to_vec_text,
        to_vec_text_dim,
        array_to_vec_dim
    ]
);
