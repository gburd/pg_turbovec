//! Casts between `tvector` and SQL array types.
//!
//! All casts are explicit (no implicit promotion). pgvector's `vector`
//! type is intentionally *not* listed here — Phase 1 stores tvectors
//! in a CBOR varlena that is not byte-compatible with pgvector. A
//! `tvector_to_pgvector_vector` helper will appear in Phase 2 once we
//! switch to the binary-compatible layout.

use pgrx::prelude::*;

use crate::tvector::Tvector;

/// `real[]` -> `tvector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn array_to_tvector(arr: Vec<Option<f32>>) -> Tvector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| error!("tvector cannot contain NULL element at index {}", i))
        })
        .collect();
    Tvector::from_vec(data)
}

/// `double precision[]` -> `tvector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn float8_array_to_tvector(arr: Vec<Option<f64>>) -> Tvector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let v =
                v.unwrap_or_else(|| error!("tvector cannot contain NULL element at index {}", i));
            if !v.is_finite() {
                error!("tvector value at index {} is not finite ({})", i, v);
            }
            v as f32
        })
        .collect();
    Tvector::from_vec(data)
}

/// `integer[]` -> `tvector` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn int4_array_to_tvector(arr: Vec<Option<i32>>) -> Tvector {
    let data: Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let v =
                v.unwrap_or_else(|| error!("tvector cannot contain NULL element at index {}", i));
            v as f32
        })
        .collect();
    Tvector::from_vec(data)
}

/// `tvector` -> `real[]` (explicit cast).
#[pg_extern(immutable, parallel_safe)]
fn tvector_to_array(v: Tvector) -> Vec<f32> {
    v.data
}

/// `to_tvector(text) -> tvector` — parse a `'[1, 2, 3]'`-formatted
/// text literal into a `tvector`. Equivalent to `text::tvector`,
/// provided as a named function to mirror pgvector's `to_vector`.
#[pg_extern(immutable, parallel_safe)]
fn to_tvector_text(s: &str) -> Tvector {
    match crate::tvector::parse_tvector(s) {
        Ok(v) => Tvector::from_vec(v),
        Err(msg) => error!("to_tvector: invalid input '{}': {}", s, msg),
    }
}

/// `to_tvector(text, integer, boolean) -> tvector` — parse and
/// optionally enforce a dimension. Matches pgvector's signature
/// for drop-in compatibility.
///
/// * `dim` — if non-zero, raise an ERROR when the parsed vector
///   doesn't have exactly this many dimensions.
/// * `_transpose` — accepted for pgvector compatibility but a
///   no-op for `tvector` (we have no row/column distinction; a
///   tvector is always a 1-D sequence).
#[pg_extern(immutable, parallel_safe)]
fn to_tvector_text_dim(s: &str, dim: i32, _transpose: bool) -> Tvector {
    let v = to_tvector_text(s);
    if dim != 0 && v.dim() != dim as usize {
        error!(
            "to_tvector: expected dim {}, got {}",
            dim,
            v.dim()
        );
    }
    v
}

/// `array_to_tvector(real[], integer, boolean) -> tvector` —
/// pgvector-compatible array-to-vector conversion with explicit
/// dim check. The two-argument form (without dim/transpose) is
/// the cast `(real[] AS tvector)`.
#[pg_extern(immutable, parallel_safe)]
fn array_to_tvector_dim(arr: Vec<Option<f32>>, dim: i32, _transpose: bool) -> Tvector {
    let v = array_to_tvector(arr);
    if dim != 0 && v.dim() != dim as usize {
        error!(
            "array_to_tvector: expected dim {}, got {}",
            dim,
            v.dim()
        );
    }
    v
}

extension_sql!(
    r"
    CREATE CAST (real[]             AS tvector) WITH FUNCTION array_to_tvector(real[]);
    CREATE CAST (double precision[] AS tvector) WITH FUNCTION float8_array_to_tvector(double precision[]);
    CREATE CAST (integer[]          AS tvector) WITH FUNCTION int4_array_to_tvector(integer[]);
    CREATE CAST (tvector AS real[])             WITH FUNCTION tvector_to_array(tvector);

    -- pgvector-style aliases. The single-argument forms are
    -- equivalent to '...'::tvector / array::tvector; the
    -- three-argument forms add an explicit dim check.
    CREATE FUNCTION to_tvector(text)                       RETURNS tvector
        AS 'MODULE_PATHNAME', 'to_tvector_text_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    CREATE FUNCTION to_tvector(text, integer, boolean)     RETURNS tvector
        AS 'MODULE_PATHNAME', 'to_tvector_text_dim_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    CREATE FUNCTION array_to_tvector(real[], integer, boolean) RETURNS tvector
        AS 'MODULE_PATHNAME', 'array_to_tvector_dim_wrapper'
        LANGUAGE c IMMUTABLE PARALLEL SAFE STRICT;
    ",
    name = "tvector_casts",
    requires = [
        Tvector,
        array_to_tvector,
        float8_array_to_tvector,
        int4_array_to_tvector,
        tvector_to_array,
        to_tvector_text,
        to_tvector_text_dim,
        array_to_tvector_dim
    ]
);
