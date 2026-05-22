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

extension_sql!(
    r"
    CREATE CAST (real[]             AS tvector) WITH FUNCTION array_to_tvector(real[]);
    CREATE CAST (double precision[] AS tvector) WITH FUNCTION float8_array_to_tvector(double precision[]);
    CREATE CAST (integer[]          AS tvector) WITH FUNCTION int4_array_to_tvector(integer[]);
    CREATE CAST (tvector AS real[])             WITH FUNCTION tvector_to_array(tvector);
    ",
    name = "tvector_casts",
    requires = [
        Tvector,
        array_to_tvector,
        float8_array_to_tvector,
        int4_array_to_tvector,
        tvector_to_array
    ]
);
