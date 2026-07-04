//! Phase 5 extras: pgvector-parity helpers that don't fit cleanly
//! into the type / distance / aggregate modules.
//!
//! - `subvector(vector, start, length) -> vector` — 1-indexed
//!   slice, mirrors pgvector's `subvector`.
//! - `vec_to_jsonb(vector)` — explicit JSON output (handy for
//!   logging, replication via JSONB columns).
//! - `jsonb_to_vec(jsonb)` — inverse.
//! - `vec_check_dim(vector, integer)` — runtime dim assertion;
//!   raises ERROR if mismatch. Cheaper than typmod plumbing.
//! - `vec_zeros(integer)` — zero-filled vector helper.
//! - `vec_to_text(vector)` — explicit text representation
//!   (the IO function's output, callable as a regular function).

use pgrx::prelude::*;
use serde_json::{json, Value};

use crate::vec::{Vector, MAX_DIM};

/// `subvector(v, start, length)` — 1-indexed slice (matches pgvector).
/// `start` and `length` must be positive and the resulting range must
/// lie within `v`.
///
/// ```ignore
/// SELECT turbovec.subvector('[10, 20, 30, 40]'::turbovec.vector, 2, 2)::text;
/// -- returns '[20, 30]'
///
/// -- Out-of-bounds raises ERROR:
/// SELECT turbovec.subvector('[1, 2, 3]'::turbovec.vector, 2, 5);
/// -- ERROR: subvector: range 2..6 is out of bounds for vector of dim 3
/// ```
#[pg_extern(immutable, parallel_safe)]
fn subvector(v: Vector, start: i32, length: i32) -> Vector {
    if start < 1 {
        error!(
            "subvector: start ({}) must be a positive 1-indexed offset",
            start
        );
    }
    if length < 1 {
        error!("subvector: length ({}) must be positive", length);
    }
    let s = (start - 1) as usize;
    let l = length as usize;
    if s + l > v.dim() {
        error!(
            "subvector: range {}..{} is out of bounds for vector of dim {}",
            start,
            start + length - 1,
            v.dim()
        );
    }
    Vector::from_vec(v.as_slice()[s..s + l].to_vec())
}

/// Materialise a `vector` as a `jsonb` array of numbers.
///
/// ```ignore
/// SELECT turbovec.vec_to_jsonb('[1, 2.5, -3]'::turbovec.vector);
/// -- returns [1, 2.5, -3]::jsonb
///
/// -- Equivalent cast form:
/// SELECT '[1, 2.5, -3]'::turbovec.vector::jsonb;
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_to_jsonb(v: Vector) -> pgrx::JsonB {
    let arr: Vec<Value> = v
        .as_slice()
        .iter()
        .map(|x| Value::from(f64::from(*x)))
        .collect();
    pgrx::JsonB(json!(arr))
}

/// Parse a `jsonb` array of numbers as a `vector`. Rejects non-array
/// inputs and non-numeric / non-finite elements.
///
/// ```ignore
/// SELECT turbovec.jsonb_to_vec('[1, 2.5, 3]'::jsonb)::text;
/// -- returns '[1, 2.5, 3]'
///
/// -- Errors:
/// SELECT turbovec.jsonb_to_vec('{"a": 1}'::jsonb);     -- ERROR (not array)
/// SELECT turbovec.jsonb_to_vec('[1, "x", 3]'::jsonb);  -- ERROR (string elem)
/// SELECT turbovec.jsonb_to_vec('[1, null, 3]'::jsonb); -- ERROR (null elem)
/// ```
#[pg_extern(immutable, parallel_safe)]
fn jsonb_to_vec(j: pgrx::JsonB) -> Vector {
    let arr = match j.0 {
        Value::Array(a) => a,
        other => error!(
            "jsonb_to_vec: expected JSON array, got {}",
            value_kind(&other)
        ),
    };
    if arr.is_empty() || arr.len() > MAX_DIM {
        error!(
            "jsonb_to_vec: dim {} out of range 1..={}",
            arr.len(),
            MAX_DIM
        );
    }
    let mut out: Vec<f32> = Vec::with_capacity(arr.len());
    for (i, v) in arr.into_iter().enumerate() {
        let n = v.as_f64().unwrap_or_else(|| {
            error!(
                "jsonb_to_vec: element {} is not a number ({})",
                i,
                value_kind(&v)
            )
        });
        if !n.is_finite() {
            error!("jsonb_to_vec: element {} is not finite ({})", i, n);
        }
        out.push(n as f32);
    }
    Vector::from_vec(out)
}

/// Returns the kind name of a `serde_json::Value` for error messages.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Raise an ERROR if `v.dim() != expected`, otherwise return `v`
/// unchanged. Useful as `CHECK (turbovec.vec_check_dim(emb, 1536))`.
///
/// ```ignore
/// CREATE TABLE docs (
///     id  bigserial PRIMARY KEY,
///     emb turbovec.vector
///         CHECK (turbovec.vector_dims(
///             turbovec.vec_check_dim(emb, 1536)) = 1536)
/// );
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_check_dim(v: Vector, expected: i32) -> Vector {
    if expected < 1 {
        error!("vec_check_dim: expected dim must be positive");
    }
    if v.dim() != expected as usize {
        error!(
            "vec_check_dim: dim mismatch (got {}, expected {})",
            v.dim(),
            expected
        );
    }
    v
}

/// Build a zero-filled `vector` of the requested dimension. Useful
/// as the identity for `sum(vector)` in extension queries.
///
/// ```ignore
/// SELECT turbovec.vector_dims(turbovec.vec_zeros(8));
/// -- returns 8
///
/// SELECT turbovec.vector_norm(turbovec.vec_zeros(8));
/// -- returns 0.0
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_zeros(dim: i32) -> Vector {
    if dim <= 0 || dim as usize > MAX_DIM {
        error!("vec_zeros: dim {} out of range 1..={}", dim, MAX_DIM);
    }
    Vector::from_vec(vec![0.0_f32; dim as usize])
}

/// Explicit text rendering of a `vector` (mirrors the type's OUTPUT
/// function but callable directly).
///
/// ```ignore
/// SELECT turbovec.vec_to_text('[1, 2.5, -3]'::turbovec.vector);
/// -- returns '[1, 2.5, -3]'
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_to_text(v: Vector) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(2 + v.dim() * 6);
    out.push('[');
    let mut first = true;
    for x in v.as_slice() {
        if !first {
            out.push_str(", ");
        }
        first = false;
        let _ = write!(out, "{x}");
    }
    out.push(']');
    out
}

/// `turbovec.index_is_degraded(regclass) -> bool` — Phase E-2
/// operator signal. Returns `true` when the given turbovec index was
/// built `WITH (lists > 0)` (an IVF index) but has degraded to a flat
/// O(n) scan (its IVF cell metadata was invalidated). A churning
/// production deployment can poll this to detect the silent latency
/// cliff and `REINDEX` before it bites.
///
/// Returns `false` for a healthy IVF index, a flat (`lists = 0`)
/// index, and a non-turbovec or empty index (nothing to degrade).
/// With the tombstone-vacuum path an IVF index survives VACUUM and
/// stays healthy, so this should normally read `false`; a `true`
/// means a fallback path fired and a REINDEX is warranted.
///
/// ```ignore
/// SELECT turbovec.index_is_degraded('my_ivf_idx'::regclass);
/// ```
#[pg_extern(stable, parallel_safe)]
fn index_is_degraded(index: pg_sys::Oid) -> bool {
    unsafe {
        let rel = pg_sys::index_open(index, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            return false;
        }
        let degraded = crate::index::relfile::read_meta(rel)
            .map(|m| m.is_degraded())
            .unwrap_or(false);
        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        degraded
    }
}

extension_sql!(
    r"
    -- jsonb <-> vector explicit casts.
    CREATE CAST (vector AS jsonb) WITH FUNCTION vec_to_jsonb(vector);
    CREATE CAST (jsonb   AS vector) WITH FUNCTION jsonb_to_vec(jsonb);
    ",
    name = "vec_jsonb_casts",
    requires = [vec_to_jsonb, jsonb_to_vec]
);
