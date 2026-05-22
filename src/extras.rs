//! Phase 5 extras: pgvector-parity helpers that don't fit cleanly
//! into the type / distance / aggregate modules.
//!
//! - `subvector(tvector, start, length) -> tvector` \u2014 1-indexed
//!   slice, mirrors pgvector's `subvector`.
//! - `tvector_to_jsonb(tvector)` \u2014 explicit JSON output (handy for
//!   logging, replication via JSONB columns).
//! - `jsonb_to_tvector(jsonb)` \u2014 inverse.
//! - `tvector_check_dim(tvector, integer)` \u2014 runtime dim assertion;
//!   raises ERROR if mismatch. Cheaper than typmod plumbing.
//! - `tvector_zeros(integer)` \u2014 zero-filled vector helper.
//! - `tvector_to_text(tvector)` \u2014 explicit text representation
//!   (the IO function's output, callable as a regular function).

use pgrx::prelude::*;
use serde_json::{json, Value};

use crate::tvector::{Tvector, MAX_DIM};

/// `subvector(v, start, length)` \u2014 1-indexed slice (matches pgvector).
/// `start` and `length` must be positive and the resulting range must
/// lie within `v`.
#[pg_extern(immutable, parallel_safe)]
fn subvector(v: Tvector, start: i32, length: i32) -> Tvector {
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
            "subvector: range {}..{} is out of bounds for tvector of dim {}",
            start,
            start + length - 1,
            v.dim()
        );
    }
    Tvector::from_vec(v.as_slice()[s..s + l].to_vec())
}

/// Materialise a `tvector` as a `jsonb` array of numbers.
#[pg_extern(immutable, parallel_safe)]
fn tvector_to_jsonb(v: Tvector) -> pgrx::JsonB {
    let arr: Vec<Value> = v
        .as_slice()
        .iter()
        .map(|x| Value::from(f64::from(*x)))
        .collect();
    pgrx::JsonB(json!(arr))
}

/// Parse a `jsonb` array of numbers as a `tvector`. Rejects non-array
/// inputs and non-numeric / non-finite elements.
#[pg_extern(immutable, parallel_safe)]
fn jsonb_to_tvector(j: pgrx::JsonB) -> Tvector {
    let arr = match j.0 {
        Value::Array(a) => a,
        other => error!(
            "jsonb_to_tvector: expected JSON array, got {}",
            value_kind(&other)
        ),
    };
    if arr.is_empty() || arr.len() > MAX_DIM {
        error!(
            "jsonb_to_tvector: dim {} out of range 1..={}",
            arr.len(),
            MAX_DIM
        );
    }
    let mut out: Vec<f32> = Vec::with_capacity(arr.len());
    for (i, v) in arr.into_iter().enumerate() {
        let n = v.as_f64().unwrap_or_else(|| {
            error!(
                "jsonb_to_tvector: element {} is not a number ({})",
                i,
                value_kind(&v)
            )
        });
        if !n.is_finite() {
            error!(
                "jsonb_to_tvector: element {} is not finite ({})",
                i, n
            );
        }
        out.push(n as f32);
    }
    Tvector::from_vec(out)
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
/// unchanged. Useful as `CHECK (turbovec.tvector_check_dim(emb, 1536))`.
#[pg_extern(immutable, parallel_safe)]
fn tvector_check_dim(v: Tvector, expected: i32) -> Tvector {
    if expected < 1 {
        error!("tvector_check_dim: expected dim must be positive");
    }
    if v.dim() != expected as usize {
        error!(
            "tvector_check_dim: dim mismatch (got {}, expected {})",
            v.dim(),
            expected
        );
    }
    v
}

/// Build a zero-filled `tvector` of the requested dimension. Useful
/// as the identity for `sum(tvector)` in extension queries.
#[pg_extern(immutable, parallel_safe)]
fn tvector_zeros(dim: i32) -> Tvector {
    if dim <= 0 || dim as usize > MAX_DIM {
        error!(
            "tvector_zeros: dim {} out of range 1..={}",
            dim, MAX_DIM
        );
    }
    Tvector::from_vec(vec![0.0_f32; dim as usize])
}

/// Explicit text rendering of a `tvector` (mirrors the type's OUTPUT
/// function but callable directly).
#[pg_extern(immutable, parallel_safe)]
fn tvector_to_text(v: Tvector) -> String {
    let mut out = String::with_capacity(2 + v.dim() * 6);
    out.push('[');
    let mut first = true;
    for x in v.as_slice() {
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push_str(&format!("{}", x));
    }
    out.push(']');
    out
}

extension_sql!(
    r#"
    -- jsonb <-> tvector explicit casts.
    CREATE CAST (tvector AS jsonb) WITH FUNCTION tvector_to_jsonb(tvector);
    CREATE CAST (jsonb   AS tvector) WITH FUNCTION jsonb_to_tvector(jsonb);
    "#,
    name = "tvector_jsonb_casts",
    requires = [tvector_to_jsonb, jsonb_to_tvector]
);
