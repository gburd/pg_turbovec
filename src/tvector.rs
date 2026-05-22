//! The `tvector` SQL type — variable-dimension `f32` vector.
//!
//! Phase 1 representation: pgrx `PostgresType` with serde-derived
//! varlena (CBOR) encoding. This is *not* binary-compatible with
//! pgvector's `vector` type — Phase 2 swaps in a hand-rolled varlena
//! layout (`vl_len_ + i16 dim + i16 reserved + f32[dim]`) that *is*
//! byte-compatible with pgvector, at which point the textual API
//! stays the same and disk format upgrades via dump/restore.
//!
//! Text representation: `'[1, 2, 3]'` — a bracketed comma-separated
//! list of finite IEEE-754 single-precision floats. Whitespace is
//! tolerated. NaN/±Inf are rejected at parse time, mirroring pgvector.

use core::ffi::CStr;

use pgrx::prelude::*;
use pgrx::{InOutFuncs, StringInfo};
use serde::{Deserialize, Serialize};

/// Hard cap on dimensionality, matching pgvector. The varlena page
/// limit and TOAST chunking effectively cap us much lower for in-line
/// storage, but values up to this dimension are accepted.
pub const MAX_DIM: usize = 16_000;

/// A turbovec `tvector` value — a variable-dimension `f32` vector.
///
/// Stored on disk as a CBOR-encoded varlena (Phase 1). The Rust
/// representation is a heap-allocated `Vec<f32>`; FromDatum/IntoDatum
/// pay one serde round-trip per call. For Phase 2 we replace this
/// with a zero-copy `&[f32]` over a hand-rolled varlena layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct Tvector {
    /// Per-coordinate values. Always finite.
    pub data: Vec<f32>,
}

impl Tvector {
    /// Build a vector from an owned `Vec<f32>`, validating that all
    /// values are finite and that the dimension is in `1..=MAX_DIM`.
    /// Raises a Postgres `ERROR` on violation.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        if data.is_empty() {
            error!("tvector must have at least one dimension");
        }
        if data.len() > MAX_DIM {
            error!("tvector cannot have more than {} dimensions", MAX_DIM);
        }
        for (i, v) in data.iter().enumerate() {
            if !v.is_finite() {
                error!("tvector value at index {} is not finite ({})", i, v);
            }
        }
        Self { data }
    }

    /// Number of dimensions.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Borrow the underlying `f32` slice.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Assert that `self` and `other` have the same dimensionality;
    /// raise a Postgres ERROR otherwise.
    #[inline]
    pub(crate) fn check_same_dim(&self, other: &Tvector, op: &str) {
        if self.dim() != other.dim() {
            error!(
                "different tvector dimensions {} and {} for operator '{}'",
                self.dim(),
                other.dim(),
                op
            );
        }
    }
}

impl InOutFuncs for Tvector {
    fn input(input: &CStr) -> Self {
        let s = input
            .to_str()
            .unwrap_or_else(|e| error!("tvector input is not valid UTF-8: {}", e));
        match parse_tvector(s) {
            Ok(v) => Tvector::from_vec(v),
            Err(msg) => error!("invalid tvector input '{}': {}", s, msg),
        }
    }

    fn output(&self, buffer: &mut StringInfo) {
        // Format: '[1, 2, 3]'. Use Rust's default f32 formatting,
        // which round-trips through f32::from_str.
        buffer.push('[');
        let mut first = true;
        for v in &self.data {
            if !first {
                buffer.push_str(", ");
            }
            first = false;
            // Use {} which prints the shortest round-trippable form.
            buffer.push_str(&format!("{}", v));
        }
        buffer.push(']');
    }

    const NULL_ERROR_MESSAGE: Option<&'static str> = Some("NULL is not a valid tvector value");
}

/// Parse a `'[a, b, c]'`-formatted tvector literal.
pub(crate) fn parse_tvector(s: &str) -> Result<Vec<f32>, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("empty input".to_string());
    }
    let stripped = trimmed
        .strip_prefix('[')
        .ok_or_else(|| "expected '[' at start".to_string())?
        .strip_suffix(']')
        .ok_or_else(|| "expected ']' at end".to_string())?;
    let body = stripped.trim();
    if body.is_empty() {
        return Err("tvector must have at least one dimension".to_string());
    }
    let mut out = Vec::with_capacity(8);
    for (i, tok) in body.split(',').enumerate() {
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(format!("empty value at position {}", i));
        }
        let v: f32 = tok.parse().map_err(|e| format!("position {}: {}", i, e))?;
        if !v.is_finite() {
            return Err(format!(
                "position {}: value '{}' is not a finite number",
                i, tok
            ));
        }
        out.push(v);
    }
    Ok(out)
}

#[cfg(any(test, feature = "pg_test"))]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn parse_basic() {
        assert_eq!(parse_tvector("[1, 2, 3]").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            parse_tvector("[ 1.5,-2.0 , 3 ]").unwrap(),
            vec![1.5, -2.0, 3.0]
        );
    }

    #[test]
    fn parse_rejects() {
        assert!(parse_tvector("").is_err());
        assert!(parse_tvector("1, 2, 3").is_err());
        assert!(parse_tvector("[1, 2,]").is_err());
        assert!(parse_tvector("[]").is_err());
        assert!(parse_tvector("[NaN]").is_err());
        assert!(parse_tvector("[inf]").is_err());
    }
}
