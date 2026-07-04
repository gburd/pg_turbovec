//! The `sparsevec` SQL type — sparse high-dimensional vectors stored
//! as `(dim, indices, values)` triples. Pgvector uses this for
//! 30k-dim TF-IDF and similar workloads where the vast majority of
//! coordinates are zero.
//!
//! On-disk: CBOR-encoded varlena (matching `vector` and `halfvec`).
//!
//! Text representation: `'{i1:v1, i2:v2, …}/dim'` — pgvector's
//! format. Indices are 1-based per pgvector convention; we store
//! them 0-based internally and translate at the I/O boundary.
//! Example: `'{1:1.5, 5:2.25}/10'::sparsevec` is a 10-dim vector
//! whose coordinates 0 (= 1-1) and 4 (= 5-1) are 1.5 and 2.25, all
//! others zero.

use core::ffi::CStr;

use pgrx::prelude::*;
use pgrx::{InOutFuncs, StringInfo};
use serde::{Deserialize, Serialize};

pub const MAX_DIM: usize = 1_000_000_000;

/// A `sparsevec` value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct Sparsevec {
    /// Total number of dimensions (zero and non-zero).
    pub dim: i32,
    /// Sorted, deduplicated 0-based indices of non-zero coordinates.
    pub indices: ::std::vec::Vec<i32>,
    /// Values aligned with `indices`. `values[i]` is the coordinate
    /// at index `indices[i]`.
    pub values: ::std::vec::Vec<f32>,
}

impl Sparsevec {
    pub fn new(dim: i32, indices: ::std::vec::Vec<i32>, values: ::std::vec::Vec<f32>) -> Self {
        if dim < 1 || dim as usize > MAX_DIM {
            error!("sparsevec dim {} out of range 1..={}", dim, MAX_DIM);
        }
        if indices.len() != values.len() {
            error!(
                "sparsevec: indices.len() ({}) != values.len() ({})",
                indices.len(),
                values.len()
            );
        }
        // Verify sorted-unique and within range.
        let mut prev: i32 = -1;
        for (k, idx) in indices.iter().enumerate() {
            if *idx <= prev {
                error!(
                    "sparsevec indices must be strictly increasing (got {} <= {} at position {})",
                    idx, prev, k
                );
            }
            if *idx < 0 || *idx >= dim {
                error!(
                    "sparsevec index {} (position {}) out of range [0, {})",
                    idx, k, dim
                );
            }
            if !values[k].is_finite() {
                error!(
                    "sparsevec value at non-zero position {} is not finite ({})",
                    k, values[k]
                );
            }
            prev = *idx;
        }
        Self {
            dim,
            indices,
            values,
        }
    }

    #[inline]
    pub fn dim(&self) -> i32 {
        self.dim
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Materialise to a dense `Vec<f32>` for kernel consumption.
    /// Allocates `dim` zeroes — beware on million-dim vectors.
    pub fn to_dense(&self) -> ::std::vec::Vec<f32> {
        let mut out = vec![0.0_f32; self.dim as usize];
        for (i, idx) in self.indices.iter().enumerate() {
            out[*idx as usize] = self.values[i];
        }
        out
    }

    pub(crate) fn check_same_dim(&self, other: &Sparsevec, op: &str) {
        if self.dim != other.dim {
            error!(
                "different sparsevec dimensions {} and {} for operator '{}'",
                self.dim, other.dim, op
            );
        }
    }
}

impl InOutFuncs for Sparsevec {
    fn input(input: &CStr) -> Self {
        let s = input
            .to_str()
            .unwrap_or_else(|e| error!("sparsevec input is not valid UTF-8: {}", e));
        match parse_sparsevec(s) {
            Ok((dim, idx, vals)) => Sparsevec::new(dim, idx, vals),
            Err(msg) => error!("invalid sparsevec input '{}': {}", s, msg),
        }
    }

    fn output(&self, buffer: &mut StringInfo) {
        buffer.push('{');
        let mut first = true;
        for (i, idx) in self.indices.iter().enumerate() {
            if !first {
                buffer.push(',');
            }
            first = false;
            // Emit 1-based per pgvector convention.
            buffer.push_str(&format!("{}:{}", idx + 1, self.values[i]));
        }
        buffer.push('}');
        buffer.push('/');
        buffer.push_str(&self.dim.to_string());
    }

    const NULL_ERROR_MESSAGE: Option<&'static str> = Some("NULL is not a valid sparsevec value");
}

/// Parse the pgvector-format `'{i1:v1, i2:v2}/dim'` literal. Returns
/// (dim, 0-based indices, values).
fn parse_sparsevec(s: &str) -> Result<(i32, ::std::vec::Vec<i32>, ::std::vec::Vec<f32>), String> {
    let trimmed = s.trim();
    let (body, dim_str) = trimmed
        .rsplit_once('/')
        .ok_or_else(|| "expected '/dim' suffix".to_string())?;
    let dim: i32 = dim_str
        .trim()
        .parse()
        .map_err(|e| format!("invalid dim '{}': {}", dim_str, e))?;
    let body = body.trim();
    let body = body
        .strip_prefix('{')
        .ok_or_else(|| "expected '{' at start".to_string())?
        .strip_suffix('}')
        .ok_or_else(|| "expected '}' at end".to_string())?
        .trim();

    if body.is_empty() {
        return Ok((dim, ::std::vec::Vec::new(), ::std::vec::Vec::new()));
    }

    let mut indices = ::std::vec::Vec::new();
    let mut values = ::std::vec::Vec::new();
    for (i, tok) in body.split(',').enumerate() {
        let tok = tok.trim();
        let (idx_str, val_str) = tok
            .split_once(':')
            .ok_or_else(|| format!("expected 'idx:val' at position {}: '{}'", i, tok))?;
        let idx_1based: i32 = idx_str
            .trim()
            .parse()
            .map_err(|e| format!("position {}: invalid index '{}': {}", i, idx_str, e))?;
        if idx_1based < 1 {
            return Err(format!(
                "position {}: index must be >= 1 (1-based), got {}",
                i, idx_1based
            ));
        }
        let val: f32 = val_str
            .trim()
            .parse()
            .map_err(|e| format!("position {}: invalid value '{}': {}", i, val_str, e))?;
        if !val.is_finite() {
            return Err(format!("position {}: non-finite value '{}'", i, val));
        }
        indices.push(idx_1based - 1);
        values.push(val);
    }
    Ok((dim, indices, values))
}
