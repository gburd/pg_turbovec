//! The `halfvec` SQL type — variable-dimension `f16` (half-precision)
//! vector. Storage is half the size of `vector` at a small precision
//! cost; useful when memory pressure dominates and your model emits
//! values that fit comfortably inside f16's ±65 504 range.
//!
//! On-disk representation: CBOR-encoded varlena (Phase 1, matching
//! `vector`). A binary-compatible-with-pgvector layout is on the
//! roadmap.
//!
//! Text representation: same as `vector` — `'[1.0, 2.0, 3.0]'`. We
//! parse `f32` and narrow to `f16`, raising on values that overflow
//! the f16 range.

use core::ffi::CStr;

use half::f16;
use pgrx::prelude::*;
use pgrx::{InOutFuncs, StringInfo};
use serde::{Deserialize, Serialize};

/// Hard cap on dimensionality, matching pgvector. f16 occupies 2
/// bytes per coord, so 16 000 dim = 32 KB raw + 8-byte header — fits
/// inline before TOAST kicks in.
pub const MAX_DIM: usize = 16_000;

/// A `halfvec` value — variable-dimension `f16` vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct Halfvec {
    /// Per-coordinate `f16` values.
    pub data: ::std::vec::Vec<f16>,
}

impl Halfvec {
    #[must_use]
    pub fn from_f16_vec(data: ::std::vec::Vec<f16>) -> Self {
        if data.is_empty() {
            error!("halfvec must have at least one dimension");
        }
        if data.len() > MAX_DIM {
            error!("halfvec cannot have more than {} dimensions", MAX_DIM);
        }
        for (i, v) in data.iter().enumerate() {
            if !v.is_finite() {
                error!(
                    "halfvec value at index {} is not finite ({})",
                    i,
                    f32::from(*v)
                );
            }
        }
        Self { data }
    }

    /// Build from `f32` values, narrowing each to `f16`. Values that
    /// overflow the f16 range raise `ERROR`.
    #[must_use]
    pub fn from_f32_vec(data: ::std::vec::Vec<f32>) -> Self {
        let mut out = ::std::vec::Vec::with_capacity(data.len());
        for (i, v) in data.iter().enumerate() {
            if !v.is_finite() {
                error!(
                    "halfvec value at index {} is not finite ({})",
                    i, v
                );
            }
            let h = f16::from_f32(*v);
            if !h.is_finite() {
                error!(
                    "halfvec value at index {} ({}) overflows the f16 range \
                     [-65504, 65504]",
                    i, v
                );
            }
            out.push(h);
        }
        Halfvec::from_f16_vec(out)
    }

    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[f16] {
        &self.data
    }

    /// Materialise to an owned `Vec<f32>` — convenience for code that
    /// wants to feed a halfvec into an f32 kernel.
    #[must_use]
    pub fn to_f32_vec(&self) -> ::std::vec::Vec<f32> {
        self.data.iter().map(|h| f32::from(*h)).collect()
    }

    pub(crate) fn check_same_dim(&self, other: &Halfvec, op: &str) {
        if self.dim() != other.dim() {
            error!(
                "different halfvec dimensions {} and {} for operator '{}'",
                self.dim(),
                other.dim(),
                op
            );
        }
    }
}

impl InOutFuncs for Halfvec {
    fn input(input: &CStr) -> Self {
        let s = input
            .to_str()
            .unwrap_or_else(|e| error!("halfvec input is not valid UTF-8: {}", e));
        match crate::vec::parse_vec(s) {
            Ok(v) => Halfvec::from_f32_vec(v),
            Err(msg) => error!("invalid halfvec input '{}': {}", s, msg),
        }
    }

    fn output(&self, buffer: &mut StringInfo) {
        buffer.push('[');
        let mut first = true;
        for v in &self.data {
            if !first {
                buffer.push_str(", ");
            }
            first = false;
            // Format via f32 so the round-trip text representation
            // matches what users typed in.
            buffer.push_str(&format!("{}", f32::from(*v)));
        }
        buffer.push(']');
    }

    const NULL_ERROR_MESSAGE: Option<&'static str> =
        Some("NULL is not a valid halfvec value");
}
