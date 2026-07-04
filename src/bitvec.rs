//! The `bitvec` SQL type — packed binary vectors stored 8 bits per
//! byte. Pgvector calls this `bit` (reusing Postgres core's `bit`
//! varlena layout); we use `bitvec` to avoid colliding with PG's
//! built-in `bit` type while keeping the same ergonomics.
//!
//! Distance ops:
//! * `<~>` Hamming distance (count of differing bits)
//! * `<%>` Jaccard distance (1 - |A ∩ B| / |A ∪ B|)
//!
//! `binary_quantize(vector)` produces a bitvec by setting each bit
//! to 1 iff the corresponding f32 coordinate is positive.

use core::ffi::CStr;

use pgrx::prelude::*;
use pgrx::{InOutFuncs, StringInfo};
use serde::{Deserialize, Serialize};

use crate::vec::Vector;

/// Hard cap matching pgvector's bit type.
pub const MAX_BITS: usize = 64_000;

/// A `bitvec` value — packed binary vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct Bitvec {
    /// Total number of *bits* (the dim of the original FP vector).
    pub n_bits: i32,
    /// Packed bytes. Byte 0 = bits 0..=7 (bit 0 = MSB of byte 0,
    /// matching Postgres core's `bit` layout). Length = ceil(n_bits / 8).
    pub bytes: ::std::vec::Vec<u8>,
}

impl Bitvec {
    pub fn new(n_bits: i32, bytes: ::std::vec::Vec<u8>) -> Self {
        if n_bits < 1 || n_bits as usize > MAX_BITS {
            error!("bitvec n_bits {} out of range 1..={}", n_bits, MAX_BITS);
        }
        let expected_len = (n_bits as usize + 7) / 8;
        if bytes.len() != expected_len {
            error!(
                "bitvec: bytes.len() {} != ceil(n_bits/8) {} for n_bits {}",
                bytes.len(),
                expected_len,
                n_bits
            );
        }
        Self { n_bits, bytes }
    }

    #[inline]
    pub fn n_bits(&self) -> i32 {
        self.n_bits
    }

    /// Number of 1-bits.
    pub fn popcount(&self) -> u64 {
        let mut count: u64 = 0;
        for b in &self.bytes {
            count += b.count_ones() as u64;
        }
        // Mask off any tail bits beyond n_bits to avoid counting them.
        let tail = self.n_bits as usize % 8;
        if tail != 0 {
            let last = *self.bytes.last().unwrap_or(&0);
            let valid_mask: u8 = !((1u8 << (8 - tail)) - 1);
            count -= (last & !valid_mask).count_ones() as u64;
        }
        count
    }

    pub(crate) fn check_same_n_bits(&self, other: &Bitvec, op: &str) {
        if self.n_bits != other.n_bits {
            error!(
                "different bitvec lengths {} and {} for operator '{}'",
                self.n_bits, other.n_bits, op
            );
        }
    }
}

impl InOutFuncs for Bitvec {
    fn input(input: &CStr) -> Self {
        let s = input
            .to_str()
            .unwrap_or_else(|e| error!("bitvec input is not valid UTF-8: {}", e));
        let trimmed = s.trim();
        if trimmed.is_empty() {
            error!("bitvec must have at least one bit");
        }
        let n = trimmed.len();
        if n > MAX_BITS {
            error!("bitvec n_bits {} exceeds {}", n, MAX_BITS);
        }
        let n_bytes = (n + 7) / 8;
        let mut bytes = vec![0u8; n_bytes];
        for (i, c) in trimmed.chars().enumerate() {
            match c {
                '0' => {}
                '1' => {
                    bytes[i / 8] |= 1u8 << (7 - (i % 8));
                }
                other => error!(
                    "bitvec input position {}: expected '0' or '1', got '{}'",
                    i, other
                ),
            }
        }
        Bitvec::new(n as i32, bytes)
    }

    fn output(&self, buffer: &mut StringInfo) {
        for i in 0..self.n_bits as usize {
            let byte = self.bytes[i / 8];
            let bit = (byte >> (7 - (i % 8))) & 1;
            buffer.push(if bit == 1 { '1' } else { '0' });
        }
    }

    const NULL_ERROR_MESSAGE: Option<&'static str> = Some("NULL is not a valid bitvec value");
}

// ---------------------------------------------------------------------
// Distance operators
// ---------------------------------------------------------------------

#[pg_extern(immutable, parallel_safe)]
fn hamming_distance(a: Bitvec, b: Bitvec) -> f64 {
    a.check_same_n_bits(&b, "<~>");
    let mut count: u64 = 0;
    for (x, y) in a.bytes.iter().zip(b.bytes.iter()) {
        count += (*x ^ *y).count_ones() as u64;
    }
    // Mask tail bits so unused trailing bits don't inflate.
    let tail = a.n_bits as usize % 8;
    if tail != 0 {
        let last_a = *a.bytes.last().unwrap_or(&0);
        let last_b = *b.bytes.last().unwrap_or(&0);
        let valid_mask: u8 = !((1u8 << (8 - tail)) - 1);
        count -= ((last_a ^ last_b) & !valid_mask).count_ones() as u64;
    }
    count as f64
}

#[pg_extern(immutable, parallel_safe)]
fn jaccard_distance(a: Bitvec, b: Bitvec) -> f64 {
    a.check_same_n_bits(&b, "<%>");
    let mut intersection: u64 = 0;
    let mut union: u64 = 0;
    for (x, y) in a.bytes.iter().zip(b.bytes.iter()) {
        intersection += (*x & *y).count_ones() as u64;
        union += (*x | *y).count_ones() as u64;
    }
    let tail = a.n_bits as usize % 8;
    if tail != 0 {
        let last_a = *a.bytes.last().unwrap_or(&0);
        let last_b = *b.bytes.last().unwrap_or(&0);
        let valid_mask: u8 = !((1u8 << (8 - tail)) - 1);
        intersection -= ((last_a & last_b) & !valid_mask).count_ones() as u64;
        union -= ((last_a | last_b) & !valid_mask).count_ones() as u64;
    }
    if union == 0 {
        return 0.0;
    }
    1.0 - (intersection as f64 / union as f64)
}

#[pg_extern(immutable, parallel_safe)]
fn bitvec_dims(v: Bitvec) -> i32 {
    v.n_bits
}

#[pg_extern(immutable, parallel_safe)]
fn bitvec_popcount(v: Bitvec) -> i64 {
    v.popcount() as i64
}

/// `binary_quantize(vector) -> bitvec` — set bit i iff the i-th
/// f32 coordinate is positive. Matches pgvector's binary_quantize.
#[pg_extern(immutable, parallel_safe)]
fn binary_quantize(v: Vector) -> Bitvec {
    let n = v.dim();
    if n > MAX_BITS {
        error!(
            "binary_quantize: vector dim {} exceeds bitvec MAX_BITS {}",
            n, MAX_BITS
        );
    }
    let mut bytes = vec![0u8; (n + 7) / 8];
    for (i, x) in v.as_slice().iter().enumerate() {
        if *x > 0.0 {
            bytes[i / 8] |= 1u8 << (7 - (i % 8));
        }
    }
    Bitvec::new(n as i32, bytes)
}

extension_sql!(
    r#"
    CREATE OPERATOR <~> (
        LEFTARG = bitvec, RIGHTARG = bitvec,
        PROCEDURE = hamming_distance,
        COMMUTATOR = '<~>'
    );
    CREATE OPERATOR <%> (
        LEFTARG = bitvec, RIGHTARG = bitvec,
        PROCEDURE = jaccard_distance,
        COMMUTATOR = '<%>'
    );
    "#,
    name = "bitvec_surface",
    requires = [Bitvec, hamming_distance, jaccard_distance, binary_quantize]
);
