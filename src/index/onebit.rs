//! 1-bit sign binary quantization (BQ) — the `WITH (bit_width = 1)`
//! index code path.
//!
//! ## Why this is NOT TurboQuant-at-1-bit
//!
//! The pinned `turbovec` crate (`IdMapIndex::new` / `new_lazy`)
//! **hard-rejects `bit_width < 2`** (`ConstructError::BitWidthOutOfRange`,
//! asserted `(2..=4).contains(&bit_width)`). turbovec cannot build a
//! 1-bit index at all, so 1-bit is a *distinct* scheme: sign-BQ, the
//! DiskANN/pgvector/Qdrant coarse code — a per-coordinate sign bit,
//! packed 8-to-a-byte, scored by Hamming (popcount of XOR), then
//! reranked exactly against the heap tuple (the existing
//! `xs_recheckorderby` path). No rotation, no Lloyd-Max codebook, no
//! per-vector scale — so the on-disk code is `dim/8` bytes/vec, exactly
//! **half** the 2-bit stride (`dim/8 * 2`). Hamming on sign bits
//! approximates ANGULAR/cosine, matching the AM's cosine/IP opclasses;
//! it is NOT an L2 code (the AM has no L2 opclass anyway).
//!
//! ## The footgun this module MUST handle (mean-centering)
//!
//! The naive "bit = 1 iff coord > 0" rule (what SQL `binary_quantize`
//! does) FAILS on non-zero-centered data: a dense-positive corpus
//! (e.g. GIST image descriptors) sets every bit to 1, so every code is
//! identical and recall collapses to 0. The fix — implemented here — is
//! to subtract the per-dimension corpus mean BEFORE the sign. On
//! already-zero-centered data (OpenAI/Cohere text embeddings) the mean
//! is ~0 and centering is a no-op; on skewed data it recovers a usable
//! code. [`is_degenerate`] additionally detects the pathological
//! all-same-sign-after-centering case so the build can refuse to ship a
//! silent all-ones landmine.

/// Pack one already-centered vector into MSB-first sign bits.
///
/// Bit `i` is `1` iff `centered[i] > 0.0`, laid out 8 bits/byte with
/// bit 0 = MSB of byte 0 — the SAME layout as [`crate::bitvec::Bitvec`]
/// and Postgres core's `bit` type, so the SQL Hamming/popcount kernels
/// and the index-side scorer share one packing convention. Output
/// length is `ceil(dim / 8)` bytes.
///
/// A coordinate of exactly `0.0` (e.g. a value that sat exactly on the
/// mean) packs as `0`, matching `binary_quantize`'s `> 0.0` rule.
pub fn pack_signs(centered: &[f32]) -> Vec<u8> {
    let dim = centered.len();
    let mut bytes = vec![0u8; dim.div_ceil(8)];
    for (i, &x) in centered.iter().enumerate() {
        if x > 0.0 {
            bytes[i / 8] |= 1u8 << (7 - (i % 8));
        }
    }
    bytes
}

/// Decode packed sign bits back to `±1.0` per coordinate.
///
/// Inverse of [`pack_signs`] up to the sign (magnitude is destroyed by
/// quantization — that is the whole point of 1-bit). Bit set ⇒ `+1.0`,
/// bit clear ⇒ `-1.0`. Used only by the round-trip property test; the
/// scan path scores packed codes directly via Hamming, never decodes.
pub fn unpack_signs(bytes: &[u8], dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let bit = (bytes[i / 8] >> (7 - (i % 8))) & 1;
            if bit == 1 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Per-dimension mean of a row-major `n x dim` corpus.
///
/// This is the centering vector: subtract it from every vector before
/// [`pack_signs`]. Persisted in the meta page (a `dim`-length `f32`
/// header, negligible next to the `n * dim/8` codes) so the SAME shift
/// is applied to query vectors at scan time. Returns all-zeros for an
/// empty corpus (centering then a no-op).
pub fn corpus_mean(flat: &[f32], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f64; dim];
    if dim == 0 {
        return Vec::new();
    }
    let n = flat.len() / dim;
    if n == 0 {
        return vec![0.0f32; dim];
    }
    for row in flat.chunks_exact(dim) {
        for (m, &x) in mean.iter_mut().zip(row) {
            *m += x as f64;
        }
    }
    let inv = 1.0 / n as f64;
    mean.iter().map(|&m| (m * inv) as f32).collect()
}

/// Subtract `mean` from `v` in place-free form, returning the centered
/// vector ready for [`pack_signs`]. `mean` must be `v.len()` long (or
/// empty, meaning "no centering").
pub fn center(v: &[f32], mean: &[f32]) -> Vec<f32> {
    if mean.is_empty() {
        return v.to_vec();
    }
    v.iter().zip(mean).map(|(&x, &m)| x - m).collect()
}

/// Detect the degenerate distribution the sign-at-zero rule can't
/// encode: after centering, EVERY vector's every coordinate has the
/// same sign, so every packed code is identical and Hamming distance is
/// uniformly 0 — recall would collapse to garbage.
///
/// Returns `true` iff, per dimension, all `n` centered values share one
/// sign (all `> 0`, or all `<= 0`). This is exactly the "every bit is 1"
/// / "every bit is 0" GIST failure the feasibility study measured
/// (R@10 = 0.0). Centering normally fixes it (subtracting the mean puts
/// ~half the mass on each side per dim); if it STILL trips, the data is
/// unusable for 1-bit and the build should error with the REINDEX-with-
/// bit_width>=2 hint rather than ship a silent landmine.
///
/// `flat` is row-major `n x dim`, ALREADY centered.
pub fn is_degenerate(flat: &[f32], dim: usize) -> bool {
    if dim == 0 {
        return false;
    }
    let n = flat.len() / dim;
    if n <= 1 {
        // A 0- or 1-row corpus can't be "collapsed" in a way that
        // hurts ranking (nothing to rank against).
        return false;
    }
    // For each dimension, are ALL rows the same sign? If even one
    // dimension splits the corpus, the codes are not all identical and
    // Hamming ranking has signal — not degenerate.
    'dim: for d in 0..dim {
        let mut saw_pos = false;
        let mut saw_nonpos = false;
        for row in flat.chunks_exact(dim) {
            if row[d] > 0.0 {
                saw_pos = true;
            } else {
                saw_nonpos = true;
            }
            if saw_pos && saw_nonpos {
                // This dim distinguishes some rows: signal exists.
                continue 'dim;
            }
        }
        // Reaching here means dim `d` did NOT split — keep checking.
    }
    // Degenerate iff NO dimension split the corpus (every code
    // identical). We detect that by checking every row's packed code
    // equals row 0's.
    let code0 = pack_signs(&flat[..dim]);
    flat.chunks_exact(dim).all(|row| pack_signs(row) == code0)
}

/// On-disk / scan-side per-vector byte width for a 1-bit index:
/// `dim/8`, no scale. Half of the 2-bit stride (`dim/8 * 2`). Used by
/// the storage assertions and the meta-page stride math.
#[inline]
pub fn codes_stride(dim: usize) -> usize {
    dim.div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: pack then unpack recovers the SIGN of every
    /// coordinate (magnitude is intentionally lost).
    #[test]
    fn pack_unpack_preserves_sign() {
        let v = [1.5f32, -0.2, 3.0, -9.0, 0.0, 0.01, -0.01, 42.0];
        let packed = pack_signs(&v);
        assert_eq!(packed.len(), 1, "8 dims -> 1 byte");
        let round = unpack_signs(&packed, v.len());
        for (i, (&orig, &back)) in v.iter().zip(&round).enumerate() {
            let orig_sign = orig > 0.0;
            let back_sign = back > 0.0;
            assert_eq!(orig_sign, back_sign, "sign mismatch at dim {i}");
        }
        // Exactly-zero packs as clear (-> -1.0 on unpack), matching the
        // `> 0.0` rule.
        assert_eq!(round[4], -1.0);
    }

    /// MSB-first packing matches bitvec's convention: bit 0 is the MSB
    /// of byte 0.
    #[test]
    fn packing_is_msb_first() {
        // Only dim 0 positive -> top bit of byte 0 set (0b1000_0000).
        let mut v = vec![-1.0f32; 8];
        v[0] = 1.0;
        assert_eq!(pack_signs(&v), vec![0b1000_0000]);
        // Only dim 7 positive -> LSB of byte 0 set.
        let mut v = vec![-1.0f32; 8];
        v[7] = 1.0;
        assert_eq!(pack_signs(&v), vec![0b0000_0001]);
    }

    /// Non-multiple-of-8 dim rounds up to a whole byte; tail bits stay
    /// clear.
    #[test]
    fn tail_bits_are_zero() {
        let v = vec![1.0f32; 3]; // 3 dims -> 1 byte, bits 0..2 set
        assert_eq!(pack_signs(&v), vec![0b1110_0000]);
    }

    /// corpus_mean is the exact per-dim average.
    #[test]
    fn mean_is_per_dim_average() {
        // 2 rows, dim 2: [[0, 10], [4, 20]] -> mean [2, 15].
        let flat = [0.0f32, 10.0, 4.0, 20.0];
        let m = corpus_mean(&flat, 2);
        assert_eq!(m, vec![2.0, 15.0]);
    }

    /// Centering an already-zero-mean corpus is (near-)identity; the
    /// key property is it does not FLIP any sign it shouldn't.
    #[test]
    fn centering_zero_mean_is_noop() {
        let flat = [1.0f32, -1.0, -1.0, 1.0]; // per-dim mean = 0
        let m = corpus_mean(&flat, 2);
        assert_eq!(m, vec![0.0, 0.0]);
        assert_eq!(center(&flat[..2], &m), vec![1.0, -1.0]);
    }

    /// THE FOOTGUN: a dense-positive corpus (every coord > 0, like
    /// GIST) has an identical all-ones code for every row under the raw
    /// sign rule — detected as degenerate. Mean-centering then FIXES it
    /// (subtracting the per-dim mean splits each dim), so the centered
    /// corpus is NOT degenerate.
    #[test]
    fn all_positive_is_degenerate_raw_but_centering_fixes_it() {
        // 3 rows, dim 4, all strictly positive but with spread.
        let flat = [
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ];
        // Raw (uncentered) every code is 0b1111... -> degenerate.
        assert!(
            is_degenerate(&flat, 4),
            "all-positive raw corpus must be flagged degenerate"
        );
        // Center, then it splits per dim -> usable.
        let mean = corpus_mean(&flat, 4);
        let centered: Vec<f32> = flat
            .chunks_exact(4)
            .flat_map(|r| center(r, &mean))
            .collect();
        assert!(
            !is_degenerate(&centered, 4),
            "mean-centering must rescue the dense-positive corpus"
        );
    }

    /// A truly constant corpus (all rows identical) stays degenerate
    /// even after centering — centering shifts it to all-zero, every
    /// code identical. This is the case the build must ERROR on.
    #[test]
    fn constant_corpus_stays_degenerate_after_centering() {
        let flat = [7.0f32; 12]; // 3 rows x dim 4, all identical
        let mean = corpus_mean(&flat, 4);
        let centered: Vec<f32> = flat
            .chunks_exact(4)
            .flat_map(|r| center(r, &mean))
            .collect();
        assert!(
            is_degenerate(&centered, 4),
            "a constant corpus is unusable for 1-bit even after centering"
        );
    }

    /// Zero-centered spread data (the production text-embedding case) is
    /// never flagged.
    #[test]
    fn zero_centered_spread_is_not_degenerate() {
        let flat = [
            1.0f32, -2.0, 3.0, -4.0, //
            -1.0, 2.0, -3.0, 4.0, //
            0.5, -0.5, 0.5, -0.5,
        ];
        assert!(!is_degenerate(&flat, 4));
    }

    /// Storage: 1-bit stride is exactly half of 2-bit.
    #[test]
    fn onebit_stride_is_half_of_twobit() {
        for dim in [8usize, 128, 768, 1536] {
            let onebit = codes_stride(dim);
            let twobit = dim / 8 * 2;
            assert_eq!(twobit, onebit * 2, "dim {dim}: 2-bit must be 2x 1-bit");
        }
    }
}
