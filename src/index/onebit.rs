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
///
/// Expressed in terms of [`accumulate_sums`] + [`finish_mean`] so the
/// whole-corpus (flat build) and streamed-in-blocks (IVF build) paths are
/// the SAME arithmetic in the same order, not two implementations that
/// could drift (gated by `streamed_mean_matches_whole_corpus_mean`).
pub fn corpus_mean(flat: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let n = flat.len() / dim;
    if n == 0 {
        return vec![0.0f32; dim];
    }
    let mut sums = vec![0.0f64; dim];
    accumulate_sums(&mut sums, flat, dim);
    finish_mean(&sums, n)
}

/// Add a row-major block's per-dimension values into `sums` (an f64
/// accumulator of length `dim`), row by row in block order.
///
/// The streaming half of [`corpus_mean`]: an out-of-core build calls this
/// once per spill block and [`finish_mean`] at the end, which sums the
/// rows in exactly the same order (spill order) as a whole-corpus
/// `corpus_mean` would, so the resulting mean is BIT-IDENTICAL.
pub fn accumulate_sums(sums: &mut [f64], flat: &[f32], dim: usize) {
    if dim == 0 || sums.len() != dim {
        return;
    }
    for row in flat.chunks_exact(dim) {
        for (m, &x) in sums.iter_mut().zip(row) {
            *m += x as f64;
        }
    }
}

/// Divide accumulated [`accumulate_sums`] totals by `n` rows to get the
/// per-dimension mean. `n == 0` yields all-zeros (centering a no-op).
pub fn finish_mean(sums: &[f64], n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![0.0f32; sums.len()];
    }
    let inv = 1.0 / n as f64;
    sums.iter().map(|&m| (m * inv) as f32).collect()
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
    // Degenerate iff EVERY row's packed code equals row 0's (no
    // dimension split the corpus, so Hamming is uniformly 0).
    let code0 = pack_signs(&flat[..dim]);
    flat.chunks_exact(dim).all(|row| pack_signs(row) == code0)
}

/// [`is_degenerate`] on ALREADY-PACKED codes: `true` iff every one of the
/// `n` code rows equals row 0, so Hamming distance is uniformly 0 and the
/// ranking would be arbitrary.
///
/// This is the streamable form the out-of-core (IVF) build needs — it
/// never has the whole centered f32 corpus resident — and it is exactly
/// equivalent to `is_degenerate` on the corresponding centered corpus,
/// because `pack_signs` is a per-row pure function (gated by
/// `packed_degeneracy_matches_centered_degeneracy`).
///
/// `n <= 1` returns `false`, matching `is_degenerate`: a 0- or 1-row
/// corpus has nothing to rank against.
pub fn codes_are_degenerate(codes: &[u8], stride: usize, n: usize) -> bool {
    if stride == 0 || n <= 1 || codes.len() < n * stride {
        return false;
    }
    let code0 = &codes[..stride];
    (1..n).all(|s| &codes[s * stride..(s + 1) * stride] == code0)
}

/// On-disk / scan-side per-vector byte width for a 1-bit index:
/// `dim/8`, no scale. Half of the 2-bit stride (`dim/8 * 2`). Used by
/// the storage assertions and the meta-page stride math.
#[inline]
pub fn codes_stride(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Hamming distance between two packed sign-code rows of equal length.
///
/// No tail masking is needed: [`pack_signs`] zeroes the unused trailing
/// bits of the last byte (unit-asserted by `tail_bits_are_zero`), so
/// equal-length codes agree on those bits and they contribute 0 to the
/// XOR. This is the same MSB-first convention as `bitvec.rs` and
/// Postgres's `bit` type, so the SQL popcount kernels and this scorer
/// cannot disagree.
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Top-`k` nearest slots to `query_code` by Hamming distance over a flat
/// chain of `n` packed codes, returned as `(distance, slot)` ascending.
///
/// Thin wrapper over [`topk_hamming_slots`] with the full slot range —
/// ONE heap implementation shared by the flat scan and the IVF
/// cell-restricted scan, so the two can never diverge on the tie-break
/// (asserted by `topk_hamming_slots_full_range_matches_flat`).
pub fn topk_hamming(query_code: &[u8], codes: &[u8], n: usize, k: usize) -> Vec<(u32, u32)> {
    debug_assert!(query_code.is_empty() || codes.len() >= n * query_code.len());
    topk_hamming_slots(query_code, codes, 0..(n as u32), k)
}

/// Top-`k` nearest slots to `query_code` by Hamming distance, considering
/// ONLY the slots yielded by `slots`, returned as `(distance, slot)`
/// ascending.
///
/// This is the IVF+BQ scan primitive: the caller yields exactly the slots
/// belonging to the probed cells (minus tombstones), so the unprobed
/// cells' codes are never scored — the scan win the cell-contiguous
/// layout exists for. [`topk_hamming`] is this same function over `0..n`.
///
/// Deliberately SCALAR. The v1.7.3 incident — where pre-AVX2 CPUs
/// returned WRONG ANN results from a mis-specialised kernel — is the
/// reason: `u8::count_ones` lowers to `POPCNT` on any modern x86_64 and
/// to the equivalent elsewhere, and a hand-vectorised version must be
/// proven bit-identical against THIS function before it can replace it.
/// A wide-SIMD popcount is a follow-up, not a prerequisite.
///
/// Ties break toward the lower slot so results are deterministic (the
/// same reason `partition::rank_nearest` does), INDEPENDENT of the order
/// `slots` yields them in. Hamming over `dim` bits has only `dim + 1`
/// distinct values, so ties are COMMON — which is exactly why the caller
/// must rerank exactly: see `guc::hi_dim_rerank_candidate_count`, which
/// treats a 1-bit index as high-dim at any `dim` so `hi_dim_rerank =
/// auto` widens the exact rerank window for BQ.
///
/// A slot whose code row would fall outside `codes` is SKIPPED rather
/// than panicking: an IVF slot list is derived from the on-disk cell
/// directory, which is only as trustworthy as the relfile, and a torn
/// read must not abort the backend.
pub fn topk_hamming_slots<I>(query_code: &[u8], codes: &[u8], slots: I, k: usize) -> Vec<(u32, u32)>
where
    I: IntoIterator<Item = u32>,
{
    let stride = query_code.len();
    if k == 0 || stride == 0 {
        return Vec::new();
    }
    // A bounded max-heap of the k best: O(candidates log k), no full sort.
    let mut heap: std::collections::BinaryHeap<(u32, u32)> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    for slot in slots {
        let start = (slot as usize) * stride;
        let end = start + stride;
        if end > codes.len() {
            continue;
        }
        let d = hamming(query_code, &codes[start..end]);
        if heap.len() < k {
            heap.push((d, slot));
        } else if let Some(&(worst, worst_slot)) = heap.peek() {
            // Strictly-better OR equal-distance-but-lower-slot, so the
            // tie-break is deterministic rather than heap-order-dependent.
            if d < worst || (d == worst && slot < worst_slot) {
                heap.pop();
                heap.push((d, slot));
            }
        }
    }
    let mut out = heap.into_vec();
    out.sort_unstable_by_key(|&(d, slot)| (d, slot));
    out
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
    fn hamming_matches_bitvec_convention() {
        // Same MSB-first packing as bitvec.rs, so XOR popcount must agree
        // with a hand count. 0b1010_0000 vs 0b1100_0000 differ in 2 bits.
        let a = pack_signs(&[1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        let b = pack_signs(&[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(a, vec![0b1010_0000]);
        assert_eq!(b, vec![0b1100_0000]);
        assert_eq!(hamming(&a, &b), 2);
        assert_eq!(hamming(&a, &a), 0, "self-distance must be zero");
    }

    #[test]
    fn hamming_ignores_zeroed_tail_bits() {
        // dim=12 -> 2 bytes with 4 unused trailing bits. pack_signs zeroes
        // them, so two codes that agree on all 12 real dims are distance 0
        // regardless of the tail.
        let v: Vec<f32> = (0..12)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let a = pack_signs(&v);
        let b = pack_signs(&v);
        assert_eq!(a.len(), 2);
        assert_eq!(hamming(&a, &b), 0);
    }

    /// The top-k heap must agree with brute force on every query. A
    /// bounded heap with a tie-break is easy to get subtly wrong, and
    /// Hamming over `dim` bits has only `dim + 1` distinct values so ties
    /// are the common case, not the edge case.
    #[test]
    fn topk_hamming_matches_brute_force() {
        let dim = 32usize;
        let n = 200usize;
        let stride = codes_stride(dim);
        // Deterministic pseudo-random corpus of packed codes.
        let mut codes = vec![0u8; n * stride];
        let mut x: u32 = 0x1234_5678;
        for b in codes.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        for qi in 0..8usize {
            let q = &codes[qi * stride..(qi + 1) * stride];
            for k in [1usize, 5, 10, 50] {
                let got = topk_hamming(q, &codes, n, k);
                // Brute force: full sort by (distance, slot).
                let mut all: Vec<(u32, u32)> = (0..n)
                    .map(|s| (hamming(q, &codes[s * stride..(s + 1) * stride]), s as u32))
                    .collect();
                all.sort_unstable_by_key(|&(d, s)| (d, s));
                all.truncate(k);
                assert_eq!(got, all, "query {qi}, k={k}");
            }
        }
    }

    #[test]
    fn topk_hamming_finds_self_first_and_handles_edges() {
        let dim = 64usize;
        let stride = codes_stride(dim);
        let n = 10usize;
        let mut codes = vec![0u8; n * stride];
        // Slot i gets i bits set, so distances from slot 0 are distinct.
        for i in 0..n {
            for bit in 0..i {
                codes[i * stride + bit / 8] |= 0x80 >> (bit % 8);
            }
        }
        let q = codes[3 * stride..4 * stride].to_vec();
        let got = topk_hamming(&q, &codes, n, 3);
        assert_eq!(got[0], (0, 3), "a vector must be its own nearest neighbour");
        // k == 0 and n == 0 must not panic and must return nothing.
        assert!(topk_hamming(&q, &codes, n, 0).is_empty());
        assert!(topk_hamming(&q, &codes, 0, 5).is_empty());
        // k larger than n clamps.
        assert_eq!(topk_hamming(&q, &codes, n, 99).len(), n);
    }

    #[test]
    fn onebit_stride_is_half_of_twobit() {
        for dim in [8usize, 128, 768, 1536] {
            let onebit = codes_stride(dim);
            let twobit = dim / 8 * 2;
            assert_eq!(twobit, onebit * 2, "dim {dim}: 2-bit must be 2x 1-bit");
        }
    }

    /// Deterministic pseudo-random packed-code chain for the slot-restricted
    /// tests below. Same LCG the brute-force test uses.
    fn synth_codes(n: usize, stride: usize, seed: u32) -> Vec<u8> {
        let mut codes = vec![0u8; n * stride];
        let mut x = seed;
        for b in codes.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        codes
    }

    /// `topk_hamming` MUST be exactly `topk_hamming_slots` over the full
    /// slot range. The flat BQ scan and the IVF+BQ cell-restricted scan
    /// share one heap implementation precisely so the two can never
    /// disagree on the tie-break; this gates that they are the same code.
    #[test]
    fn topk_hamming_slots_full_range_matches_flat() {
        let dim = 40usize;
        let stride = codes_stride(dim);
        let n = 150usize;
        let codes = synth_codes(n, stride, 0x0BAD_F00D);
        for qi in [0usize, 7, 99] {
            let q = &codes[qi * stride..(qi + 1) * stride];
            for k in [1usize, 3, 25, 200] {
                assert_eq!(
                    topk_hamming(q, &codes, n, k),
                    topk_hamming_slots(q, &codes, 0..(n as u32), k),
                    "query {qi}, k={k}"
                );
            }
        }
    }

    /// The IVF+BQ scan primitive: restricting to a subset of slots must
    /// return exactly the brute-force top-k OVER THAT SUBSET -- not the
    /// global top-k filtered afterwards (which would under-fill k when the
    /// global winners are all outside the probed cells). This is the
    /// property that makes probing a fraction of cells CORRECT rather than
    /// just cheap.
    #[test]
    fn topk_hamming_slots_matches_brute_force_over_the_subset() {
        let dim = 32usize;
        let stride = codes_stride(dim);
        let n = 200usize;
        let codes = synth_codes(n, stride, 0x1234_5678);
        // Three subsets: a contiguous "cell" range (what the IVF layout
        // actually produces), a sparse stride, and a single slot.
        let subsets: Vec<Vec<u32>> = vec![
            (40u32..90).collect(),
            (0u32..n as u32).filter(|s| s % 7 == 3).collect(),
            vec![123],
        ];
        for (si, subset) in subsets.iter().enumerate() {
            for qi in [0usize, 55, 140] {
                let q = &codes[qi * stride..(qi + 1) * stride];
                for k in [1usize, 4, 20, 500] {
                    let got = topk_hamming_slots(q, &codes, subset.iter().copied(), k);
                    let mut want: Vec<(u32, u32)> = subset
                        .iter()
                        .map(|&s| {
                            let row = &codes[s as usize * stride..(s as usize + 1) * stride];
                            (hamming(q, row), s)
                        })
                        .collect();
                    want.sort_unstable_by_key(|&(d, s)| (d, s));
                    want.truncate(k);
                    assert_eq!(got, want, "subset {si}, query {qi}, k={k}");
                }
            }
        }
    }

    /// The tie-break must not depend on the ORDER the slot iterator yields
    /// slots in. A probe set is built from `coarse_probe`'s
    /// distance-ordered cell list, so slots arrive out of ascending order;
    /// if the heap's tie-break were order-sensitive, the same query would
    /// return different rows depending on which cell happened to be
    /// nearest.
    #[test]
    fn topk_hamming_slots_is_order_independent() {
        // The sharpest case first: EVERY distance identical (an all-ties
        // corpus), slots arriving in DESCENDING order. Ties must resolve to
        // the lowest slots regardless. A bounded heap that evicts only on
        // `d < worst` (dropping the `slot < worst_slot` half of the
        // tie-break) returns the HIGHEST arriving slots here instead -- the
        // exact non-determinism this asserts against. Verified to fail if
        // that half is removed.
        {
            let stride = codes_stride(8);
            let codes = vec![0u8; 6 * stride]; // all identical => all d = 0
            let q = vec![0u8; stride];
            assert_eq!(
                topk_hamming_slots(&q, &codes, [5u32, 4, 3, 2, 1, 0], 2),
                vec![(0, 0), (0, 1)],
                "an all-ties corpus must resolve to the LOWEST slots, not the first-arriving"
            );
        }
        let dim = 24usize; // few bits => MANY ties, the case that matters
        let stride = codes_stride(dim);
        let n = 120usize;
        let codes = synth_codes(n, stride, 0xFEED_BEEF);
        let ascending: Vec<u32> = (10u32..80).collect();
        let mut shuffled = ascending.clone();
        // Deterministic "shuffle": reverse then rotate, so the order is
        // definitely not ascending and definitely not the heap's.
        shuffled.reverse();
        shuffled.rotate_left(17);
        for qi in [0usize, 33] {
            let q = &codes[qi * stride..(qi + 1) * stride];
            for k in [1usize, 5, 30] {
                assert_eq!(
                    topk_hamming_slots(q, &codes, ascending.iter().copied(), k),
                    topk_hamming_slots(q, &codes, shuffled.iter().copied(), k),
                    "query {qi}, k={k}: tie-break must be slot-id based, not arrival-order based"
                );
            }
        }
    }

    /// An out-of-range slot (a torn cell directory) must be SKIPPED, not
    /// panic across the FFI boundary -- the same reasoning as
    /// `ReadOnlyIndex::id_at_slot_checked`. A backend abort under load is
    /// strictly worse than a short result.
    #[test]
    fn topk_hamming_slots_skips_out_of_range_slots() {
        let dim = 16usize;
        let stride = codes_stride(dim);
        let n = 10usize;
        let codes = synth_codes(n, stride, 42);
        let q = codes[0..stride].to_vec();
        // Slots 5, 9 are valid; 10, 999, u32::MAX are not.
        let got = topk_hamming_slots(&q, &codes, [5u32, 10, 9, 999, u32::MAX], 5);
        assert_eq!(
            got.iter().map(|&(_, s)| s).collect::<Vec<_>>(),
            {
                let mut want = vec![
                    (hamming(&q, &codes[5 * stride..6 * stride]), 5u32),
                    (hamming(&q, &codes[9 * stride..10 * stride]), 9u32),
                ];
                want.sort_unstable_by_key(|&(d, s)| (d, s));
                want.iter().map(|&(_, s)| s).collect::<Vec<_>>()
            },
            "only the in-range slots may be scored"
        );
    }

    /// The streamed (out-of-core / IVF) mean MUST be bit-identical to the
    /// whole-corpus mean. `bq_ivf_build_and_write` accumulates per spill
    /// block; `bq_build_and_write` computes over the resident corpus. If
    /// they diverged, the same table would get different sign codes
    /// depending on `maintenance_work_mem`, i.e. the on-disk bytes would
    /// depend on a runtime knob.
    #[test]
    fn streamed_mean_matches_whole_corpus_mean() {
        let dim = 12usize;
        let n = 997usize; // prime, so blocks don't divide it evenly
        let flat: Vec<f32> = (0..n * dim)
            .map(|i| (((i * 37) % 211) as f32) * 0.031 - 3.0)
            .collect();
        let whole = corpus_mean(&flat, dim);
        for block_rows in [1usize, 7, 64, 512, 997, 4096] {
            let mut sums = vec![0.0f64; dim];
            let mut start = 0usize;
            while start < n {
                let rows = (n - start).min(block_rows);
                accumulate_sums(&mut sums, &flat[start * dim..(start + rows) * dim], dim);
                start += rows;
            }
            let streamed = finish_mean(&sums, n);
            assert_eq!(
                whole.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                streamed.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "block_rows={block_rows}: streamed mean must be BIT-identical"
            );
        }
    }

    /// The packed-codes degeneracy check (what the streaming IVF build can
    /// afford) must agree with the centered-corpus one (what the flat build
    /// uses) on every case that matters: the rescued dense-positive corpus,
    /// the unindexable constant corpus, and normal spread data.
    #[test]
    fn packed_degeneracy_matches_centered_degeneracy() {
        let dim = 8usize;
        let stride = codes_stride(dim);
        let cases: Vec<(&str, Vec<f32>)> = vec![
            // Dense-positive with spread: degenerate RAW, fine centered.
            (
                "all_positive_spread",
                (0..5 * dim).map(|i| 10.0 + (i as f32) * 0.5).collect(),
            ),
            // Constant: degenerate even after centering.
            ("constant", vec![7.0f32; 5 * dim]),
            // Zero-centered spread: never degenerate.
            (
                "spread",
                (0..5 * dim)
                    .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
                    .collect(),
            ),
            // Single row: never "degenerate" (nothing to rank against).
            ("one_row", vec![1.0f32; dim]),
        ];
        for (name, flat) in cases {
            let n = flat.len() / dim;
            let mean = corpus_mean(&flat, dim);
            let centered: Vec<f32> = flat
                .chunks_exact(dim)
                .flat_map(|r| center(r, &mean))
                .collect();
            let codes: Vec<u8> = centered
                .chunks_exact(dim)
                .flat_map(|r| pack_signs(r))
                .collect();
            assert_eq!(
                is_degenerate(&centered, dim),
                codes_are_degenerate(&codes, stride, n),
                "case {name}: the two degeneracy checks must agree"
            );
        }
    }
}
