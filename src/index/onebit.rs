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

/// Hamming distance between two packed sign-code rows of equal length.
///
/// Folds 8 bytes at a time through `u64::count_ones` (one `POPCNT` per
/// 8 bytes instead of per byte), with a byte-wise tail for the trailing
/// `len % 8` bytes. That tail loop IS the original scalar kernel and is
/// the ONLY path for codes shorter than 8 bytes (`dim < 64`), so it stays
/// live and covered.
///
/// ## Why wide words and NOT hand-written SIMD intrinsics
///
/// This is a **deliberately CPU-feature-INDEPENDENT** kernel: no
/// `is_x86_feature_detected!`, no `target_feature`, no `unsafe`, no
/// runtime dispatch. Every machine executes the same instruction
/// sequence over the same word decomposition, so it cannot become a
/// second v1.7.3 (where a mis-specialised kernel returned WRONG ANN
/// results on pre-AVX2 CPUs). An AVX2 `pshufb`-nibble-LUT variant WAS
/// written, proven bit-identical, and benchmarked; it captured only the
/// remaining ~27% of the achievable latency reduction and only above
/// `dim >= 512`, losing to this function below that. It was declined.
/// The measurements and the condition that would justify revisiting it
/// are in `docs/ONEBIT_BQ.md` §7.
///
/// ## Why this is bit-identical to the byte-wise count
///
/// Hamming is `popcount(a XOR b)` — a sum over independent bits, so any
/// partition of the bytes into groups gives the same total. `from_ne_bytes`
/// applies the SAME byte permutation to both operands, and XOR is
/// element-wise, so the multiset of XORed bits per word is identical on
/// big- and little-endian alike: the count is endianness-independent.
/// `hamming_agrees_with_bitwise_reference_across_dims` asserts this
/// against a bit-by-bit reference over 5720 random pairs at 143 dims.
///
/// No tail masking is needed: [`pack_signs`] zeroes the unused trailing
/// bits of the last byte (unit-asserted by `tail_bits_are_zero`), so
/// equal-length codes agree on those bits and they contribute 0 to the
/// XOR. This is the same MSB-first convention as `bitvec.rs` and
/// Postgres's `bit` type, so the SQL popcount kernels and this scorer
/// cannot disagree.
///
/// Mismatched lengths (a `debug_assert` violation) truncate to the
/// shorter operand, exactly as the previous `zip`-based kernel did.
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0u32;
    let mut aw = a.chunks_exact(8);
    let mut bw = b.chunks_exact(8);
    for (x, y) in aw.by_ref().zip(bw.by_ref()) {
        // `chunks_exact(8)` yields exactly 8 bytes, so the conversion
        // cannot fail.
        let xv = u64::from_ne_bytes(x.try_into().unwrap());
        let yv = u64::from_ne_bytes(y.try_into().unwrap());
        acc += (xv ^ yv).count_ones();
    }
    // Scalar byte tail: the sole path when `len < 8`.
    for (x, y) in aw.remainder().iter().zip(bw.remainder().iter()) {
        acc += (x ^ y).count_ones();
    }
    acc
}

/// Top-`k` nearest slots to `query_code` by Hamming distance over a flat
/// chain of `n` packed codes, returned as `(distance, slot)` ascending.
///
/// The distance is [`hamming`] (wide-word `POPCNT`, CPU-feature
/// INDEPENDENT); the top-k SELECTION stays scalar on purpose. Vectorising
/// the selection would put the tie-break — which decides the visible
/// result order, since Hamming over `dim` bits has only `dim + 1` distinct
/// values and ties are therefore the common case — inside a lane-shuffle,
/// and the distance loop is where all the measured time goes anyway
/// (`docs/ONEBIT_BQ.md` §7).
///
/// The v1.7.3 incident — where pre-AVX2 CPUs returned WRONG ANN results
/// from a mis-specialised kernel — is why there is no runtime feature
/// dispatch anywhere in this module: every machine runs the identical
/// instruction sequence. See [`hamming`] for why the AVX2 variant that
/// was written and proven bit-identical was still declined.
///
/// Ties break toward the lower slot so results are deterministic (the
/// same reason `partition::rank_nearest` does). Since ties are common,
/// the caller must rerank exactly: see
/// `guc::hi_dim_rerank_candidate_count`, which treats a 1-bit index as
/// high-dim at any `dim` so `hi_dim_rerank = auto` widens the exact
/// rerank window for BQ.
pub fn topk_hamming(query_code: &[u8], codes: &[u8], n: usize, k: usize) -> Vec<(u32, u32)> {
    let stride = query_code.len();
    if k == 0 || n == 0 || stride == 0 {
        return Vec::new();
    }
    debug_assert!(codes.len() >= n * stride);
    // A bounded max-heap of the k best: O(n log k), no full sort of n.
    let mut heap: std::collections::BinaryHeap<(u32, u32)> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    for slot in 0..n {
        let row = &codes[slot * stride..(slot + 1) * stride];
        let d = hamming(query_code, row);
        if heap.len() < k {
            heap.push((d, slot as u32));
        } else if let Some(&(worst, worst_slot)) = heap.peek() {
            // This condition is exactly "`(d, slot)` is lexicographically
            // smaller than the heap's lexicographic max" — the textbook
            // bounded-max-heap top-k, so the result is the k smallest by
            // `(distance, slot)` for ANY visit order.
            //
            // Under the CURRENT ascending order the `d == worst` half
            // never actually fires (measured: 0 firings in 4.2M
            // evaluations over tie-saturated corpora), because a tie is
            // already resolved by arriving later — `heap.peek()` on a
            // `BinaryHeap<(u32, u32)>` is the lexicographic max, so
            // `worst_slot` is the highest slot at distance `worst`, and
            // every slot already in the heap is below the current one.
            // It is kept because it is what makes the tie-break a
            // property of the COMPARISON rather than of the loop order:
            // `topk_tie_break_prefers_the_lower_slot` still passes if the
            // loop is reversed, and fails if either the clause or the
            // order is broken alone. A future chunked/parallel scan can
            // therefore reorder safely.
            if d < worst || (d == worst && (slot as u32) < worst_slot) {
                heap.pop();
                heap.push((d, slot as u32));
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

    // -----------------------------------------------------------------
    // Wide-word Hamming: exact-agreement proof.
    //
    // `hamming` folds 8 bytes at a time through `u64::count_ones`. That
    // is a pure refactor of the byte-wise count — but "pure refactor" is
    // exactly what was believed about the kernel that shipped WRONG ANN
    // results on pre-AVX2 CPUs in v1.7.3. So it is PROVEN here, not
    // asserted: against an independent bit-by-bit reference, over
    // thousands of random pairs, at dims that are and are not multiples
    // of 8 and of 64, including every dim below one word.
    //
    // These are plain `#[test]`s (no cluster), so they run in CI under
    // `cargo pgrx test` on every matrix lane and under a bare
    // `cargo test --lib`.
    // -----------------------------------------------------------------

    /// xorshift64* — deterministic, dependency-free, and far better
    /// distributed than the LCG-high-byte trick used above (which only
    /// varies the top 8 bits per step).
    struct Xs(u64);
    impl Xs {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for c in buf.chunks_mut(8) {
                let v = self.next_u64().to_le_bytes();
                let len = c.len();
                c.copy_from_slice(&v[..len]);
            }
        }
    }

    /// The reference: count differing bits ONE AT A TIME, MSB-first,
    /// reading each bit exactly the way [`unpack_signs`] does. Shares no
    /// code and no word decomposition with [`hamming`], so an error in
    /// either cannot hide in the other.
    fn hamming_bitwise_reference(a: &[u8], b: &[u8], dim: usize) -> u32 {
        (0..dim)
            .map(|i| {
                let m = 0x80u8 >> (i % 8);
                u32::from((a[i / 8] & m) != (b[i / 8] & m))
            })
            .sum()
    }

    /// The byte-wise count this kernel used to be, kept as a second
    /// independent oracle (it decomposes into 1-byte words where
    /// `hamming` uses 8-byte words).
    fn hamming_bytewise(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    /// Dims spanning every alignment case: 1..=130 covers sub-byte,
    /// sub-word, and every `dim % 8` / `dim % 64` residue; the tail
    /// covers realistic embedding widths on both sides of a word
    /// boundary.
    fn agreement_dims() -> Vec<usize> {
        (1usize..=130)
            .chain([
                255, 256, 257, 383, 384, 511, 512, 768, 960, 1000, 1024, 1536, 3072,
            ])
            .collect()
    }

    /// THE PROOF: `hamming` equals a bit-by-bit count, exactly, on
    /// thousands of random code pairs across 143 dims.
    #[test]
    fn hamming_agrees_with_bitwise_reference_across_dims() {
        let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
        let mut trials = 0usize;
        for dim in agreement_dims() {
            let stride = codes_stride(dim);
            for _ in 0..40 {
                let mut a = vec![0u8; stride];
                let mut b = vec![0u8; stride];
                rng.fill(&mut a);
                rng.fill(&mut b);
                // Zero the unused tail bits, exactly as `pack_signs`
                // does, so the fixtures are shaped like real codes.
                let used = dim % 8;
                if used != 0 {
                    let mask = !0u8 << (8 - used);
                    a[stride - 1] &= mask;
                    b[stride - 1] &= mask;
                }
                let want = hamming_bitwise_reference(&a, &b, dim);
                assert_eq!(hamming(&a, &b), want, "wide-word != bitwise, dim={dim}");
                assert_eq!(
                    hamming_bytewise(&a, &b),
                    want,
                    "bytewise != bitwise, dim={dim}"
                );
                // Symmetry and self-distance, for free, on every fixture.
                assert_eq!(hamming(&b, &a), want, "asymmetric, dim={dim}");
                assert_eq!(hamming(&a, &a), 0, "self-distance nonzero, dim={dim}");
                trials += 1;
            }
        }
        assert!(trials >= 5000, "expected thousands of trials, ran {trials}");
    }

    /// Degenerate operands the random fixtures will essentially never
    /// draw: all-zero vs all-ones must be exactly `dim`, and the answer
    /// must not depend on where the word boundary falls.
    #[test]
    fn hamming_extremes_agree_at_every_word_boundary() {
        for dim in [
            1usize, 7, 8, 9, 15, 16, 17, 63, 64, 65, 71, 72, 127, 128, 129,
        ] {
            let stride = codes_stride(dim);
            let zero = vec![0u8; stride];
            let ones = pack_signs(&vec![1.0f32; dim]);
            assert_eq!(
                hamming(&zero, &ones) as usize,
                dim,
                "all-ones vs all-zero must be dim, dim={dim}"
            );
            assert_eq!(
                hamming(&zero, &ones),
                hamming_bitwise_reference(&zero, &ones, dim)
            );
            assert_eq!(hamming(&zero, &zero), 0);
            assert_eq!(hamming(&ones, &ones), 0);
        }
    }

    /// `topk_hamming` must return the identical `(distance, slot)`
    /// sequence — INCLUDING tie order — as a brute-force sort scored by
    /// the independent bit-by-bit reference. Ties are the whole risk
    /// here: Hamming over `dim` bits has only `dim + 1` distinct values,
    /// so an all-zero query at low dim collides constantly (asserted
    /// separately by `topk_fixtures_really_are_tie_dense`).
    #[test]
    fn topk_hamming_agrees_with_bitwise_brute_force_including_ties() {
        let mut rng = Xs(0xDEAD_BEEF_CAFE_1234);
        let mut cases = 0usize;
        for &dim in &[3usize, 8, 12, 16, 31, 64, 65, 100, 128, 768, 1536] {
            let stride = codes_stride(dim);
            let used = dim % 8;
            let mask = if used == 0 {
                0xffu8
            } else {
                !0u8 << (8 - used)
            };
            for &n in &[1usize, 2, 7, 50, 333] {
                let mut codes = vec![0u8; n * stride];
                rng.fill(&mut codes);
                for s in 0..n {
                    codes[(s + 1) * stride - 1] &= mask;
                }
                // Queries: a corpus member (guarantees a distance-0 hit),
                // a fresh random code, and the all-zero code (maximally
                // tie-saturated, since the distance is then just the
                // row's own popcount).
                let mut queries: Vec<Vec<u8>> = vec![codes[..stride].to_vec()];
                let mut fresh = vec![0u8; stride];
                rng.fill(&mut fresh);
                fresh[stride - 1] &= mask;
                queries.push(fresh);
                queries.push(vec![0u8; stride]);
                for q in &queries {
                    let mut want: Vec<(u32, u32)> = (0..n)
                        .map(|s| {
                            (
                                hamming_bitwise_reference(
                                    q,
                                    &codes[s * stride..(s + 1) * stride],
                                    dim,
                                ),
                                s as u32,
                            )
                        })
                        .collect();
                    want.sort_unstable_by_key(|&(d, s)| (d, s));
                    for &k in &[1usize, 3, 10, 64, 400] {
                        let mut expect = want.clone();
                        expect.truncate(k);
                        assert_eq!(
                            topk_hamming(q, &codes, n, k),
                            expect,
                            "topk != bitwise brute force, dim={dim} n={n} k={k}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 500, "expected many top-k cases, ran {cases}");
    }

    /// Guard against the test above passing vacuously: confirm the
    /// all-zero-query fixtures really are saturated with ties, so the
    /// lower-slot tie-break is genuinely under test.
    #[test]
    fn topk_fixtures_really_are_tie_dense() {
        let dim = 16usize;
        let stride = codes_stride(dim);
        let n = 333usize;
        let mut rng = Xs(7);
        let mut codes = vec![0u8; n * stride];
        rng.fill(&mut codes);
        let q = vec![0u8; stride];
        let mut d: Vec<u32> = (0..n)
            .map(|s| hamming(&q, &codes[s * stride..(s + 1) * stride]))
            .collect();
        d.sort_unstable();
        d.dedup();
        assert!(
            d.len() <= dim + 1 && d.len() * 10 < n,
            "expected heavy ties: {} distinct distances over {n} rows",
            d.len()
        );
    }

    /// The tie-break contract, isolated: on an ALL-EQUAL-distance corpus
    /// every slot ties, so the `k` returned slots must be exactly the
    /// `k` LOWEST, in ascending order. This is the property the scan's
    /// determinism rests on, and it holds for ANY visit order because the
    /// heap condition is a full lexicographic `(distance, slot)`
    /// comparison — verified by mutation: reversing the loop alone still
    /// passes, while dropping the tie clause or flipping its direction
    /// fails here.
    #[test]
    fn topk_tie_break_prefers_the_lower_slot() {
        let dim = 64usize;
        let stride = codes_stride(dim);
        let n = 200usize;
        // Every row identical => every distance identical => total ties.
        let codes = vec![0xA5u8; n * stride];
        let q = vec![0x5Au8; stride];
        for k in [1usize, 2, 7, 50, 199, 200] {
            let got = topk_hamming(&q, &codes, n, k);
            let want: Vec<(u32, u32)> = (0..k as u32).map(|s| (dim as u32, s)).collect();
            assert_eq!(got, want, "all-ties must yield the k lowest slots, k={k}");
        }
        // Same, with a distinct better row planted at a HIGH slot: it must
        // come first, then the lowest of the tied remainder.
        let mut codes2 = codes.clone();
        codes2[190 * stride..191 * stride].copy_from_slice(&q);
        let got = topk_hamming(&q, &codes2, n, 3);
        assert_eq!(got, vec![(0, 190), (dim as u32, 0), (dim as u32, 1)]);
    }

    /// End-to-end at the shape the scan path actually uses: pack real
    /// centered f32 vectors, then confirm the packed-code Hamming equals
    /// a direct sign-disagreement count on the f32s. This ties the kernel
    /// back to the *semantics* (how many coordinates disagree in sign),
    /// not just to another bit-counting loop.
    #[test]
    fn hamming_counts_sign_disagreements_on_real_vectors() {
        let mut rng = Xs(0x0BAD_C0DE_0BAD_C0DE);
        for dim in [1usize, 5, 8, 63, 64, 65, 100, 768] {
            for _ in 0..50 {
                let mk = |rng: &mut Xs| -> Vec<f32> {
                    (0..dim)
                        .map(|_| {
                            // Values in [-1, 1), including exact 0.0 often
                            // enough to exercise the `> 0.0` rule.
                            let r = (rng.next_u64() >> 40) as i64 - 8_388_608;
                            (r / 4096) as f32 / 512.0
                        })
                        .collect()
                };
                let x = mk(&mut rng);
                let y = mk(&mut rng);
                let want: u32 = x
                    .iter()
                    .zip(&y)
                    .map(|(&p, &q)| u32::from((p > 0.0) != (q > 0.0)))
                    .sum();
                assert_eq!(
                    hamming(&pack_signs(&x), &pack_signs(&y)),
                    want,
                    "packed Hamming != sign-disagreement count, dim={dim}"
                );
            }
        }
    }
}
