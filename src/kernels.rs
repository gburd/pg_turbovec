//! Pure-Rust math kernels — no Postgres dependency.
//!
//! All distance functions in `distance.rs` and the normalisation
//! helper in `normalize.rs` delegate to these. Keeping the kernels
//! Postgres-free means we can exercise them under plain `cargo test`,
//! prove their correctness in isolation, and benchmark them with
//! `criterion` without booting a cluster.
//!
//! All functions assume the caller has already validated equal
//! lengths / dimensionality (on a mismatch they use the shorter one).
//!
//! # Precision
//!
//! `dot`, `norm2`, `l2_sq` and `l1_abs` sum in 16 independent `f32`
//! lanes, which LLVM vectorizes (SSE2 on the portable x86-64 target,
//! NEON on aarch64), and flush the lanes into `f64` every 128 elements.
//! Whatever the dimension, a term goes through at most 9 `f32`
//! roundings before it reaches `f64` (8 for `dot`, `norm2`, `l1_abs`),
//! so the error is at most `γ₉ ≈ 5.4e-7` times `Σ|term|`, plus ~1e-14
//! from the `f64` part (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, §3.1, §4.2). For `norm2`, `l2_sq` and `l1_abs` every
//! term is non-negative, so that is a relative error. For `dot` it is
//! relative to `Σ|aᵢbᵢ|`, the usual condition-scaled bound: near an
//! exact zero no finite-precision dot product, the old serial `f64` one
//! included, has a bounded relative error. Measured max against an
//! exact compensated-`f64` reference, dims 1 to 16 000, unit-norm and
//! ±1e3 data: 7.4e-8; adversarial constant vectors: 2.3e-7
//! (`kernels_match_exact_reference`).
//!
//! A lane result that is non-finite or below 1e-20 (an `f32` term may
//! have overflowed or underflowed: inputs beyond ~1e19 or below ~1e-19)
//! is recomputed with the old serial `f64` loop, so such inputs give the
//! same answers as before. Dimensions below 16 never enter the `f32`
//! lanes and are bit-identical to the old serial `f64` loop. A cosine
//! distance below 1e-3, where `1 - cos θ` cancels, is recomputed in
//! serial `f64` too (see [`cosine_distance_with_qnorm`]).
//!
//! What this does NOT cover: `dot` on near-duplicate unit vectors. Two
//! inner products that differ by less than ~1e-7 relative (e.g. a row
//! and its 1e-10-distant near-duplicate under `<#>`) can now compare in
//! either order; the old serial `f64` sum resolved ~1e-16. pgvector's
//! `f32` inner product has the same limit. L2 has no cancellation (it
//! subtracts before squaring) and keeps full relative precision.
//!
//! These kernels used to accumulate in serial `f64` on the grounds that
//! "`f32` accumulation drops 2–3 decimal digits on corpora of ≥ 10⁶
//! vectors". That conflated a per-vector sum (at most 16 000 terms) with
//! a corpus-sized one. The single dependent add chain cannot be
//! reordered by LLVM, so it was ~4-10x slower at 1024-d.

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(a, b, |x, y| x * y, |x, y| x * y)
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(
        a,
        b,
        |x, y| {
            let d = x - y;
            d * d
        },
        |x, y| {
            let d = x - y;
            d * d
        },
    )
}

#[inline]
pub fn l1_abs(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(a, b, |x, y| (x - y).abs(), |x, y| (x - y).abs())
}

/// Squared L2 norm.
#[inline]
pub fn norm2(a: &[f32]) -> f64 {
    lane_sum(a, a, |x, _| x * x, |x, _| x * x)
}

/// Cosine distance: `1 - cos θ`. Returns `NaN` if either operand has
/// zero L2 norm. Clamps `cos θ` to `[-1, 1]` to defend against
/// numerical drift past the unit circle.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    cosine_distance_with_qnorm(a, b, norm2(b))
}

/// [`cosine_distance`] with the squared norm of `b` supplied by the
/// caller (`qnorm2 == norm2(b)`), so a scan scoring one query against
/// many rows computes it once. Same semantics: `NaN` if either norm is
/// zero, `cos θ` clamped to `[-1, 1]`. The norm of `a` is always
/// computed; rows are not assumed to be unit length.
///
/// `1 - cos θ` cancels as the distance approaches 0, and the lane
/// kernels' absolute error in `cos θ` (at most ~1e-6, measured ≤ 2.1e-7)
/// would then swamp it: near-duplicates 1e-10 apart all come out as
/// exactly 0 and a self-match query returns an arbitrary one of them. A
/// result below [`COSINE_EXACT_BELOW`] is therefore recomputed with the
/// old serial `f64` code and is bit-identical to it. Neighbours in real
/// embeddings (cosine distance ≳ 0.05) never take that path.
///
/// `dot` and `norm2(a)` are two passes rather than one fused loop on
/// purpose: on the portable SSE2 target the fused loop needs 32 live
/// accumulator registers, spills, and measured ~20% slower than two
/// passes over a vector that is already in L1.
#[inline]
pub fn cosine_distance_with_qnorm(a: &[f32], b: &[f32], qnorm2: f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let na = norm2(a);
    if na == 0.0 || qnorm2 == 0.0 {
        return f64::NAN;
    }
    let dist = 1.0 - (dot(a, b) / (na.sqrt() * qnorm2.sqrt())).clamp(-1.0, 1.0);
    if dist < COSINE_EXACT_BELOW {
        cosine_distance_serial(a, b)
    } else {
        dist
    }
}

/// Cosine distances below this are recomputed in serial `f64` (see
/// [`cosine_distance_with_qnorm`]). The fast path's absolute error is
/// ≤ ~1e-6, so above this it is ≤ 0.1% relative.
const COSINE_EXACT_BELOW: f64 = 1e-3;

/// The pre-2.11.1 cosine distance, bit for bit: three serial `f64` sums.
#[cold]
#[inline(never)]
fn cosine_distance_serial(a: &[f32], b: &[f32]) -> f64 {
    let na = serial_sum(a, a, |x, _| x * x);
    let nb = serial_sum(b, b, |x, _| x * x);
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    let cos = (serial_sum(a, b, |x, y| x * y) / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
    1.0 - cos
}

/// Independent `f32` accumulator lanes. 16 = four 128-bit registers,
/// enough independent add chains to hide the FP-add latency.
const LANES: usize = 16;
/// Elements summed in `f32` before the lanes are flushed into `f64`.
const BLOCK: usize = 8 * LANES;

/// Below this magnitude a lane result may have lost terms to `f32`
/// underflow. Each `f32` operation loses at most `2^-126 ≈ 1.2e-38`
/// absolute to underflow, even with flush-to-zero set (which a library
/// built with `-ffast-math` can turn on in the backend). With at most
/// ~32 000 operations (16 000 dims) that is < 4e-34 absolute, so a
/// result above 1e-20 carries < 4e-14 relative underflow error.
const LANE_TINY: f64 = 1e-20;

/// `Σ term(a[i], b[i])`: blocks of [`BLOCK`] in `f32` lanes flushed into
/// `f64`, then 16-wide chunks, then the last < 16 elements as exact
/// `f64` terms via `exact`.
///
/// If the lane result is non-finite (an `f32` term or lane overflowed:
/// with finite inputs that is the only way to get inf/NaN, and it cannot
/// be masked), zero, or below [`LANE_TINY`] (terms may have
/// underflowed), it is recomputed by [`serial_sum`], the old serial
/// `f64` loop, which returns its exact old bits. Real embeddings never
/// take that branch; it keeps inputs beyond ~1e19 or below ~1e-19, which
/// the type accepts, giving the same answers as before.
#[inline]
fn lane_sum(
    a: &[f32],
    b: &[f32],
    term: impl Fn(f32, f32) -> f32 + Copy,
    exact: impl Fn(f64, f64) -> f64 + Copy,
) -> f64 {
    let n = a.len().min(b.len());
    let (blocks_a, tail_a) = a[..n].as_chunks::<BLOCK>();
    let (blocks_b, tail_b) = b[..n].as_chunks::<BLOCK>();
    let mut wide = [0.0_f64; LANES];
    for (x, y) in blocks_a.iter().zip(blocks_b) {
        let p = block_lanes(x, y, term);
        for i in 0..LANES {
            wide[i] += f64::from(p[i]);
        }
    }
    let (chunks_a, rest_a) = tail_a.as_chunks::<LANES>();
    let (chunks_b, rest_b) = tail_b.as_chunks::<LANES>();
    let mut lanes = [0.0_f32; LANES];
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for i in 0..LANES {
            lanes[i] += term(x[i], y[i]);
        }
    }
    let mut s = 0.0_f64;
    for i in 0..LANES {
        s += wide[i] + f64::from(lanes[i]);
    }
    for (x, y) in rest_a.iter().zip(rest_b) {
        s += exact(f64::from(*x), f64::from(*y));
    }
    if s.abs() >= LANE_TINY && s.abs() < f64::INFINITY {
        s
    } else {
        serial_sum(&a[..n], &b[..n], exact)
    }
}

/// One [`BLOCK`] in [`LANES`] independent `f32` accumulators.
///
/// Kept out of line on purpose: compiled on its own, the fixed-size body
/// vectorizes to packed SIMD on both x86-64 (SSE2) and aarch64 (NEON) for
/// every `term`. Inlined into its caller, LLVM's SLP vectorizer was seen
/// to fall back to 2-wide shuffled loads for `dot` alone, at 2.2x the
/// cost. One call per 128 elements is noise next to that.
#[inline(never)]
fn block_lanes(x: &[f32; BLOCK], y: &[f32; BLOCK], term: impl Fn(f32, f32) -> f32) -> [f32; LANES] {
    let mut lanes = [0.0_f32; LANES];
    for k in (0..BLOCK).step_by(LANES) {
        for i in 0..LANES {
            lanes[i] += term(x[k + i], y[k + i]);
        }
    }
    lanes
}

/// The pre-2.11.1 kernel: one serial `f64` add chain.
#[cold]
#[inline(never)]
fn serial_sum(a: &[f32], b: &[f32], exact: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        acc += exact(f64::from(*x), f64::from(*y));
    }
    acc
}

/// Write a unit-normalised copy of `src` into `dst`. If `src` is the
/// zero vector, `dst` is filled with `src` unchanged. Returns the
/// L2 norm of the input (caller may want it for further bookkeeping).
///
/// The norm is summed in serial `f64`, not with the lane kernel: the
/// output is what gets quantized and persisted, so it must stay
/// bit-identical across releases (an ulp of difference can move a code
/// across a quantization boundary). It runs once per row or query, not
/// per candidate, so its speed does not matter.
pub fn normalise_into(dst: &mut [f32], src: &[f32]) -> f64 {
    debug_assert_eq!(dst.len(), src.len());
    let n2: f64 = src
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .fold(0.0, |s, t| s + t);
    if n2 == 0.0 {
        dst.copy_from_slice(src);
        return 0.0;
    }
    let norm = n2.sqrt();
    // Divide in f64 and cast each RESULT to f32. Casting the reciprocal
    // `(1.0/norm) as f32` first overflows to +inf when `norm` is a tiny
    // (but nonzero) f64 — a vector of near-underflow elements — which
    // then poisons every element with inf (norm2 == inf). Per-element
    // f64 division keeps each `src/norm` finite (|src/norm| <= |src|
    // since norm >= |src_max| for a real vector), so the f32 cast is
    // always in range. `normalise_on_insert` runs this on every row.
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = (f64::from(*s) / norm) as f32;
    }
    norm
}

/// Allocate a unit-normalised copy of `src`.
pub fn normalise_to_vec(src: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.len()];
    normalise_into(&mut out, src);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Slightly looser tolerance for f32 round-tripped through f64 —
    /// 0.2 is not exactly representable in binary, so 3*0.2 + 4*0.2
    /// drifts ~1e-7 from 5.0.
    fn approx_f32(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn dot_basic() {
        assert!(approx(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
        assert!(approx(dot(&[0.0; 4], &[1.0; 4]), 0.0));
    }

    #[test]
    fn l2_basic() {
        assert!(approx(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0));
        assert!(approx(l2_sq(&[1.0; 8], &[1.0; 8]), 0.0));
    }

    #[test]
    fn l1_basic() {
        assert!(approx(l1_abs(&[0.0, 0.0], &[3.0, 4.0]), 7.0));
        assert!(approx(l1_abs(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0));
    }

    #[test]
    fn norm2_basic() {
        assert!(approx(norm2(&[3.0, 4.0]), 25.0));
        assert!(approx(norm2(&[]), 0.0));
    }

    /// Regression: a tiny-but-nonzero-norm vector must normalise to a
    /// FINITE unit vector, not +inf. The old `(1.0/norm) as f32`
    /// overflowed to +inf when `norm` was a tiny f64 (elements near f32
    /// underflow), poisoning every element; per-element f64 division
    /// fixes it. `normalise_on_insert` runs on every indexed row, so an
    /// inf here would corrupt the codes.
    #[test]
    fn normalise_tiny_norm_stays_finite() {
        // 8 elements each ~1e-22: norm ~ 2.8e-22 (nonzero, doesn't
        // underflow n2 to 0), reciprocal ~3.5e21 which is FINITE as f32
        // only if we divide in f64 (1/2.8e-22 = 3.5e21 < f32::MAX 3.4e38,
        // actually fine here) — use a smaller value to force the old
        // overflow: 1e-30 elements -> norm ~2.8e-30 -> 1/norm ~3.5e29
        // (still < f32 max)… the true overflow is a LARGE-dim tiny-elem
        // vector. Construct 256 elements of 1e-20: n2 = 256*1e-40 = 2.56e-38
        // -> norm = 1.6e-19 -> 1/norm = 6.25e18 (finite). The genuine
        // reciprocal-overflow case the property test hit: a vector whose
        // norm sqrt is < ~2.9e-39 so 1/norm > f32::MAX. Build it directly:
        let v = vec![1.0e-23_f32; 4]; // n2 = 4e-46, norm = 2e-23, 1/norm = 5e22 (finite f32)
        let out = normalise_to_vec(&v);
        let n = norm2(&out).sqrt();
        assert!(
            out.iter().all(|x| x.is_finite()),
            "normalised elements must be finite, got {out:?}"
        );
        assert!(
            (n - 1.0).abs() < 1e-3 || n == 0.0,
            "tiny-norm vector must normalise to unit or zero, got norm {n}"
        );
        // The actual f32-reciprocal-overflow trigger: norm so small that
        // 1.0/norm > f32::MAX (3.4e38), i.e. norm < 2.94e-39. A single
        // element of 2e-39 gives norm 2e-39 -> old (1/norm)as f32 = +inf.
        let tiny = vec![2.0e-39_f32, 0.0, 0.0, 0.0];
        let ot = normalise_to_vec(&tiny);
        assert!(
            ot.iter().all(|x| x.is_finite()),
            "reciprocal-overflow input must not produce inf, got {ot:?}"
        );
    }

    #[test]
    fn cosine_basic() {
        assert!(approx(cosine_distance(&[1.0, 0.0], &[1.0, 0.0]), 0.0));
        assert!(approx(cosine_distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0));
        assert!(approx(cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]), 2.0));
        // zero -> NaN
        assert!(cosine_distance(&[0.0; 3], &[1.0, 2.0, 3.0]).is_nan());
    }

    #[test]
    fn normalise_unit_norm() {
        let v = normalise_to_vec(&[3.0, 4.0]);
        assert!(approx_f32(norm2(&v).sqrt(), 1.0));
        // 3-4-5 triangle: components become 0.6 and 0.8.
        assert!(approx_f32(f64::from(v[0]), 0.6));
        assert!(approx_f32(f64::from(v[1]), 0.8));
    }

    #[test]
    fn normalise_zero_passthrough() {
        let v = normalise_to_vec(&[0.0; 5]);
        assert_eq!(v, vec![0.0; 5]);
    }

    // -----------------------------------------------------------------
    // Precision / equivalence vs an exact reference. The oracle is
    // Neumaier-compensated f64 summation of exact f64 terms (an f32*f32
    // product is exact in f64), independent of the code under test.
    // -----------------------------------------------------------------

    /// Neumaier-compensated sum of `term(x, y)` over the pairs.
    #[allow(clippy::many_single_char_names)]
    fn oracle(a: &[f32], b: &[f32], term: impl Fn(f64, f64) -> f64) -> f64 {
        let (mut s, mut c) = (0.0_f64, 0.0_f64);
        for (x, y) in a.iter().zip(b) {
            let t = term(f64::from(*x), f64::from(*y));
            let u = s + t;
            c += if s.abs() >= t.abs() {
                (s - u) + t
            } else {
                (t - u) + s
            };
            s = u;
        }
        s + c
    }

    // The pre-2.11.1 kernels, verbatim: serial f64 accumulation.
    fn dot_old(a: &[f32], b: &[f32]) -> f64 {
        let mut acc: f64 = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += f64::from(*x) * f64::from(*y);
        }
        acc
    }
    fn l2_sq_old(a: &[f32], b: &[f32]) -> f64 {
        let mut acc: f64 = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = f64::from(*x) - f64::from(*y);
            acc += d * d;
        }
        acc
    }
    fn l1_abs_old(a: &[f32], b: &[f32]) -> f64 {
        let mut acc: f64 = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += (f64::from(*x) - f64::from(*y)).abs();
        }
        acc
    }
    fn norm2_old(a: &[f32]) -> f64 {
        let mut acc: f64 = 0.0;
        for x in a {
            acc += f64::from(*x) * f64::from(*x);
        }
        acc
    }
    fn cosine_old(a: &[f32], b: &[f32]) -> f64 {
        let na = norm2_old(a);
        let nb = norm2_old(b);
        if na == 0.0 || nb == 0.0 {
            return f64::NAN;
        }
        let cos = (dot_old(a, b) / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
        1.0 - cos
    }

    /// Below 16 dims nothing enters the f32 lanes: results are
    /// bit-identical to the old serial f64 kernels, so small-dim SQL
    /// outputs and tie orders cannot move.
    #[test]
    #[allow(clippy::float_cmp)] // exact results are the point
    fn small_dims_bit_identical_to_old_kernels() {
        let mut rng = Rng(42);
        for kind in ["unit", "signed1e3", "positive1e3"] {
            for dim in 0..16 {
                for _ in 0..50 {
                    let a = rng.vector(kind, dim);
                    let b = rng.vector(kind, dim);
                    assert_eq!(dot(&a, &b).to_bits(), dot_old(&a, &b).to_bits());
                    assert_eq!(norm2(&a).to_bits(), norm2_old(&a).to_bits());
                    assert_eq!(l2_sq(&a, &b).to_bits(), l2_sq_old(&a, &b).to_bits());
                    assert_eq!(l1_abs(&a, &b).to_bits(), l1_abs_old(&a, &b).to_bits());
                    let (c, co) = (cosine_distance(&a, &b), cosine_old(&a, &b));
                    assert!(c.to_bits() == co.to_bits() || (c.is_nan() && co.is_nan()));
                }
            }
        }
    }

    /// `normalise_into` must produce the same f32 bits as before: its
    /// output is what gets quantized and persisted, so a change here
    /// would make the same row encode differently across versions.
    #[test]
    fn normalise_bit_identical_to_old() {
        let mut rng = Rng(9);
        for &dim in &PRECISION_DIMS {
            for kind in ["unit", "signed1e3", "positive1e3", "constant"] {
                let v = rng.vector(kind, dim);
                let n = norm2_old(&v).sqrt();
                let want: Vec<u32> = v
                    .iter()
                    .map(|x| ((f64::from(*x) / n) as f32).to_bits())
                    .collect();
                let got: Vec<u32> = normalise_to_vec(&v).iter().map(|x| x.to_bits()).collect();
                assert_eq!(got, want, "{kind} dim {dim}");
            }
        }
    }

    /// Deterministic xorshift64 PRNG (no test-time dependency on `rand`).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unif(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1_u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u = self.unif().max(1e-300);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unif()).cos()
        }
        fn vector(&mut self, kind: &str, dim: usize) -> Vec<f32> {
            match kind {
                "unit" => {
                    let v: Vec<f64> = (0..dim).map(|_| self.gauss()).collect();
                    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.iter().map(|x| (x / n) as f32).collect()
                }
                "signed1e3" => (0..dim)
                    .map(|_| ((self.unif() * 2.0 - 1.0) * 1e3) as f32)
                    .collect(),
                "positive1e3" => (0..dim).map(|_| (self.unif() * 1e3) as f32).collect(),
                // Same value in every slot: every partial sum rounds the
                // same way, the worst case for a long f32 accumulation.
                "constant" => vec![(0.1 + self.unif()) as f32; dim],
                _ => unreachable!(),
            }
        }
    }

    const PRECISION_DIMS: [usize; 14] = [
        1, 7, 8, 15, 16, 17, 31, 64, 384, 768, 1024, 1536, 3072, 16_000,
    ];

    /// Bound on the error of every kernel, relative to the sum of the
    /// absolute values of its terms. For norm2 / l2_sq / l1_abs every
    /// term is non-negative, so this IS the ordinary relative error. For
    /// dot it is the standard condition-scaled error: relative error
    /// against a near-zero dot (near-orthogonal vectors) is unbounded for
    /// any finite-precision sum, the old f64 one included.
    const KERNEL_REL_BOUND: f64 = 1e-6;
    /// Absolute bound on `cosine_distance` (a value in [0, 2]).
    const COSINE_ABS_BOUND: f64 = 2e-6;

    #[test]
    fn kernels_match_exact_reference() {
        use std::fmt::Write as _;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut report = String::new();
        for kind in ["unit", "signed1e3", "positive1e3", "constant"] {
            for &dim in &PRECISION_DIMS {
                let trials = if dim >= 3072 { 8 } else { 40 };
                let mut worst = [0.0_f64; 5];
                for _ in 0..trials {
                    let a = rng.vector(kind, dim);
                    let b = rng.vector(kind, dim);
                    let abs_dot = oracle(&a, &b, |x, y| (x * y).abs());
                    let errs = [
                        (dot(&a, &b) - oracle(&a, &b, |x, y| x * y)).abs() / abs_dot,
                        {
                            let e = oracle(&a, &a, |x, _| x * x);
                            (norm2(&a) - e).abs() / e
                        },
                        {
                            let e = oracle(&a, &b, |x, y| (x - y) * (x - y));
                            if e == 0.0 {
                                l2_sq(&a, &b)
                            } else {
                                (l2_sq(&a, &b) - e).abs() / e
                            }
                        },
                        {
                            let e = oracle(&a, &b, |x, y| (x - y).abs());
                            if e == 0.0 {
                                l1_abs(&a, &b)
                            } else {
                                (l1_abs(&a, &b) - e).abs() / e
                            }
                        },
                        {
                            let d = oracle(&a, &b, |x, y| x * y);
                            let na = oracle(&a, &a, |x, _| x * x);
                            let nb = oracle(&b, &b, |x, _| x * x);
                            let exact = 1.0 - (d / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
                            (cosine_distance(&a, &b) - exact).abs()
                        },
                    ];
                    for (w, e) in worst.iter_mut().zip(errs) {
                        assert!(e.is_finite(), "{kind} dim {dim}: non-finite error");
                        *w = w.max(e);
                    }
                }
                writeln!(
                    report,
                    "{kind:>11} dim {dim:>5}: dot {:.1e} norm2 {:.1e} l2_sq {:.1e} l1 {:.1e} cosine(abs) {:.1e}",
                    worst[0], worst[1], worst[2], worst[3], worst[4]
                )
                .unwrap();
                for (name, w) in ["dot", "norm2", "l2_sq", "l1_abs"].iter().zip(&worst[..4]) {
                    assert!(
                        *w <= KERNEL_REL_BOUND,
                        "{name} {kind} dim {dim}: rel err {w:e} > {KERNEL_REL_BOUND:e}"
                    );
                }
                assert!(
                    worst[4] <= COSINE_ABS_BOUND,
                    "cosine {kind} dim {dim}: abs err {:e} > {COSINE_ABS_BOUND:e}",
                    worst[4]
                );
            }
        }
        println!("{report}");
    }

    /// Zero vector, empty slices, length 1: exact, and cosine is NaN.
    #[test]
    #[allow(clippy::float_cmp)] // exact results are the point
    fn kernels_zero_empty_and_singleton() {
        let z = [0.0_f32; 33];
        let o = [1.0_f32; 33];
        assert_eq!(dot(&z, &o), 0.0);
        assert_eq!(norm2(&z), 0.0);
        assert_eq!(l2_sq(&z, &z), 0.0);
        assert_eq!(l1_abs(&z, &z), 0.0);
        assert!(cosine_distance(&z, &o).is_nan());
        assert!(cosine_distance(&o, &z).is_nan());
        assert_eq!(dot(&[], &[]), 0.0);
        assert_eq!(norm2(&[]), 0.0);
        assert_eq!(l2_sq(&[], &[]), 0.0);
        assert_eq!(l1_abs(&[], &[]), 0.0);
        assert!(cosine_distance(&[], &[]).is_nan());
        assert_eq!(dot(&[-3.0], &[5.0]), -15.0);
        assert_eq!(norm2(&[-3.0]), 9.0);
        assert_eq!(l2_sq(&[-3.0], &[5.0]), 64.0);
        assert_eq!(l1_abs(&[-3.0], &[5.0]), 8.0);
        assert_eq!(cosine_distance(&[-3.0], &[5.0]), 2.0);
        assert_eq!(cosine_distance(&[3.0], &[5.0]), 0.0);
    }

    /// `1 - cos θ` cancels near 0, so an f32-lane cosine (absolute error
    /// ~1e-7) cannot resolve near-duplicates: rows 3e-10 apart all come
    /// out as exactly 0 and a self-match query (`ORDER BY emb <=> (SELECT
    /// emb ... WHERE id = k)`) returns an arbitrary near-duplicate.
    /// Pure-Rust reproduction of the `onebit_all_positive_corpus_*`
    /// fixture, where row 29 is 3.3e-10 from row 100. Below the
    /// near-duplicate threshold the result must be the old serial f64
    /// value, bit for bit.
    #[test]
    fn cosine_resolves_near_duplicates_like_old() {
        let row = |g: i64| -> Vec<f32> {
            (1..=32)
                .map(|s| (10.0 + ((g * 31 + s * 17) % 100) as f64 / 100.0) as f32)
                .collect()
        };
        let q = row(100);
        let rows: Vec<Vec<f32>> = (1..=500).map(row).collect();
        let nearest = |f: &dyn Fn(&[f32], &[f32]) -> f64| -> i64 {
            let mut best = (f64::INFINITY, 0_i64);
            for (i, r) in rows.iter().enumerate() {
                let d = f(r, &q);
                if d < best.0 {
                    best = (d, i as i64 + 1);
                }
            }
            best.1
        };
        assert_eq!(nearest(&cosine_old), 100, "fixture: old kernel finds self");
        assert_eq!(
            nearest(&cosine_distance),
            100,
            "new kernel must find self too"
        );
        let qn = norm2(&q);
        assert_eq!(
            nearest(&|a, b| cosine_distance_with_qnorm(a, b, qn)),
            100,
            "with_qnorm must find self too"
        );
        // Random near-duplicates at every dim: perturbations from 1e-7 to
        // 1e-3 relative, so true distances span ~1e-14 to ~1e-6.
        let mut rng = Rng(11);
        for &dim in &PRECISION_DIMS {
            for kind in ["unit", "signed1e3", "positive1e3"] {
                let a = rng.vector(kind, dim);
                for eps in [1e-7, 1e-5, 1e-3] {
                    let b: Vec<f32> = a
                        .iter()
                        .map(|x| x * (1.0 + (eps * (rng.unif() * 2.0 - 1.0)) as f32))
                        .collect();
                    for (x, y) in [(&a, &b), (&b, &a), (&a, &a)] {
                        let (new, old) = (cosine_distance(x, y), cosine_old(x, y));
                        assert_eq!(new.to_bits(), old.to_bits(), "{kind} dim {dim} eps {eps}");
                        let qn = norm2(y);
                        let nq = cosine_distance_with_qnorm(x, y, qn);
                        assert_eq!(nq.to_bits(), old.to_bits(), "{kind} dim {dim} eps {eps}");
                    }
                }
            }
        }
    }

    /// Cosine stays inside [0, 2] for parallel / antiparallel inputs,
    /// where rounding pushes |cos| past 1 without the clamp.
    #[test]
    fn cosine_clamps_to_valid_range() {
        let mut rng = Rng(7);
        for dim in [3, 100, 1024, 1537] {
            let a = rng.vector("signed1e3", dim);
            let neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let scaled: Vec<f32> = a.iter().map(|x| x * 3.0).collect();
            for (b, want) in [(&a, 0.0), (&scaled, 0.0), (&neg, 2.0)] {
                let d = cosine_distance(&a, b);
                assert!((0.0..=2.0).contains(&d), "dim {dim}: {d} outside [0,2]");
                assert!(
                    (d - want).abs() <= COSINE_ABS_BOUND,
                    "dim {dim}: {d} vs {want}"
                );
            }
        }
    }

    /// Magnitudes whose squares leave the f32 range (overflow above
    /// ~1.8e19, underflow / subnormal below ~1e-19) and subnormal inputs
    /// must give the same FINITE answers the f64 kernels always gave:
    /// `normalise_on_insert` and every distance operator run through
    /// these, and an inf / 0 here would silently mis-encode or mis-rank.
    #[test]
    fn kernels_survive_extreme_magnitudes() {
        let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![1e15; 37], vec![-1e15; 37]),
            (vec![1e20; 37], vec![1e20; 37]),
            (vec![3e38, -3e38, 1.0], vec![-3e38, 3e38, 2.0]),
            (vec![1e-23; 37], vec![1e-23; 37]),
            (vec![1e-40; 37], vec![2e-40; 37]),
            (vec![f32::MIN_POSITIVE; 130], vec![f32::MIN_POSITIVE; 130]),
            (
                {
                    let mut v = vec![1e-30_f32; 300];
                    v[7] = 1e20;
                    v
                },
                vec![1e19; 300],
            ),
            // Most terms underflow f32 but the sum is dominated by
            // normal ones: stays on the lane path, error ~1e-48.
            (
                {
                    let mut v = vec![1e-25_f32; 300];
                    v[299] = 1.0;
                    v
                },
                vec![1e-25; 300],
            ),
        ];
        for (a, b) in &cases {
            let pairs = [
                ("dot", dot(a, b), oracle(a, b, |x, y| x * y)),
                ("norm2", norm2(a), oracle(a, a, |x, _| x * x)),
                ("l2_sq", l2_sq(a, b), oracle(a, b, |x, y| (x - y) * (x - y))),
                ("l1_abs", l1_abs(a, b), oracle(a, b, |x, y| (x - y).abs())),
            ];
            let abs_dot = oracle(a, b, |x, y| (x * y).abs());
            for (name, got, want) in pairs {
                let scale = if name == "dot" { abs_dot } else { want };
                assert!(got.is_finite(), "{name}({:e}..) = {got}", a[0]);
                assert!(
                    (got - want).abs() <= KERNEL_REL_BOUND * scale,
                    "{name}({:e}..): {got:e} vs exact {want:e}",
                    a[0]
                );
            }
            let c = cosine_distance(a, b);
            let co = cosine_old(a, b);
            assert!(
                c.is_finite() == co.is_finite(),
                "cosine({:e}..): {c} vs old {co}",
                a[0]
            );
            assert!(
                (c - co).abs() <= COSINE_ABS_BOUND,
                "cosine({:e}..): {c} vs old {co}",
                a[0]
            );
            let unit = normalise_to_vec(a);
            let n = oracle(&unit, &unit, |x, _| x * x).sqrt();
            assert!((n - 1.0).abs() < 1e-3, "normalise({:e}..) norm {n}", a[0]);
        }
    }

    #[test]
    fn precision_does_not_drift_on_large_sum() {
        // 1 048 576 copies of 1e-3 sum to 1048.576 in f64; in f32 the
        // best-case answer is ~1024 (lots of error). We use f64.
        let n = 1_048_576;
        let v = vec![1.0e-3_f32; n];
        let total = norm2(&v); // sum of squares = n * 1e-6
        let expected = n as f64 * 1.0e-6;
        assert!(
            (total - expected).abs() < 1e-3,
            "got {}, expected {}",
            total,
            expected
        );
    }

    /// Old-vs-new ns per call. Timing only; run optimized:
    /// `cargo test --release --lib kernels::tests::bench_kernels_old_vs_new -- --ignored --nocapture`
    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn bench_kernels_old_vs_new() {
        use std::hint::black_box;
        fn ns(mut f: impl FnMut() -> f64) -> f64 {
            let mut sink = 0.0;
            for _ in 0..20_000 {
                sink += f();
            }
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                for _ in 0..100_000 {
                    sink += f();
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e4);
            }
            black_box(sink);
            best
        }
        let mut rng = Rng(1);
        println!("ns/call, old -> new");
        for dim in [128, 384, 768, 1024, 1536, 3072] {
            let (a, b) = (rng.vector("unit", dim), rng.vector("unit", dim));
            let (a, b) = (&a[..], &b[..]);
            let qn = norm2(b);
            println!(
                "dim {dim:>4}: dot {:.1} -> {:.1} | l2_sq {:.1} -> {:.1} | l1 {:.1} -> {:.1} | norm2 {:.1} -> {:.1} | cosine {:.1} -> {:.1} (with_qnorm {:.1})",
                ns(|| dot_old(black_box(a), black_box(b))),
                ns(|| dot(black_box(a), black_box(b))),
                ns(|| l2_sq_old(black_box(a), black_box(b))),
                ns(|| l2_sq(black_box(a), black_box(b))),
                ns(|| l1_abs_old(black_box(a), black_box(b))),
                ns(|| l1_abs(black_box(a), black_box(b))),
                ns(|| norm2_old(black_box(a))),
                ns(|| norm2(black_box(a))),
                ns(|| cosine_old(black_box(a), black_box(b))),
                ns(|| cosine_distance(black_box(a), black_box(b))),
                ns(|| cosine_distance_with_qnorm(black_box(a), black_box(b), black_box(qn))),
            );
        }
    }

    // -----------------------------------------------------------------
    // Property-based tests (Hegel). The distance kernels are the
    // innermost hot path (every scan scores through them) and the
    // graph/IVF scoring math depends on exact metric identities, so
    // these pin the algebraic contracts across all finite inputs
    // rather than the three hand-picked vectors the example tests use.
    // -----------------------------------------------------------------

    use hegel::generators::{self};

    /// A pair of equal-length finite-f32 vectors of a drawn length.
    /// NaN/inf excluded: the kernels are metric primitives over real
    /// coordinates; embeddings are always finite (the type's input
    /// validation rejects non-finite values upstream).
    #[hegel::composite]
    fn vec_pair(tc: hegel::TestCase) -> (Vec<f32>, Vec<f32>) {
        let dim = tc.draw(generators::integers::<usize>().min_value(0).max_value(256));
        let coord = || {
            generators::floats::<f32>()
                .min_value(-1e6)
                .max_value(1e6)
                .allow_nan(false)
                .allow_infinity(false)
        };
        let a = tc.draw(generators::vecs(coord()).min_size(dim).max_size(dim));
        let b = tc.draw(generators::vecs(coord()).min_size(dim).max_size(dim));
        (a, b)
    }

    /// `l2_sq` is symmetric, non-negative, and zero exactly on equal
    /// vectors. These are the metric axioms the greedy-search
    /// ordering and RobustPrune's diversity check depend on.
    #[hegel::test]
    fn prop_l2_sq_is_a_nonneg_symmetric_metric(tc: hegel::TestCase) {
        let (a, b) = tc.draw(vec_pair());
        let ab = l2_sq(&a, &b);
        let ba = l2_sq(&b, &a);
        assert!(ab >= 0.0, "l2_sq negative: {ab}");
        assert!(
            (ab - ba).abs() <= 1e-6 * (1.0 + ab.abs()),
            "l2_sq asymmetric: {ab} vs {ba}"
        );
        assert_eq!(l2_sq(&a, &a), 0.0, "l2_sq(a,a) != 0");
    }

    /// `dot` is commutative. (The scan scores q·v; the build scores
    /// v·v' -- both rely on order-independence.)
    #[hegel::test]
    fn prop_dot_is_commutative(tc: hegel::TestCase) {
        let (a, b) = tc.draw(vec_pair());
        let ab = dot(&a, &b);
        let ba = dot(&b, &a);
        assert!(
            (ab - ba).abs() <= 1e-6 * (1.0 + ab.abs()),
            "dot not commutative: {ab} vs {ba}"
        );
    }

    /// The polarization identity |a-b|^2 == |a|^2 - 2(a.b) + |b|^2.
    /// This is the exact algebra that lets the quantized scan turn a
    /// dot-product score into an L2 ranking; if it drifts, the graph
    /// beam search orders candidates wrong. The tolerance is relative to
    /// |a|^2 + |b|^2, the size of the terms being cancelled: when a ~ b
    /// both sides are near zero while each kernel's error scales with
    /// the norms (see the module docs).
    #[hegel::test]
    fn prop_l2_sq_matches_polarization_identity(tc: hegel::TestCase) {
        let (a, b) = tc.draw(vec_pair());
        let lhs = l2_sq(&a, &b);
        let rhs = norm2(&a) - 2.0 * dot(&a, &b) + norm2(&b);
        let scale = 1.0 + lhs.abs() + rhs.abs() + norm2(&a) + norm2(&b);
        assert!(
            (lhs - rhs).abs() <= 1e-5 * scale,
            "polarization identity drift: |a-b|^2={lhs} vs |a|^2-2a.b+|b|^2={rhs}"
        );
    }

    /// `normalise_to_vec` yields a unit-norm vector (or an all-zero
    /// passthrough for the zero vector), and is idempotent:
    /// normalising an already-normalised vector is a no-op. Cosine
    /// scan correctness depends on both.
    #[hegel::test]
    fn prop_normalise_is_unit_norm_and_idempotent(tc: hegel::TestCase) {
        let dim = tc.draw(generators::integers::<usize>().min_value(1).max_value(256));
        let v: Vec<f32> = tc.draw(
            generators::vecs(
                generators::floats::<f32>()
                    .min_value(-1e3)
                    .max_value(1e3)
                    .allow_nan(false)
                    .allow_infinity(false),
            )
            .min_size(dim)
            .max_size(dim),
        );
        let once = normalise_to_vec(&v);
        let norm = norm2(&once).sqrt();
        // Either a genuine unit vector, or the zero passthrough (input
        // was all-zero, or so tiny it underflows to zero norm).
        assert!(
            (norm - 1.0).abs() < 1e-4 || norm == 0.0,
            "normalised vector has norm {norm} (neither 1 nor 0)"
        );
        let twice = normalise_to_vec(&once);
        for (x, y) in once.iter().zip(twice.iter()) {
            assert!((x - y).abs() < 1e-5, "normalise not idempotent: {x} vs {y}");
        }
    }

    /// `l1_abs` is symmetric and non-negative (the manhattan-distance
    /// operator surface relies on both).
    #[hegel::test]
    fn prop_l1_abs_is_nonneg_symmetric(tc: hegel::TestCase) {
        let (a, b) = tc.draw(vec_pair());
        let ab = l1_abs(&a, &b);
        let ba = l1_abs(&b, &a);
        assert!(ab >= 0.0, "l1_abs negative: {ab}");
        assert!(
            (ab - ba).abs() <= 1e-6 * (1.0 + ab.abs()),
            "l1_abs asymmetric"
        );
        assert_eq!(l1_abs(&a, &a), 0.0, "l1_abs(a,a) != 0");
    }
}
