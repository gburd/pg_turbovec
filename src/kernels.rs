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
//! These compute the EXACT distance the `ORDER BY` recheck ranks by, so
//! they accumulate in `f64`. Each `f32` input widens to `f64` exactly (and
//! an `f32 * f32` product is exact in `f64`), so the only error is the
//! `f64` summation for `dot`/`norm2`; `l2_sq`/`l1_abs` add at most one
//! rounding per term (≤ 1.1e-16 relative). The largest square of a
//! finite `f32` (~1.2e77) cannot overflow `f64`. The sum runs over 8 independent `f64` lanes so LLVM can
//! vectorize it (packed `cvtps2pd`/`mulpd`/`addpd`, 2 doubles per op, at
//! the portable x86-64 SSE2 target; `fcvtl`/`fmul`/`fadd .2d` on aarch64
//! NEON). The old single serial chain could not be reordered, so it ran
//! scalar. Splitting the chain 8 ways only shortens it: the error bound
//! drops from `γₙ` to `γ₍ₙ/₈₊₁₃₎` times `Σ|term|` (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, §4.2).
//!
//! Measured against an exact compensated-`f64` reference (dims 1 to
//! 16 000; unit-norm, ±1e3 and constant vectors; 1e15, 3e38 and
//! subnormal magnitudes), the max relative error is 5.5e-14 for constant
//! vectors (the worst case) and ≤ 3.4e-15 otherwise
//! (`kernels_match_exact_reference`); the old serial loop measured up to
//! 4.4e-13 on the same data. `dot`'s error is relative to
//! `Σ|aᵢbᵢ|`: near an exact zero no finite-precision dot product has a
//! bounded relative error. Dimensions below 8 never enter the lanes and
//! are bit-identical to the old serial loop.
//!
//! The old serial loop was justified as "`f32` accumulation drops 2–3
//! decimal digits on corpora of ≥ 10⁶ vectors". That conflated a
//! per-vector sum (at most 16 000 terms) with a corpus-sized one. `f32`
//! lanes were tried and rejected: their ~1e-7 relative error reordered
//! inner-product top-10s on tight clusters and collapsed near-duplicate
//! cosine distances to 0 (`inner_product_top10_order_exact_on_tight_cluster`,
//! `cosine_resolves_near_duplicates`).

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(a, b, |x, y| x * y)
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(a, b, |x, y| (x - y) * (x - y))
}

#[inline]
pub fn l1_abs(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    lane_sum(a, b, |x, y| (x - y).abs())
}

/// Squared L2 norm.
#[inline]
pub fn norm2(a: &[f32]) -> f64 {
    lane_sum(a, a, |x, _| x * x)
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
/// `dot` and `norm2(a)` are two passes rather than one fused loop on
/// purpose: on the portable SSE2 target the fused loop is about 2x
/// slower than two passes over a vector that is already in L1.
#[inline]
pub fn cosine_distance_with_qnorm(a: &[f32], b: &[f32], qnorm2: f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(
        qnorm2.to_bits(),
        norm2(b).to_bits(),
        "stale or foreign qnorm2"
    );
    let na = norm2(a);
    if na == 0.0 || qnorm2 == 0.0 {
        return f64::NAN;
    }
    let cos = (dot(a, b) / (na.sqrt() * qnorm2.sqrt())).clamp(-1.0, 1.0);
    1.0 - cos
}

/// Independent `f64` accumulator lanes. 8 measured fastest or tied
/// (4/8/16/32 tried) on Sapphire Rapids at 384-3072 dims; 32 spills.
const LANES: usize = 8;

/// `Σ term(a[i], b[i])` in `f64`, over [`LANES`] independent lanes, then
/// the last < [`LANES`] elements.
#[inline]
fn lane_sum(a: &[f32], b: &[f32], term: impl Fn(f64, f64) -> f64) -> f64 {
    let n = a.len().min(b.len());
    let (chunks_a, rest_a) = a[..n].as_chunks::<LANES>();
    let (chunks_b, rest_b) = b[..n].as_chunks::<LANES>();
    let mut lanes = [0.0_f64; LANES];
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for i in 0..LANES {
            lanes[i] += term(f64::from(x[i]), f64::from(y[i]));
        }
    }
    let mut s = 0.0_f64;
    for v in lanes {
        s += v;
    }
    for (x, y) in rest_a.iter().zip(rest_b) {
        s += term(f64::from(*x), f64::from(*y));
    }
    s
}

/// Write a unit-normalised copy of `src` into `dst`. If `src` is the
/// zero vector, `dst` is filled with `src` unchanged. Returns the
/// L2 norm of the input (caller may want it for further bookkeeping).
///
/// The norm is summed in one serial `f64` chain, not with [`norm2`]'s
/// lanes: the output is what gets quantized and persisted, so it must
/// stay bit-identical across releases (an ulp of difference can move a
/// code across a quantization boundary). It runs once per row or query,
/// not per candidate, so its speed does not matter.
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

    // The pre-2.12.0 kernels, verbatim: serial f64 accumulation.
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

    /// Below 8 dims nothing enters the lanes: results are bit-identical
    /// to the old serial f64 kernels, so small-dim SQL outputs and tie
    /// orders cannot move.
    #[test]
    #[allow(clippy::float_cmp)] // exact results are the point
    fn small_dims_bit_identical_to_old_kernels() {
        let mut rng = Rng(42);
        for kind in ["unit", "signed1e3", "positive1e3"] {
            for dim in 0..LANES {
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
    /// The returned f64 norm is checked bit-for-bit too: a changed
    /// summation order moves the norm by an ulp in almost every vector
    /// but flips an f32 output only about once per 2^29 elements, so the
    /// outputs alone cannot catch it here (they would at corpus scale).
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
                let mut out = vec![0.0_f32; v.len()];
                let norm = normalise_into(&mut out, &v);
                assert_eq!(norm.to_bits(), n.to_bits(), "{kind} dim {dim}: norm");
                let got: Vec<u32> = out.iter().map(|x| x.to_bits()).collect();
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
    /// any finite-precision sum, the old f64 one included. Rigorous bound
    /// at 16 000 dims: γ₂₀₁₃ ≈ 2.2e-13. 1e-13 is an empirical regression
    /// bound for this seed (measured 5.5e-14), deliberately below the
    /// rigorous bound.
    const KERNEL_REL_BOUND: f64 = 1e-13;
    /// Absolute bound on `cosine_distance` (a value in [0, 2]).
    const COSINE_ABS_BOUND: f64 = 1e-13;

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

    /// Pins the rounding change against the 2.11.0 kernels that the
    /// CHANGELOG and docs/UPGRADING.md publish: max |new - old| <= 4.5e-13
    /// relative for dot (relative to sum |a_i b_i|) and l2_sq, <= 6.5e-13
    /// absolute for cosine; worst case is constant vectors at 16000-d. If a
    /// kernel change pushes past these, the published bound is stale.
    #[test]
    fn lanes_vs_old_change_is_within_published_bound() {
        let mut rng = Rng(0x2120_0000_0000_0001);
        let (mut dot_rel, mut l2_rel, mut cos_abs) = (0.0f64, 0.0f64, 0.0f64);
        for &dim in &[8usize, 64, 1024, 3072, 16000] {
            for trial in 0..200 {
                let (ca, cb) = (0.1 + rng.unif() as f32, 0.1 + rng.unif() as f32);
                let (a, b): (Vec<f32>, Vec<f32>) = if trial % 2 == 0 {
                    (vec![ca; dim], vec![cb; dim])
                } else {
                    (
                        (0..dim).map(|_| rng.gauss() as f32).collect(),
                        (0..dim).map(|_| rng.gauss() as f32).collect(),
                    )
                };
                let mag: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (f64::from(*x) * f64::from(*y)).abs())
                    .sum();
                dot_rel = dot_rel.max((dot(&a, &b) - dot_old(&a, &b)).abs() / mag);
                let l2o = l2_sq_old(&a, &b);
                if l2o > 0.0 {
                    l2_rel = l2_rel.max((l2_sq(&a, &b) - l2o).abs() / l2o);
                }
                cos_abs = cos_abs.max((cosine_distance(&a, &b) - cosine_old(&a, &b)).abs());
            }
        }
        assert!(dot_rel <= 4.5e-13, "dot max rel change {dot_rel:e}");
        assert!(l2_rel <= 4.5e-13, "l2_sq max rel change {l2_rel:e}");
        assert!(cos_abs <= 6.5e-13, "cosine max abs change {cos_abs:e}");
        // The sample must actually reach the worst case, or the test pins
        // nothing (a weaker sample is how the first published bound was 2x low).
        assert!(
            dot_rel > 2.5e-13,
            "sample missed the worst case: {dot_rel:e}"
        );
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

    /// Exact cosine distance from compensated sums.
    fn cosine_exact(a: &[f32], b: &[f32]) -> f64 {
        let d = oracle(a, b, |x, y| x * y);
        let (na, nb) = (oracle(a, a, |x, _| x * x), oracle(b, b, |x, _| x * x));
        1.0 - (d / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
    }

    /// `1 - cos θ` cancels near 0, so a cosine with absolute error ~1e-7
    /// (f32 accumulation) cannot resolve near-duplicates: rows 3e-10
    /// apart all come out as exactly 0 and a self-match query (`ORDER BY
    /// emb <=> (SELECT emb ... WHERE id = k)`) returns an arbitrary
    /// near-duplicate. Pure-Rust reproduction of the
    /// `onebit_all_positive_corpus_*` fixture, where row 29 is 3.3e-10
    /// from row 100; then perturbations of 1e-6..1e-3 at every dim, whose
    /// distances (~1e-13..1e-7) must come out in exact order.
    #[test]
    fn cosine_resolves_near_duplicates() {
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
        assert_eq!(nearest(&cosine_distance), 100, "new kernel must find self");
        let qn = norm2(&q);
        assert_eq!(
            nearest(&|a, b| cosine_distance_with_qnorm(a, b, qn)),
            100,
            "with_qnorm must find self"
        );
        let mut rng = Rng(11);
        let (mut worst, mut worst_old) = (0.0_f64, 0.0_f64);
        for &dim in &PRECISION_DIMS[4..] {
            for kind in ["unit", "signed1e3", "positive1e3"] {
                let a = rng.vector(kind, dim);
                let u: Vec<f64> = (0..dim).map(|_| rng.unif() * 2.0 - 1.0).collect();
                let mut prev = (-1.0_f64, -1.0_f64);
                for eps in [1e-6, 1e-5, 1e-4, 1e-3] {
                    let b: Vec<f32> = a
                        .iter()
                        .zip(&u)
                        .map(|(x, w)| (f64::from(*x) * (1.0 + eps * w)) as f32)
                        .collect();
                    let (got, exact) = (cosine_distance(&b, &a), cosine_exact(&b, &a));
                    let qn = norm2(&a);
                    assert_eq!(
                        cosine_distance_with_qnorm(&b, &a, qn).to_bits(),
                        got.to_bits()
                    );
                    // `1 - cos` inherits cos's few-ulp-of-1 error (old
                    // serial kernel included): bound it absolutely.
                    let old = cosine_old(&b, &a);
                    worst = worst.max((got - exact).abs());
                    worst_old = worst_old.max((old - exact).abs());
                    assert!(
                        (got - exact).abs() <= 1e-14,
                        "{kind} dim {dim} eps {eps}: {got:e} vs exact {exact:e}"
                    );
                    assert!(
                        got > prev.0 && exact > prev.1,
                        "{kind} dim {dim} eps {eps}: order lost ({got:e} after {:e})",
                        prev.0
                    );
                    prev = (got, exact);
                }
            }
        }
        println!("near-dup cosine abs err: new {worst:e}, old serial {worst_old:e}");
    }

    /// Inner product on a tight cluster: unit vectors around a few
    /// centres, nearest cosine distance ~7e-5, adjacent top-10 scores
    /// ~1e-10 apart. f32 accumulation (relative error ~1e-7) reordered
    /// the top-10 in 7 of 100 queries here; the exact recheck must not.
    /// The top-10 ORDER by `-dot` must match the compensated-f64 oracle
    /// for every query (ties broken by row index in both).
    #[test]
    fn inner_product_top10_order_exact_on_tight_cluster() {
        let (n, dim, centres, spread) = (1000_usize, 384_usize, 10_usize, 0.01_f64);
        let mut rng = Rng(5);
        let c: Vec<Vec<f64>> = (0..centres)
            .map(|_| (0..dim).map(|_| rng.gauss()).collect())
            .collect();
        let sample = |rng: &mut Rng| -> Vec<f32> {
            let k = (rng.next() % centres as u64) as usize;
            let v: Vec<f64> = c[k].iter().map(|x| x + spread * rng.gauss()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter().map(|x| (x / norm) as f32).collect()
        };
        let corpus: Vec<Vec<f32>> = (0..n).map(|_| sample(&mut rng)).collect();
        let top10 = |scores: &[f64]| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..scores.len()).collect();
            idx.sort_by(|&i, &j| scores[i].total_cmp(&scores[j]).then(i.cmp(&j)));
            idx.truncate(10);
            idx
        };
        let mut nearest = f64::INFINITY;
        for qi in 0..100 {
            let q = sample(&mut rng);
            let exact: Vec<f64> = corpus
                .iter()
                .map(|v| -oracle(v, &q, |x, y| x * y))
                .collect();
            let got: Vec<f64> = corpus.iter().map(|v| -dot(v, &q)).collect();
            let want = top10(&exact);
            nearest = nearest.min(1.0 + exact[want[0]]);
            assert_eq!(
                top10(&got),
                want,
                "query {qi}: top-10 order differs from exact"
            );
        }
        assert!(
            nearest < 2e-4,
            "fixture not tight: nearest cos distance {nearest:e}"
        );
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
    /// must give FINITE, exact-to-rounding answers: f64 accumulation has
    /// the range for all of them (3e38^2 ~ 1e77), with no fallback.
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
            // Terms whose squares underflow f32, next to normal ones.
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
            ab.to_bits() == ba.to_bits(),
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
            ab.to_bits() == ba.to_bits(),
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
            (lhs - rhs).abs() <= 1e-12 * scale,
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
            ab.to_bits() == ba.to_bits(),
            "l1_abs asymmetric: {ab} vs {ba}"
        );
        assert_eq!(l1_abs(&a, &a), 0.0, "l1_abs(a,a) != 0");
    }
}
