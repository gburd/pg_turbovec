//! Pure-Rust math kernels — no Postgres dependency.
//!
//! All distance functions in `distance.rs` and the normalisation
//! helper in `normalize.rs` delegate to these. Keeping the kernels
//! Postgres-free means we can exercise them under plain `cargo test`,
//! prove their correctness in isolation, and benchmark them with
//! `criterion` without booting a cluster.
//!
//! All functions assume the caller has already validated equal
//! lengths / dimensionality. They use `f64` accumulators because
//! `f32` accumulation drops 2–3 decimal digits of precision on
//! corpora of ≥ 10⁶ vectors.

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += f64::from(*x) * f64::from(*y);
    }
    acc
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        acc += d * d;
    }
    acc
}

#[inline]
pub fn l1_abs(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (f64::from(*x) - f64::from(*y)).abs();
    }
    acc
}

#[inline]
pub fn norm2(a: &[f32]) -> f64 {
    let mut acc: f64 = 0.0;
    for x in a {
        acc += f64::from(*x) * f64::from(*x);
    }
    acc
}

/// Cosine distance: `1 - cos θ`. Returns `NaN` if either operand has
/// zero L2 norm. Clamps `cos θ` to `[-1, 1]` to defend against
/// numerical drift past the unit circle.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let na = norm2(a);
    let nb = norm2(b);
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    let cos = (dot(a, b) / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0);
    1.0 - cos
}

/// Write a unit-normalised copy of `src` into `dst`. If `src` is the
/// zero vector, `dst` is filled with `src` unchanged. Returns the
/// L2 norm of the input (caller may want it for further bookkeeping).
pub fn normalise_into(dst: &mut [f32], src: &[f32]) -> f64 {
    debug_assert_eq!(dst.len(), src.len());
    let n2 = norm2(src);
    if n2 == 0.0 {
        dst.copy_from_slice(src);
        return 0.0;
    }
    let inv = (1.0_f64 / n2.sqrt()) as f32;
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s * inv;
    }
    n2.sqrt()
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
    /// beam search orders candidates wrong. A weak relative tolerance
    /// accounts for f64 summation on large-magnitude coords.
    #[hegel::test]
    fn prop_l2_sq_matches_polarization_identity(tc: hegel::TestCase) {
        let (a, b) = tc.draw(vec_pair());
        let lhs = l2_sq(&a, &b);
        let rhs = norm2(&a) - 2.0 * dot(&a, &b) + norm2(&b);
        let scale = 1.0 + lhs.abs() + rhs.abs();
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
