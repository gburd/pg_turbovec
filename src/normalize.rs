//! Unit-norm normalisation helpers and direct turbovec round-trip
//! probes used by tests.
//!
//! Math kernels live in `crate::kernels`.

use pgrx::prelude::*;

use crate::kernels;
use crate::tvector::Tvector;

/// L2-normalise a `tvector` to unit length. Required by the TurboQuant
/// kernel; we expose it as a SQL function so users can pre-normalise
/// vectors in pipelines that don't run through the index.
///
/// Returns the input unchanged when the L2 norm is zero (avoids
/// producing NaNs).
#[pg_extern(immutable, parallel_safe)]
pub fn tvector_normalize(v: Tvector) -> Tvector {
    Tvector::from_vec(kernels::normalise_to_vec(v.as_slice()))
}

/// Diagnostic: encode a `tvector` through the turbovec quantiser and
/// score it back against itself, returning the inner-product self-score.
///
/// This is intentionally narrow — it is *not* a full ANN function. Its
/// purpose is to surface upstream regressions: if the version of
/// turbovec we link against starts producing different scores, this
/// will catch it.
///
/// Constraints:
/// - `dim` must be a multiple of 8 (turbovec assertion).
/// - `bit_width` must be in `2..=4`.
#[pg_extern(immutable, parallel_safe)]
pub fn turbovec_self_score(v: Tvector, bit_width: i32) -> f64 {
    use turbovec::IdMapIndex;

    if v.dim() % 8 != 0 {
        error!(
            "turbovec_self_score: dim must be a multiple of 8 (got {})",
            v.dim()
        );
    }
    if !(2..=4).contains(&bit_width) {
        error!(
            "turbovec_self_score: bit_width must be in 2..=4 (got {})",
            bit_width
        );
    }
    let dim = v.dim();
    let mut idx = IdMapIndex::new(dim, bit_width as usize);
    if let Err(e) = idx.add_with_ids(v.as_slice(), &[1u64]) {
        error!("turbovec_self_score: add_with_ids failed: {:?}", e);
    }
    let (scores, ids) = idx.search(v.as_slice(), 1);
    if ids.is_empty() {
        error!("turbovec_self_score: search returned no results");
    }
    f64::from(scores[0])
}

/// Build a random unit-norm `tvector` of dimension `dim`. Useful for
/// benchmarks and recall tests; deliberately *not* deterministic
/// across calls (we want different vectors each invocation).
///
/// `dim` must be in `1..=16000`. Distribution is i.i.d. standard
/// normal followed by L2 normalisation — the canonical "uniform on
/// the sphere" sampling.
#[pg_extern(volatile, parallel_safe)]
pub fn tvector_random_unit(dim: i32) -> Tvector {
    use rand::Rng;

    if dim <= 0 || dim as usize > crate::tvector::MAX_DIM {
        error!(
            "tvector_random_unit: dim must be in 1..={} (got {})",
            crate::tvector::MAX_DIM,
            dim
        );
    }
    let mut rng = rand::thread_rng();
    let raw: Vec<f32> = (0..dim)
        .map(|_| {
            // Box-Muller via rng's standard_normal would need rand_distr;
            // a uniform [-1, 1) is good enough for benchmarking and avoids
            // the extra dependency.
            let u: f32 = rng.gen_range(-1.0_f32..1.0_f32);
            u
        })
        .collect();
    Tvector::from_vec(kernels::normalise_to_vec(&raw))
}
