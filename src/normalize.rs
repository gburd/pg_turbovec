//! Unit-norm normalisation helpers and direct turbovec round-trip
//! probes used by tests.

use pgrx::prelude::*;

use crate::tvector::Tvector;

/// L2-normalise a `tvector` to unit length. Required by the TurboQuant
/// kernel; for v0.1 we expose it as a SQL function so users can pre-
/// normalise vectors in pipelines that don't run through the index.
///
/// Returns the input unchanged when the L2 norm is zero (avoids
/// producing NaNs).
#[pg_extern(immutable, parallel_safe)]
pub fn tvector_normalize(v: Tvector) -> Tvector {
    let mut acc: f64 = 0.0;
    for x in v.as_slice() {
        acc += f64::from(*x) * f64::from(*x);
    }
    if acc == 0.0 {
        return v;
    }
    let n = acc.sqrt() as f32;
    let data: Vec<f32> = v.data.iter().map(|x| *x / n).collect();
    Tvector::from_vec(data)
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
