//! Phase D — breadth-parity SQL surface for hybrid / multivector search.
//!
//! Two genuinely-useful, low-risk primitives that close the breadth
//! gap vs VectorChord/Qdrant *at the SQL layer*, without touching the
//! wire format or the index AM:
//!
//! - **MaxSim late interaction** (`max_sim`, `max_sim_cosine`):
//!   ColBERT-style scoring of a (query, doc) pair where both sides are
//!   *arrays* of per-token `vector`s. This is a RE-RANK primitive —
//!   ANN-retrieve candidate docs on a pooled/centroid vector, then
//!   MaxSim-rerank the top-N. pg_turbovec still indexes a single
//!   vector per row; index-native late interaction (per-token storage
//!   + a MaxSim-aware scan) is a documented future phase, NOT this.
//!
//! - **Reciprocal Rank Fusion** (`rrf_score`): the scalar
//!   `1 / (k + rank)` term used to fuse a dense ANN ranking with a
//!   sparse / keyword ranking. The arithmetic is trivial; the value
//!   is having a single tested, documented helper so users stop
//!   hand-rolling `1.0/(60+rank)` in every query. The full fusion
//!   lives in a documented CTE recipe (see docs/HYBRID_SEARCH.md) —
//!   the roadmap's "build server-side fusion only on demand".
//!
//! See `docs/HYBRID_SEARCH.md` for the usage patterns, conventions,
//! and the honest limitations.

use pgrx::prelude::*;

use crate::kernels;
use crate::vec::Vector;

/// Validate that every vector in both arrays shares one dimension.
///
/// Returns that common dimension. Empty arrays contribute no
/// constraint (an empty query or empty doc is handled by the caller
/// before this is consulted). ERRORs — mirroring `check_same_dim` in
/// `sparsevec_ops.rs` — on the first mismatch, naming both dims.
fn check_uniform_dim(query: &[Vector], doc: &[Vector], op: &str) -> Option<usize> {
    let mut dim: Option<usize> = None;
    for v in query.iter().chain(doc.iter()) {
        match dim {
            None => dim = Some(v.dim()),
            Some(d) if d != v.dim() => {
                error!(
                    "different vector dimensions {} and {} in '{}': all token \
                     vectors in both arrays must share one dimension",
                    d,
                    v.dim(),
                    op
                );
            }
            _ => {}
        }
    }
    dim
}

/// Core MaxSim: `sum over q in Q of ( max over d in D of sim(q, d) )`.
///
/// `sim` is the per-pair similarity closure. The outer sum walks the
/// query tokens in array order (no reassociation) so the result is
/// bit-for-bit deterministic for a given input.
fn max_sim_with<F: Fn(&[f32], &[f32]) -> f64>(query: &[Vector], doc: &[Vector], sim: F) -> f64 {
    // ColBERT convention: an empty query scores 0 (nothing to match).
    // An empty doc gives every query token a max over the empty set,
    // which we also define as 0 (the doc supports no token).
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let mut total: f64 = 0.0;
    for q in query {
        let qs = q.as_slice();
        let mut best = f64::NEG_INFINITY;
        for d in doc {
            let s = sim(qs, d.as_slice());
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
}

/// ColBERT-style MaxSim using **dot-product** similarity.
///
/// `MaxSim(Q, D) = sum_{q in Q} max_{d in D} dot(q, d)`.
///
/// Both arguments are `vector[]` — one entry per token. All token
/// vectors (across both arrays) must share one dimension; mismatches
/// ERROR. An empty query or empty doc scores `0.0` (ColBERT
/// convention: nothing to match / nothing to match against).
///
/// This is the right variant when token vectors are already
/// L2-normalised (the usual ColBERT setup), where dot == cosine
/// similarity. It is a RE-RANK primitive — see `docs/HYBRID_SEARCH.md`.
///
/// ```ignore
/// SELECT turbovec.max_sim(
///   ARRAY['[1,0]','[0,1]']::turbovec.vector[],   -- query tokens
///   ARRAY['[1,0]','[0,1]','[1,1]']::turbovec.vector[]  -- doc tokens
/// );
/// -- max over doc of dot(q1=[1,0]) = 1 ; of dot(q2=[0,1]) = 1 ; sum = 2
/// ```
#[pg_extern(immutable, parallel_safe)]
fn max_sim(query: Vec<Vector>, doc: Vec<Vector>) -> f64 {
    check_uniform_dim(&query, &doc, "max_sim");
    max_sim_with(&query, &doc, kernels::dot)
}

/// ColBERT-style MaxSim using **cosine similarity**.
///
/// `MaxSim(Q, D) = sum_{q in Q} max_{d in D} (1 - cosine_distance(q, d))`.
///
/// Identical contract to [`max_sim`] but normalises each pair, so it
/// is correct for un-normalised token vectors. A zero-norm token
/// makes its pair's cosine `NaN`; `NaN` never wins a `>` comparison,
/// so a zero token simply never becomes the per-query max (and if
/// *every* doc token is zero, that query token contributes `NaN`,
/// propagating to the sum — document your tokens are non-zero).
///
/// Empty query or empty doc scores `0.0`.
#[pg_extern(immutable, parallel_safe)]
fn max_sim_cosine(query: Vec<Vector>, doc: Vec<Vector>) -> f64 {
    check_uniform_dim(&query, &doc, "max_sim_cosine");
    max_sim_with(&query, &doc, |q, d| 1.0 - kernels::cosine_distance(q, d))
}

/// Reciprocal Rank Fusion term: `1.0 / (k + rank)`.
///
/// `rank` is the 0-based (or 1-based — pick one convention and keep
/// it consistent across rankers) position of a document in one
/// ranker's output; `k` damps the contribution of low ranks (default
/// 60, the value from the original RRF paper, Cormack et al. 2009).
/// Fuse two rankings by summing each document's `rrf_score` across
/// rankers and ordering by the sum descending.
///
/// `k + rank` must be positive; otherwise ERROR (a negative or zero
/// denominator is a caller bug, not a meaningful score).
///
/// ```ignore
/// SELECT turbovec.rrf_score(0);        -- 1/60  ≈ 0.01667
/// SELECT turbovec.rrf_score(1, 10);    -- 1/11  ≈ 0.09091
/// ```
///
/// See `docs/HYBRID_SEARCH.md` for the full dense+sparse CTE recipe.
#[pg_extern(immutable, parallel_safe)]
fn rrf_score(rank: i32, k: default!(i32, 60)) -> f64 {
    let denom = i64::from(k) + i64::from(rank);
    if denom <= 0 {
        error!(
            "rrf_score: k + rank must be positive, got k={} rank={} (denom {})",
            k, rank, denom
        );
    }
    1.0 / denom as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // Postgres-free unit tests: construct `Vector` via the struct
    // literal (NOT `Vector::from_vec`, which calls `error!` and would
    // pull pgrx FFI symbols into the plain `lib test` binary). The
    // SQL-surface behaviour (dim mismatch, empty arrays, the cosine
    // variant, rrf_score) is covered by `#[pg_test]`s in src/lib.rs,
    // which run inside the postmaster.
    fn v(data: &[f32]) -> Vector {
        Vector { data: data.to_vec() }
    }

    #[test]
    fn max_sim_dot_sum_of_maxes() {
        // Q = { [1,0], [0,1] }, D = { [1,0], [0,1], [1,1] }
        // q1=[1,0]: dot with doc = {1, 0, 1} -> max 1
        // q2=[0,1]: dot with doc = {0, 1, 1} -> max 1
        // sum = 2
        let q = vec![v(&[1.0, 0.0]), v(&[0.0, 1.0])];
        let d = vec![v(&[1.0, 0.0]), v(&[0.0, 1.0]), v(&[1.0, 1.0])];
        let got = max_sim_with(&q, &d, kernels::dot);
        assert!((got - 2.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn max_sim_dot_picks_largest() {
        // single query token, doc tokens with increasing dot
        let q = vec![v(&[1.0, 1.0])];
        let d = vec![v(&[1.0, 0.0]), v(&[2.0, 2.0]), v(&[0.0, 1.0])];
        // dot = {1, 4, 1} -> max 4
        let got = max_sim_with(&q, &d, kernels::dot);
        assert!((got - 4.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn max_sim_empty_is_zero() {
        let some = vec![v(&[1.0, 2.0])];
        let dot = kernels::dot as fn(&[f32], &[f32]) -> f64;
        assert_eq!(max_sim_with(&[], &some, dot), 0.0);
        assert_eq!(max_sim_with(&some, &[], dot), 0.0);
        assert_eq!(max_sim_with(&[], &[], dot), 0.0);
    }

    #[test]
    fn max_sim_cosine_normalised_equals_dot() {
        // unit vectors: cosine sim == dot
        let q = vec![v(&[1.0, 0.0])];
        let d = vec![v(&[0.6, 0.8]), v(&[1.0, 0.0])];
        let cos = max_sim_with(&q, &d, |a, b| 1.0 - kernels::cosine_distance(a, b));
        let dot = max_sim_with(&q, &d, kernels::dot);
        assert!((cos - dot).abs() < 1e-6, "cos {cos} dot {dot}");
        assert!((cos - 1.0).abs() < 1e-6, "cos {cos}");
    }

    #[test]
    fn rrf_term_is_decreasing() {
        // Pure arithmetic check of the RRF formula (the SQL function's
        // ERROR path + default are covered by `rrf_score_values` in
        // src/lib.rs). 1/(60+0) > 1/(60+1) > 1/(60+2).
        let term = |rank: i64, k: i64| 1.0 / (k + rank) as f64;
        assert!((term(0, 60) - 1.0 / 60.0).abs() < 1e-12);
        assert!((term(1, 10) - 1.0 / 11.0).abs() < 1e-12);
        assert!(term(0, 60) > term(1, 60) && term(1, 60) > term(2, 60));
    }
}
