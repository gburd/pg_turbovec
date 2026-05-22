//! Aggregates `avg(tvector)` and `sum(tvector)`.
//!
//! Internal state is a running element-wise sum stored as `f64[dim]`
//! plus a `count`. Using `f64` for the accumulator matters: 1 M `f32`
//! values can lose ~3 decimal digits of precision in a naive `f32`
//! sum, and pgvector exhibits the same drift. Final coercion back to
//! `tvector` happens only in the `finalfn`.
//!
//! Both aggregates are parallel-safe: `combinefn` merges two partial
//! states. We declare them via `extension_sql!` so we have full
//! control over the C-side declaration (pgrx's `#[pg_aggregate]`
//! macro is awkward for stateful types whose dim is determined at
//! runtime).
//!
//! ```ignore
//! -- Element-wise mean over a corpus.
//! SELECT avg(emb) FROM docs;
//!
//! -- Element-wise sum (e.g. for centroid computation).
//! SELECT sum(emb) FROM docs;
//!
//! -- An empty input or all-NULL input returns NULL (SQL spec).
//! SELECT avg(emb) FROM docs WHERE FALSE;  -- NULL
//!
//! -- Mixed-dim rows raise ERROR mid-aggregate:
//! SELECT avg(v) FROM (VALUES
//!     ('[1,2,3]'::turbovec.tvector),
//!     ('[1,2,3,4]'::turbovec.tvector)
//! ) t(v);
//! -- ERROR: tvector_accum: cannot accumulate tvectors of different dimensions (3 vs 4)
//! ```

use pgrx::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tvector::{Tvector, MAX_DIM};

/// Internal state for `avg(tvector)` and `sum(tvector)`.
///
/// Storage is a CBOR varlena (auto-derived). The first `transfn` call
/// initialises `sum` to a zero vector of the input's dimension; later
/// calls validate dim and accumulate.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PostgresType)]
pub struct TvectorAccum {
    /// Element-wise running sum, in `f64`.
    pub sum: Vec<f64>,
    /// Number of values accumulated.
    pub count: i64,
}

impl TvectorAccum {
    fn ensure_dim(&mut self, dim: usize, op: &str) {
        if self.sum.is_empty() {
            if dim == 0 || dim > MAX_DIM {
                error!(
                    "{}: invalid tvector dimension {} (must be 1..={})",
                    op, dim, MAX_DIM
                );
            }
            self.sum = vec![0.0_f64; dim];
        } else if self.sum.len() != dim {
            error!(
                "{}: cannot accumulate tvectors of different dimensions ({} vs {})",
                op,
                self.sum.len(),
                dim
            );
        }
    }
}

/// `state := tvector_accum(state, value)`. Accepts `Option<TvectorAccum>`
/// so pgrx generates a non-strict SQL function that handles the
/// initial NULL state — otherwise CREATE AGGREGATE rejects the
/// definition with "must not omit initial value when transition
/// function is strict and transition type is not compatible with
/// input type".
#[pg_extern(immutable, parallel_safe)]
fn tvector_accum(state: Option<TvectorAccum>, value: Tvector) -> TvectorAccum {
    let mut state = state.unwrap_or_default();
    state.ensure_dim(value.dim(), "tvector_accum");
    for (s, v) in state.sum.iter_mut().zip(value.as_slice().iter()) {
        *s += f64::from(*v);
    }
    state.count += 1;
    state
}

/// `state := tvector_combine(s1, s2)` for parallel aggregation.
/// Both operands are nullable for symmetry with the SQL machinery.
#[pg_extern(immutable, parallel_safe)]
fn tvector_combine(
    s1: Option<TvectorAccum>,
    s2: Option<TvectorAccum>,
) -> Option<TvectorAccum> {
    match (s1, s2) {
        (None, None) => None,
        (Some(s), None) | (None, Some(s)) => Some(s),
        (Some(a), Some(b)) => {
            if a.count == 0 {
                return Some(b);
            }
            if b.count == 0 {
                return Some(a);
            }
            let mut out = a;
            if out.sum.len() != b.sum.len() {
                error!(
                    "tvector_combine: cannot merge accumulators of different dimensions ({} vs {})",
                    out.sum.len(),
                    b.sum.len()
                );
            }
            for (x, y) in out.sum.iter_mut().zip(b.sum.iter()) {
                *x += *y;
            }
            out.count += b.count;
            Some(out)
        }
    }
}

/// Final function for `avg(tvector)` — divides the running sum by
/// `count`, then narrows from `f64` back to `f32`.
#[pg_extern(immutable, parallel_safe)]
fn tvector_avg_finalfn(state: TvectorAccum) -> Option<Tvector> {
    if state.count == 0 {
        return None;
    }
    let count = state.count as f64;
    let data: Vec<f32> = state.sum.iter().map(|s| (*s / count) as f32).collect();
    Some(Tvector::from_vec(data))
}

/// Final function for `sum(tvector)` — narrows the running `f64` sum
/// back to `f32`.
#[pg_extern(immutable, parallel_safe)]
fn tvector_sum_finalfn(state: TvectorAccum) -> Option<Tvector> {
    if state.count == 0 {
        return None;
    }
    let data: Vec<f32> = state.sum.iter().map(|s| *s as f32).collect();
    Some(Tvector::from_vec(data))
}

extension_sql!(
    r"
    CREATE AGGREGATE avg(tvector) (
        SFUNC = tvector_accum,
        STYPE = TvectorAccum,
        FINALFUNC = tvector_avg_finalfn,
        COMBINEFUNC = tvector_combine,
        PARALLEL = SAFE
    );

    CREATE AGGREGATE sum(tvector) (
        SFUNC = tvector_accum,
        STYPE = TvectorAccum,
        FINALFUNC = tvector_sum_finalfn,
        COMBINEFUNC = tvector_combine,
        PARALLEL = SAFE
    );
    ",
    name = "tvector_aggregates",
    requires = [
        TvectorAccum,
        tvector_accum,
        tvector_combine,
        tvector_avg_finalfn,
        tvector_sum_finalfn
    ]
);
