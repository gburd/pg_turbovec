//! Distance operators, casts, and aggregates for `halfvec`.
//!
//! The kernel here computes in `f64` over `f32`-promoted values, so
//! results are bit-identical to running the same operation on the
//! `vector` representation of the same data. The only semantic
//! difference is precision loss on insert (f32 → f16 narrow).

use half::f16;
use pgrx::prelude::*;
use serde::{Deserialize, Serialize};

use crate::halfvec::{Halfvec, MAX_DIM};
use crate::kernels;
use crate::vec::Vector;

// ---------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------

#[pg_extern(immutable, parallel_safe)]
fn halfvec_l2_distance(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "<->");
    kernels::l2_sq(&a.to_f32_vec(), &b.to_f32_vec()).sqrt()
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_l2_squared_distance(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "l2_squared_distance");
    kernels::l2_sq(&a.to_f32_vec(), &b.to_f32_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_inner_product(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "inner_product");
    kernels::dot(&a.to_f32_vec(), &b.to_f32_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_negative_inner_product(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "<#>");
    -kernels::dot(&a.to_f32_vec(), &b.to_f32_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_cosine_distance(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "<=>");
    kernels::cosine_distance(&a.to_f32_vec(), &b.to_f32_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_l1_distance(a: Halfvec, b: Halfvec) -> f64 {
    a.check_same_dim(&b, "<+>");
    kernels::l1_abs(&a.to_f32_vec(), &b.to_f32_vec())
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_dims(v: Halfvec) -> i32 {
    v.dim() as i32
}

/// pgvector-compat overload: `vector_dims(halfvec) -> integer`.
#[pg_extern(name = "vector_dims", immutable, parallel_safe)]
fn vector_dims_halfvec(v: Halfvec) -> i32 {
    v.dim() as i32
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_norm(v: Halfvec) -> f64 {
    kernels::norm2(&v.to_f32_vec()).sqrt()
}

/// pgvector-compat overload: `vector_norm(halfvec) -> double precision`.
#[pg_extern(name = "vector_norm", immutable, parallel_safe)]
fn vector_norm_halfvec(v: Halfvec) -> f64 {
    kernels::norm2(&v.to_f32_vec()).sqrt()
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_l2_normalize(v: Halfvec) -> Halfvec {
    let f32_unit = kernels::normalise_to_vec(&v.to_f32_vec());
    Halfvec::from_f32_vec(f32_unit)
}

// ---------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------

/// `vector::halfvec` — narrow each f32 to f16. Overflows raise.
#[pg_extern(immutable, parallel_safe)]
fn vector_to_halfvec(v: Vector) -> Halfvec {
    Halfvec::from_f32_vec(v.data)
}

/// `halfvec::vector` — widen each f16 to f32 (lossless).
#[pg_extern(immutable, parallel_safe)]
fn halfvec_to_vector(v: Halfvec) -> Vector {
    Vector::from_vec(v.to_f32_vec())
}

/// `real[]::halfvec`
#[pg_extern(immutable, parallel_safe)]
fn array_to_halfvec(arr: ::std::vec::Vec<Option<f32>>) -> Halfvec {
    let data: ::std::vec::Vec<f32> = arr
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| {
                error!("halfvec cannot contain NULL element at index {}", i)
            })
        })
        .collect();
    Halfvec::from_f32_vec(data)
}

/// `halfvec::real[]`
#[pg_extern(immutable, parallel_safe)]
fn halfvec_to_array(v: Halfvec) -> ::std::vec::Vec<f32> {
    v.to_f32_vec()
}

// ---------------------------------------------------------------------
// Aggregate state and transitions
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize, PostgresType)]
pub struct HalfvecAccum {
    pub sum: ::std::vec::Vec<f64>,
    pub count: i64,
}

impl HalfvecAccum {
    fn ensure_dim(&mut self, dim: usize, op: &str) {
        if self.sum.is_empty() {
            if dim == 0 || dim > MAX_DIM {
                error!(
                    "{}: invalid halfvec dimension {} (must be 1..={})",
                    op, dim, MAX_DIM
                );
            }
            self.sum = vec![0.0_f64; dim];
        } else if self.sum.len() != dim {
            error!(
                "{}: cannot accumulate halfvecs of different dimensions ({} vs {})",
                op,
                self.sum.len(),
                dim
            );
        }
    }
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_accum(state: Option<HalfvecAccum>, value: Halfvec) -> HalfvecAccum {
    let mut state = state.unwrap_or_default();
    state.ensure_dim(value.dim(), "halfvec_accum");
    for (s, v) in state.sum.iter_mut().zip(value.as_slice().iter()) {
        *s += f64::from(f32::from(*v));
    }
    state.count += 1;
    state
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_combine(
    s1: Option<HalfvecAccum>,
    s2: Option<HalfvecAccum>,
) -> Option<HalfvecAccum> {
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
                    "halfvec_combine: cannot merge accumulators of different dimensions ({} vs {})",
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

#[pg_extern(immutable, parallel_safe)]
fn halfvec_avg_finalfn(state: HalfvecAccum) -> Option<Halfvec> {
    if state.count == 0 {
        return None;
    }
    let count = state.count as f64;
    let data: ::std::vec::Vec<f16> = state
        .sum
        .iter()
        .map(|s| f16::from_f32((*s / count) as f32))
        .collect();
    Some(Halfvec::from_f16_vec(data))
}

#[pg_extern(immutable, parallel_safe)]
fn halfvec_sum_finalfn(state: HalfvecAccum) -> Option<Halfvec> {
    if state.count == 0 {
        return None;
    }
    let data: ::std::vec::Vec<f16> = state
        .sum
        .iter()
        .map(|s| f16::from_f32(*s as f32))
        .collect();
    Some(Halfvec::from_f16_vec(data))
}

// ---------------------------------------------------------------------
// SQL declarations
// ---------------------------------------------------------------------

extension_sql!(
    r#"
    -- Operators dispatch by argument type, so reusing <-> <#> <=> <+>
    -- is collision-free with both vector and pgvector.vector.
    CREATE OPERATOR <-> (
        LEFTARG = halfvec, RIGHTARG = halfvec,
        PROCEDURE = halfvec_l2_distance,
        COMMUTATOR = '<->'
    );
    CREATE OPERATOR <#> (
        LEFTARG = halfvec, RIGHTARG = halfvec,
        PROCEDURE = halfvec_negative_inner_product,
        COMMUTATOR = '<#>'
    );
    CREATE OPERATOR <=> (
        LEFTARG = halfvec, RIGHTARG = halfvec,
        PROCEDURE = halfvec_cosine_distance,
        COMMUTATOR = '<=>'
    );
    CREATE OPERATOR <+> (
        LEFTARG = halfvec, RIGHTARG = halfvec,
        PROCEDURE = halfvec_l1_distance,
        COMMUTATOR = '<+>'
    );

    -- Casts between vector and halfvec; both directions explicit.
    CREATE CAST (vector  AS halfvec) WITH FUNCTION vector_to_halfvec(vector);
    CREATE CAST (halfvec AS vector)  WITH FUNCTION halfvec_to_vector(halfvec);
    CREATE CAST (real[]  AS halfvec) WITH FUNCTION array_to_halfvec(real[]);
    CREATE CAST (halfvec AS real[])  WITH FUNCTION halfvec_to_array(halfvec);

    -- Aggregates.
    CREATE AGGREGATE avg(halfvec) (
        SFUNC = halfvec_accum,
        STYPE = HalfvecAccum,
        FINALFUNC = halfvec_avg_finalfn,
        COMBINEFUNC = halfvec_combine,
        PARALLEL = SAFE
    );
    CREATE AGGREGATE sum(halfvec) (
        SFUNC = halfvec_accum,
        STYPE = HalfvecAccum,
        FINALFUNC = halfvec_sum_finalfn,
        COMBINEFUNC = halfvec_combine,
        PARALLEL = SAFE
    );
    "#,
    name = "halfvec_surface",
    requires = [
        Halfvec,
        HalfvecAccum,
        halfvec_l2_distance,
        halfvec_negative_inner_product,
        halfvec_cosine_distance,
        halfvec_l1_distance,
        vector_to_halfvec,
        halfvec_to_vector,
        array_to_halfvec,
        halfvec_to_array,
        halfvec_accum,
        halfvec_combine,
        halfvec_avg_finalfn,
        halfvec_sum_finalfn
    ]
);
