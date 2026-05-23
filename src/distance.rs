//! Distance functions and operators for `vector`.
//!
//! All distance functions are immutable and parallel-safe. Operators
//! are wired up via `extension_sql!` so they appear in `pg_operator`
//! and become available for `ORDER BY embedding <#> $1` style queries.
//!
//! Math kernels live in `crate::kernels` (no Postgres dependency,
//! exercised by `cargo test` directly).
//!
//! | Operator | Postgres semantics                | Function           |
//! |----------|-----------------------------------|--------------------|
//! | `<->`    | Euclidean (L2) distance           | `l2_distance`      |
//! | `<#>`    | *Negative* inner product          | `negative_inner_product` |
//! | `<=>`    | Cosine distance (1 - cos θ)       | `cosine_distance`  |
//! | `<+>`    | Taxicab (L1) distance             | `l1_distance`      |
//!
//! `<#>` returns the *negative* inner product so that `ORDER BY a <#> b`
//! sorts most-similar-first under ascending order, matching pgvector.

use pgrx::prelude::*;

use crate::kernels;
use crate::vec::Vector;

// ---------------------------------------------------------------------
// SQL-callable distance functions (mirrors pgvector's named functions).
// ---------------------------------------------------------------------

/// Euclidean (L2) distance between two equal-dimension `vector`s.
///
/// ```ignore
/// SELECT turbovec.l2_distance(
///     '[1,2,3]'::turbovec.vector,
///     '[4,6,3]'::turbovec.vector
/// );
/// -- returns 5.0  (sqrt(9 + 16 + 0))
/// ```
///
/// Both arguments must have the same dim; mismatch raises an ERROR.
#[pg_extern(immutable, parallel_safe)]
fn l2_distance(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "<->");
    kernels::l2_sq(a.as_slice(), b.as_slice()).sqrt()
}

/// Squared Euclidean distance — useful when you only need order, not
/// magnitudes. Matches pgvector's `vector_l2_squared_distance`.
///
/// ```ignore
/// SELECT turbovec.l2_squared_distance(
///     '[1,2,3]'::turbovec.vector,
///     '[4,6,3]'::turbovec.vector
/// );
/// -- returns 25.0  (= l2_distance(...) ^ 2)
/// ```
#[pg_extern(immutable, parallel_safe)]
fn l2_squared_distance(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "l2_squared_distance");
    kernels::l2_sq(a.as_slice(), b.as_slice())
}

/// Inner (dot) product.
///
/// ```ignore
/// SELECT turbovec.inner_product(
///     '[1,2,3]'::turbovec.vector,
///     '[4,5,6]'::turbovec.vector
/// );
/// -- returns 32.0
/// ```
#[pg_extern(immutable, parallel_safe)]
fn inner_product(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "inner_product");
    kernels::dot(a.as_slice(), b.as_slice())
}

/// Negative inner product — used by the `<#>` operator and by the
/// `vec_ip_ops` index opclass so that ascending sort returns the
/// most-similar rows first.
///
/// ```ignore
/// SELECT turbovec.negative_inner_product(
///     '[1,2,3]'::turbovec.vector,
///     '[4,5,6]'::turbovec.vector
/// );
/// -- returns -32.0
///
/// -- Equivalent operator form (most-similar-first ASC):
/// SELECT id
/// FROM   docs
/// ORDER  BY emb <#> '[...]'::vector
/// LIMIT  10;
/// ```
#[pg_extern(immutable, parallel_safe)]
fn negative_inner_product(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "<#>");
    -kernels::dot(a.as_slice(), b.as_slice())
}

/// Cosine distance: `1 - cos θ` where `cos θ = dot(a, b) / (||a|| * ||b||)`.
/// Returns `NaN` if either operand is the zero vector, matching pgvector.
///
/// ```ignore
/// SELECT turbovec.cosine_distance(
///     '[1,0]'::turbovec.vector,
///     '[0,1]'::turbovec.vector
/// );
/// -- returns 1.0  (perpendicular: cos = 0, distance = 1 - 0)
///
/// SELECT turbovec.cosine_distance(
///     '[0,0,0]'::turbovec.vector,
///     '[1,2,3]'::turbovec.vector
/// );
/// -- returns NaN
/// ```
#[pg_extern(immutable, parallel_safe)]
fn cosine_distance(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "<=>");
    kernels::cosine_distance(a.as_slice(), b.as_slice())
}

/// Taxicab (L1) distance.
///
/// ```ignore
/// SELECT turbovec.l1_distance(
///     '[1,2,3]'::turbovec.vector,
///     '[4,6,3]'::turbovec.vector
/// );
/// -- returns 7.0  (|1-4| + |2-6| + |3-3|)
/// ```
#[pg_extern(immutable, parallel_safe)]
fn l1_distance(a: Vector, b: Vector) -> f64 {
    a.check_same_dim(&b, "<+>");
    kernels::l1_abs(a.as_slice(), b.as_slice())
}

/// Number of dimensions in a `vector`.
///
/// ```ignore
/// SELECT turbovec.vector_dims('[1,2,3,4,5]'::turbovec.vector);
/// -- returns 5
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vector_dims(v: Vector) -> i32 {
    v.dim() as i32
}

/// Euclidean (L2) norm of a `vector`.
///
/// ```ignore
/// SELECT turbovec.vector_norm('[3,4]'::turbovec.vector);
/// -- returns 5.0
///
/// SELECT turbovec.vector_norm('[0,0,0]'::turbovec.vector);
/// -- returns 0.0
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vector_norm(v: Vector) -> f64 {
    kernels::norm2(v.as_slice()).sqrt()
}

/// Element-wise sum of two equal-dimension `vector`s.
#[pg_extern(immutable, parallel_safe)]
fn vec_add(a: Vector, b: Vector) -> Vector {
    a.check_same_dim(&b, "+");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x + *y);
    }
    Vector::from_vec(out)
}

/// Element-wise difference of two equal-dimension `vector`s.
#[pg_extern(immutable, parallel_safe)]
fn vec_sub(a: Vector, b: Vector) -> Vector {
    a.check_same_dim(&b, "-");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x - *y);
    }
    Vector::from_vec(out)
}

/// Element-wise (Hadamard) product of two equal-dimension `vector`s.
#[pg_extern(immutable, parallel_safe)]
fn vec_mul(a: Vector, b: Vector) -> Vector {
    a.check_same_dim(&b, "*");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x * *y);
    }
    Vector::from_vec(out)
}

// ---------------------------------------------------------------------
// Operators. We declare these via `extension_sql!` so the SQL is
// emitted exactly as we want it (with COMMUTATOR / NEGATOR clauses
// where appropriate).
// ---------------------------------------------------------------------

extension_sql!(
    r"
    CREATE OPERATOR <-> (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = l2_distance,
        COMMUTATOR = '<->'
    );

    CREATE OPERATOR <#> (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = negative_inner_product,
        COMMUTATOR = '<#>'
    );

    CREATE OPERATOR <=> (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = cosine_distance,
        COMMUTATOR = '<=>'
    );

    CREATE OPERATOR <+> (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = l1_distance,
        COMMUTATOR = '<+>'
    );

    CREATE OPERATOR + (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = vec_add,
        COMMUTATOR = '+'
    );

    CREATE OPERATOR - (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = vec_sub
    );

    CREATE OPERATOR * (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = vec_mul,
        COMMUTATOR = '*'
    );
    ",
    name = "vec_operators",
    requires = [
        Vector,
        l2_distance,
        negative_inner_product,
        cosine_distance,
        l1_distance,
        vec_add,
        vec_sub,
        vec_mul
    ]
);
