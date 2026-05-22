//! Distance functions and operators for `tvector`.
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
use crate::tvector::Tvector;

// ---------------------------------------------------------------------
// SQL-callable distance functions (mirrors pgvector's named functions).
// ---------------------------------------------------------------------

/// Euclidean (L2) distance between two equal-dimension `tvector`s.
#[pg_extern(immutable, parallel_safe)]
fn l2_distance(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "<->");
    kernels::l2_sq(a.as_slice(), b.as_slice()).sqrt()
}

/// Squared Euclidean distance — useful when you only need order, not
/// magnitudes. Matches pgvector's `vector_l2_squared_distance`.
#[pg_extern(immutable, parallel_safe)]
fn l2_squared_distance(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "l2_squared_distance");
    kernels::l2_sq(a.as_slice(), b.as_slice())
}

/// Inner (dot) product.
#[pg_extern(immutable, parallel_safe)]
fn inner_product(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "inner_product");
    kernels::dot(a.as_slice(), b.as_slice())
}

/// Negative inner product — used by the `<#>` operator and by the
/// `tvector_ip_ops` index opclass so that ascending sort returns the
/// most-similar rows first.
#[pg_extern(immutable, parallel_safe)]
fn negative_inner_product(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "<#>");
    -kernels::dot(a.as_slice(), b.as_slice())
}

/// Cosine distance: `1 - cos θ` where `cos θ = dot(a, b) / (||a|| * ||b||)`.
/// Returns `NaN` if either operand is the zero vector, matching pgvector.
#[pg_extern(immutable, parallel_safe)]
fn cosine_distance(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "<=>");
    kernels::cosine_distance(a.as_slice(), b.as_slice())
}

/// Taxicab (L1) distance.
#[pg_extern(immutable, parallel_safe)]
fn l1_distance(a: Tvector, b: Tvector) -> f64 {
    a.check_same_dim(&b, "<+>");
    kernels::l1_abs(a.as_slice(), b.as_slice())
}

/// Number of dimensions in a `tvector`.
#[pg_extern(immutable, parallel_safe)]
fn vector_dims(v: Tvector) -> i32 {
    v.dim() as i32
}

/// Euclidean (L2) norm of a `tvector`.
#[pg_extern(immutable, parallel_safe)]
fn vector_norm(v: Tvector) -> f64 {
    kernels::norm2(v.as_slice()).sqrt()
}

/// Element-wise sum of two equal-dimension `tvector`s.
#[pg_extern(immutable, parallel_safe)]
fn tvector_add(a: Tvector, b: Tvector) -> Tvector {
    a.check_same_dim(&b, "+");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x + *y);
    }
    Tvector::from_vec(out)
}

/// Element-wise difference of two equal-dimension `tvector`s.
#[pg_extern(immutable, parallel_safe)]
fn tvector_sub(a: Tvector, b: Tvector) -> Tvector {
    a.check_same_dim(&b, "-");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x - *y);
    }
    Tvector::from_vec(out)
}

/// Element-wise (Hadamard) product of two equal-dimension `tvector`s.
#[pg_extern(immutable, parallel_safe)]
fn tvector_mul(a: Tvector, b: Tvector) -> Tvector {
    a.check_same_dim(&b, "*");
    let mut out = Vec::with_capacity(a.dim());
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        out.push(*x * *y);
    }
    Tvector::from_vec(out)
}

// ---------------------------------------------------------------------
// Operators. We declare these via `extension_sql!` so the SQL is
// emitted exactly as we want it (with COMMUTATOR / NEGATOR clauses
// where appropriate).
// ---------------------------------------------------------------------

extension_sql!(
    r"
    CREATE OPERATOR <-> (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = l2_distance,
        COMMUTATOR = '<->'
    );

    CREATE OPERATOR <#> (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = negative_inner_product,
        COMMUTATOR = '<#>'
    );

    CREATE OPERATOR <=> (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = cosine_distance,
        COMMUTATOR = '<=>'
    );

    CREATE OPERATOR <+> (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = l1_distance,
        COMMUTATOR = '<+>'
    );

    CREATE OPERATOR + (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = tvector_add,
        COMMUTATOR = '+'
    );

    CREATE OPERATOR - (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = tvector_sub
    );

    CREATE OPERATOR * (
        LEFTARG = tvector, RIGHTARG = tvector,
        PROCEDURE = tvector_mul,
        COMMUTATOR = '*'
    );
    ",
    name = "tvector_operators",
    requires = [
        Tvector,
        l2_distance,
        negative_inner_product,
        cosine_distance,
        l1_distance,
        tvector_add,
        tvector_sub,
        tvector_mul
    ]
);
