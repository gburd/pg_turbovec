//! Distance operators and casts for `sparsevec`.
//!
//! Distance kernels exploit the sparsity: for two sparse vectors a
//! and b, inner-product is computed by walking the union of their
//! index sets, which is O(nnz_a + nnz_b) vs O(dim) for the dense
//! kernel. L2 / L1 are similar.

use pgrx::prelude::*;

use crate::sparsevec::Sparsevec;
use crate::vec::Vector;

/// Two-pointer walk over the union of two sorted-unique index sets.
fn sparse_walk<F: FnMut(f32, f32)>(a: &Sparsevec, b: &Sparsevec, mut visit: F) {
    let (ai, av) = (&a.indices, &a.values);
    let (bi, bv) = (&b.indices, &b.values);
    let (mut i, mut j) = (0usize, 0usize);
    while i < ai.len() && j < bi.len() {
        match ai[i].cmp(&bi[j]) {
            std::cmp::Ordering::Equal => {
                visit(av[i], bv[j]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                visit(av[i], 0.0);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                visit(0.0, bv[j]);
                j += 1;
            }
        }
    }
    while i < ai.len() {
        visit(av[i], 0.0);
        i += 1;
    }
    while j < bi.len() {
        visit(0.0, bv[j]);
        j += 1;
    }
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_inner_product(a: Sparsevec, b: Sparsevec) -> f64 {
    a.check_same_dim(&b, "inner_product");
    let mut acc: f64 = 0.0;
    sparse_walk(&a, &b, |x, y| acc += f64::from(x) * f64::from(y));
    acc
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_negative_inner_product(a: Sparsevec, b: Sparsevec) -> f64 {
    -sparsevec_inner_product(a, b)
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_l2_squared_distance(a: Sparsevec, b: Sparsevec) -> f64 {
    a.check_same_dim(&b, "l2_squared_distance");
    let mut acc: f64 = 0.0;
    sparse_walk(&a, &b, |x, y| {
        let d = f64::from(x) - f64::from(y);
        acc += d * d;
    });
    acc
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_l2_distance(a: Sparsevec, b: Sparsevec) -> f64 {
    sparsevec_l2_squared_distance(a, b).sqrt()
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_l1_distance(a: Sparsevec, b: Sparsevec) -> f64 {
    a.check_same_dim(&b, "l1_distance");
    let mut acc: f64 = 0.0;
    sparse_walk(&a, &b, |x, y| {
        acc += (f64::from(x) - f64::from(y)).abs()
    });
    acc
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_norm(v: Sparsevec) -> f64 {
    let mut acc: f64 = 0.0;
    for x in &v.values {
        acc += f64::from(*x) * f64::from(*x);
    }
    acc.sqrt()
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_cosine_distance(a: Sparsevec, b: Sparsevec) -> f64 {
    a.check_same_dim(&b, "<=>");
    let na = sparsevec_norm(a.clone());
    let nb = sparsevec_norm(b.clone());
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    let ip = sparsevec_inner_product(a, b);
    (1.0 - (ip / (na * nb)).clamp(-1.0, 1.0)).max(0.0)
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_dims(v: Sparsevec) -> i32 {
    v.dim()
}

#[pg_extern(immutable, parallel_safe)]
fn sparsevec_nnz(v: Sparsevec) -> i32 {
    v.nnz() as i32
}

// ---------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------

/// `vector::sparsevec` — convert dense to sparse, keeping only
/// non-zero coordinates.
#[pg_extern(immutable, parallel_safe)]
fn vector_to_sparsevec(v: Vector) -> Sparsevec {
    let dim = v.dim() as i32;
    let mut indices = ::std::vec::Vec::new();
    let mut values = ::std::vec::Vec::new();
    for (i, x) in v.as_slice().iter().enumerate() {
        if *x != 0.0 {
            indices.push(i as i32);
            values.push(*x);
        }
    }
    Sparsevec::new(dim, indices, values)
}

/// `sparsevec::vector` — materialise the dense form. Allocates
/// `dim * 4` bytes; beware on million-dim sparsevecs.
#[pg_extern(immutable, parallel_safe)]
fn sparsevec_to_vector(v: Sparsevec) -> Vector {
    Vector::from_vec(v.to_dense())
}

extension_sql!(
    r#"
    CREATE OPERATOR <-> (
        LEFTARG = sparsevec, RIGHTARG = sparsevec,
        PROCEDURE = sparsevec_l2_distance,
        COMMUTATOR = '<->'
    );
    CREATE OPERATOR <#> (
        LEFTARG = sparsevec, RIGHTARG = sparsevec,
        PROCEDURE = sparsevec_negative_inner_product,
        COMMUTATOR = '<#>'
    );
    CREATE OPERATOR <=> (
        LEFTARG = sparsevec, RIGHTARG = sparsevec,
        PROCEDURE = sparsevec_cosine_distance,
        COMMUTATOR = '<=>'
    );
    CREATE OPERATOR <+> (
        LEFTARG = sparsevec, RIGHTARG = sparsevec,
        PROCEDURE = sparsevec_l1_distance,
        COMMUTATOR = '<+>'
    );

    CREATE CAST (vector    AS sparsevec) WITH FUNCTION vector_to_sparsevec(vector);
    CREATE CAST (sparsevec AS vector)    WITH FUNCTION sparsevec_to_vector(sparsevec);
    "#,
    name = "sparsevec_surface",
    requires = [
        Sparsevec,
        sparsevec_l2_distance,
        sparsevec_negative_inner_product,
        sparsevec_cosine_distance,
        sparsevec_l1_distance,
        vector_to_sparsevec,
        sparsevec_to_vector
    ]
);
