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

use pgrx::callconv::{Arg, ArgAbi};
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, ReturnsError, ReturnsRef, SqlMappingRef, SqlTranslatable, TypeOrigin,
};
use pgrx::prelude::*;
use pgrx::{pg_func_extra, vardata_any, varsize_any_exhdr};

use crate::kernels;
use crate::vec::{MAX_DIM, Vector};

// ---------------------------------------------------------------------
// Operand decode cache.
//
// pgrx CBOR-decodes every by-value `Vector` argument on every call. In
// an ORDER BY recheck or `sum(emb <=> q)` the query operand is the same
// datum for every row, so it was decoded once per candidate (~3.7 us at
// 1024-d on an AVX2 laptop, ~2.3-2.5 us on EC2 c7i; see
// benches/results/perf_abc_20261006/fix_b/). The distance functions
// below take `VectorArg` (the raw datum)
// and decode through a per-FmgrInfo cache in `fn_extra` instead.
//
// Staleness is impossible by construction: a slot is reused only when
// the incoming argument's detoasted varlena payload is byte-identical to
// the payload the cached value was decoded from, and CBOR decode is a
// pure function of those bytes. Pointer identity is never trusted
// (per-tuple memory is reused at the same address with new contents).
// ---------------------------------------------------------------------

/// A `vector` argument handed to the function undecoded. Same SQL type
/// as [`Vector`]; only the Rust-side unboxing differs.
///
/// PGRX COUPLING: `ArgAbi` and `SqlTranslatable` are pgrx-internal
/// traits ("very subject to change between versions"). These impls are
/// sound for pgrx =0.19.1, which Cargo.toml pins exactly: unboxing
/// mirrors what `#[derive(PostgresType)]` generates for `Vector` minus
/// the CBOR decode, and the `SqlTranslatable` consts are `Vector`'s, so
/// the generated CREATE FUNCTION is unchanged (pinned by
/// `distance_cache_null_and_markings_unchanged`). On any pgrx bump,
/// diff `cargo pgrx schema` against the last release (order-independent)
/// and rerun the `distance_cache_*` tests.
pub struct VectorArg(pg_sys::Datum);

// SAFETY: the six functions taking `VectorArg` are declared STRICT
// (inferred by pgrx because no argument is `Option`), so Postgres never
// calls them with a NULL here; a NULL would still panic (ERROR), not be
// read. The value is kept as a raw `Datum` and only ever read through
// `pg_detoast_datum_packed` + `Vector::from_datum` in `Slot::get`, the
// same path pgrx's derived `FromDatum` for `Vector` takes, while the
// call (and so the argument's memory) is live.
unsafe impl<'fcx> ArgAbi<'fcx> for VectorArg {
    unsafe fn unbox_arg_unchecked(arg: Arg<'_, 'fcx>) -> Self {
        let index = arg.index();
        unsafe { arg.unbox_arg_using_from_datum::<pg_sys::Datum>() }
            .map(VectorArg)
            .unwrap_or_else(|| panic!("argument {index} must not be null"))
    }
}

// SAFETY: every const is `Vector`'s, so SQL sees exactly the `vector`
// type `Vector` maps to; a `VectorArg` is only ever unboxed from a datum
// of that type (see the `ArgAbi` impl). It is argument-only: no function
// returns one.
unsafe impl SqlTranslatable for VectorArg {
    const TYPE_IDENT: &'static str = <Vector as SqlTranslatable>::TYPE_IDENT;
    const TYPE_ORIGIN: TypeOrigin = <Vector as SqlTranslatable>::TYPE_ORIGIN;
    const ARGUMENT_SQL: Result<SqlMappingRef, ArgumentError> =
        <Vector as SqlTranslatable>::ARGUMENT_SQL;
    const RETURN_SQL: Result<ReturnsRef, ReturnsError> = <Vector as SqlTranslatable>::RETURN_SQL;
}

/// Test-only counters: CBOR decodes performed, and distance calls made.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) static DECODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(any(test, feature = "pg_test"))]
pub(crate) static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Test-only: `OperandCache`s currently alive (created minus dropped), to
/// prove the `fn_mcxt` reset callback really frees them.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) static LIVE_CACHES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// One decoded operand and the exact payload bytes it was decoded from.
#[derive(Default)]
struct Slot {
    raw: Vec<u8>,
    vec: Option<Vector>,
}

impl Slot {
    /// The decoded `vector` for `datum`, re-decoding only when its
    /// payload differs from the cached one.
    ///
    /// ponytail: a slot whose argument varies per row (the candidate
    /// side) misses every call and pays a memcmp-to-first-difference
    /// plus a payload memcpy (~5 KB at 1024-d) on top of the decode it
    /// always paid; stop caching a slot after N straight misses if that
    /// ever shows up in a profile.
    unsafe fn get(&mut self, datum: pg_sys::Datum) -> &Vector {
        unsafe {
            // Same detoast pgrx's cbor_decode does; a no-op for inline values.
            let p = pg_sys::pg_detoast_datum_packed(datum.cast_mut_ptr());
            let bytes =
                std::slice::from_raw_parts(vardata_any(p).cast::<u8>(), varsize_any_exhdr(p));
            if self.vec.is_none() || self.raw.as_slice() != bytes {
                #[cfg(any(test, feature = "pg_test"))]
                DECODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // pgrx's own FromDatum (cbor_decode), so values are
                // bit-identical to the old by-value `Vector` argument.
                let v = Vector::from_datum(pg_sys::Datum::from(p), false)
                    .expect("vector argument must not be null");
                self.vec = None;
                self.raw.clear();
                self.raw.extend_from_slice(bytes);
                self.vec = Some(v);
            }
            self.vec.as_ref().expect("slot filled above")
        }
    }
}

/// Per-FmgrInfo cache, one slot per argument position, so the constant
/// operand hits whether it is written on the left or the right.
struct OperandCache([Slot; 2]);

impl Default for OperandCache {
    fn default() -> Self {
        #[cfg(any(test, feature = "pg_test"))]
        LIVE_CACHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        OperandCache(Default::default())
    }
}

#[cfg(any(test, feature = "pg_test"))]
impl Drop for OperandCache {
    fn drop(&mut self) {
        LIVE_CACHES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Decode both operands (through the `fn_extra` cache when there is an
/// FmgrInfo to hang it on) and run `f` on them.
fn with_operands<R>(
    fcinfo: pg_sys::FunctionCallInfo,
    a: VectorArg,
    b: VectorArg,
    f: impl FnOnce(&Vector, &Vector) -> R,
) -> R {
    #[cfg(any(test, feature = "pg_test"))]
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    unsafe {
        if (*fcinfo).flinfo.is_null() {
            // DirectFunctionCall-style invocation: nothing to cache in.
            let (mut x, mut y) = (Slot::default(), Slot::default());
            return f(x.get(a.0), y.get(b.0));
        }
        // Lives in fn_mcxt; pgrx drops it (freeing the Rust Vecs) when
        // that context is reset or deleted.
        let mut cache = pg_func_extra(fcinfo, OperandCache::default);
        let [x, y] = &mut cache.0;
        f(x.get(a.0), y.get(b.0))
    }
}

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
fn l2_distance(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "<->");
        kernels::l2_sq(a.as_slice(), b.as_slice()).sqrt()
    })
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
fn l2_squared_distance(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "l2_squared_distance");
        kernels::l2_sq(a.as_slice(), b.as_slice())
    })
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
fn inner_product(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "inner_product");
        kernels::dot(a.as_slice(), b.as_slice())
    })
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
fn negative_inner_product(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "<#>");
        -kernels::dot(a.as_slice(), b.as_slice())
    })
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
fn cosine_distance(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "<=>");
        // TODO(fix-a seam): once kernels::cosine_distance_with_qnorm lands,
        // cache the query's norm2 in the Slot (computed when the slot is
        // filled) and pass it here for the side that hit the cache.
        kernels::cosine_distance(a.as_slice(), b.as_slice())
    })
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
fn l1_distance(a: VectorArg, b: VectorArg, fcinfo: pg_sys::FunctionCallInfo) -> f64 {
    with_operands(fcinfo, a, b, |a, b| {
        a.check_same_dim(b, "<+>");
        kernels::l1_abs(a.as_slice(), b.as_slice())
    })
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

/// Concatenate two `vector`s into one of dim `dim(a) + dim(b)`.
/// Mirrors pgvector's `vector_concat`. Errors if the combined dim
/// exceeds `MAX_DIM`.
#[pg_extern(name = "vector_concat", immutable, parallel_safe)]
fn vec_concat(a: Vector, b: Vector) -> Vector {
    if a.dim() + b.dim() > MAX_DIM {
        error!(
            "operand dimensions {} + {} exceed maximum {} for vector concatenation",
            a.dim(),
            b.dim(),
            MAX_DIM
        );
    }
    let mut out = Vec::with_capacity(a.dim() + b.dim());
    out.extend_from_slice(a.as_slice());
    out.extend_from_slice(b.as_slice());
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

    CREATE OPERATOR || (
        LEFTARG = vector, RIGHTARG = vector,
        PROCEDURE = vector_concat
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
        vec_mul,
        vec_concat
    ]
);
