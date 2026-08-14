//! Phase 5 extras: pgvector-parity helpers that don't fit cleanly
//! into the type / distance / aggregate modules.
//!
//! - `subvector(vector, start, length) -> vector` — 1-indexed
//!   slice, mirrors pgvector's `subvector`.
//! - `vec_to_jsonb(vector)` — explicit JSON output (handy for
//!   logging, replication via JSONB columns).
//! - `jsonb_to_vec(jsonb)` — inverse.
//! - `vec_check_dim(vector, integer)` — runtime dim assertion;
//!   raises ERROR if mismatch. Cheaper than typmod plumbing.
//! - `vec_zeros(integer)` — zero-filled vector helper.
//! - `vec_to_text(vector)` — explicit text representation
//!   (the IO function's output, callable as a regular function).

use pgrx::prelude::*;
use serde_json::{Value, json};

use crate::vec::{MAX_DIM, Vector};

/// `subvector(v, start, length)` — 1-indexed slice (matches pgvector).
/// `start` and `length` must be positive and the resulting range must
/// lie within `v`.
///
/// ```ignore
/// SELECT turbovec.subvector('[10, 20, 30, 40]'::turbovec.vector, 2, 2)::text;
/// -- returns '[20, 30]'
///
/// -- Out-of-bounds raises ERROR:
/// SELECT turbovec.subvector('[1, 2, 3]'::turbovec.vector, 2, 5);
/// -- ERROR: subvector: range 2..6 is out of bounds for vector of dim 3
/// ```
#[pg_extern(immutable, parallel_safe)]
fn subvector(v: Vector, start: i32, length: i32) -> Vector {
    if start < 1 {
        error!(
            "subvector: start ({}) must be a positive 1-indexed offset",
            start
        );
    }
    if length < 1 {
        error!("subvector: length ({}) must be positive", length);
    }
    let s = (start - 1) as usize;
    let l = length as usize;
    if s + l > v.dim() {
        error!(
            "subvector: range {}..{} is out of bounds for vector of dim {}",
            start,
            start + length - 1,
            v.dim()
        );
    }
    Vector::from_vec(v.as_slice()[s..s + l].to_vec())
}

/// Materialise a `vector` as a `jsonb` array of numbers.
///
/// ```ignore
/// SELECT turbovec.vec_to_jsonb('[1, 2.5, -3]'::turbovec.vector);
/// -- returns [1, 2.5, -3]::jsonb
///
/// -- Equivalent cast form:
/// SELECT '[1, 2.5, -3]'::turbovec.vector::jsonb;
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_to_jsonb(v: Vector) -> pgrx::JsonB {
    let arr: Vec<Value> = v
        .as_slice()
        .iter()
        .map(|x| Value::from(f64::from(*x)))
        .collect();
    pgrx::JsonB(json!(arr))
}

/// Parse a `jsonb` array of numbers as a `vector`. Rejects non-array
/// inputs and non-numeric / non-finite elements.
///
/// ```ignore
/// SELECT turbovec.jsonb_to_vec('[1, 2.5, 3]'::jsonb)::text;
/// -- returns '[1, 2.5, 3]'
///
/// -- Errors:
/// SELECT turbovec.jsonb_to_vec('{"a": 1}'::jsonb);     -- ERROR (not array)
/// SELECT turbovec.jsonb_to_vec('[1, "x", 3]'::jsonb);  -- ERROR (string elem)
/// SELECT turbovec.jsonb_to_vec('[1, null, 3]'::jsonb); -- ERROR (null elem)
/// ```
#[pg_extern(immutable, parallel_safe)]
fn jsonb_to_vec(j: pgrx::JsonB) -> Vector {
    let arr = match j.0 {
        Value::Array(a) => a,
        other => error!(
            "jsonb_to_vec: expected JSON array, got {}",
            value_kind(&other)
        ),
    };
    if arr.is_empty() || arr.len() > MAX_DIM {
        error!(
            "jsonb_to_vec: dim {} out of range 1..={}",
            arr.len(),
            MAX_DIM
        );
    }
    let mut out: Vec<f32> = Vec::with_capacity(arr.len());
    for (i, v) in arr.into_iter().enumerate() {
        let n = v.as_f64().unwrap_or_else(|| {
            error!(
                "jsonb_to_vec: element {} is not a number ({})",
                i,
                value_kind(&v)
            )
        });
        if !n.is_finite() {
            error!("jsonb_to_vec: element {} is not finite ({})", i, n);
        }
        out.push(n as f32);
    }
    Vector::from_vec(out)
}

/// Returns the kind name of a `serde_json::Value` for error messages.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Raise an ERROR if `v.dim() != expected`, otherwise return `v`
/// unchanged. Useful as `CHECK (turbovec.vec_check_dim(emb, 1536))`.
///
/// ```ignore
/// CREATE TABLE docs (
///     id  bigserial PRIMARY KEY,
///     emb turbovec.vector
///         CHECK (turbovec.vector_dims(
///             turbovec.vec_check_dim(emb, 1536)) = 1536)
/// );
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_check_dim(v: Vector, expected: i32) -> Vector {
    if expected < 1 {
        error!("vec_check_dim: expected dim must be positive");
    }
    if v.dim() != expected as usize {
        error!(
            "vec_check_dim: dim mismatch (got {}, expected {})",
            v.dim(),
            expected
        );
    }
    v
}

/// Build a zero-filled `vector` of the requested dimension. Useful
/// as the identity for `sum(vector)` in extension queries.
///
/// ```ignore
/// SELECT turbovec.vector_dims(turbovec.vec_zeros(8));
/// -- returns 8
///
/// SELECT turbovec.vector_norm(turbovec.vec_zeros(8));
/// -- returns 0.0
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_zeros(dim: i32) -> Vector {
    if dim <= 0 || dim as usize > MAX_DIM {
        error!("vec_zeros: dim {} out of range 1..={}", dim, MAX_DIM);
    }
    Vector::from_vec(vec![0.0_f32; dim as usize])
}

/// Explicit text rendering of a `vector` (mirrors the type's OUTPUT
/// function but callable directly).
///
/// ```ignore
/// SELECT turbovec.vec_to_text('[1, 2.5, -3]'::turbovec.vector);
/// -- returns '[1, 2.5, -3]'
/// ```
#[pg_extern(immutable, parallel_safe)]
fn vec_to_text(v: Vector) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(2 + v.dim() * 6);
    out.push('[');
    let mut first = true;
    for x in v.as_slice() {
        if !first {
            out.push_str(", ");
        }
        first = false;
        let _ = write!(out, "{x}");
    }
    out.push(']');
    out
}

/// `turbovec.index_is_degraded(regclass) -> bool` — Phase E-2
/// operator signal. Returns `true` when the given turbovec index was
/// built `WITH (lists > 0)` (an IVF index) but has degraded to a flat
/// O(n) scan (its IVF cell metadata was invalidated). A churning
/// production deployment can poll this to detect the silent latency
/// cliff and `REINDEX` before it bites.
///
/// Returns `false` for a healthy IVF index, a flat (`lists = 0`)
/// index, and a non-turbovec or empty index (nothing to degrade).
/// With the tombstone-vacuum path an IVF index survives VACUUM and
/// stays healthy, so this should normally read `false`; a `true`
/// means a fallback path fired and a REINDEX is warranted.
///
/// ```ignore
/// SELECT turbovec.index_is_degraded('my_ivf_idx'::regclass);
/// ```
#[pg_extern(stable, parallel_safe)]
fn index_is_degraded(index: pg_sys::Oid) -> bool {
    unsafe {
        let rel = pg_sys::index_open(index, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            return false;
        }
        let degraded = crate::index::relfile::read_meta(rel)
            .map(|m| m.is_degraded())
            .unwrap_or(false);
        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        degraded
    }
}

/// Ownership check portable across PG13-19: `object_ownercheck`
/// (PG16+) vs the older `pg_class_ownercheck` (PG13-15). Superusers
/// always pass. `pg_class`'s catalog OID is 1259
/// (`RelationRelationId`). ERRORs (never returns `false`) when the
/// caller doesn't own the relation, matching how PG's own
/// maintenance functions gate access.
unsafe fn require_index_owner(index: pg_sys::Oid) {
    let roleid = pg_sys::GetUserId();
    if pg_sys::superuser() {
        return;
    }
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
    let owns = pg_sys::pg_class_ownercheck(index, roleid);
    #[cfg(not(any(feature = "pg13", feature = "pg14", feature = "pg15")))]
    let owns = pg_sys::object_ownercheck(pg_sys::RelationRelationId, index, roleid);
    if !owns {
        error!("turbovec_check: permission denied — must own the index");
    }
}

/// `turbovec.turbovec_check(regclass)` — read-only integrity report
/// for a turbovec index. Reads the meta page + ids chain and reports
/// enough for an operator to detect the ".tvim" duplicate-id
/// corruption WITHOUT attempting a write (which is the only signal
/// the pre-1.28.4 detection gave, and which blocks the whole table).
///
/// Columns:
/// - `wire_version` — `MetaPageData::version` (7 for current builds;
///   `< 7` is a pre-Phase-Q-0 legacy index needing REINDEX).
/// - `kind` — `single` / `colbert` / `graph`.
/// - `n_vectors` — the row count the meta page claims.
/// - `slot_count` — the number of ids actually present in the ids
///   chain. MUST equal `n_vectors`; a mismatch means the meta
///   over/under-counts the ids chain (the exact drift that produces
///   "id 0 in more than one slot").
/// - `count_matches` — `n_vectors == slot_count`.
/// - `duplicate_id` — the first id appearing in more than one slot,
///   or NULL when the ids form a clean bijection (via
///   `scan::first_duplicate_id`). Only meaningful for the flat
///   (`lists = 0`) kind; an IVF index legitimately repeats ids
///   across cells, so this is left NULL there.
/// - `is_corrupt` — `true` when a flat index has a duplicate id OR
///   the counts disagree. This is the one column monitoring should
///   alert on.
/// - `tombstone_density` — fraction of slots tombstoned by VACUUM
///   (0.0 for a freshly built index).
///
/// Ownership-checked (like PG's own maintenance functions):
/// non-owners get a permission-denied ERROR. Takes only
/// `AccessShareLock`, so it never blocks writers.
///
/// ```ignore
/// SELECT * FROM turbovec.turbovec_check('my_idx'::regclass);
/// ```
#[pg_extern(stable, parallel_safe)]
fn turbovec_check(
    index: pg_sys::Oid,
) -> TableIterator<
    'static,
    (
        name!(wire_version, i32),
        name!(kind, String),
        name!(n_vectors, i64),
        name!(slot_count, i64),
        name!(count_matches, bool),
        name!(duplicate_id, Option<i64>),
        name!(is_corrupt, bool),
        name!(tombstone_density, f64),
    ),
> {
    use crate::index::page::{KIND_COLBERT, KIND_GRAPH};
    unsafe {
        require_index_owner(index);
        let rel = pg_sys::index_open(index, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            error!("turbovec_check: could not open index {:?}", index);
        }
        let Some(meta) = crate::index::relfile::read_meta(rel) else {
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            error!(
                "turbovec_check: relation {:?} has no turbovec meta page (not a turbovec index, or empty)",
                index
            );
        };

        let kind = match meta.kind {
            KIND_COLBERT => "colbert",
            KIND_GRAPH => "graph",
            _ => "single",
        }
        .to_string();

        // Read only the ids chain (via `read_ids_only`); we don't
        // need the much larger codes/scales chains for an integrity
        // report.
        //
        // v1.29.1 corruption fix (monitor consistency): `read_meta`
        // above and `read_ids_only` below must observe the SAME
        // relfile snapshot, or a concurrent insert-flush / vacuum
        // committing between them makes `count_matches` a spurious
        // `false` (meta.n_vectors from before the rewrite vs an ids
        // chain from after). Re-read the meta UNDER the shared
        // rewrite lock, atomically with the ids, and compare THOSE.
        // `read_ids_only` takes + re-reads meta under the ShareLock;
        // we take our own ShareLock around BOTH the meta re-read and
        // the ids read so they're one consistent observation. Same-
        // xact ShareLock nesting does not self-conflict.
        crate::index::relfile::lock_relfile_read(rel);
        let meta_consistent = crate::index::relfile::read_meta(rel).unwrap_or(meta);
        let ids = crate::index::relfile::read_ids_only(rel, &meta_consistent);
        crate::index::relfile::unlock_relfile_read(rel);
        let meta = meta_consistent;
        let slot_count = ids.len() as i64;
        let count_matches = meta.n_vectors == ids.len() as u64;

        // Duplicate-id check is only a corruption signal for the
        // bijective flat kind. IVF (lists > 0) repeats ids across
        // cells by design (see scan::assert_ids_unique_or_reindex).
        let dup = if meta.lists == 0 {
            crate::index::scan::first_duplicate_id(&ids)
        } else {
            None
        };
        let duplicate_id = dup.map(|id| id as i64);
        let is_corrupt = dup.is_some() || !count_matches;

        let tombstones = crate::index::relfile::read_tombstones(rel, &meta);
        let dead = tombstones
            .iter()
            .map(|b| b.count_ones() as u64)
            .sum::<u64>();
        let tombstone_density = if meta.n_vectors == 0 {
            0.0
        } else {
            dead as f64 / meta.n_vectors as f64
        };

        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);

        TableIterator::once((
            meta.version as i32,
            kind,
            meta.n_vectors as i64,
            slot_count,
            count_matches,
            duplicate_id,
            is_corrupt,
            tombstone_density,
        ))
    }
}

/// Test-only: acquire the EXCLUSIVE relfile-rewrite lock on `index`
/// and hold it until the end of the current transaction. Lets a
/// `#[pg_test]` (via `dblink` from another backend) prove that a
/// concurrent reader is SERIALIZED against a rewrite — the v1.29.1
/// corruption fix's core invariant. Not part of the shipped SQL
/// surface (only compiled under `pg_test`), so it never appears in
/// the extension schema.
///
/// # Safety
/// pgrx wrapper; opens/holds the index relation and takes a
/// heavyweight page lock that the lock manager releases at xact end.
#[cfg(any(test, feature = "pg_test"))]
#[pg_extern]
fn _turbovec_test_hold_rewrite_lock(index: pg_sys::Oid) -> bool {
    unsafe {
        let rel = pg_sys::index_open(index, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            error!(
                "_turbovec_test_hold_rewrite_lock: could not open index {:?}",
                index
            );
        }
        crate::index::relfile::lock_relfile_write(rel);
        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
    }
    true
}

/// DIAGNOSTIC (temporary, corruption investigation): return every
/// slot whose external id is 0 in the flat ids chain, with its
/// neighbours, plus the total id-0 count. Read-only, AccessShareLock.
#[pg_extern]
fn turbovec_dump_id0(
    index: pg_sys::Oid,
) -> TableIterator<
    'static,
    (
        name!(slot, i64),
        name!(id, i64),
        name!(prev_id, i64),
        name!(next_id, i64),
    ),
> {
    unsafe {
        let rel = pg_sys::index_open(index, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            error!("turbovec_dump_id0: could not open index {:?}", index);
        }
        let meta = crate::index::relfile::read_meta(rel)
            .unwrap_or_else(|| error!("turbovec_dump_id0: no meta page"));
        let ids = crate::index::relfile::read_ids_only(rel, &meta);
        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        let n = ids.len();
        let mut rows: Vec<(i64, i64, i64, i64)> = Vec::new();
        for (slot, &id) in ids.iter().enumerate() {
            if id == 0 {
                let prev = if slot > 0 { ids[slot - 1] as i64 } else { -1 };
                let next = if slot + 1 < n {
                    ids[slot + 1] as i64
                } else {
                    -1
                };
                rows.push((slot as i64, id as i64, prev, next));
            }
        }
        TableIterator::new(rows.into_iter())
    }
}

extension_sql!(
    r"
    -- jsonb <-> vector explicit casts.
    CREATE CAST (vector AS jsonb) WITH FUNCTION vec_to_jsonb(vector);
    CREATE CAST (jsonb   AS vector) WITH FUNCTION jsonb_to_vec(jsonb);
    ",
    name = "vec_jsonb_casts",
    requires = [vec_to_jsonb, jsonb_to_vec]
);
