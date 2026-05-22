//! `ambeginscan` / `amrescan` / `amgettuple` / `amendscan` — the
//! query path. Lazily loads the persisted IdMapIndex on first
//! `amgettuple`, runs a single batch search, then drains results
//! one TID per call.

use std::ffi::c_int;

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::index::persist;
use crate::kernels;
use crate::tvector::Tvector;

/// Scan-private state. Lives in the scan's memory context (allocated
/// by `palloc0` so all fields start zeroed).
#[repr(C)]
pub(crate) struct ScanOpaque {
    /// Cached query vector — set by `amrescan`, consumed by the
    /// first `amgettuple`.
    query: Vec<f32>,
    /// Search results, populated lazily on first `amgettuple`.
    /// Each entry is a u64 in pgrx's canonical CTID encoding.
    results: Vec<u64>,
    /// Per-result distance scores in the same order as `results`.
    /// Used to populate `scan->xs_orderbyvals` so the executor can
    /// pipeline the order-by under `amcanorderbyop = true`.
    distances: Vec<f64>,
    /// Cursor into `results` / `distances`.
    cursor: usize,
    /// Whether the search has been executed yet.
    fetched: bool,
}

/// `ambeginscan`: allocate the IndexScanDesc and attach our opaque.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: c_int,
    norderbys: c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index_relation, nkeys, norderbys);
    if scan.is_null() {
        error!("turbovec: RelationGetIndexScan returned null");
    }

    // PostgreSQL leaves xs_orderbyvals / xs_orderbynulls null when
    // RelationGetIndexScan returns; AMs that advertise
    // `amcanorderbyop = true` must allocate them themselves.
    if norderbys > 0 {
        (*scan).xs_orderbyvals = pg_sys::palloc0(
            std::mem::size_of::<pg_sys::Datum>() * (norderbys as usize),
        ) as *mut pg_sys::Datum;
        (*scan).xs_orderbynulls = pg_sys::palloc0(
            std::mem::size_of::<bool>() * (norderbys as usize),
        ) as *mut bool;
        for i in 0..(norderbys as usize) {
            *(*scan).xs_orderbynulls.add(i) = true;
        }
    }

    // Allocate ScanOpaque inside the scan's memory context.
    let opaque_ptr = pg_sys::palloc0(std::mem::size_of::<ScanOpaque>()) as *mut ScanOpaque;
    if opaque_ptr.is_null() {
        error!("turbovec: failed to palloc ScanOpaque");
    }
    std::ptr::write(
        opaque_ptr,
        ScanOpaque {
            query: Vec::new(),
            results: Vec::new(),
            distances: Vec::new(),
            cursor: 0,
            fetched: false,
        },
    );

    (*scan).opaque = opaque_ptr as *mut std::ffi::c_void;
    scan
}

/// `amrescan`: pull the order-by query out of `orderbys[0]` and
/// stash it in our opaque.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    nkeys: c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: c_int,
) {
    if !keys.is_null() && nkeys > 0 {
        // Standard pattern: copy keys into the scan slot. We don't
        // use scan keys (only order-by), but we still memcpy to keep
        // the ScanDesc consistent.
        std::ptr::copy_nonoverlapping(
            keys,
            (*scan).keyData,
            (nkeys as usize) * std::mem::size_of::<pg_sys::ScanKeyData>(),
        );
    }

    if !orderbys.is_null() && norderbys > 0 {
        std::ptr::copy_nonoverlapping(
            orderbys,
            (*scan).orderByData,
            (norderbys as usize) * std::mem::size_of::<pg_sys::ScanKeyData>(),
        );
    }

    let opaque = (*scan).opaque as *mut ScanOpaque;
    if opaque.is_null() {
        error!("turbovec amrescan: opaque is null");
    }

    // We support exactly one order-by (the distance operator). The
    // operand is `(*orderbys).sk_argument`.
    if norderbys < 1 || orderbys.is_null() {
        // No ORDER BY \u2014 nothing to scan. Phase 5 may add a "scan
        // everything" mode for plain `WHERE` predicates; v0.4
        // intentionally rejects this combination.
        error!(
            "turbovec: index scan requires an ORDER BY <operator> <query> clause"
        );
    }
    let order = orderbys.add(0);
    let datum = (*order).sk_argument;
    let isnull = ((*order).sk_flags as u32) & pg_sys::SK_ISNULL != 0;
    if isnull {
        error!("turbovec: ORDER BY query datum is NULL");
    }

    let query: Tvector = match pgrx::FromDatum::from_datum(datum, false) {
        Some(v) => v,
        None => error!("turbovec: ORDER BY datum did not decode to tvector"),
    };

    let normalise = guc::NORMALIZE_ON_INSERT.get();
    (*opaque).query = if normalise {
        kernels::normalise_to_vec(query.as_slice())
    } else {
        query.as_slice().to_vec()
    };
    (*opaque).results.clear();
    (*opaque).distances.clear();
    (*opaque).cursor = 0;
    (*opaque).fetched = false;
}

/// `amgettuple`: on first call run the search and cache results;
/// subsequent calls drain one TID at a time.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let opaque = (*scan).opaque as *mut ScanOpaque;
    if opaque.is_null() {
        error!("turbovec amgettuple: opaque is null");
    }

    if !(*opaque).fetched {
        let indexrelid = (*(*scan).indexRelation).rd_id;
        let stored = match persist::load(indexrelid) {
            Some(s) => s,
            None => {
                (*opaque).fetched = true;
                return false;
            }
        };
        let dim = stored.dim as usize;
        if (*opaque).query.len() != dim {
            error!(
                "turbovec amgettuple: query dim {} != index dim {}",
                (*opaque).query.len(),
                dim
            );
        }
        let n_in_index = stored.n_vectors.max(0) as usize;
        if n_in_index == 0 {
            (*opaque).fetched = true;
            return false;
        }
        let k = 1024.min(n_in_index).max(1);
        let (scores, ids) = stored.index.search(&(*opaque).query, k);
        let dists: Vec<f64> = scores
            .iter()
            .map(|s| (1.0 - f64::from(*s)).clamp(0.0, 2.0))
            .collect();
        (*opaque).results = ids;
        (*opaque).distances = dists;
        (*opaque).cursor = 0;
        (*opaque).fetched = true;
    }

    if (*opaque).cursor >= (*opaque).results.len() {
        return false;
    }
    let id = {
        let cursor = (*opaque).cursor;
        (&(*opaque).results)[cursor]
    };
    let dist = {
        let cursor = (*opaque).cursor;
        (&(*opaque).distances)[cursor]
    };
    (*opaque).cursor += 1;
    pgrx::itemptr::u64_to_item_pointer(id, &mut (*scan).xs_heaptid);
    let _ = dist;

    // Quantised inner-product ranks approximate cosine distances.
    // Setting xs_recheckorderby = true makes the executor recompute
    // the orderby expression on the heap tuple, restoring exact
    // distances for the final sort. We deliberately do NOT write
    // xs_orderbyvals[0]: with recheck the executor recomputes
    // anyway, and the write site has historically caused a glibc
    // "free(): invalid pointer" abort that we have not yet
    // tracked down (probably a memory-allocator mismatch between
    // pgrx and the executor's expected free path).
    (*scan).xs_recheckorderby = true;
    (*scan).xs_recheck = true;
    true
}

/// `amendscan`: nothing to do — palloc'd memory is freed by the scan
/// memory context teardown.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {}
