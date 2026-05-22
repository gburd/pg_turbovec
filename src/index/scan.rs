//! `ambeginscan` / `amrescan` / `amgettuple` / `amendscan` — the
//! query path. Lazily loads the persisted IdMapIndex on first
//! `amgettuple`, runs a single batch search, then drains results
//! one TID per call.

use std::ffi::c_int;

use pgrx::pg_sys;
use pgrx::prelude::*;

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
    /// Cursor into `results`.
    cursor: usize,
    /// Whether the search has been executed yet.
    fetched: bool,
}

/// `ambeginscan`: allocate the IndexScanDesc and attach our opaque.
pub(crate) unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: c_int,
    norderbys: c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index_relation, nkeys, norderbys);
    if scan.is_null() {
        error!("turbovec: RelationGetIndexScan returned null");
    }

    // Allocate ScanOpaque inside the scan's memory context. We can't
    // just Box::new because the box would be in the Rust heap and
    // not freed by `amendscan`'s palloc cleanup.
    let opaque_ptr = pg_sys::palloc0(std::mem::size_of::<ScanOpaque>()) as *mut ScanOpaque;
    if opaque_ptr.is_null() {
        error!("turbovec: failed to palloc ScanOpaque");
    }
    // Initialise the Vecs in place. We'll never `drop_in_place` this
    // (palloc'd memory is released wholesale), so the Vec destructors
    // will not run — that is fine because the Vec heap memory itself
    // is std-managed and will be released when the Vec is moved out
    // (we use std::ptr::write to install initial values).
    std::ptr::write(
        opaque_ptr,
        ScanOpaque {
            query: Vec::new(),
            results: Vec::new(),
            cursor: 0,
            fetched: false,
        },
    );

    (*scan).opaque = opaque_ptr as *mut std::ffi::c_void;
    scan
}

/// `amrescan`: pull the order-by query out of `orderbys[0]` and
/// stash it in our opaque.
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
    (*opaque).cursor = 0;
    (*opaque).fetched = false;
}

/// `amgettuple`: on first call run the search and cache results;
/// subsequent calls drain one TID at a time.
pub(crate) unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let opaque = (*scan).opaque as *mut ScanOpaque;
    if opaque.is_null() {
        error!("turbovec amgettuple: opaque is null");
    }

    if !(*opaque).fetched {
        // Lazily run the search.
        let indexrelid = (*(*scan).indexRelation).rd_id;
        let stored = match persist::load(indexrelid) {
            Some(s) => s,
            None => {
                // Empty / unbuilt index — return no rows.
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
        // Phase 4 returns up to 1 024 results per scan; the executor
        // will discard everything beyond LIMIT. Phase 5 should pull
        // `numLimit` from the IndexScanState if available.
        let k = 1024.min(stored.n_vectors as usize).max(1);
        let (_scores, ids) = stored.index.search(&(*opaque).query, k);
        (*opaque).results = ids;
        (*opaque).cursor = 0;
        (*opaque).fetched = true;
    }

    if (*opaque).cursor >= (*opaque).results.len() {
        return false;
    }
    let id = (*opaque).results[(*opaque).cursor];
    (*opaque).cursor += 1;
    pgrx::itemptr::u64_to_item_pointer(id, &mut (*scan).xs_heaptid);
    (*scan).xs_recheckorderby = false;
    (*scan).xs_recheck = false;
    true
}

/// `amendscan`: nothing to do — palloc'd memory is freed by the scan
/// memory context teardown.
pub(crate) unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {}
