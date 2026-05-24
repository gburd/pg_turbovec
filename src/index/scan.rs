//! `ambeginscan` / `amrescan` / `amgettuple` / `amendscan` — the
//! query path. Lazily loads the persisted IdMapIndex on first
//! `amgettuple`, runs a single batch search, then drains results
//! one TID per call.

use std::ffi::c_int;
use std::sync::Arc;

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::index::persist;
use crate::kernels;
use crate::vec::Vector;

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
        (*scan).xs_orderbyvals =
            pg_sys::palloc0(std::mem::size_of::<pg_sys::Datum>() * (norderbys as usize))
                as *mut pg_sys::Datum;
        (*scan).xs_orderbynulls =
            pg_sys::palloc0(std::mem::size_of::<bool>() * (norderbys as usize)) as *mut bool;
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
        // Copy scan keys into the IndexScanDesc's pre-allocated slot.
        // We don't use them (turbovec is order-by-only), but the
        // executor expects them to be present and consistent.
        //
        // NB: `copy_nonoverlapping::<T>(src, dst, count)` takes `count`
        // in **elements of T**, not bytes. Passing
        // `nkeys * size_of::<ScanKeyData>()` (as v0.4..v1.0-rc.1 did)
        // overruns the destination by ~`nkeys * sizeof(ScanKeyData)`
        // *extra* bytes, smashing the IndexScanDesc and any heap
        // chunks that follow it. That was the root cause of the
        // `munmap_chunk(): invalid pointer` abort tracked through
        // Phase 17 as the "forced-index-scan crash". The other 39
        // tests never tripped it because the planner kept
        // small-table queries on a sequential scan, never calling
        // `amrescan` with `norderbys > 0`.
        std::ptr::copy_nonoverlapping(keys, (*scan).keyData, nkeys as usize);
    }

    if !orderbys.is_null() && norderbys > 0 {
        std::ptr::copy_nonoverlapping(orderbys, (*scan).orderByData, norderbys as usize);
    }

    let opaque = (*scan).opaque as *mut ScanOpaque;
    if opaque.is_null() {
        error!("turbovec amrescan: opaque is null");
    }

    // If the planner chose us without an ORDER BY operator (e.g. a
    // count(*) over an indexed column), produce an empty result
    // rather than ERROR. The executor falls through to whatever else
    // can satisfy the query; fetched=true on entry to amgettuple
    // short-circuits to an immediate `false` return.
    if norderbys < 1 || orderbys.is_null() {
        (*opaque).query.clear();
        (*opaque).results.clear();
        (*opaque).distances.clear();
        (*opaque).cursor = 0;
        (*opaque).fetched = true;
        return;
    }
    let order = orderbys.add(0);
    let datum = (*order).sk_argument;
    let isnull = ((*order).sk_flags as u32) & pg_sys::SK_ISNULL != 0;
    if isnull {
        error!("turbovec: ORDER BY query datum is NULL");
    }

    let query: Vector = match pgrx::FromDatum::from_datum(datum, false) {
        Some(v) => v,
        None => error!("turbovec: ORDER BY datum did not decode to vector"),
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

        // Cheap metadata-only fetch: lets us build the cache key and
        // compute a freshness signal without dragging the (possibly
        // hundreds-of-MiB) payload bytea across SPI on every query.
        let meta = match persist::load_meta(indexrelid) {
            Some(m) => m,
            None => {
                (*opaque).fetched = true;
                return false;
            }
        };
        let dim = meta.dim as usize;
        if (*opaque).query.len() != dim {
            error!(
                "turbovec amgettuple: query dim {} != index dim {}",
                (*opaque).query.len(),
                dim
            );
        }
        let n_in_index = meta.n_vectors.max(0) as usize;
        if n_in_index == 0 {
            (*opaque).fetched = true;
            return false;
        }

        // Cache lookup. The AM path uses `attnum = 0` by convention
        // (the index relation owns a single attribute and we don't
        // disambiguate further); the kernel path uses positive heap
        // attnums, so the namespaces never collide. The freshness
        // tuple is `(relfilenode, version)` — `meta.version` is
        // bumped on every `aminsert` / `ambuild` / `ambulkdelete`,
        // and we stash it into the cache's `n_rows` slot since the
        // cache only compares for equality.
        let key = CacheKey {
            rel_oid: indexrelid,
            attnum: 0,
            bit_width: meta.bit_width as u8,
            dim: meta.dim as u32,
        };
        let relfile = cache::current_relfilenode(indexrelid);
        let version_as_i64 = meta.version as i64;

        let arc: Arc<IdMapIndex> = match cache::lookup(key, relfile, version_as_i64) {
            Some(a) => a,
            None => {
                // Miss: pay the SPI + tmpfile + IdMapIndex::load cost
                // once, then publish the Arc into the cache so every
                // subsequent scan on this index version is free.
                let stored = match persist::load(indexrelid) {
                    Some(s) => s,
                    None => {
                        (*opaque).fetched = true;
                        return false;
                    }
                };
                // Approximate bytes: packed_codes (dim*bit_width/8
                // per vector) plus per-vector scale + id-map overhead
                // heuristic, mirroring the knn.rs estimate so a
                // single `cache_size_mb` budget governs both paths.
                let bytes_per_vec =
                    (meta.dim as usize * meta.bit_width as usize) / 8 + 4 + 64;
                let total_bytes = bytes_per_vec * (n_in_index.max(1));
                cache::insert(
                    key,
                    stored.index,
                    total_bytes,
                    relfile,
                    version_as_i64,
                )
            }
        };

        // The K knob: how many candidates to fetch per scan. v1.0
        // shipped a hard 1024 which made every ORDER BY on a million-
        // row index ~17 s. Default lowered to 100 (turbovec.search_k
        // GUC) — tune up for high LIMITs or higher recall, down for
        // sub-ms latency.
        let k_pref = crate::guc::SEARCH_K.get() as usize;
        let k = k_pref.min(n_in_index).max(1);
        let (scores, ids) = arc.search(&(*opaque).query, k);
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

    // The executor's reorder-queue path (`IndexNextWithReorder` in
    // `nodeIndexscan.c`) compares our advertised orderby distance
    // against the recomputed exact distance and `elog(ERROR,
    // "index returned tuples in wrong order")` if the recomputed
    // value is *less than* what we claimed. To be robust across
    // every operator class — cosine (range [0, 2]), inner-product
    // (-dot, unbounded) and any future addition — we advertise
    // `f64::NEG_INFINITY`, a universal lower bound. This forces
    // every tuple onto the reorder queue and drains it in exact
    // order at end-of-scan. The performance cost is negligible
    // because we only return up to `k` tuples per scan anyway.
    if !(*scan).xs_orderbyvals.is_null() && !(*scan).xs_orderbynulls.is_null() {
        let lb_bits = f64::NEG_INFINITY.to_bits();
        *(*scan).xs_orderbyvals.add(0) = pg_sys::Datum::from(lb_bits);
        *(*scan).xs_orderbynulls.add(0) = false;
    }
    let _ = dist;

    // Force the executor to recheck — our quantised distance is
    // approximate. The recheck recomputes the orderby expression
    // against the heap tuple, restoring exact distances.
    (*scan).xs_recheckorderby = true;
    (*scan).xs_recheck = false;
    true
}

/// `amendscan`: nothing to do — palloc'd memory is freed by the scan
/// memory context teardown.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {}
