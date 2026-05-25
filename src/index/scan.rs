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
#[cfg(not(feature = "relfile_storage"))]
use crate::index::persist;
#[cfg(feature = "relfile_storage")]
use crate::index::relfile;
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

    // Migration HINT (Phase L hardening item 5 + Phase P): when
    // the running binary has --features relfile_storage but is
    // opening an index that was either:
    //   (a) built under the older side-table path — main fork
    //       meta page is empty / never initialised; or
    //   (b) built under the v1 relfile preview — meta page is
    //       valid but lacks the prepared SIMD-blocked chain and
    //       inline codebook that Phase P relies on for fast
    //       cold scans.
    // In both cases the scan still works (the v1 path falls
    // through to per-backend `pack::repack`), but we surface a
    // NOTICE pointing the user at REINDEX so they can pick up
    // the persisted layout and skip the ~26 s cold-scan cost.
    #[cfg(feature = "relfile_storage")]
    {
        let indexrelid = (*index_relation).rd_id;
        let relfile_meta = crate::index::relfile::read_meta(index_relation);
        match &relfile_meta {
            None => {
                // Empty relfile — fall back to side-table
                // detection (case (a)).
                if let Some(legacy) = crate::index::persist::load_meta(indexrelid) {
                    if legacy.n_vectors > 0 {
                        pgrx::ereport!(
                            pgrx::PgLogLevel::NOTICE,
                            pgrx::PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                            "turbovec index appears to be in the legacy side-table format",
                            "This binary was built with --features relfile_storage but the index uses the v1.0.x / v1.1.0 side-table layout. The scan will return no rows. Run `REINDEX INDEX <name>;` to migrate."
                        );
                    }
                }
            }
            Some(m) if m.dim == 0 && m.n_vectors == 0 => {
                // Stub meta page (only ambuildempty ran). Same
                // side-table check as before.
                if let Some(legacy) = crate::index::persist::load_meta(indexrelid) {
                    if legacy.n_vectors > 0 {
                        pgrx::ereport!(
                            pgrx::PgLogLevel::NOTICE,
                            pgrx::PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                            "turbovec index appears to be in the legacy side-table format",
                            "This binary was built with --features relfile_storage but the index uses the v1.0.x / v1.1.0 side-table layout. The scan will return no rows. Run `REINDEX INDEX <name>;` to migrate."
                        );
                    }
                }
            }
            Some(m) if m.is_legacy_v1() && m.n_vectors > 0 => {
                // Case (b): populated v1 relfile. Scan will
                // succeed via per-backend repack, but cold
                // latency is ~26 s on 1 M × 1536-d. Recommend
                // REINDEX so the prepared layout gets baked in.
                pgrx::ereport!(
                    pgrx::PgLogLevel::NOTICE,
                    pgrx::PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                    "turbovec index uses an older relfile layout (v1) without the prepared SIMD-blocked cache",
                    "This index was built before Phase P and lacks the persisted SIMD-blocked layout + Lloyd-Max codebook. Cold scans will pay ~12-15 s of pack::repack and ~5-8 s of codebook compute per fresh backend. Run `REINDEX INDEX <name>;` to migrate to the v2 layout."
                );
            }
            _ => {}
        }
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
#[allow(clippy::too_many_lines)] // single linear flow; splitting hides logic
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

        // Phase L: when relfile storage is enabled we read the
        // meta page directly from the index relation's main fork
        // via the buffer manager. shared_buffers caches it
        // cluster-wide; subsequent backends pay only the buffer-
        // pool hit cost, not the SPI fetch + TOAST + parse cost.
        #[cfg(feature = "relfile_storage")]
        let (bit_width_u8, dim_u32, n_vectors_u64, am_version_u32) = {
            let m = match relfile::read_meta((*scan).indexRelation) {
                Some(m) => m,
                None => {
                    (*opaque).fetched = true;
                    return false;
                }
            };
            (m.bit_width, m.dim, m.n_vectors, m.am_version)
        };
        #[cfg(not(feature = "relfile_storage"))]
        let (bit_width_u8, dim_u32, n_vectors_u64, am_version_u32) = {
            // Cheap metadata-only fetch: lets us build the cache
            // key and compute a freshness signal without dragging
            // the (possibly hundreds-of-MiB) payload bytea across
            // SPI on every query.
            let meta = match persist::load_meta(indexrelid) {
                Some(m) => m,
                None => {
                    (*opaque).fetched = true;
                    return false;
                }
            };
            (
                meta.bit_width as u8,
                meta.dim as u32,
                meta.n_vectors.max(0) as u64,
                meta.version as u32,
            )
        };

        let dim = dim_u32 as usize;
        let n_in_index = n_vectors_u64 as usize;

        // Deferred-commit fallback: an aminsert in the same xact has
        // mutated the in-memory cache but the side-table reflects
        // the pre-mutation state. If the on-disk meta says empty
        // (or doesn't match our query dim) but the cache holds a
        // dirty mirror, use that instead.
        #[cfg(not(feature = "relfile_storage"))]
        let dirty_fallback = (n_in_index == 0 || dim != (*opaque).query.len())
            .then(|| cache::am_find_dirty_by_rel(indexrelid))
            .flatten();
        #[cfg(feature = "relfile_storage")]
        let dirty_fallback: Option<(crate::cache::CacheKey, Arc<parking_lot::RwLock<IdMapIndex>>, crate::cache::PersistState)> = None;

        if dirty_fallback.is_none() {
            if (*opaque).query.len() != dim {
                error!(
                    "turbovec amgettuple: query dim {} != index dim {}",
                    (*opaque).query.len(),
                    dim
                );
            }
            if n_in_index == 0 {
                (*opaque).fetched = true;
                return false;
            }
        }

        // Cache lookup. The AM path uses `attnum = 0` by convention
        // (the index relation owns a single attribute and we don't
        // disambiguate further); the kernel path uses positive heap
        // attnums, so the namespaces never collide. The freshness
        // tuple is `(relfilenode, am_version)` so the cache
        // invalidates whenever the relfile changes (REINDEX) or the
        // version is bumped (aminsert / ambulkdelete).
        let key = CacheKey {
            rel_oid: indexrelid,
            attnum: 0,
            bit_width: bit_width_u8,
            dim: dim_u32,
        };
        let relfile_node = cache::current_relfilenode(indexrelid);
        let version_as_i64 = am_version_u32 as i64;

        let arc: Arc<parking_lot::RwLock<IdMapIndex>> = match dirty_fallback {
            Some((_, a, _)) => a,
            None => match cache::lookup(key, relfile_node, version_as_i64) {
            Some(a) => a,
            None => {
                #[cfg(feature = "relfile_storage")]
                let stored_index = {
                    let meta = relfile::read_meta((*scan).indexRelation)
                        .expect("meta disappeared mid-scan");
                    let (codes, scales, ids) = relfile::read_full((*scan).indexRelation, &meta);

                    // Phase P: when the v2 layout is populated,
                    // skip the per-backend `pack::repack` + Lloyd-
                    // Max compute by reading the prepared parts
                    // off disk and feeding them back into the
                    // OnceLocks via from_id_map_parts_with_prepared.
                    let load_result = if meta.has_prepared_layout() {
                        let blocked = relfile::read_blocked(
                            (*scan).indexRelation,
                            &meta,
                        );
                        let centroids = meta.centroids_slice().to_vec();
                        let boundaries = meta.boundaries_slice().to_vec();
                        IdMapIndex::from_id_map_parts_with_prepared(
                            meta.bit_width as usize,
                            meta.dim as usize,
                            meta.n_vectors as usize,
                            codes,
                            scales,
                            ids,
                            blocked,
                            meta.n_blocks_blocked as usize,
                            centroids,
                            boundaries,
                        )
                    } else {
                        IdMapIndex::from_id_map_parts(
                            meta.bit_width as usize,
                            meta.dim as usize,
                            meta.n_vectors as usize,
                            codes,
                            scales,
                            ids,
                        )
                    };
                    match load_result {
                        Ok(idx) => idx,
                        Err(e) => error!(
                            "turbovec relfile: corrupt page chain for {:?}: {}",
                            indexrelid, e
                        ),
                    }
                };
                #[cfg(not(feature = "relfile_storage"))]
                let stored_index = match persist::load(indexrelid) {
                    Some(s) => s.index,
                    None => {
                        (*opaque).fetched = true;
                        return false;
                    }
                };
                let bytes_per_vec = (dim_u32 as usize * bit_width_u8 as usize) / 8 + 4 + 64;
                let total_bytes = bytes_per_vec * (n_in_index.max(1));
                cache::insert(key, stored_index, total_bytes, relfile_node, version_as_i64)
            }
        }};
        let n_live = arc.read().len();
        if n_live == 0 {
            (*opaque).fetched = true;
            return false;
        }

        // The K knob: how many candidates to fetch per scan. v1.0
        // shipped a hard 1024 which made every ORDER BY on a million-
        // row index ~17 s. Default lowered to 100 (turbovec.search_k
        // GUC) — tune up for high LIMITs or higher recall, down for
        // sub-ms latency.
        let k_pref = crate::guc::SEARCH_K.get() as usize;
        let k = k_pref.min(n_live.max(1)).max(1);
        let (scores, ids) = arc.read().search(&(*opaque).query, k);
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
