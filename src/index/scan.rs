//! `ambeginscan` / `amrescan` / `amgettuple` / `amendscan` - the
//! query path. Lazily loads the persisted IdMapIndex on first
//! `amgettuple`, runs a single batch search, then drains results
//! one TID per call.

use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::c_int;

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::PreparedCachesBorrowed;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::index::{mmap_static, relfile};
use crate::kernels;
use crate::vec::Vector;

/// Scan-private state. Lives in the scan's memory context (allocated
/// by `palloc0` so all fields start zeroed).
#[repr(C)]
pub(crate) struct ScanOpaque {
    /// Cached query vector - set by `amrescan`, consumed by the
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
    /// Iterative-scan state. The materialised index handle is cached
    /// on the first search so refills don't re-do the cache lookup /
    /// relfile load. `None` until the first `amgettuple` populates it.
    arc: Option<cache::ScanHandle>,
    /// The `k` used for the most recent `arc.search(query, k)`. Starts
    /// at `search_k` and doubles on each refill, capped at
    /// `min(max_scan_tuples, n_live)`.
    current_k: usize,
    /// Number of live vectors in the index (`arc.read().len()`),
    /// cached so refills don't re-lock.
    n_live: usize,
    /// TIDs already emitted to the executor, across all refill
    /// batches. Used to dedup: `search(query, 2k)` re-ranks the whole
    /// corpus with `sort_unstable`, so the top-`2k` is NOT guaranteed
    /// to contain the previous top-`k` as a stable prefix when scores
    /// tie at the boundary. The set is robust regardless; it is
    /// bounded by `max_scan_tuples` (default 20k entries).
    emitted: HashSet<u64>,
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

    // Phase Q (v1.3.0) + Phase R-2 (v1.4.0): hard migration
    // boundary. We refuse to serve scans against an index built
    // under any pre-Phase-R-2 wire format. Three cases:
    //   (a) main fork is empty / never initialised — the index
    //       was built under v1.0.x..v1.1 (side-table only) or
    //       under a v1.2 binary that bailed before writing the
    //       relfile.
    //   (b) main fork is populated but the meta page is the v1
    //       (Phase L preview) layout that lacks the persisted
    //       SIMD-blocked chain + Lloyd-Max codebook Phase P
    //       relies on.
    //   (c) main fork is populated as v2 (Phase P, v1.3.x) but
    //       lacks the persisted rotation chain Phase R-2
    //       relies on. The lazy QR was the warm-scan hotspot.
    // All three are unrecoverable from the running binary;
    // require the user to REINDEX. We emit ERROR (not NOTICE)
    // so a half-broken state can't silently return zero rows.
    {
        let relfile_meta = relfile::read_meta(index_relation);
        match &relfile_meta {
            None => {
                ereport!(
                    PgLogLevel::ERROR,
                    PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                    "turbovec index has an empty main fork (built under pg_turbovec ≤ 1.2)",
                    "Run `REINDEX INDEX <name>;` to migrate the index to the v1.4.0 wire format."
                );
            }
            Some(m) if m.is_legacy_v1() && m.n_vectors > 0 => {
                ereport!(
                    PgLogLevel::ERROR,
                    PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                    "turbovec index uses the legacy v1 relfile layout (built under pg_turbovec 1.2)",
                    "This index lacks the persisted SIMD-blocked layout + Lloyd-Max codebook required by pg_turbovec ≥ 1.3.0. Run `REINDEX INDEX <name>;` to migrate."
                );
            }
            Some(m) if m.is_legacy_v2() && m.n_vectors > 0 => {
                ereport!(
                    PgLogLevel::ERROR,
                    PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                    "turbovec index built under pg_turbovec ≤ 1.3 cannot be scanned by pg_turbovec 1.4+",
                    "Run `REINDEX INDEX <name>;` to migrate. See docs/UPGRADING.md for details."
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
            arc: None,
            current_k: 0,
            n_live: 0,
            emitted: HashSet::new(),
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
        (*opaque).arc = None;
        (*opaque).current_k = 0;
        (*opaque).n_live = 0;
        (*opaque).emitted.clear();
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
    (*opaque).arc = None;
    (*opaque).current_k = 0;
    (*opaque).n_live = 0;
    (*opaque).emitted.clear();
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

        // Read the meta page directly from the index relation's
        // main fork via the buffer manager. shared_buffers caches
        // it cluster-wide; subsequent backends pay only the
        // buffer-pool hit cost.
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

        let dim = dim_u32 as usize;
        let n_in_index = n_vectors_u64 as usize;

        // Deferred-commit fallback: an aminsert in the same xact
        // mutated the in-memory cache but the relfile meta page
        // reflects the pre-mutation state. If the on-disk meta
        // says empty (or doesn't match our query dim) but the
        // cache holds a dirty mirror, use that instead.
        let dirty_fallback = (n_in_index == 0 || dim != (*opaque).query.len())
            .then(|| cache::am_find_dirty_by_rel(indexrelid))
            .flatten();

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
        let relfile_node = cache::relfilenode_from_relation((*scan).indexRelation);
        let version_as_i64 = am_version_u32 as i64;

        let arc: cache::ScanHandle = match dirty_fallback {
            Some((_, a, _)) => cache::ScanHandle::Mutable(a),
            None => match cache::scan_lookup(key, relfile_node, version_as_i64) {
            Some(a) => a,
            None => {
                // Phase R-3: cache miss. Try the mmap fast path
                // for the deterministic-after-`ambuild` static
                // regions (blocked codes + rotation matrix +
                // inline codebook); fall back to the buffer-
                // manager `read_full` if mmap isn't available
                // (mmap_static_blocked GUC off, non-default
                // tablespace, or open(2) raced a REINDEX).
                //
                // We materialise a read-only `ReadOnlyIndex` here,
                // NOT a full `IdMapIndex`: the scan path only needs
                // slot->id translation (a `Vec` index), so we skip
                // the O(n) `id_to_slot` `HashMap` build. The first
                // `aminsert` in this backend rebuilds a full
                // `IdMapIndex` via `am_install` (deferred HashMap).
                let meta = relfile::read_meta((*scan).indexRelation)
                    .expect("meta disappeared mid-scan");
                let (codes, scales, ids) = relfile::read_full((*scan).indexRelation, &meta);
                let mmap_enabled = guc::MMAP_STATIC_BLOCKED.get();
                let mmap_load = if mmap_enabled && meta.has_prepared_layout() {
                    mmap_static::load_static_regions((*scan).indexRelation, &meta)
                } else {
                    None
                };

                let (stored_index, mmap_handle): (cache::ReadOnlyIndex, _) =
                    if let Some((handle, parts)) = mmap_load {
                    // Mmap path: hand the chain bytes to turbovec
                    // via the borrowed-cache constructor as
                    // `Cow::Owned` (the chains had to be copied
                    // off the mmap because the on-disk layout has
                    // 24-byte page-header gaps every 8168 bytes).
                    // The Mmap handle still lives in the cache
                    // entry per the isolation contract.
                    let prepared = PreparedCachesBorrowed {
                        blocked_codes: Some(Cow::Owned(parts.blocked_codes)),
                        n_blocks: parts.n_blocks,
                        centroids: Some(Cow::Owned(parts.centroids)),
                        boundaries: Some(Cow::Owned(parts.boundaries)),
                        rotation: Some(Cow::Owned(parts.rotation)),
                    };
                    let idx = cache::ReadOnlyIndex::from_prepared_parts_borrowed(
                        meta.bit_width as usize,
                        meta.dim as usize,
                        meta.n_vectors as usize,
                        Cow::Owned(codes),
                        Cow::Owned(scales),
                        ids,
                        prepared,
                    );
                    (idx, Some(handle))
                } else if meta.has_prepared_layout() {
                    // Buffer-manager fallback path with prepared
                    // layout (matches v1.4.0 behaviour).
                    let blocked = relfile::read_blocked(
                        (*scan).indexRelation,
                        &meta,
                    );
                    let centroids = meta.centroids_slice().to_vec();
                    let boundaries = meta.boundaries_slice().to_vec();
                    let rotation = relfile::read_rotation(
                        (*scan).indexRelation,
                        &meta,
                    );
                    let rotation_opt =
                        if rotation.is_empty() { None } else { Some(rotation) };
                    let idx = cache::ReadOnlyIndex::from_prepared_parts(
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
                        rotation_opt,
                    );
                    (idx, None)
                } else {
                    // No prepared layout (legacy path; should be
                    // unreachable post-Phase Q because
                    // ambeginscan ERRORs on legacy_v1 / legacy_v2
                    // up-front, but keep the fall-through for
                    // defence in depth).
                    let idx = cache::ReadOnlyIndex::from_parts(
                        meta.bit_width as usize,
                        meta.dim as usize,
                        meta.n_vectors as usize,
                        codes,
                        scales,
                        ids,
                    );
                    (idx, None)
                };
                let bytes_per_vec = (dim_u32 as usize * bit_width_u8 as usize) / 8 + 4 + 64;
                let total_bytes = bytes_per_vec * (n_in_index.max(1));
                cache::scan_install(
                    key,
                    stored_index,
                    total_bytes,
                    relfile_node,
                    version_as_i64,
                    mmap_handle,
                )
            }
        }};
        let n_live = arc.len();
        if n_live == 0 {
            (*opaque).fetched = true;
            return false;
        }

        // The K knob: how many candidates to fetch per scan. v1.0
        // shipped a hard 1024 which made every ORDER BY on a million-
        // row index ~17 s. Default lowered to 100 (turbovec.search_k
        // GUC) - tune up for high LIMITs or higher recall, down for
        // sub-ms latency.
        //
        // Iterative scan (v1.8.0): `search_k` is only the *first*
        // batch. When the executor drains it and asks for more (and
        // turbovec.iterative_scan != off), the drain block below
        // doubles k and refills. We stash the arc + n_live so refills
        // don't repeat the cache lookup / relfile load.
        let k_pref = crate::guc::SEARCH_K.get() as usize;
        let k = k_pref.min(n_live.max(1)).max(1);
        let (scores, ids) = arc.search(&(*opaque).query, k);
        (*opaque).arc = Some(arc);
        (*opaque).n_live = n_live;
        (*opaque).current_k = k;
        populate_batch(opaque, &scores, &ids);
        (*opaque).cursor = 0;
        (*opaque).fetched = true;
    }

    // Drain / refill loop. We may have to refill several times before
    // we either find an unemitted candidate or exhaust the index, so
    // loop rather than tail-recurse.
    loop {
        if (*opaque).cursor < (*opaque).results.len() {
            break;
        }
        // Current batch drained. Try to refill if iterative scan is on
        // and we haven't hit the caps.
        if !try_refill(opaque) {
            return false;
        }
        // try_refill either appended new candidates (loop continues to
        // the break above) or appended nothing because every new
        // candidate was already emitted; in the latter case cursor is
        // still == results.len() and we'll attempt another refill with
        // a larger k. try_refill returns false once k is maxed out.
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
    (*opaque).emitted.insert(id);
    pgrx::itemptr::u64_to_item_pointer(id, &mut (*scan).xs_heaptid);

    // The executor's reorder-queue path (`IndexNextWithReorder` in
    // `nodeIndexscan.c`) compares our advertised orderby distance
    // against the recomputed exact distance and `elog(ERROR,
    // "index returned tuples in wrong order")` if the recomputed
    // value is *less than* what we claimed. To be robust across
    // every operator class - cosine (range [0, 2]), inner-product
    // (-dot, unbounded) and any future addition - we advertise
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

    // Force the executor to recheck - our quantised distance is
    // approximate. The recheck recomputes the orderby expression
    // against the heap tuple, restoring exact distances.
    (*scan).xs_recheckorderby = true;
    (*scan).xs_recheck = false;
    true
}

/// `amendscan`: nothing to do - palloc'd memory is freed by the scan
/// memory context teardown.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {}

/// Append the candidates in `(scores, ids)` that haven't already been
/// emitted to the current batch (`results` / `distances`), converting
/// turbovec cosine scores to executor-facing distances. Used for both
/// the initial search and every iterative refill. Skipping
/// already-emitted TIDs is what makes refills dedup-safe regardless of
/// turbovec's `search(q, 2k)` prefix (in)stability.
#[inline]
unsafe fn populate_batch(opaque: *mut ScanOpaque, scores: &[f32], ids: &[u64]) {
    for (s, id) in scores.iter().zip(ids.iter()) {
        if (*opaque).emitted.contains(id) {
            continue;
        }
        (*opaque).results.push(*id);
        let dist = (1.0 - f64::from(*s)).clamp(0.0, 2.0);
        (*opaque).distances.push(dist);
    }
}

/// Iterative refill (turbovec.iterative_scan != off). Grows `k` (doubling,
/// capped at `min(max_scan_tuples, n_live)`), re-runs the search, and
/// appends the newly-seen candidates to the current batch.
///
/// Returns `true` if the caller should keep draining (either new
/// candidates were appended, or another, larger refill is still
/// possible). Returns `false` only when the scan is exhausted: either
/// iterative scan is off, or `k` has reached its ceiling.
unsafe fn try_refill(opaque: *mut ScanOpaque) -> bool {
    if crate::guc::ITERATIVE_SCAN.get() == crate::guc::IterativeScanMode::Off {
        return false;
    }
    let arc = match &(*opaque).arc {
        Some(a) => a.clone(),
        None => return false,
    };
    let n_live = (*opaque).n_live;
    let max_scan = (crate::guc::MAX_SCAN_TUPLES.get() as usize).max(1);
    // Hard ceiling on k: never look past the live corpus, never past
    // the max_scan_tuples budget.
    let k_ceiling = n_live.min(max_scan);
    let old_k = (*opaque).current_k;
    if old_k >= k_ceiling {
        // Already examined as many candidates as we're allowed to.
        return false;
    }
    let new_k = (old_k.saturating_mul(2)).min(k_ceiling).max(old_k + 1);
    let before = (*opaque).results.len();
    let (scores, ids) = arc.search(&(*opaque).query, new_k);
    (*opaque).current_k = new_k;
    populate_batch(opaque, &scores, &ids);
    if (*opaque).results.len() > before {
        // New candidates appended; caller can drain them.
        true
    } else {
        // The larger search surfaced nothing we hadn't already
        // emitted (heavy tie clustering, or the corpus is exhausted).
        // If k can still grow, signal the caller to try again with a
        // bigger k; otherwise we're done.
        new_k < k_ceiling
    }
}
