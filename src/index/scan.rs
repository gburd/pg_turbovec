//! `ambeginscan` / `amrescan` / `amgettuple` / `amendscan` - the
//! query path. Lazily loads the persisted IdMapIndex on first
//! `amgettuple`, runs a single batch search, then drains results
//! one TID per call.

use std::collections::HashSet;
use std::ffi::c_int;

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::index::relfile;
use crate::kernels;
use crate::vec::Vector;

thread_local! {
    /// Index oids this backend has already warned about being
    /// degraded (Phase E-2). Throttles the scan-time WARNING to once
    /// per backend per index so a churning deployment gets a clear
    /// signal without flooding the log on every query.
    static DEGRADED_WARNED: std::cell::RefCell<HashSet<pg_sys::Oid>> =
        std::cell::RefCell::new(HashSet::new());
}

/// Emit a throttled WARNING that an IVF index has degraded to a flat
/// O(n) scan (Phase E-2). Fires at most once per backend per index.
///
/// # Safety
///
/// `rel` must be a live index relation reference.
unsafe fn warn_index_degraded(rel: pg_sys::Relation, indexrelid: pg_sys::Oid) {
    let already = DEGRADED_WARNED.with(|s| !s.borrow_mut().insert(indexrelid));
    if already {
        return;
    }
    let name = {
        let rd_rel = (*rel).rd_rel;
        if rd_rel.is_null() {
            "<unknown>".to_string()
        } else {
            // relname is a NameData (fixed-size, NUL-terminated char
            // array). RelationGetRelationName is a C macro
            // (NameStr(rel->rd_rel->relname)) with no FFI binding, so
            // read the field directly.
            let name_ptr = std::ptr::addr_of!((*rd_rel).relname) as *const std::os::raw::c_char;
            std::ffi::CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
        }
    };
    ereport!(
        PgLogLevel::WARNING,
        PgSqlErrorCode::ERRCODE_WARNING,
        format!(
            "turbovec index \"{name}\" was built WITH (lists > 0) but has degraded to a flat scan after VACUUM"
        ),
        "REINDEX INDEX to restore IVF (cell-restricted) query performance. See docs/PRODUCTION.md."
    );
}

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
    /// IVF-2/IVF-3 cell-restriction context. `Some(ctx)` when this is
    /// an IVF index (`meta.has_ivf()`) backed by a read-only handle
    /// whose slot order matches the on-disk cell directory: every
    /// `arc.search*` for this scan goes through `search_masked` so
    /// turbovec's blocked kernel skips the unprobed (contiguous) cell
    /// ranges, and the cached coarse-search state lets an iterative
    /// refill WIDEN the probe set (IVF-3) without re-reading the
    /// relfile. `None` for flat indexes, vacuum-degraded IVF indexes
    /// (`has_ivf() == false`), or a mutable handle (slot order
    /// diverged from the cell directory) — all of which use the flat
    /// `arc.search` path unchanged.
    ivf: Option<IvfScanCtx>,
    /// Phase C operator-path allowlist: the parsed `turbovec.allowlist`
    /// id-set for this scan, or `None` when the GUC is empty/unset
    /// (the unfiltered hot path — zero added work). Parsed ONCE on
    /// the first `amgettuple` (not re-parsed per refill) and reused
    /// for every (re)built mask, so the slot-bool build is the only
    /// per-search cost and only when an allowlist is active.
    allow: Option<HashSet<u64>>,
}

/// IVF coarse-search state cached across an iterative scan so a
/// probe-widening refill (IVF-3) doesn't re-read the relfile. Built
/// once by [`ivf_setup_and_search`] on the first `amgettuple`.
struct IvfScanCtx {
    /// Row-major `lists * dim` coarse centroids (rotated space).
    centroids: Vec<f32>,
    /// The cell directory (`code_offset` + `n_vectors` per cell).
    directory: crate::index::ivf::CellDirectory,
    /// Query already normalised + rotated into the clustering space,
    /// so a refill re-runs `coarse_probe` without re-rotating.
    q_rot: Vec<f32>,
    /// Number of coarse cells in the index.
    lists: usize,
    /// Current probe-set width. Starts at `turbovec.probes` (clamped
    /// to `lists`) and doubles on each widening refill, capped at
    /// `min(max_probes, lists)`.
    probes: usize,
    /// The current slot mask for the `probes` nearest cells, with any
    /// tombstoned (vacuum-deleted) slots already excluded. Rebuilt
    /// each time `probes` widens.
    mask: Vec<bool>,
    /// Per-slot tombstone bitmap (LSB-first, bit set ⇒ slot is dead),
    /// `ceil(n_live / 8)` bytes, or empty when no rows have been
    /// vacuum-deleted. ANDed into every (re)built probe mask so a
    /// tombstoned row is never scored or returned (Phase E-2). Cached
    /// here so a probe-widening refill doesn't re-read the chain.
    tombstones: Vec<u8>,
    /// Phase B-1/B-2: when `Some`, this is the out-of-core
    /// cell-scoped IVF index. The fine search goes through
    /// [`cache::OocIvfIndex::search_ooc`] (gather probed cells off
    /// the mmap) instead of `arc.search_masked` over a whole-index
    /// handle, so the resident set stays O(probes*cell_size). The
    /// `mask` field is unused on this path (the gather applies the
    /// probe/tombstone restriction directly); we keep the probed
    /// cell list in `probed_cells` for the refill.
    ooc: Option<std::sync::Arc<cache::OocIvfIndex>>,
    /// The probed cell ids for the current `probes` width (OOC path
    /// only). Re-derived on each widen.
    probed_cells: Vec<u32>,
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
            // Phase F-2: a ColBERT / multivector token index (kind = 1,
            // wire v5) has NO single-vector order-by semantics. Its
            // opclass registers no order-by operator, so the planner
            // should never pick it for `ORDER BY ... <=> q`; but a
            // forced index scan (or a future planner change) could
            // still reach the AM scan path. We REJECT it here with a
            // clear HINT rather than crash or return garbage. The
            // ColBERT query path is `turbovec.colbert_search(...)`,
            // which reads the persistent index directly (cache /
            // relfile) and never enters `ambeginscan`.
            Some(m) if m.is_colbert() => {
                ereport!(
                    PgLogLevel::ERROR,
                    PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                    "this is a ColBERT (multivector) turbovec index; it has no ORDER BY semantics",
                    "Query it with turbovec.colbert_search(rel, id_col, token_col, query, k), not an ORDER BY <=> scan. See an internal design note."
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
            ivf: None,
            allow: None,
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
        (*opaque).allow = None;
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
    (*opaque).ivf = None;
    (*opaque).allow = None;
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
            // Phase E-2: surface a silent IVF->flat degradation. An
            // index built WITH (lists > 0) whose IVF metadata was
            // invalidated (the pre-E-2 vacuum landmine, or any future
            // safety-net path) scans O(n) flat instead of
            // cell-restricted. Warn the operator ONCE per backend per
            // index so a churning deployment notices the latency
            // cliff instead of eating it silently. The queryable
            // signal is `turbovec.index_is_degraded(regclass)`.
            if m.is_degraded() {
                warn_index_degraded((*scan).indexRelation, indexrelid);
            }
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
                    // Cache miss: build the read-only handle from the
                    // relfile via the buffer manager (all index data is
                    // read through `ReadBufferExtended`; see
                    // docs/BUFFER_CACHE_ONLY_DESIGN.md).
                    //
                    // We materialise a read-only `ReadOnlyIndex` here,
                    // NOT a full `IdMapIndex`: the scan path only needs
                    // slot->id translation (a `Vec` index), so we skip
                    // the O(n) `id_to_slot` `HashMap` build. The first
                    // `aminsert` in this backend rebuilds a full
                    // `IdMapIndex` via `am_install` (deferred HashMap).
                    let meta = relfile::read_meta((*scan).indexRelation)
                        .expect("meta disappeared mid-scan");

                    // Phase B-1/B-2: out-of-core cell-scoped IVF. When
                    // `turbovec.out_of_core` selects it AND this is a
                    // live IVF index (lists > 0, not vacuum-degraded),
                    // install a bounded-resident `OocIvfIndex` instead
                    // of the whole-index `ReadOnlyIndex`. The codes
                    // buffer is gathered per probed cell through the
                    // buffer manager (`gather_codes_ranges`), so the
                    // resident set is O(probes*cell_size) not O(n) — a
                    // >RAM index can be served. Flat / degraded indexes
                    // fall through to the whole-index load (they have no
                    // cells to scope). `auto` (default) goes cell-scoped
                    // only when the codes are large vs
                    // turbovec.cache_size_mb; `on` always, `off` never.
                    let codes_bytes = (meta.n_vectors as u64)
                        .saturating_mul(((meta.dim as u64) * (meta.bit_width as u64)) / 8);
                    if guc::out_of_core_cell_scoped(codes_bytes)
                        && meta.has_ivf()
                        && meta.has_prepared_layout()
                    {
                        if let Some(handle) = try_install_ooc(
                            (*scan).indexRelation,
                            &meta,
                            key,
                            relfile_node,
                            version_as_i64,
                        ) {
                            handle
                        } else {
                            install_whole_index(
                                (*scan).indexRelation,
                                &meta,
                                key,
                                relfile_node,
                                version_as_i64,
                                n_in_index,
                                dim_u32,
                                bit_width_u8,
                            )
                        }
                    } else {
                        install_whole_index(
                            (*scan).indexRelation,
                            &meta,
                            key,
                            relfile_node,
                            version_as_i64,
                            n_in_index,
                            dim_u32,
                            bit_width_u8,
                        )
                    }
                }
            },
        };
        let n_live = arc.len();
        if n_live == 0 {
            (*opaque).fetched = true;
            return false;
        }

        // The K knob: how many candidates to fetch per scan. v1.0
        // shipped a hard 1024 which made every ORDER BY on a million-
        // row index ~17 s. Default lowered to 100, then to 32 in
        // v1.18 (turbovec.search_k GUC) once the recall-vs-search_k
        // frontier showed recall@10 plateaus by ~25 (the per-query
        // floor is the reorder-recheck of all search_k candidates:
        // a heap fetch + exact recompute each). Tune up for high
        // LIMITs or higher recall, down (toward 16) for lower latency.
        //
        // Iterative scan (v1.8.0): `search_k` is only the *first*
        // batch. When the executor drains it and asks for more (and
        // turbovec.iterative_scan != off), the drain block below
        // doubles k and refills. We stash the arc + n_live so refills
        // don't repeat the cache lookup / relfile load.
        //
        // Oversampling (differentiator #5): `turbovec.oversample`
        // widens the *initial* candidate set to ceil(search_k *
        // oversample). The lossy quantized ranking can place a true
        // neighbour just outside search_k; oversampling pulls it back
        // into the candidate set, and the reorder queue
        // (xs_recheckorderby) re-ranks the whole set by exact distance.
        // Default 1.0 = no oversampling = pre-feature behaviour.
        // Iterative refill still doubles from this oversampled floor.
        let k_pref = crate::guc::SEARCH_K.get() as usize;
        let oversample = crate::guc::OVERSAMPLE.get().clamp(1.0, 100.0);
        // ceil(k_pref * oversample) without float-rounding surprises;
        // k_pref <= 100_000 and oversample <= 100 so the product fits
        // an f64 exactly well within u64 range.
        let k_oversampled = (k_pref as f64 * oversample).ceil() as usize;
        let k = k_oversampled.min(n_live.max(1)).max(1);

        // Phase C operator-path allowlist: parse turbovec.allowlist
        // ONCE per scan (not per refill). None = unfiltered hot path.
        (*opaque).allow = parse_allowlist();

        // IVF-2: if this is an IVF index (has_ivf() == true, i.e.
        // lists > 0 AND the v4 cell metadata is present — NOT a
        // vacuum-degraded index, which blanks those fields), do the
        // coarse search and build a cell-restriction mask. The mask
        // is true for exactly the slots in the `probes` nearest cells;
        // turbovec's blocked kernel skips the contiguous unprobed
        // ranges. The Phase C allowlist (when active) is ANDed into
        // that mask so the block-skip also drops blocks with no
        // allowed slot. Falls back to the flat path when:
        //   - the index is flat or vacuum-degraded (has_ivf() false),
        //   - the handle is Mutable (post-insert / dirty-xact mirror,
        //     whose slot order has diverged from the cell directory —
        //     search_masked returns None and we drop the mask).
        let ivf_results = ivf_setup_and_search(
            scan,
            &arc,
            &(*opaque).query,
            k,
            n_live,
            (*opaque).allow.as_ref(),
        );
        let (scores, ids) = match ivf_results {
            Some((ctx, scores, ids)) => {
                (*opaque).ivf = Some(ctx);
                (scores, ids)
            }
            None => {
                (*opaque).ivf = None;
                flat_search(&arc, &(*opaque).query, k, (*opaque).allow.as_ref())
            }
        };
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
        if !try_refill(scan, opaque) {
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
    //
    // Tier-1 #1b (investigated 2026-06, REJECTED as a no-op): we
    // considered advertising a TIGHTER valid lower bound (e.g. 0.0
    // for cosine/L2, which are non-negative) so the executor could
    // skip rechecking some candidates. It cannot. Under
    // `xs_recheckorderby = true`, `IndexNextWithReorder`
    // UNCONDITIONALLY fetches the heap tuple (in
    // `index_getnext_slot` -> `index_fetch_heap`, before it reads
    // our advertised value) and recomputes the exact distance
    // (`EvalOrderByExpressions`) for EVERY tuple `amgettuple`
    // returns. The advertised value only governs the wrong-order
    // ERROR and the final drain ordering, never whether a recheck
    // happens. So a tighter bound is legal but buys nothing; the
    // ONLY lever that cuts the per-query recheck floor (heap fetch +
    // exact recompute per candidate) is returning fewer candidates,
    // i.e. `turbovec.search_k` (Tier-1 #1a, default lowered to 32).
    // (Behaviour is identical across PG 13-18.) Keep NEG_INFINITY:
    // simplest, opclass-agnostic, provably safe.
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

/// Build and install the whole-index `ReadOnlyIndex` cache entry
/// (the pre-Phase-B-1 behaviour: O(n) resident). Used for flat
/// indexes, vacuum-degraded IVF, and as the IVF fallback when
/// out-of-core is off or the mmap is unavailable. Returns the
/// installed [`cache::ScanHandle`].
///
/// # Safety
///
/// `rel` holds a live index relation reference; `meta` came from
/// `relfile::read_meta(rel)` on the same relation.
#[allow(clippy::too_many_arguments)]
unsafe fn install_whole_index(
    rel: pg_sys::Relation,
    meta: &crate::index::page::MetaPageData,
    key: CacheKey,
    relfile_node: u32,
    version_as_i64: i64,
    n_in_index: usize,
    dim_u32: u32,
    bit_width_u8: u8,
) -> cache::ScanHandle {
    let (codes, scales, ids) = relfile::read_full(rel, meta);

    // All index data is read through PostgreSQL's buffer manager
    // (`ReadBufferExtended`) — there is no direct relfile mmap. Heap
    // visibility + `xs_recheckorderby` remain the correctness
    // backstops; the buffer manager is the single source of truth
    // for page access (consistent pinning/locking, crash + streaming-
    // replication semantics). See docs/BUFFER_CACHE_ONLY_DESIGN.md.
    let stored_index: cache::ReadOnlyIndex = if meta.has_prepared_layout() {
        // Prepared (SIMD-blocked) layout: read the blocked codes +
        // rotation chains through the buffer manager. The result is
        // cached in this per-backend `ReadOnlyIndex`, so the
        // per-page pin/lock/copy cost is paid once per (backend,
        // am_version) — warm queries hit the resident buffers, never
        // the buffer manager.
        let blocked = relfile::read_blocked(rel, meta);
        let centroids = meta.centroids_slice().to_vec();
        let boundaries = meta.boundaries_slice().to_vec();
        let rotation = relfile::read_rotation(rel, meta);
        let rotation_opt = if rotation.is_empty() {
            None
        } else {
            Some(rotation)
        };
        cache::ReadOnlyIndex::from_prepared_parts(
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
        )
    } else {
        // No prepared layout (legacy path; should be unreachable
        // post-Phase Q because ambeginscan ERRORs on legacy_v1 /
        // legacy_v2 up-front, but keep the fall-through for
        // defence in depth).
        cache::ReadOnlyIndex::from_parts(
            meta.bit_width as usize,
            meta.dim as usize,
            meta.n_vectors as usize,
            codes,
            scales,
            ids,
        )
    };
    let bytes_per_vec = (dim_u32 as usize * bit_width_u8 as usize) / 8 + 4 + 64;
    let total_bytes = bytes_per_vec * (n_in_index.max(1));
    cache::scan_install(key, stored_index, total_bytes, relfile_node, version_as_i64)
}

/// Phase B-1/B-2: build and install an out-of-core cell-scoped
/// [`cache::OocIvfIndex`] for a live IVF index. Returns
/// `Some(handle)` on success, or `None` (caller falls back to
/// [`install_whole_index`]) if the mmap can't be opened, the static
/// regions disagree with the buffer-managed meta page (a concurrent
/// rewrite raced our `AccessShareLock`), or any bounded region is
/// missing / inconsistent.
///
/// Only the BOUNDED regions are read here: coarse centroids, cell
/// directory, rotation, codebook, and the small per-slot scales /
/// ids tables. The big codes chain is NOT read — it's faulted per
/// probed cell off the mmap at query time. That is what bounds the
/// resident set to O(probes*cell_size) and lets a >RAM index be
/// served.
///
/// # Safety
///
/// `rel` holds a live index relation reference; `meta` came from
/// `relfile::read_meta(rel)` on the same relation. Caller holds at
/// least `AccessShareLock`.
unsafe fn try_install_ooc(
    rel: pg_sys::Relation,
    meta: &crate::index::page::MetaPageData,
    key: CacheKey,
    relfile_node: u32,
    version_as_i64: i64,
) -> Option<cache::ScanHandle> {
    let dim = meta.dim as usize;
    let lists = meta.lists as usize;
    let n_vectors = meta.n_vectors as usize;
    if dim == 0 || lists == 0 || n_vectors == 0 {
        return None;
    }

    // All index data is read through the buffer manager. The
    // per-query gather (`OocIvfIndex::search_ooc` ->
    // `relfile::gather_codes_ranges`) reads ONLY the probed cells'
    // code pages via `ReadBufferExtended`, so the resident set stays
    // bounded at O(probes * cell_size) — out-of-core serving needs
    // cell-contiguous layout + range-scoped reads, NOT mmap. See
    // docs/BUFFER_CACHE_ONLY_DESIGN.md.

    // Bounded regions: coarse centroids + rotation + cell directory.
    let coarse_centroids = relfile::read_coarse_centroids(rel, meta);
    if coarse_centroids.len() != lists * dim {
        return None;
    }
    let rotation = relfile::read_rotation(rel, meta);
    if rotation.len() != dim * dim {
        return None;
    }
    let directory = relfile::read_cell_directory(rel, meta)?;
    if directory.total_vectors() != n_vectors as u64 {
        return None;
    }

    // Codebook (inline in the meta page).
    let codebook_centroids = meta.centroids_slice().to_vec();
    let codebook_boundaries = meta.boundaries_slice().to_vec();

    // Small per-slot tables (scales 4 B/vec, ids 8 B/vec). These are
    // O(n) but tiny next to the codes (e.g. 768 B/vec at 1536-d
    // 4-bit); keeping them resident lets the per-query gather pull
    // only the codes off the mmap. The codes chain itself is NOT
    // read here — that is what bounds the resident set.
    let scales = relfile::read_scales_only(rel, meta);
    let ids = relfile::read_ids_only(rel, meta);
    if scales.len() != n_vectors || ids.len() != n_vectors {
        return None;
    }

    let ooc = cache::OocIvfIndex::new(
        meta.bit_width as usize,
        dim,
        n_vectors,
        meta.codes_first,
        meta.stride_bytes,
        meta.rows_per_codes_page,
        coarse_centroids,
        lists,
        rotation,
        directory,
        codebook_centroids,
        codebook_boundaries,
        scales,
        ids,
    );

    // Bounded resident-byte estimate for the LRU cap: centroids +
    // rotation + directory + scales + ids (NOT O(n) codes).
    let bytes = lists * dim * 4 + dim * dim * 4 + lists * 12 + n_vectors * 4 + n_vectors * 8 + 4096;
    Some(cache::scan_install_ooc(
        key,
        ooc,
        bytes,
        relfile_node,
        version_as_i64,
    ))
}

/// IVF-2/IVF-3 coarse search + cell-restricted fine search. Returns
/// `Some((ctx, scores, ids))` when the index is a live IVF index
/// (`meta.has_ivf()`) backed by a read-only handle whose slot order
/// matches the on-disk cell directory; the caller stashes `ctx`
/// (centroids + directory + rotated query + current probes/mask) for
/// iterative **probe-widening** refills (IVF-3) and uses `(scores,
/// ids)` as the first batch.
///
/// Returns `None` — signalling the caller to take the flat
/// `arc.search` path — when:
///   - the index is flat (`lists == 0`) or vacuum-degraded
///     (`has_ivf()` is false because the v4 cell fields were blanked
///     by `write_meta_shrink_in_place`),
///   - the coarse centroids / rotation / cell directory are missing
///     or inconsistent (defensive; should not happen for a healthy
///     IVF index),
///   - the handle is [`cache::ScanHandle::Mutable`] (slot order has
///     diverged from the build-time cell layout; `search_masked`
///     returns `None`).
///
/// # Safety
///
/// `scan` holds a live index relation reference for the duration.
unsafe fn ivf_setup_and_search(
    scan: pg_sys::IndexScanDesc,
    arc: &cache::ScanHandle,
    query: &[f32],
    k: usize,
    n_live: usize,
    allow: Option<&HashSet<u64>>,
) -> Option<(IvfScanCtx, Vec<f32>, Vec<u64>)> {
    // Phase B-1/B-2: out-of-core cell-scoped path. The handle owns
    // the cached centroids / directory / rotation / codebook + the
    // mmap; coarse-probe, then gather only the probed cells off the
    // mmap (no whole-index buffer). Results match the whole-load
    // IVF path exactly: same coarse_probe, same cells, same fine
    // ranking over the same codes (just gathered into a compact
    // buffer instead of masked in place).
    if let Some(ooc) = arc.ooc() {
        return ivf_setup_and_search_ooc(scan, &ooc, query, k, allow);
    }

    let meta = relfile::read_meta((*scan).indexRelation)?;
    if !meta.has_ivf() {
        return None;
    }
    let dim = meta.dim as usize;
    let lists = meta.lists as usize;

    // Coarse centroids (f32, rotated space) + rotation matrix + cell
    // directory. Any missing piece ⇒ flat fallback (defensive).
    let centroids = relfile::read_coarse_centroids((*scan).indexRelation, &meta);
    if centroids.len() != lists * dim {
        return None;
    }
    let rotation = relfile::read_rotation((*scan).indexRelation, &meta);
    if rotation.len() != dim * dim {
        return None;
    }
    let directory = relfile::read_cell_directory((*scan).indexRelation, &meta)?;
    // The cell directory must partition exactly n_live slots; if it
    // doesn't, the mask would be wrong — fall back to flat.
    if directory.total_vectors() != n_live as u64 {
        return None;
    }

    // Phase E-2: per-slot tombstone bitmap from VACUUM. Empty when
    // nothing has been deleted. ANDed into the probe mask below so a
    // vacuum-deleted (tombstoned) row is never scored or returned.
    let tombstones = relfile::read_tombstones((*scan).indexRelation, &meta);

    // Coarse search: normalise + rotate the query into the clustering
    // space the build used, then pick the `probes` nearest cells.
    // The build always normalises before assigning to a cell (see
    // ivf_build_and_write), so the coarse search must normalise too,
    // regardless of turbovec.normalize_on_insert.
    //
    // Phase G-1 scoping note: the whole-load path re-reads
    // `centroids` fresh from the relfile on every scan-open (there is
    // no per-backend cache of THIS struct today), so building an
    // O(lists^2) `CentroidGraph` here per scan would not be the
    // "build once per backend" amortised cost the plan requires --
    // it would be pure overhead on every query. The whole-load path
    // is also gated to comfortably-RAM-resident indexes
    // (`turbovec.out_of_core`), i.e. the SMALL-`lists` regime where
    // the linear scan is already cheap. G-1's graph therefore only
    // backs the OOC path (`OocIvfIndex::coarse_probe_cells`, cached
    // once per backend by the existing scan-install machinery), which
    // is also the >RAM / large-`lists` regime the plan targets. This
    // path keeps the exact linear `coarse_probe`.
    let unit = kernels::normalise_to_vec(query);
    let q_rot = crate::index::ivf::rotate_query(&rotation, &unit, dim);
    let probes = (crate::guc::PROBES.get() as usize).clamp(1, lists);
    let probed = crate::index::ivf::coarse_probe(&centroids, lists, dim, &q_rot, probes);

    // Build the slot mask for the probed cells and run the
    // cell-restricted fine search. search_masked returns None for a
    // Mutable handle (slot order diverged) → flat fallback.
    let mut mask = directory.probe_mask(&probed, n_live);
    apply_tombstones(&mut mask, &tombstones);
    // Phase C: AND the operator-path allowlist into the probe mask
    // (alongside tombstones) BEFORE search_masked, so the blocked
    // kernel's 32-vector block-skip drops blocks with no
    // probed+allowed+live slot — the real latency win on a selective
    // allowlist. Built only when an allowlist is active.
    apply_allowlist(&mut mask, arc, allow);
    let (scores, ids) = arc.search_masked(query, k, &mask)?;
    let ctx = IvfScanCtx {
        centroids,
        directory,
        q_rot,
        lists,
        probes,
        mask,
        tombstones,
        ooc: None,
        probed_cells: Vec::new(),
    };
    Some((ctx, scores, ids))
}

/// Out-of-core (Phase B-1/B-2) IVF setup + first cell-scoped search.
/// The [`cache::OocIvfIndex`] owns the cached centroids / directory /
/// rotation / codebook + the relfile mmap; this coarse-probes the
/// cells and gathers ONLY the probed cells off the mmap to score
/// them, so the resident set stays O(probes*cell_size). The
/// per-slot tombstone bitmap (Phase E-2) is read here (small) and
/// applied during the gather so dead rows are never scored.
///
/// Returns `None` (caller falls back to the flat `arc.search` path)
/// only on a defensive gather failure (corrupt index / post-truncate
/// race) — but since the OOC handle has no whole-index buffer, that
/// fallback would be an empty `arc.search`; in practice the gather
/// succeeds for a healthy index.
///
/// # Safety
///
/// `scan` holds a live index relation reference for the duration.
unsafe fn ivf_setup_and_search_ooc(
    scan: pg_sys::IndexScanDesc,
    ooc: &std::sync::Arc<cache::OocIvfIndex>,
    query: &[f32],
    k: usize,
    allow: Option<&HashSet<u64>>,
) -> Option<(IvfScanCtx, Vec<f32>, Vec<u64>)> {
    let meta = relfile::read_meta((*scan).indexRelation)?;
    let lists = ooc.lists();
    // Per-slot tombstone bitmap (small). Applied during the gather.
    let tombstones = relfile::read_tombstones((*scan).indexRelation, &meta);

    // Coarse search: normalise + rotate the query, pick the probed
    // cells — IDENTICAL to the whole-load path (same coarse_probe,
    // same centroids, same rotation) so results match.
    let unit = kernels::normalise_to_vec(query);
    let probes = (crate::guc::PROBES.get() as usize).clamp(1, lists);
    let probed = ooc.coarse_probe_cells(&unit, probes);

    // Phase C: the allowlist is masked INSIDE search_ooc (it builds a
    // compact-slot mask over the gathered cells and pushes it into
    // the blocked kernel, so the block-skip applies on the OOC path
    // too — not just a post-filter).
    let (scores, ids) =
        ooc.search_ooc((*scan).indexRelation, query, k, &probed, &tombstones, allow)?;
    let ctx = IvfScanCtx {
        // Unused on the OOC path (the OOC index holds its own copies);
        // kept empty to avoid duplicating the O(lists*dim) centroids.
        centroids: Vec::new(),
        directory: crate::index::ivf::CellDirectory {
            entries: Vec::new(),
        },
        q_rot: Vec::new(),
        lists,
        probes,
        mask: Vec::new(),
        tombstones,
        ooc: Some(ooc.clone()),
        probed_cells: probed,
    };
    Some((ctx, scores, ids))
}

/// Clear (set to `false`) every mask slot whose tombstone bit is set,
/// so a vacuum-deleted row never contributes to the top-k. `tombstones`
/// is the LSB-first per-slot bitmap (`bit set ⇒ dead`); an empty slice
/// means no rows have been deleted and the mask is left unchanged.
/// Phase E-2.
#[inline]
fn apply_tombstones(mask: &mut [bool], tombstones: &[u8]) {
    if tombstones.is_empty() {
        return;
    }
    for (slot, m) in mask.iter_mut().enumerate() {
        if !*m {
            continue;
        }
        let byte = slot / 8;
        if byte < tombstones.len() && (tombstones[byte] >> (slot % 8)) & 1 != 0 {
            *m = false;
        }
    }
}

/// AND the Phase C operator-path allowlist into a (probe+tombstone)
/// slot mask in place. A `None` allowlist (the unfiltered hot path)
/// leaves the mask untouched and costs nothing. For an allowlist over
/// a [`cache::ScanHandle::ReadOnly`] handle, clear every mask slot
/// whose external id is not in the allowlist; the blocked kernel then
/// skips blocks with no surviving slot. The `arc.allow_slot_mask`
/// returns `None` for non-ReadOnly handles (Mutable / Ooc), which the
/// IVF path never reaches with a mask (Mutable falls back to flat;
/// Ooc masks inside `search_ooc`).
#[inline]
fn apply_allowlist(mask: &mut [bool], arc: &cache::ScanHandle, allow: Option<&HashSet<u64>>) {
    let Some(set) = allow else {
        return;
    };
    let Some(allow_slot) = arc.allow_slot_mask(set) else {
        return;
    };
    for (m, a) in mask.iter_mut().zip(allow_slot.iter()) {
        *m = *m && *a;
    }
}

/// Parse the `turbovec.allowlist` GUC (a CSV of heap TIDs) into a
/// `HashSet<u64>` for this scan (Phase C operator-path allowlist).
///
/// The index AM keys every vector by its heap **TID** (the
/// `pgrx::itemptr::item_pointer_to_u64` encoding, `(block << 32) |
/// offset`), not by any heap `id` column — the index only ever sees
/// item pointers. So the allowlist is a set of those TID-encoded
/// bigints; the scan ANDs them into the slot mask by matching against
/// the slot→TID table the read path keeps. Callers build the set from
/// `ctid` (see docs/FILTERING.md § 3.5 for the SQL).
///
/// Returns `None` when the GUC is unset or empty (after trimming) —
/// the unfiltered hot path, so the caller pays nothing. Whitespace
/// around tokens is tolerated; empty tokens (e.g. a trailing comma)
/// are ignored. A non-integer token ERRORs the scan clearly. Tokens
/// are parsed as `i64` (the SQL bigint domain) then reinterpreted as
/// the `u64` TID domain.
fn parse_allowlist() -> Option<HashSet<u64>> {
    let raw = guc::ALLOWLIST.get()?;
    let s = raw.to_string_lossy();
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut set = HashSet::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        match tok.parse::<i64>() {
            Ok(v) => {
                set.insert(v as u64);
            }
            Err(_) => error!(
                "turbovec.allowlist: '{}' is not a valid bigint id (allowlist must be a CSV of bigint ids)",
                tok
            ),
        }
    }
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

/// Encode a heap `ctid` into the bigint TID domain the
/// `turbovec.allowlist` GUC expects (Phase C operator-path
/// allowlist). This is the ergonomic front door for building an
/// allowlist from `ctid`: instead of the `(block << 32) | offset`
/// `split_part` incantation, write
///
/// ```sql
/// SELECT set_config('turbovec.allowlist',
///     (SELECT string_agg(turbovec.tid_to_bigint(ctid)::text, ',')
///      FROM items WHERE tenant_id = $1), false);
/// ```
///
/// The encoding matches `pgrx::itemptr::item_pointer_to_u64`
/// (`(block << 32) | offset`), i.e. exactly the value the AM stores
/// per slot, so the scan's allow-mask match is exact. Reinterpreted
/// as `i64` for the SQL `bigint` domain (TIDs never set the high
/// bit in practice, but the round-trip is bit-preserving either
/// way: the GUC parser reads `i64` then casts back to `u64`).
#[pg_extern(immutable, parallel_safe)]
fn tid_to_bigint(ctid: pg_sys::ItemPointerData) -> i64 {
    pgrx::itemptr::item_pointer_to_u64(ctid) as i64
}

/// Flat (non-IVF) top-`k` search, honouring the Phase C operator-path
/// allowlist. When `allow` is `None` this is exactly `arc.search` —
/// the unfiltered hot path, zero added work. When `allow` is `Some`:
///   - On a [`cache::ScanHandle::ReadOnly`] handle, build a by-slot
///     allowlist mask and route through `search_masked`, so the
///     blocked kernel skips 32-vector blocks with no allowed slot
///     (the in-kernel block-skip, on the operator path).
///   - On a `Mutable` handle (post-insert / dirty-xact mirror, whose
///     slot order has diverged so no slot mask applies), fall back to
///     a plain search and post-filter the returned ids by the
///     allowlist — a correctness backstop (the rarer path). To keep
///     the post-filtered top-`k` correct, the unfiltered search is
///     widened so the allowlisted neighbours that would survive aren't
///     starved by non-allowed ids ranking ahead of them.
fn flat_search(
    arc: &cache::ScanHandle,
    query: &[f32],
    k: usize,
    allow: Option<&HashSet<u64>>,
) -> (Vec<f32>, Vec<u64>) {
    let Some(set) = allow else {
        return arc.search(query, k);
    };
    if let Some(mask) = arc.allow_slot_mask(set) {
        // ReadOnly: in-kernel block-skip over the allowlist mask.
        if let Some(res) = arc.search_masked(query, k, &mask) {
            return res;
        }
    }
    // Mutable / Ooc fallback: search wide, then post-filter by the
    // allowlist. Widen so allowed neighbours ranked behind non-allowed
    // ids still surface (cap at the live corpus).
    let n = arc.len();
    let wide = k.saturating_mul(8).min(n.max(1)).max(k);
    let (scores, ids) = arc.search(query, wide);
    let mut out_s = Vec::with_capacity(k);
    let mut out_i = Vec::with_capacity(k);
    for (s, id) in scores.iter().zip(ids.iter()) {
        if set.contains(id) {
            out_s.push(*s);
            out_i.push(*id);
            if out_i.len() == k {
                break;
            }
        }
    }
    (out_s, out_i)
}

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

/// Iterative refill (`turbovec.iterative_scan != off`).
///
/// **Flat indexes (v1.8.0 behaviour, unchanged):** grow `k`
/// (doubling, capped at `min(max_scan_tuples, n_live)`), re-run the
/// flat search, append the newly-seen candidates.
///
/// **IVF indexes (IVF-3):** WIDEN the probe set first — double the
/// number of probed cells (`probes`, `2*probes`, `4*probes`, …) up to
/// `min(max_probes, lists)`, re-run the coarse search to pick that
/// many nearest cells, rebuild the cell mask, and re-run the
/// cell-restricted fine search. `k` (the candidate count *within* the
/// probed cells) also grows so the wider cell set can surface more
/// candidates. This is the IVF analogue of `ivfflat.max_probes`: it
/// fixes under-return when a selective `WHERE` filter's matches live
/// in cells that weren't in the initial `probes` nearest set —
/// growing `k` alone (IVF-2) could never reach them because they
/// weren't in the probed cells at all. When `probes` reaches `lists`
/// the whole corpus has been scanned; once `k` is also maxed the scan
/// is exhausted. `max_scan_tuples` still caps `k` as a backstop.
///
/// Dedup across widening batches reuses the existing emitted-TID set
/// in [`populate_batch`], so re-probing a wider cell set that includes
/// already-returned ids never double-emits.
///
/// Returns `true` if the caller should keep draining (new candidates
/// appended, or another widening/k-growth is still possible). Returns
/// `false` only when the scan is exhausted: iterative scan is off, or
/// the probe set is at its cap AND `k` has reached its ceiling.
unsafe fn try_refill(scan: pg_sys::IndexScanDesc, opaque: *mut ScanOpaque) -> bool {
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

    // IVF path: widen the probe set, then re-run the cell-restricted
    // search. Borrow-check: take the ctx out, mutate, put it back.
    if (*opaque).ivf.is_some() {
        let mut ctx = (*opaque).ivf.take().unwrap();
        // Cap on probe widening: min(max_probes, lists).
        let max_probes = (crate::guc::MAX_PROBES.get() as usize).clamp(1, ctx.lists);
        let can_widen = ctx.probes < max_probes;
        let can_grow_k = old_k < k_ceiling;
        if !can_widen && !can_grow_k {
            // Probe set is at its cap and k is maxed — exhausted.
            // (When max_probes == lists this means the whole corpus
            // has been scanned.) Put the ctx back so a later call
            // sees a consistent state.
            (*opaque).ivf = Some(ctx);
            return false;
        }

        // Phase B-1/B-2 out-of-core refill: widen the probe set (the
        // OOC index re-coarse-probes from its cached centroids),
        // then gather the wider cell set off the mmap. Same widening
        // schedule + same k-growth as the whole-load path, so the
        // iterative results match.
        if let Some(ooc) = ctx.ooc.clone() {
            if can_widen {
                let new_probes = (ctx.probes.saturating_mul(2))
                    .min(max_probes)
                    .max(ctx.probes + 1);
                let unit = kernels::normalise_to_vec(&(*opaque).query);
                ctx.probed_cells = ooc.coarse_probe_cells(&unit, new_probes);
                ctx.probes = new_probes;
            }
            let new_k = (old_k.saturating_mul(2)).min(k_ceiling).max(old_k + 1);
            let before = (*opaque).results.len();
            let (scores, ids) = ooc
                .search_ooc(
                    (*scan).indexRelation,
                    &(*opaque).query,
                    new_k,
                    &ctx.probed_cells,
                    &ctx.tombstones,
                    (*opaque).allow.as_ref(),
                )
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            (*opaque).current_k = new_k;
            let probes_now = ctx.probes;
            (*opaque).ivf = Some(ctx);
            populate_batch(opaque, &scores, &ids);
            if (*opaque).results.len() > before {
                return true;
            }
            return probes_now < max_probes || (*opaque).current_k < k_ceiling;
        }

        // Widen probes (double, capped). Rebuild the mask only when
        // the probe set actually grew.
        if can_widen {
            let new_probes = (ctx.probes.saturating_mul(2))
                .min(max_probes)
                .max(ctx.probes + 1);
            let probed = crate::index::ivf::coarse_probe(
                &ctx.centroids,
                ctx.lists,
                ctx.q_rot.len(),
                &ctx.q_rot,
                new_probes,
            );
            ctx.mask = ctx.directory.probe_mask(&probed, n_live);
            apply_tombstones(&mut ctx.mask, &ctx.tombstones);
            // Phase C: re-AND the allowlist into the rebuilt mask so
            // the widened probe set stays restricted to allowed slots.
            apply_allowlist(&mut ctx.mask, &arc, (*opaque).allow.as_ref());
            ctx.probes = new_probes;
        }
        // Grow k within the (possibly wider) cell set so the extra
        // cells can contribute candidates. Floor at old_k + 1 to
        // guarantee forward progress even when probes alone widened.
        let new_k = (old_k.saturating_mul(2)).min(k_ceiling).max(old_k + 1);
        let before = (*opaque).results.len();
        let (scores, ids) = arc
            .search_masked(&(*opaque).query, new_k, &ctx.mask)
            // Defensive: the ReadOnly arm this ctx came from always
            // returns Some; fall back to flat only if that ever
            // changes (e.g. handle swapped mid-scan).
            .unwrap_or_else(|| {
                flat_search(&arc, &(*opaque).query, new_k, (*opaque).allow.as_ref())
            });
        (*opaque).current_k = new_k;
        (*opaque).ivf = Some(ctx);
        populate_batch(opaque, &scores, &ids);
        if (*opaque).results.len() > before {
            return true;
        }
        // Nothing new this round. Keep going only if the probe set or
        // k can still grow on the next attempt.
        let ctx_ref = (*opaque).ivf.as_ref().unwrap();
        let max_probes2 = (crate::guc::MAX_PROBES.get() as usize).clamp(1, ctx_ref.lists);
        return ctx_ref.probes < max_probes2 || (*opaque).current_k < k_ceiling;
    }

    // Flat path (v1.8.0): grow k only.
    if old_k >= k_ceiling {
        // Already examined as many candidates as we're allowed to.
        return false;
    }
    let new_k = (old_k.saturating_mul(2)).min(k_ceiling).max(old_k + 1);
    let before = (*opaque).results.len();
    let (scores, ids) = flat_search(&arc, &(*opaque).query, new_k, (*opaque).allow.as_ref());
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
