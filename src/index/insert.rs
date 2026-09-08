//! `aminsert` — incremental insert into an existing turbovec index.
//!
//! Mutates the cached `IdMapIndex` under a `parking_lot::RwLock`
//! write guard, marks the entry dirty, and defers the relfile-page
//! write to the `PreCommit` xact callback (see `src/xact.rs`). One
//! relfile rewrite per transaction, regardless of how many rows
//! were inserted.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::cache::ReadOnlyIndex;
use crate::guc;
use crate::index::graph;
use crate::index::{options, relfile};
use crate::kernels;
use crate::vec::Vector;
use turbovec::IdMapIndex;

/// `aminsert` callback. Returns `true` if the index now contains the
/// row; `false` if we deliberately skipped it. We never skip
/// without an explicit reason (NULL embeddings, decode failures);
/// any unexpected condition produces an `ERROR` instead.
///
/// The callback signature changed in PG 14 to add the
/// `indexUnchanged` flag (used by HOT chain elision); pg13 has the
/// 7-arg form. We expose two thin wrappers and pick which one to
/// install in `register_am`.
#[cfg(not(feature = "pg13"))]
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_impl(index_relation, values, isnull, heap_tid)
}

/// PG 13 `aminsert` shape — no `indexUnchanged` parameter.
#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_impl(index_relation, values, isnull, heap_tid)
}

unsafe fn aminsert_impl(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
) -> bool {
    let indexrelid = (*index_relation).rd_id;

    // Single-column indexes only — values[0] / isnull[0].
    if *isnull {
        // NULL embeddings simply don't get indexed (matches pgvector).
        return false;
    }
    let datum: pg_sys::Datum = *values;
    let value: Option<Vector> = pgrx::FromDatum::from_datum(datum, false);
    let Some(value) = value else {
        return false;
    };

    let dim = value.dim();
    if dim % 8 != 0 {
        error!(
            "turbovec aminsert: dim must be a multiple of 8 (got {})",
            dim
        );
    }

    // Encode CTID into u64 using the canonical pgrx layout.
    let id = pgrx::itemptr::item_pointer_to_u64(*heap_tid);

    // Phase G-2b: real incremental insert into a Vamana graph index.
    // A whole-relfile rewrite (read everything back, mutate in RAM,
    // write everything back), NOT the deferred RwLock/xact-callback
    // caching every other kind's aminsert uses -- documented,
    // deliberate O(n) cost per insert (see `insert_graph_row`'s doc
    // comment). Graph inserts are expected to be RARE/one-at-a-time
    // (bulk-load via REINDEX, per the reloption's own guidance); this
    // is the simplest CORRECT implementation, not the fastest one --
    // matching G-2a/G-2b's whole "correctness-first" scope.
    if let Some(meta) = relfile::read_meta(index_relation) {
        if meta.is_graph() {
            return insert_graph_row(index_relation, value, id);
        }
        if meta.is_bq() {
            return insert_bq_row(index_relation, value, id);
        }
    }

    let normalise = guc::NORMALIZE_ON_INSERT.get();
    let (bit_width, _, _, _, _graph) = options::read(index_relation);

    aminsert_relfile(
        index_relation,
        indexrelid,
        bit_width,
        dim,
        normalise,
        value,
        id,
    )
}

unsafe fn aminsert_relfile(
    index_relation: pg_sys::Relation,
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: usize,
    normalise: bool,
    value: Vector,
    id: u64,
) -> bool {
    use crate::cache::{self, CacheKey, PersistState};

    let buf = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    let key = CacheKey {
        rel_oid: indexrelid,
        attnum: 0,
        bit_width: bit_width as u8,
        dim: dim as u32,
    };

    let relfile_node = cache::relfilenode_from_relation(index_relation);
    let arc = match cache::am_lookup_for_mutation(key, relfile_node) {
        Some(a) => a,
        None => {
            // First mutation in this tx: load from relfile pages.
            let (idx_index, n_vectors_existing, version_existing) =
                match relfile::read_meta(index_relation) {
                    Some(meta) if meta.n_vectors > 0 => {
                        // v1.29.1 corruption fix: read meta + chains
                        // atomically under the shared rewrite lock and
                        // build the index from the RETURNED fresh meta
                        // (a concurrent flush/vacuum may have rewritten
                        // the relfile since our `read_meta` above).
                        let (meta, codes, scales, ids) =
                            relfile::read_full_consistent(index_relation, &meta);
                        // Duplicate-id corrupt relfile: fail loudly
                        // with an actionable REINDEX hint (same as the
                        // read path) BEFORE turbovec rejects it with
                        // an opaque "duplicate ids in .tvim file", so a
                        // backfill loop gets a clear signal instead of
                        // retrying an opaque error forever. Reported
                        // 2026-07-30 (agora). Only for the bijective
                        // flat kind (`lists == 0`): an IVF index
                        // (`lists > 0`) legitimately repeats ids across
                        // cells (soft-assignment), so uniqueness must
                        // NOT be asserted there. See
                        // scan::assert_ids_unique_or_reindex.
                        if meta.lists == 0 {
                            crate::index::scan::assert_ids_unique_or_reindex(index_relation, &ids);
                        }
                        // wire v8: per-index TQ+ (empty = identity; a
                        // pg_turbovec index never calibrates today).
                        let (tqplus_shift, tqplus_scale) =
                            relfile::read_tqplus(index_relation, &meta);
                        let idx = IdMapIndex::from_id_map_parts(
                            meta.bit_width as usize,
                            meta.dim as usize,
                            meta.n_vectors as usize,
                            codes,
                            scales,
                            ids,
                            tqplus_shift,
                            tqplus_scale,
                        )
                        .unwrap_or_else(|e| {
                            error!("turbovec aminsert: corrupt relfile pages: {}", e)
                        });
                        (idx, meta.n_vectors as i64, meta.am_version as i32)
                    }
                    _ => (
                        IdMapIndex::new(dim, bit_width as usize).expect(
                            "turbovec aminsert: invalid (dim, bit_width) for IdMapIndex::new",
                        ),
                        0,
                        0,
                    ),
                };
            let bytes_per_vec = (dim * bit_width as usize) / 8 + 4 + 64;
            let total_bytes = bytes_per_vec * n_vectors_existing.max(1) as usize;
            let live_ids = idx_index.slot_to_id().to_vec();
            let persist_state = PersistState {
                bit_width,
                dim: dim as i32,
                n_vectors: n_vectors_existing,
                version: version_existing,
                live_ids,
                touched_ids: Vec::new(),
            };
            cache::am_install(
                key,
                idx_index,
                total_bytes,
                relfile_node,
                version_existing as i64,
                persist_state,
            )
        }
    };

    let mut id_already_present = false;
    {
        let mut guard = arc.write();
        if guard.dim() != 0 && guard.dim() != dim {
            error!(
                "turbovec aminsert: dim mismatch — index expects {}, row has {}",
                guard.dim(),
                dim
            );
        }
        match guard.add_with_ids(&buf, &[id]) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{:?}", e);
                if msg.contains("IdAlreadyPresent") {
                    guard.remove(id);
                    if let Err(e2) = guard.add_with_ids(&buf, &[id]) {
                        error!("turbovec aminsert: re-add after remove failed: {:?}", e2);
                    }
                    id_already_present = true;
                } else {
                    error!("turbovec aminsert: add_with_ids failed: {:?}", e);
                }
            }
        }
    }

    let updated = cache::am_mark_dirty(key, |p| {
        // v1.29.1 corruption fix #2: record EVERY upserted id (new or
        // ON-CONFLICT re-add) so the PreCommit flush can splice just
        // these onto fresh on-disk state instead of blind-overwriting
        // from this backend's stale snapshot. Both the fresh-insert
        // and the IdAlreadyPresent remove+re-add branch above land
        // the id in this backend's in-memory index, so both must be
        // reconciled.
        p.touched_ids.push(id);
        if !id_already_present {
            // Keep the `live_ids` / `n_vectors` mirror in step with the
            // in-memory index's `slot_to_id`. As of v1.28.4 the flush
            // path (`xact::flush_to_relfile`) DERIVES the persisted
            // count from `idx.slot_to_id().len()`, not from
            // `p.n_vectors` — so this mirror is now only a
            // cross-check (hard-guarded at flush) and the scan-
            // visibility snapshot (`am_find_dirty_by_rel`). Keeping it
            // exact preserves that guard's usefulness.
            p.live_ids.push(id);
            p.n_vectors += 1;
        }
        p.version += 1;
    });
    if !updated {
        error!("turbovec aminsert: cache entry vanished between install and mark_dirty");
    }

    crate::xact::ensure_xact_callbacks_registered();

    true
}

/// Phase G-2b: real incremental insert into an existing Vamana graph
/// index (`meta.is_graph()`). Whole-relfile rewrite — read every
/// existing chain back, quantize + append the new row via the SAME
/// `IdMapIndex::add_with_ids` path every other kind's build/insert
/// already uses, run [`graph::insert_one_node_via_oracle`] to extend
/// the adjacency, then persist everything back via
/// `relfile::write_full_with_prepared_graph_and_tombstones` (the
/// tombstone-aware twin of the function `build.rs`'s
/// `graph_build_and_write` uses).
///
/// ## C1 (v2.0.x): the whole read-modify-write is under ONE lock
///
/// This function used to `read_full` WITHOUT the relfile rewrite lock,
/// do the (expensive) quantize + Vamana-insert CPU work, then BLINDLY
/// `write_full_with_prepared_graph` from that stale snapshot — the
/// exact lost-update pattern v1.29.1/v1.29.4 fixed for flat/IVF but
/// which was never applied to the graph kind. Two concurrent graph
/// inserters both read `n_vectors = N`, both compute an adjacency for
/// `N + 1`, and the second write lands a meta page whose
/// `graph_offsets_bytes` describe `N + 2` nodes over a neighbors chain
/// the first writer sized for `N + 1` — reproduced as
/// `corrupt graph adjacency chain: graph offsets[n]=54272 !=
/// neighbors.len()=54240`, plus silently lost rows (the loser's row is
/// gone from the index while its heap tuple committed).
///
/// The fix is structural, not a retry loop: take
/// [`relfile::lock_relfile_write`] FIRST, re-read the meta + every
/// chain UNDER it, and hold it across recompute + write. That makes
/// the whole insert serialize against another inserter, a VACUUM, and
/// a deferred flat flush — all of which already take the same
/// exclusive page lock. The lock is a heavyweight page lock, so it is
/// released by the lock manager on a longjmp / xact end even if
/// something below `error!`s.
///
/// The tombstone bitmap is re-persisted in the SAME rewrite (see
/// `write_full_with_prepared_graph_and_tombstones`) instead of by a
/// second `write_tombstones_and_meta` call, closing the M2 window
/// where a crash between the two writes silently un-tombstoned every
/// VACUUM delete.
///
/// **Cost, documented explicitly**: this is `O(n)` per single-row
/// insert (every existing chain is read AND rewritten), not the
/// deferred-per-transaction-batch, O(1)-amortized path every other
/// kind's `aminsert` gets via `cache::am_mark_dirty` +
/// `xact::ensure_xact_callbacks_registered`. A graph index built
/// `WITH (graph = true)` is documented (reloption help, CHANGELOG)
/// as a build-then-query-mostly structure; bulk-loading many rows
/// into an EXISTING graph index one at a time will be slow by
/// design, not by oversight — REINDEX after a bulk load, per the
/// same guidance the (now-removed) hard-error message used to give.
/// Holding the rewrite lock across the CPU work makes concurrent
/// graph inserts strictly serial, which is a throughput ceiling, not
/// a new one: they were already O(n) each, and correctness is not
/// negotiable here (AGENTS.md's hard mandate).
/// A proper fix (touching only the handful of adjacency lists that
/// actually change per insert, batching multiple inserts into one
/// relfile rewrite per transaction like the flat/IVF path does) is
/// real future work, not attempted here — G-2b's scope is
/// correctness, not this performance profile.
/// `aminsert` for a 1-bit sign-BQ index (`kind = KIND_BQ`).
///
/// Like the graph kind, this is a deliberate O(n)-per-row whole-relfile
/// rewrite under ONE exclusive lock across read -> recompute -> write.
/// That locking shape is not a style choice: an unlocked read-modify-write
/// is exactly the bug that SILENTLY lost rows from graph indexes before
/// v2.1.0 (while `turbovec_check()` reported `is_corrupt = false`).
///
/// The corpus mean is deliberately NOT recomputed on insert. Recomputing
/// it would invalidate every previously-packed sign code (they were taken
/// relative to the OLD mean), so a single insert would silently corrupt
/// the ranking of the whole index. The persisted build-time mean is
/// therefore treated as fixed, and drift is a REINDEX concern -- the same
/// build-then-serve model the reloption's guidance already sets out.
///
/// ## IVF + 1-bit (`WITH (lists = N, bit_width = 1)`) DEGRADES to flat
///
/// The new row's cell is unknown to this path (the coarse centroids are on
/// disk, but a cell-contiguous insert would have to SHIFT every slot after
/// the target cell and renumber the whole cell directory -- an O(n)
/// rewrite that also invalidates the tombstone bitmap's slot indices).
/// Rather than get that subtly wrong, the insert appends the row at the
/// END and drops the cell metadata, so the index falls back to a flat
/// Hamming scan over all slots: SLOWER, never wrong (a full scan can only
/// improve recall) and never silently lost.
///
/// Crucially this is OBSERVABLE, not silent: `lists` is preserved and
/// `ivf_degraded` is stamped, so `turbovec.index_is_degraded(rel)` reports
/// `true` and `ambeginscan` emits the throttled degradation WARNING
/// naming the index -- REINDEX restores the cells. (The TurboQuant IVF
/// insert path degrades too, but blanks `lists` outright and is therefore
/// NOT reportable; see the note in `xact::validate_flush_snapshot`.)
/// Real cell-aware incremental insert is future work, not attempted here.
unsafe fn insert_bq_row(index_relation: pg_sys::Relation, value: Vector, id: u64) -> bool {
    use crate::index::onebit;

    let normalise = guc::NORMALIZE_ON_INSERT.get();
    relfile::lock_relfile_write(index_relation);

    let meta = match relfile::read_meta(index_relation) {
        Some(m) if m.is_bq() => m,
        Some(_) => {
            relfile::unlock_relfile_write(index_relation);
            error!(
                "turbovec aminsert (1-bit): index is no longer a 1-bit index (concurrent REINDEX?); retry the statement"
            );
        }
        None => {
            relfile::unlock_relfile_write(index_relation);
            error!("turbovec aminsert (1-bit): meta page vanished (concurrent REINDEX?)");
        }
    };

    // An EMPTY BQ build stamps `dim = 0` (no rows and no reloption-pinned
    // dim), just like the flat path's empty meta page. The flat path copes
    // because it takes `dim` from the INCOMING ROW rather than from the meta
    // page; this path read it from meta, so `CREATE TABLE` -> `CREATE INDEX`
    // (on the empty table) -> `INSERT` failed with "index expects 0, row has
    // N". Found on a real 1024-d corpus host, not by a unit test: every
    // in-tree BQ fixture happened to index an already-populated table.
    let empty_unpinned = meta.dim == 0 && meta.n_vectors == 0;
    let dim = if empty_unpinned {
        value.dim()
    } else {
        meta.dim as usize
    };
    if value.dim() != dim {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (1-bit): dim mismatch — index expects {}, row has {}",
            dim,
            value.dim()
        );
    }

    // A 0-row index has no persisted mean yet; this row's own centring is
    // the identity (a 1-row corpus mean IS that row, so every centred
    // coordinate is 0). Seed the mean from the first row so the codes stay
    // self-consistent, matching what a rebuild over one row would produce.
    let mean = if empty_unpinned {
        vec![0.0f32; dim]
    } else {
        relfile::read_bq_mean(index_relation, &meta)
    };
    if mean.len() != dim {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (1-bit): index is missing its corpus-mean vector; REINDEX INDEX to rebuild it"
        );
    }

    // Read the current codes + ids + tombstones, append this row, write it
    // all back.
    let mut codes = relfile::read_chain(
        index_relation,
        meta.codes_first,
        meta.stride_bytes,
        meta.rows_per_codes_page,
        meta.n_vectors,
    );
    let mut ids = relfile::read_ids_only(index_relation, &meta);
    let tombstones = relfile::read_tombstones(index_relation, &meta);

    // Hard guard, the same shape as the graph path's: the chains we just
    // read must agree with the row count we are about to extend by one. A
    // mismatch means the on-disk state is already torn, and reconciling
    // onto it would ENTRENCH the corruption in a freshly written meta.
    let n_disk = meta.n_vectors as usize;
    let stride = meta.stride_bytes as usize;
    if ids.len() != n_disk || (stride > 0 && codes.len() != n_disk * stride) {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (1-bit): on-disk row count drift — meta says {} rows but the ids chain has {} and the codes chain has {} bytes ({} expected); the index appears corrupt. REINDEX INDEX to rebuild it from the heap",
            n_disk,
            ids.len(),
            codes.len(),
            n_disk * stride,
        );
    }

    let v = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };
    let centered = onebit::center(&v, &mean);
    let new_code = onebit::pack_signs(&centered);

    // M2 (the graph kind's v2.1.0 lesson, applied here): the tombstone
    // bitmap must be re-persisted in the SAME rewrite. `plan_bq` plans a
    // fresh meta with the tombstone fields zeroed, so writing without
    // carrying the bitmap forward would silently RESURRECT every
    // VACUUM-deleted row.
    let mut new_tombstones = tombstones;

    // Re-insert of an EXISTING heap TID (what the flat TurboQuant path
    // handles as `IdAlreadyPresent` via remove + re-add): overwrite that
    // slot's code in place instead of appending a second slot for the same
    // logical row. Appending would grow the index without bound under
    // repeated upserts of the same row, and would put the same TID in two
    // slots -- which is exactly the shape the id-table bijection guard
    // treats as corruption for the flat kind.
    //
    // If that slot had been TOMBSTONED, clear its bit: the row is being
    // written back, so it is live again. Leaving the bit set would make an
    // upserted row permanently invisible.
    //
    // On an IVF-built (soft-assigned) index the SAME id can legitimately
    // occupy several slots, so update EVERY one of them -- they all
    // describe the same row and must not disagree.
    let existing: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter_map(|(slot, &existing_id)| (existing_id == id).then_some(slot))
        .collect();
    if existing.is_empty() {
        codes.extend_from_slice(&new_code);
        ids.push(id);
        if !new_tombstones.is_empty() {
            // One more slot; the new row is live (bit 0).
            new_tombstones.resize(ids.len().div_ceil(8), 0);
        }
    } else {
        for &slot in &existing {
            codes[slot * stride..(slot + 1) * stride].copy_from_slice(&new_code);
            if let Some(byte) = new_tombstones.get_mut(slot / 8) {
                *byte &= !(1u8 << (slot % 8));
            }
        }
    }

    relfile::write_full_bq_parts(
        index_relation,
        dim as u32,
        ids.len() as u64,
        &codes,
        &ids,
        &mean,
        meta.am_version + 1,
        // No cell metadata: an APPENDED slot is not in any cell, so the
        // directory can no longer partition the slots. See the doc above.
        // (Even a pure in-place overwrite takes this path: a code change
        // can move the row to a different cell, so the old directory would
        // be describing a membership that no longer holds.)
        None,
        &new_tombstones,
        // Preserve `lists` + stamp `ivf_degraded` so an IVF+BQ index's
        // degradation is REPORTABLE rather than silent. `0` for a flat BQ
        // index, which was never IVF and is not degraded by an insert.
        meta.lists,
    );
    relfile::unlock_relfile_write(index_relation);
    crate::cache::invalidate_all();
    false
}

unsafe fn insert_graph_row(index_relation: pg_sys::Relation, value: Vector, id: u64) -> bool {
    let normalise = guc::NORMALIZE_ON_INSERT.get();

    // C1: ONE exclusive rewrite lock across read -> recompute -> write.
    // Taken BEFORE the meta re-read so the `meta` every step below uses
    // is the state no other rewriter can change under us. Same-xact
    // re-entry (the write path takes it again) does not self-conflict.
    relfile::lock_relfile_write(index_relation);

    // Re-read the meta UNDER the lock: the caller's `read_meta` was
    // taken unlocked, so a concurrent insert/VACUUM may have rewritten
    // the relfile (new n_vectors, new chain offsets) since.
    let meta = match relfile::read_meta(index_relation) {
        Some(m) if m.is_graph() => m,
        Some(_) => {
            // The index stopped being a graph index between the caller's
            // unlocked probe and this locked re-read (only a concurrent
            // REINDEX/build can do that, and it holds AccessExclusive on
            // the relation, so this is unreachable in practice). Bail out
            // loudly rather than write a graph chain over a flat index.
            relfile::unlock_relfile_write(index_relation);
            error!(
                "turbovec aminsert (graph): index is no longer a graph index (concurrent REINDEX?); retry the statement"
            );
        }
        None => {
            relfile::unlock_relfile_write(index_relation);
            error!("turbovec aminsert (graph): meta page vanished (concurrent REINDEX?)");
        }
    };

    let dim = meta.dim as usize;
    if value.dim() != dim {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (graph): dim mismatch — index expects {}, row has {}",
            dim,
            value.dim()
        );
    }
    let new_vec = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    // Read every existing chain back, still under the same lock. Uses
    // `read_full` (which takes the SHARED side; same-xact share-after-
    // exclusive does not self-conflict and adds no window, since we
    // never drop the exclusive side).
    let (codes, scales, ids) = relfile::read_full(index_relation, &meta);
    if ids.contains(&id) {
        // Matches the flat path's `IdAlreadyPresent` handling: a
        // re-insert of the same heap TID (e.g. a HOT update that
        // still touches the indexed column) is not a new row.
        // Rejecting cleanly here (rather than silently duplicating a
        // slot or corrupting the adjacency chain by assuming
        // `n_vectors` grew by one) is the safe, simple choice for a
        // whole-rewrite path — REINDEX recovers cleanly either way.
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (graph): heap tid {} already present in this graph index (re-insert of an existing row is not supported for the graph kind); REINDEX INDEX to rebuild if the underlying table changed",
            id
        );
    }
    // C1 hard guard: the chains we just read must agree with the meta
    // row count we are about to extend by one. A mismatch means the
    // on-disk state is already torn (pre-fix binary, or a truncated
    // relfile); reconciling onto it would ENTRENCH the corruption in a
    // freshly written meta page. Abort with an actionable hint instead.
    let n_disk = meta.n_vectors as usize;
    if ids.len() != n_disk || scales.len() != n_disk {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (graph): on-disk row count drift — meta says {} rows but the ids chain has {} and the scales chain has {}; the index appears corrupt. REINDEX INDEX to rebuild it from the heap",
            n_disk,
            ids.len(),
            scales.len()
        );
    }
    let stored_index: ReadOnlyIndex = if meta.has_prepared_layout() {
        // wire v8 / turbovec 1.0.0: codebook + rotation derived from
        // (bit_width, dim); the SIMD-blocked layout is recomputed
        // inside `from_prepared_parts`. Only the per-index TQ+ pair is
        // read back (empty = identity today).
        let (tqplus_shift, tqplus_scale) = relfile::read_tqplus(index_relation, &meta);
        ReadOnlyIndex::from_prepared_parts(
            meta.bit_width as usize,
            dim,
            meta.n_vectors as usize,
            codes.clone(),
            scales.clone(),
            ids.clone(),
            tqplus_shift,
            tqplus_scale,
        )
    } else {
        ReadOnlyIndex::from_parts(
            meta.bit_width as usize,
            dim,
            meta.n_vectors as usize,
            codes.clone(),
            scales.clone(),
            ids.clone(),
        )
    };
    let adjacency = match relfile::read_graph_adjacency(index_relation, &meta) {
        Some(a) => a,
        None => {
            relfile::unlock_relfile_write(index_relation);
            error!(
                "turbovec aminsert (graph): meta.is_graph() was true but the adjacency chain is missing; REINDEX INDEX to rebuild"
            );
        }
    };
    // C1 hard guard #2: the adjacency must describe exactly the rows
    // the meta claims. `read_graph_adjacency` already rejects an
    // offsets/neighbors length mismatch; this additionally pins the
    // node count against `n_vectors`, so an adjacency built for a
    // DIFFERENT row count (the exact artefact the unlocked
    // read-modify-write produced) can never be extended in place.
    if adjacency.n() != n_disk {
        relfile::unlock_relfile_write(index_relation);
        error!(
            "turbovec aminsert (graph): adjacency describes {} nodes but the meta says {} rows; the index appears corrupt. REINDEX INDEX to rebuild it from the heap",
            adjacency.n(),
            n_disk
        );
    }
    let tombstones = relfile::read_tombstones(index_relation, &meta);

    // Score oracle for `insert_one_node_via_oracle`: the new row's
    // raw f32 vector against a batch of EXISTING slot ids, via the
    // exact same quantized-code kernel the scan path already trusts
    // (`ReadOnlyIndex::score_slots`). Tombstoned slots are excluded
    // from consideration up front (never offered as insertion
    // candidates) rather than filtered post-hoc inside the oracle —
    // simpler, and `graph::insert_one_node_via_oracle`'s caller
    // contract doesn't need tombstone-awareness itself (VACUUM and
    // insert are serialized by the relfile rewrite lock this function
    // now holds for its whole body).
    let live_mask: Vec<bool> = if tombstones.is_empty() {
        vec![true; n_disk]
    } else {
        (0..n_disk)
            .map(|slot| {
                tombstones
                    .get(slot / 8)
                    .is_none_or(|&b| b & (1 << (slot % 8)) == 0)
            })
            .collect()
    };
    let entry = if live_mask.get(meta.graph_entry_point as usize) == Some(&true) {
        meta.graph_entry_point
    } else {
        // Entry point itself is tombstoned (VACUUM should have
        // already picked a fallback in the meta page — see
        // `vacuum.rs` — but defend here too rather than trust that
        // invariant blindly across a code path this far from where
        // it's enforced).
        live_mask
            .iter()
            .position(|&live| live)
            .map(|s| s as u32)
            .unwrap_or(0)
    };
    let score_existing = |query: &[f32], batch_ids: &[u32]| -> Vec<f32> {
        stored_index.score_slots(query, batch_ids)
    };
    let new_adjacency =
        graph::insert_one_node_via_oracle(&adjacency, entry, &new_vec, score_existing);

    // Quantize + append the new row via the SAME `IdMapIndex`
    // encode path every other kind's build/insert already uses.
    // Synthetic slot id = the new last index (matches
    // `graph_build_and_write`'s "slot ids == 0..n_vectors, real
    // external ids kept in a parallel array" convention).
    // wire v8: per-index TQ+ (empty = identity; never calibrated today).
    let (tqplus_shift, tqplus_scale) = relfile::read_tqplus(index_relation, &meta);
    let mut idx = IdMapIndex::from_id_map_parts(
        meta.bit_width as usize,
        dim,
        meta.n_vectors as usize,
        codes,
        scales,
        (0..meta.n_vectors).collect(),
        tqplus_shift,
        tqplus_scale,
    )
    .unwrap_or_else(|e| error!("turbovec aminsert (graph): corrupt relfile pages: {}", e));
    let new_slot = meta.n_vectors;
    idx.add_with_ids(&new_vec, &[new_slot])
        .unwrap_or_else(|e| error!("turbovec aminsert (graph): add_with_ids failed: {:?}", e));
    let mut real_ids = ids;
    real_ids.push(id);

    idx.prepare();
    let prepared = relfile::PreparedParts {
        tqplus_shift: idx.tqplus_shift(),
        tqplus_scale: idx.tqplus_scale(),
    };
    let offsets_bytes = new_adjacency.encode_offsets();
    let neighbors_bytes = new_adjacency.encode_neighbors();
    let graph_parts = relfile::GraphParts {
        offsets_bytes: &offsets_bytes,
        neighbors_bytes: &neighbors_bytes,
        entry_point: entry,
    };
    // C1 / M2: ONE rewrite that carries the tombstone bitmap too.
    //
    // Before this, `write_full_with_prepared_graph` planned a brand-new
    // meta page from scratch (`MetaPageData::plan_with_blocked`), which
    // does NOT carry forward an existing tombstone chain — so the write
    // silently reset `tombstone_count`/`tombstone_first`/
    // `tombstone_bytes` to 0, and a SECOND `write_tombstones_and_meta`
    // call had to put them back. Two meta commits means a crash /
    // cancel / FATAL between them permanently un-tombstones every
    // VACUUM delete. Folding the bitmap into the single rewrite closes
    // that window: the meta page is written exactly once, last, already
    // referencing the tombstone chain.
    //
    // The bitmap is extended by one bit for the new slot (defaulting to
    // 0 = live; the new row is obviously not dead). An index that never
    // had tombstones passes an empty slice and gets the previous
    // no-tombstone-chain layout, byte-identical to before.
    let new_tombstones: Vec<u8> = if tombstones.is_empty() {
        Vec::new()
    } else {
        let mut b = tombstones;
        b.resize((new_slot as usize + 2).div_ceil(8), 0);
        b
    };
    relfile::write_full_with_prepared_graph_and_tombstones(
        index_relation,
        meta.bit_width,
        dim as u32,
        new_slot + 1,
        idx.packed_codes(),
        idx.scales(),
        &real_ids,
        meta.am_version.saturating_add(1),
        prepared,
        graph_parts,
        &new_tombstones,
    );
    relfile::unlock_relfile_write(index_relation);
    true
}
