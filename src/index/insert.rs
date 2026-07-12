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
use crate::index::page::MetaPageData;
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
            return insert_graph_row(index_relation, &meta, value, id);
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
                        let (codes, scales, ids) = relfile::read_full(index_relation, &meta);
                        let idx = IdMapIndex::from_id_map_parts(
                            meta.bit_width as usize,
                            meta.dim as usize,
                            meta.n_vectors as usize,
                            codes,
                            scales,
                            ids,
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
        if !id_already_present {
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
/// `relfile::write_full_with_prepared_graph` (the exact same
/// function `build.rs`'s `graph_build_and_write` uses).
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
/// A proper fix (touching only the handful of adjacency lists that
/// actually change per insert, batching multiple inserts into one
/// relfile rewrite per transaction like the flat/IVF path does) is
/// real future work, not attempted here — G-2b's scope is
/// correctness, not this performance profile.
unsafe fn insert_graph_row(
    index_relation: pg_sys::Relation,
    meta: &MetaPageData,
    value: Vector,
    id: u64,
) -> bool {
    let dim = meta.dim as usize;
    if value.dim() != dim {
        error!(
            "turbovec aminsert (graph): dim mismatch — index expects {}, row has {}",
            dim,
            value.dim()
        );
    }
    let normalise = guc::NORMALIZE_ON_INSERT.get();
    let new_vec = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    // Read every existing chain back (same pattern
    // `scan.rs::install_graph_index` uses to build a `ReadOnlyIndex`
    // for the scan path — reused here, not reinvented).
    let (codes, scales, ids) = relfile::read_full(index_relation, meta);
    if ids.contains(&id) {
        // Matches the flat path's `IdAlreadyPresent` handling: a
        // re-insert of the same heap TID (e.g. a HOT update that
        // still touches the indexed column) is not a new row.
        // Rejecting cleanly here (rather than silently duplicating a
        // slot or corrupting the adjacency chain by assuming
        // `n_vectors` grew by one) is the safe, simple choice for a
        // whole-rewrite path — REINDEX recovers cleanly either way.
        error!(
            "turbovec aminsert (graph): heap tid {} already present in this graph index (re-insert of an existing row is not supported for the graph kind); REINDEX INDEX to rebuild if the underlying table changed",
            id
        );
    }
    let stored_index: ReadOnlyIndex = if meta.has_prepared_layout() {
        // Phase Q-0 (v7): recompute the SIMD-blocked layout from the
        // packed codes (no longer persisted on disk).
        let (blocked, n_blocks) = turbovec::pack::repack(
            &codes,
            meta.n_vectors as usize,
            meta.bit_width as usize,
            dim,
        );
        let centroids = meta.centroids_slice().to_vec();
        let boundaries = meta.boundaries_slice().to_vec();
        let rotation = relfile::read_rotation(index_relation, meta);
        let rotation_opt = if rotation.is_empty() {
            None
        } else {
            Some(rotation)
        };
        ReadOnlyIndex::from_prepared_parts(
            meta.bit_width as usize,
            dim,
            meta.n_vectors as usize,
            codes.clone(),
            scales.clone(),
            ids.clone(),
            blocked,
            n_blocks,
            centroids,
            boundaries,
            rotation_opt,
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
    let adjacency = relfile::read_graph_adjacency(index_relation, meta)
        .expect("insert_graph_row: meta.is_graph() was true but the adjacency chain is missing");
    let tombstones = relfile::read_tombstones(index_relation, meta);

    // Score oracle for `insert_one_node_via_oracle`: the new row's
    // raw f32 vector against a batch of EXISTING slot ids, via the
    // exact same quantized-code kernel the scan path already trusts
    // (`ReadOnlyIndex::score_slots`). Tombstoned slots are excluded
    // from consideration up front (never offered as insertion
    // candidates) rather than filtered post-hoc inside the oracle —
    // simpler, and `graph::insert_one_node_via_oracle`'s caller
    // contract doesn't need tombstone-awareness itself (VACUUM and
    // insert are already serialized by the same exclusive relation
    // lock every other kind's mutation path holds).
    let live_mask: Vec<bool> = if tombstones.is_empty() {
        vec![true; meta.n_vectors as usize]
    } else {
        (0..meta.n_vectors as usize)
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
    let mut idx = IdMapIndex::from_id_map_parts(
        meta.bit_width as usize,
        dim,
        meta.n_vectors as usize,
        codes,
        scales,
        (0..meta.n_vectors).collect(),
    )
    .unwrap_or_else(|e| error!("turbovec aminsert (graph): corrupt relfile pages: {}", e));
    let new_slot = meta.n_vectors;
    idx.add_with_ids(&new_vec, &[new_slot])
        .unwrap_or_else(|e| error!("turbovec aminsert (graph): add_with_ids failed: {:?}", e));
    let mut real_ids = ids;
    real_ids.push(id);

    idx.prepare_eager();
    let rotation = idx.rotation();
    let prepared = relfile::PreparedParts {
        centroids: idx.centroids(),
        boundaries: idx.boundaries(),
        rotation,
    };
    let offsets_bytes = new_adjacency.encode_offsets();
    let neighbors_bytes = new_adjacency.encode_neighbors();
    let graph_parts = relfile::GraphParts {
        offsets_bytes: &offsets_bytes,
        neighbors_bytes: &neighbors_bytes,
        entry_point: entry,
    };
    relfile::write_full_with_prepared_graph(
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
    );
    // BUG FOUND + FIXED during G-2b's own test-writing:
    // `write_full_with_prepared_graph` -> `write_full_inner` always
    // plans a BRAND-NEW meta page from scratch (`MetaPageData::
    // plan_with_blocked`), which does NOT carry forward an existing
    // tombstone chain -- the write above just silently reset
    // `tombstone_count`/`tombstone_first`/`tombstone_bytes` to 0,
    // and the OLD tombstone bytes are now unreferenced (still
    // physically on disk in old pages, but nothing points at them).
    // Re-persist the (possibly nonexistent) tombstone bitmap NOW,
    // extended by one bit for the new slot (defaulting to 0 = live,
    // the new row is obviously not dead), via the SAME
    // `write_tombstones_and_meta` VACUUM already uses -- this is a
    // second small meta-page write immediately after the first, not
    // a single atomic operation, but `write_full_with_prepared_graph`
    // just finished (the relation is internally consistent at this
    // point, just missing the tombstone reference) and this
    // function holds the same exclusive lock the whole time, so
    // there is no window where a concurrent reader could observe
    // the intermediate (correct-but-tombstone-less) state.
    if !tombstones.is_empty() {
        let new_n_bytes = (new_slot as usize + 2).div_ceil(8);
        let mut new_bitmap = tombstones.clone();
        new_bitmap.resize(new_n_bytes, 0);
        // Re-read the meta this write just produced (fresh chain
        // offsets/counts) rather than reuse the stale `meta` this
        // function was called with.
        let fresh_meta = relfile::read_meta(index_relation)
            .expect("insert_graph_row: meta vanished immediately after our own write");
        relfile::write_tombstones_and_meta(
            index_relation,
            &fresh_meta,
            &new_bitmap,
            fresh_meta.am_version.saturating_add(1),
        );
    }
    // `write_full_with_prepared_graph` only touches the
    // codes/scales/ids/blocked/graph chains, never the tombstone
    // chain — confirmed by reading that function's body, not
    // assumed.
    true
}
