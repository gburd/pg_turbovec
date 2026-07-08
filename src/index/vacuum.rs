//! `ambulkdelete` / `amvacuumcleanup`.
//!
//! ## In-place page walk (Phase L hardening item 6)
//!
//! Earlier relfile builds rebuilt the entire `IdMapIndex` from
//! disk, removed dead ids in memory, and rewrote every chain page
//! via `relfile::write_full`. That cost O(n_vectors) of disk I/O
//! and WAL on every VACUUM regardless of how few rows died — a
//! single dead row in a 1 M-vector index rewrote ~200 MiB.
//!
//! The current implementation walks the existing chain pages and
//! does in-place swap-removes (analogue of `IdMapIndex::remove`):
//!
//! 1. Read the ids chain only (~8 MiB on 1 M rows) and ask the
//!    callback for each id; collect the dead slot indices.
//! 2. Sort dead slots **descending** and process from the back.
//!    For each dead slot `s` with `last = alive_count - 1`:
//!    - if `s == last`: nothing to copy, just decrement.
//!    - else: copy slot `last` → slot `s` on the codes / scales /
//!      ids chains (3 page writes per swap, each WAL-logged via
//!      `GenericXLog`), then decrement.
//!    Descending order guarantees the source `last` row is always
//!    a still-live row whose data hasn't been moved by an earlier
//!    iteration.
//! 3. Rewrite the meta page with the smaller `n_vectors` and a
//!    bumped `am_version` (drives the cache freshness check in
//!    `scan.rs`).
//! 4. `RelationTruncate` to release trailing ids-chain pages
//!    that the shrink left orphaned. Mid-file gaps between the
//!    codes / scales / ids chains are left in place; the next
//!    `write_full` (build / aminsert commit) re-packs.
//!
//! Cost: O(deleted) page writes + 1 meta write + 1 truncate, vs.
//! the old O(total) full rewrite. WAL volume scales the same way.
//!
//! ## IVF indexes tombstone instead of swap-removing (Phase E-2)
//!
//! Swap-remove moves the global last live slot into a deleted hole.
//! For an IVF index (`lists > 0`) the codes/scales/ids chains are
//! cell-contiguous, so that move crosses cell boundaries and breaks
//! the cell directory's `[code_offset .. +n_vectors)` partition. The
//! pre-E-2 code blanked the v4 IVF meta fields after a swap-remove,
//! silently degrading the index to an O(n) flat scan — a production
//! latency landmine for a churning index.
//!
//! The IVF path instead leaves every row in place, leaves
//! `n_vectors` and the cell directory untouched, and ORs the dead
//! slots into a persisted per-slot tombstone bitmap (a new v4-additive
//! chain). Cells stay contiguous, the directory stays valid,
//! `has_ivf()` stays true, and the scan masks the tombstoned slots
//! out of its cell-restriction mask so dead rows are never returned.
//! A REINDEX compacts the dead space. See `docs/PRODUCTION.md`.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::page::MetaPageData;
use crate::index::relfile;

/// `ambulkdelete`: process dead-tuple removal.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let stats = if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult
    } else {
        stats
    };
    if stats.is_null() {
        error!("turbovec: failed to allocate IndexBulkDeleteResult");
    }

    ambulkdelete_relfile((*info).index, stats, callback, callback_state)
}

unsafe fn ambulkdelete_relfile(
    index: pg_sys::Relation,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let meta = match relfile::read_meta(index) {
        Some(m) if m.n_vectors > 0 => m,
        _ => return stats,
    };

    let Some(cb) = callback else {
        // No callback supplied (cleanup pass without dead
        // tuples) — nothing to remove.
        (*stats).num_index_tuples = meta.n_vectors as f64;
        return stats;
    };

    // Pass 1: read the ids chain only (cheap — 8 bytes per row
    // vs. stride_bytes for codes) and collect dead slot indices.
    let ids = relfile::read_ids_only(index, &meta);
    debug_assert_eq!(ids.len() as u64, meta.n_vectors);

    let mut dead_slots: Vec<usize> = Vec::new();
    for (slot, &id) in ids.iter().enumerate() {
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::u64_to_item_pointer(id, &mut tid);
        let is_dead = (cb)(&mut tid as *mut _, callback_state);
        if is_dead {
            dead_slots.push(slot);
        }
    }

    let dead_count = dead_slots.len();
    if dead_count == 0 {
        // Nothing to remove. Don't bump am_version (no on-disk
        // change) and don't rewrite the meta page — we want a
        // pure read-only VACUUM pass to avoid emitting WAL.
        (*stats).num_index_tuples = meta.n_vectors as f64;
        return stats;
    }

    // Phase E-2: IVF indexes TOMBSTONE instead of swap-removing.
    //
    // Swap-remove (the flat path below) moves the global last live
    // slot into a deleted hole. For an IVF index the slots are
    // cell-contiguous, so that move crosses cell boundaries and
    // breaks the cell directory's `[code_offset .. +n_vectors)`
    // partition. The pre-E-2 code "fixed" this by blanking the v4
    // IVF meta fields, silently degrading the index to an O(n) flat
    // scan — a production latency landmine for a churning index.
    //
    // Instead we leave every row exactly where it is, leave
    // `n_vectors` and the cell directory untouched, and OR the newly
    // dead slots into a persisted per-slot tombstone bitmap. The
    // cells stay contiguous, the directory stays valid, `has_ivf()`
    // stays true, and the scan masks the tombstoned slots out. A
    // future REINDEX compacts the dead space.
    if meta.has_ivf() {
        let next_version = meta.am_version.saturating_add(1);
        let total_dead = ivf_tombstone_dead(index, &meta, &dead_slots, next_version);
        // n_vectors is unchanged (dead rows stay on disk as
        // tombstones); report the live count for the planner.
        let live = meta.n_vectors.saturating_sub(total_dead);
        (*stats).num_index_tuples = live as f64;
        (*stats).tuples_removed += dead_count as f64;
        return stats;
    }

    // Phase G-2b: graph indexes tombstone exactly like IVF (see the
    // module doc's "IVF indexes tombstone instead of swap-removing"
    // section above, and an internal design note's "VACUUM (all
    // graph phases)" note). Swap-remove would move the last live
    // slot into a deleted hole, silently invalidating every
    // adjacency-chain neighbor id that pointed at the moved-from or
    // moved-to slot (the graph has no notion of "this slot's
    // identity changed"). Instead we leave every row and the whole
    // adjacency chain untouched and OR the newly dead slots into the
    // SAME per-slot tombstone bitmap chain IVF uses (the storage
    // helpers in relfile.rs are already generic over slot index,
    // not IVF-specific). The scan path (`scan.rs` / `graph.rs`)
    // excludes tombstoned slots from traversal and results.
    if meta.is_graph() {
        let next_version = meta.am_version.saturating_add(1);
        let total_dead = graph_tombstone_dead(index, &meta, &dead_slots, next_version);
        let live = meta.n_vectors.saturating_sub(total_dead);
        (*stats).num_index_tuples = live as f64;
        (*stats).tuples_removed += dead_count as f64;
        return stats;
    }

    // Pass 2 (FLAT path only): swap-remove from the back. dead_slots is built in
    // ascending order (we walked slot 0..n); reverse-iterate to
    // process highest-index dead first. This invariant lets us
    // copy from `alive - 1` (which is always either an unmoved
    // original live row, or a row that was moved into a position
    // still > current dead slot — either way the data is the
    // canonical live data we want to preserve).
    let mut alive: u64 = meta.n_vectors;
    for &dead_slot in dead_slots.iter().rev() {
        let s = dead_slot as u64;
        let last = alive - 1;
        if s != last {
            // Codes chain.
            relfile::copy_slot_in_chain(
                index,
                meta.codes_first,
                meta.stride_bytes,
                meta.rows_per_codes_page,
                last,
                s,
            );
            // Scales chain.
            relfile::copy_slot_in_chain(
                index,
                meta.scales_first,
                std::mem::size_of::<f32>() as u32,
                meta.rows_per_scales_page,
                last,
                s,
            );
            // Ids chain. Updating the on-disk id at slot `s`
            // matters for the next ambulkdelete pass: the
            // callback walks ids chain -> ItemPointer -> heap
            // tuple lookup.
            relfile::copy_slot_in_chain(
                index,
                meta.ids_first,
                std::mem::size_of::<u64>() as u32,
                meta.rows_per_ids_page,
                last,
                s,
            );
        }
        alive -= 1;
    }
    debug_assert_eq!(alive, meta.n_vectors - dead_count as u64);

    // Pass 3: persist the new n_vectors via the meta page. The
    // chain layout (codes_first / scales_first / ids_first /
    // rows_per_*_page / stride_bytes) is preserved so the swap-
    // removed slots remain at their on-disk positions for
    // subsequent reads. Bump am_version so the per-backend cache
    // (cache.rs) re-loads next scan.
    let new_n = alive;
    let next_version = meta.am_version.saturating_add(1);
    relfile::write_meta_shrink_in_place(index, &meta, new_n, next_version);

    // Pass 4: release trailing ids-chain pages that the shrink
    // left orphaned. RelationTruncate emits XLOG_SMGR_TRUNCATE
    // and is crash-safe.
    let shrunk_meta = MetaPageData {
        n_vectors: new_n,
        ..meta
    };
    relfile::truncate_ids_tail(index, &shrunk_meta);

    (*stats).num_index_tuples = new_n as f64;
    (*stats).tuples_removed += dead_count as f64;
    stats
}

/// IVF tombstone path (Phase E-2). Reads any existing tombstone
/// bitmap, ORs in the `dead_slots` (slot indices into the
/// cell-contiguous chains), and persists the merged bitmap via
/// [`relfile::write_tombstones_and_meta`] — without moving a single
/// row or touching `n_vectors`, the cell directory, or the coarse
/// centroids. The index stays IVF across the VACUUM.
///
/// Returns the TOTAL number of tombstoned (dead) slots after the
/// merge, so the caller can report the live count
/// (`n_vectors - total_dead`) to the planner.
///
/// # Safety
///
/// Caller holds an exclusive relation lock (VACUUM does).
unsafe fn ivf_tombstone_dead(
    index: pg_sys::Relation,
    meta: &MetaPageData,
    dead_slots: &[usize],
    new_am_version: u32,
) -> u64 {
    // One bit per slot, LSB-first. ceil(n_vectors / 8) bytes.
    let n_bytes = (meta.n_vectors as usize).div_ceil(8);
    let mut bitmap = relfile::read_tombstones(index, meta);
    if bitmap.len() < n_bytes {
        // First vacuum (empty) or a grown corpus: size up, preserving
        // any bits already set.
        bitmap.resize(n_bytes, 0u8);
    }
    for &slot in dead_slots {
        if slot >= meta.n_vectors as usize {
            continue; // defensive: ignore out-of-range slot
        }
        bitmap[slot / 8] |= 1u8 << (slot % 8);
    }
    let total_dead: u64 = bitmap.iter().map(|b| b.count_ones() as u64).sum();
    relfile::write_tombstones_and_meta(index, meta, &bitmap, new_am_version);
    total_dead
}

/// Graph tombstone path (Phase G-2b). Identical shape to
/// [`ivf_tombstone_dead`] — reads any existing tombstone bitmap, ORs
/// in the newly `dead_slots`, and persists the merged bitmap via
/// [`relfile::write_tombstones_and_meta`] without moving a row or
/// touching the adjacency chain.
///
/// One graph-specific wrinkle: if the tombstoned set includes the
/// current `graph_entry_point` (the slot `graph_search` starts every
/// traversal from), a dead entry point would leave the scan path
/// starting from a node it must never return or expand through. We
/// pick a fallback here — the first still-live slot (ascending slot
/// order) — and stamp it onto the meta page in the SAME write as the
/// tombstone bitmap, so there is never a moment where a persisted
/// meta page points at a dead entry point. If every slot is dead the
/// entry point is left at 0 (meaningless but harmless: any query
/// against a fully-dead corpus is degenerate regardless).
///
/// Returns the TOTAL number of tombstoned (dead) slots after the
/// merge.
///
/// # Safety
///
/// Caller holds an exclusive relation lock (VACUUM does).
unsafe fn graph_tombstone_dead(
    index: pg_sys::Relation,
    meta: &MetaPageData,
    dead_slots: &[usize],
    new_am_version: u32,
) -> u64 {
    let n_bytes = (meta.n_vectors as usize).div_ceil(8);
    let mut bitmap = relfile::read_tombstones(index, meta);
    if bitmap.len() < n_bytes {
        bitmap.resize(n_bytes, 0u8);
    }
    for &slot in dead_slots {
        if slot >= meta.n_vectors as usize {
            continue;
        }
        bitmap[slot / 8] |= 1u8 << (slot % 8);
    }
    let total_dead: u64 = bitmap.iter().map(|b| b.count_ones() as u64).sum();

    let is_dead = |slot: usize| -> bool {
        let byte = slot / 8;
        byte < bitmap.len() && (bitmap[byte] >> (slot % 8)) & 1 != 0
    };
    // Robustness fix for a real degenerate case in the entry-point
    // fallback: checking ONLY "is the entry point itself dead" is
    // not enough. `graph_search` starts its FIRST hop by expanding
    // the entry point's out-edges; if the entry point survives but
    // EVERY one of its neighbors is now tombstoned, that first
    // expansion yields zero live candidates and the search
    // terminates immediately, returning (at best) just the entry
    // point itself. Treat "entry point has no live out-neighbor" as
    // equally disqualifying as "entry point itself is dead".
    // (Discovered while chasing a separate test-data-generation bug
    // -- an uncorrelated `random()` subquery that made every test
    // row identical -- but this guard stands on its own: a
    // low-degree entry point whose only edges get tombstoned is a
    // genuine dead-end regardless of the corpus.)
    let adjacency = relfile::read_graph_adjacency(index, meta);
    let entry_has_live_neighbor = adjacency.as_ref().is_some_and(|adj| {
        let ep = meta.graph_entry_point as usize;
        ep < adj.n() && adj.neighbors_of(ep).iter().any(|&nb| !is_dead(nb as usize))
    });
    let entry_needs_fallback = is_dead(meta.graph_entry_point as usize) || !entry_has_live_neighbor;
    let entry_meta = if entry_needs_fallback {
        // Fallback candidate: the first live slot that ALSO has at
        // least one live out-neighbor (so the new entry point can
        // actually expand on its first hop), falling back further to
        // just "the first live slot" if no such node exists (a
        // heavily-fragmented graph -- still better than a
        // provably-dead-end entry point, and `graph_search` itself
        // degrades gracefully to "just the entry point" rather than
        // panicking either way).
        let fallback = adjacency
            .as_ref()
            .and_then(|adj| {
                (0..meta.n_vectors as usize).find(|&s| {
                    !is_dead(s) && adj.neighbors_of(s).iter().any(|&nb| !is_dead(nb as usize))
                })
            })
            .or_else(|| (0..meta.n_vectors as usize).find(|&s| !is_dead(s)))
            .unwrap_or(0) as u32;
        MetaPageData {
            graph_entry_point: fallback,
            ..*meta
        }
    } else {
        *meta
    };
    relfile::write_tombstones_and_meta(index, &entry_meta, &bitmap, new_am_version);
    total_dead
}

/// `amvacuumcleanup`: nothing to do beyond what `ambulkdelete`
/// already wrote.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    stats
}
