//! `ambulkdelete` / `amvacuumcleanup`.
//!
//! `ambulkdelete` walks every live id in the index and asks the
//! provided callback whether the underlying heap tuple is dead. If
//! it is, we drop the id from the IdMapIndex and from the
//! `live_ids` side-list. The whole thing is persisted at the end of
//! the call.
//!
//! v0.15: actual removal (was a no-op stub in v0.4..v0.14).

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::persist;
#[cfg(feature = "relfile_storage")]
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

    #[cfg(feature = "relfile_storage")]
    {
        ambulkdelete_relfile((*info).index, stats, callback, callback_state)
    }
    #[cfg(not(feature = "relfile_storage"))]
    ambulkdelete_sidetable((*(*info).index).rd_id, stats, callback, callback_state)
}

#[cfg(feature = "relfile_storage")]
unsafe fn ambulkdelete_relfile(
    index: pg_sys::Relation,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    use turbovec::IdMapIndex;

    let indexrelid = (*index).rd_id;
    let meta = match relfile::read_meta(index) {
        Some(m) if m.n_vectors > 0 => m,
        _ => return stats,
    };

    let (codes, scales, ids) = relfile::read_full(index, &meta);
    let mut idx = match IdMapIndex::from_id_map_parts(
        meta.bit_width as usize,
        meta.dim as usize,
        meta.n_vectors as usize,
        codes,
        scales,
        ids,
    ) {
        Ok(i) => i,
        Err(e) => error!("turbovec ambulkdelete: corrupt relfile: {}", e),
    };

    let Some(cb) = callback else {
        (*stats).num_index_tuples = meta.n_vectors as f64;
        return stats;
    };

    // Snapshot the live ids before we start mutating idx (remove()
    // does swap-remove, so iterating idx.slot_to_id() during
    // mutation would skip / repeat slots).
    let live_ids: Vec<u64> = idx.slot_to_id().to_vec();
    let mut dead_count: i64 = 0;
    for id in &live_ids {
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::u64_to_item_pointer(*id, &mut tid);
        let is_dead = (cb)(&mut tid as *mut _, callback_state);
        if is_dead {
            idx.remove(*id);
            dead_count += 1;
        }
    }

    let n_after = idx.len() as u64;
    let next_version = meta.am_version.saturating_add(1);
    // `relfile::write_full` calls `RelationTruncate` itself when
    // the new layout is smaller than the old one (Phase L
    // hardening item 3), so a VACUUM that consolidates dead rows
    // actually shrinks the on-disk file instead of leaving orphan
    // trailing pages.
    relfile::write_full(
        index,
        meta.bit_width,
        meta.dim,
        n_after,
        idx.packed_codes(),
        idx.scales(),
        idx.slot_to_id(),
        next_version,
    );
    persist::save_empty_with_count(
        indexrelid,
        meta.bit_width as i32,
        meta.dim as i32,
        n_after as i64,
    );

    (*stats).num_index_tuples = n_after as f64;
    (*stats).tuples_removed += dead_count as f64;
    stats
}

#[cfg(not(feature = "relfile_storage"))]
unsafe fn ambulkdelete_sidetable(
    indexrelid: pg_sys::Oid,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let Some(mut state) = persist::load(indexrelid) else {
        return stats;
    };

    let Some(cb) = callback else {
        // No callback supplied (cleanup pass without dead
        // tuples) — nothing to remove. Still persist a fresh
        // updated_at so VACUUM stats reflect the visit.
        persist::save(
            indexrelid,
            state.bit_width,
            state.dim,
            state.n_vectors,
            &state.index,
            state.version,
            &state.live_ids,
        );
        (*stats).num_index_tuples = state.n_vectors as f64;
        return stats;
    };

    let mut dead_count: i64 = 0;
    let mut survivors: Vec<u64> = Vec::with_capacity(state.live_ids.len());

    for id in &state.live_ids {
        let mut tid = pg_sys::ItemPointerData::default();
        pgrx::itemptr::u64_to_item_pointer(*id, &mut tid);
        let is_dead = (cb)(&mut tid as *mut _, callback_state);
        if is_dead {
            state.index.remove(*id);
            dead_count += 1;
        } else {
            survivors.push(*id);
        }
    }

    state.live_ids = survivors;
    state.n_vectors = state.live_ids.len() as i64;
    state.version += 1;

    persist::save(
        indexrelid,
        state.bit_width,
        state.dim,
        state.n_vectors,
        &state.index,
        state.version,
        &state.live_ids,
    );

    (*stats).num_index_tuples = state.n_vectors as f64;
    (*stats).tuples_removed += dead_count as f64;
    stats
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
