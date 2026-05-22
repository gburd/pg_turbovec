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

    let indexrelid = (*(*info).index).rd_id;
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
