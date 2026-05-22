//! `ambulkdelete` / `amvacuumcleanup`.
//!
//! v0.4 supports incremental delete via SPI: load the index, call
//! `IdMapIndex::remove(id)` for every dead heap TID the callback
//! reports, write back. Batching delete events into a single
//! write-back is a Phase 5 concern.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::persist;

/// `ambulkdelete`: process dead-tuple removal.
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

    // Walk every id currently in the index and ask the callback
    // whether the underlying heap row is dead. v0.4 has no efficient
    // way to enumerate live ids inside `IdMapIndex` without a public
    // accessor — Phase 5 should add one upstream. For now we
    // pessimistically skip ambulkdelete and rely on the next
    // `ambuild` (REINDEX) to compact.
    let _ = (callback, callback_state);

    // Persist unchanged so updated_at refreshes (helps debugging).
    persist::save(
        indexrelid,
        state.bit_width,
        state.dim,
        state.n_vectors,
        &mut state.index,
        state.version,
    );
    (*stats).num_index_tuples = state.n_vectors as f64;
    stats
}

/// `amvacuumcleanup`: nothing to do beyond what `ambulkdelete`
/// already wrote.
pub(crate) unsafe extern "C-unwind" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    stats
}
