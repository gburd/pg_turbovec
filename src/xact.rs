//! Transaction-callback registration for the deferred-commit
//! `aminsert` path.
//!
//! See `src/index/insert.rs` for the strategy: mutate cached
//! `IdMapIndex` under `RwLock`, mark dirty, defer the relfile
//! rewrite to `PreCommit`. This module owns the once-per-
//! transaction callback wiring.

use std::cell::Cell;

use pgrx::callbacks::{
    PgSubXactCallbackEvent, PgXactCallbackEvent, register_subxact_callback, register_xact_callback,
};
use pgrx::pg_sys;

use crate::cache;

/// Reconcile the persisted-count mirror (`PersistState.n_vectors`)
/// against the authoritative row count (`slot_to_id.len()`) and
/// return the count that MUST be written to disk.
///
/// `slot_to_id` is the single source of truth: it's the array whose
/// bytes actually land in the ids chain, so the meta page's
/// `n_vectors` must equal its length or `read_full` over/under-reads
/// the chain on reload. The pre-1.28.4 bug was passing
/// `PersistState.n_vectors` (a SEPARATELY-incremented counter) as the
/// persisted count; if the two ever drifted (`IdAlreadyPresent`
/// remove+re-add, a mark_dirty closure that didn't run, a future
/// insert-path refactor), a meta page claiming MORE rows than the ids
/// chain held would reload as "id 0 in more than one slot" (the
/// reported corruption — CTIDs never encode to 0, so an id-0 slot is
/// always zeroed trailing bytes).
///
/// Returns `Ok(count)` when the mirror agrees with the authoritative
/// count (the count == `slot_to_id.len()`), or
/// `Err((mirror, authoritative))` when they've drifted so the caller
/// can abort the transaction LOUDLY instead of persisting a corrupt
/// relfile. Factored out as a pure function so the drift guard is
/// unit-testable without a live cluster.
pub(crate) fn reconciled_row_count(
    mirror_n_vectors: i64,
    slot_to_id_len: usize,
) -> Result<u64, (i64, usize)> {
    if mirror_n_vectors < 0 || mirror_n_vectors as u64 != slot_to_id_len as u64 {
        return Err((mirror_n_vectors, slot_to_id_len));
    }
    Ok(slot_to_id_len as u64)
}

#[cfg(test)]
mod reconcile_tests {
    use super::reconciled_row_count;

    // Load-INDEPENDENT drift-guard test (runs under plain
    // `cargo test --lib`, no PostgreSQL cluster needed). This is the
    // regression gate for the v1.28.4 dual-counter fix (issue B).

    #[test]
    fn agreeing_counts_return_slot_len() {
        assert_eq!(reconciled_row_count(0, 0), Ok(0));
        assert_eq!(reconciled_row_count(128, 128), Ok(128));
        assert_eq!(reconciled_row_count(1_740_000, 1_740_000), Ok(1_740_000));
    }

    #[test]
    fn mirror_over_counting_is_rejected() {
        // The exact pre-fix drift: the mirror claims MORE rows than
        // the ids array holds. Persisting `129` while `slot_to_id`
        // has 128 ids would leave the 129th id-chain slot reading
        // zeroed bytes => "id 0 in more than one slot" on reload.
        // The guard MUST reject it, not silently persist 129.
        assert_eq!(reconciled_row_count(129, 128), Err((129, 128)));
    }

    #[test]
    fn mirror_under_counting_is_rejected() {
        // The reverse drift (mirror lags the array) would truncate
        // the ids chain on write — also corruption. Reject it too.
        assert_eq!(reconciled_row_count(127, 128), Err((127, 128)));
    }

    #[test]
    fn authoritative_count_wins_when_they_agree() {
        // When mirror == slot_to_id.len(), the RETURNED count is the
        // slot length (the source of truth), never something derived
        // from the mirror — so a future refactor that stops updating
        // the mirror can't change what lands on disk as long as it
        // keeps the guard.
        let slot_len = 42usize;
        assert_eq!(reconciled_row_count(42, slot_len), Ok(slot_len as u64));
    }
}

/// PreCommit flush sink: re-opens the index relation by oid (the
/// original `Relation` from aminsert was dropped at end of the
/// executor's tuple loop), writes the cached `IdMapIndex` out as
/// relfile pages, then closes. WAL-logged via the `GenericXLog`
/// path inside `relfile::write_full_with_prepared`.
/// C-NEW-1: pure (no-I/O) validation of one dirty entry's in-memory
/// snapshot, run over EVERY dirty index in the PreCommit loop BEFORE
/// any relfile is physically written. Trips exactly the conditions a
/// well-formed transaction could hit (row-count drift; an id-0 or
/// duplicate external id in the in-memory `slot_to_id` for the
/// bijective flat/single kind). Raising here aborts the whole txn
/// with zero page writes, so a multi-index commit can never leave one
/// index physically flushed while a sibling's flush fails. The
/// on-disk-dup guard inside `reconcile_and_write_flush` is retained
/// separately (it detects pre-existing on-disk corruption, which this
/// pure pass cannot see).
fn validate_flush_snapshot(
    _indexrelid: pg_sys::Oid,
    idx: &turbovec::IdMapIndex,
    state: &cache::PersistState,
) {
    reconciled_row_count(state.n_vectors, idx.slot_to_id().len()).unwrap_or_else(
        |(mirror, authoritative)| {
            pgrx::error!(
                "turbovec persist (pre-flush validation): row-count drift \
                 (PersistState.n_vectors = {mirror}, slot_to_id.len() = {authoritative}); \
                 aborting BEFORE any index is written to avoid a corrupt relfile."
            )
        },
    );
    // In-memory bijection check. The deferred-flush path (flat AND
    // IVF `aminsert`) reloads the whole index into a FLAT `IdMapIndex`
    // in memory (an IVF insert degrades the index to flat — see
    // `index_is_degraded`), so the in-memory `slot_to_id` is bijective
    // for every kind that reaches here (graph uses a separate
    // `insert_graph_row` path that never marks this cache dirty). A
    // duplicate or id-0 in this in-memory table is therefore always a
    // real invariant violation, safe to check unconditionally.
    if let Some(dup) = crate::index::scan::first_duplicate_id(idx.slot_to_id()) {
        pgrx::error!(
            "turbovec persist (pre-flush validation): in-memory id table has id {dup} \
             in more than one slot (id 0 = a zeroed/uninitialised slot); aborting BEFORE \
             any index is written. This is an internal invariant violation."
        );
    }
}

unsafe fn flush_to_relfile(
    indexrelid: pg_sys::Oid,
    idx: &turbovec::IdMapIndex,
    state: &cache::PersistState,
) {
    // RowExclusiveLock is sufficient — VACUUM holds
    // ShareUpdateExclusiveLock, REINDEX holds AccessExclusiveLock,
    // and our writer must NOT block readers.
    let rel = pg_sys::index_open(indexrelid, pg_sys::RowExclusiveLock as i32);
    if rel.is_null() {
        // Index was dropped between the aminsert and the PreCommit
        // (e.g. user did INSERT then DROP INDEX in the same tx).
        // Bail silently; the heap rows aren't indexed but that's
        // already the user's stated intent.
        return;
    }
    // v1.29.1 corruption fix #2 (deferred-flush lost-update):
    // RECONCILE this transaction's upserts onto the CURRENT on-disk
    // state instead of blindly rewriting the whole relfile from this
    // backend's stale in-memory snapshot. `write_full_with_prepared`
    // used to overwrite from `idx` wholesale; if a concurrent VACUUM
    // shrank the relfile since `idx` was loaded, that overwrite
    // resurrected the deleted rows (the id-0 / duplicate-id source).
    //
    // `reconcile_and_write_flush` takes the exclusive rewrite lock,
    // re-reads the current on-disk (codes, scales, ids), splices in
    // ONLY the ids this txn touched (`state.touched_ids`, sourced
    // from `idx`'s own code/scale/id arrays), and writes the merged
    // image — preserving VACUUM's deletes and other backends'
    // committed inserts. The drift guard is retained purely as a
    // cross-check that `idx`'s own arrays are internally consistent.
    let _n_vectors = reconciled_row_count(state.n_vectors, idx.slot_to_id().len()).unwrap_or_else(
        |(mirror, authoritative)| {
            pgrx::error!(
                "turbovec persist: row-count drift detected \
                 (PersistState.n_vectors = {mirror}, slot_to_id.len() = {authoritative}); \
                 aborting to avoid writing a corrupt relfile. This is an \
                 internal invariant violation; please report it."
            )
        },
    );
    idx.prepare();
    crate::index::relfile::reconcile_and_write_flush(
        rel,
        state.bit_width as u8,
        state.dim as u32,
        idx.packed_codes(),
        idx.scales(),
        idx.slot_to_id(),
        &state.touched_ids,
        state.version as u32,
        crate::index::relfile::PreparedParts {
            // wire v8: persist only the per-index TQ+ pair (empty =
            // identity today); codebook + rotation derived at open.
            tqplus_shift: idx.tqplus_shift(),
            tqplus_scale: idx.tqplus_scale(),
        },
    );
    pg_sys::index_close(rel, pg_sys::RowExclusiveLock as i32);
}

/// Test-only handle to drive [`flush_to_relfile`] within a
/// `#[pg_test]` transaction (the write sticks via GenericXLog, so
/// the caller can immediately re-read the relfile it just wrote).
/// Lets the v1.28.4 corruption reproduction test flush a
/// deliberately-drifted `PersistState` and prove the guard aborts
/// BEFORE a corrupt relfile lands — something the normal deferred
/// aminsert path can't demonstrate in-band because a `#[pg_test]`'s
/// outer transaction always rolls back before PreCommit fires.
///
/// # Safety
/// Caller holds no conflicting lock on `indexrelid`; the index must
/// exist. Same contract as [`flush_to_relfile`].
#[cfg(any(test, feature = "pg_test"))]
pub(crate) unsafe fn flush_to_relfile_for_test(
    indexrelid: pg_sys::Oid,
    idx: &turbovec::IdMapIndex,
    state: &cache::PersistState,
) {
    flush_to_relfile(indexrelid, idx, state);
}

thread_local! {
    /// Tracks whether the `PreCommit` / `Abort` xact callbacks have
    /// already been registered for the current top-level transaction.
    /// pgrx clears its registered callbacks on transaction end, so
    /// this flag must be cleared in lockstep — both callbacks set it
    /// to `false` themselves so the next transaction re-registers.
    static XACT_CB_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

/// Register `PreCommit` (deferred persist) and `Abort` (cache
/// invalidation) hooks exactly once per transaction in this
/// backend. Subsequent calls within the same transaction are
/// no-ops. Idempotent across REPEATABLE READ and READ COMMITTED.
pub(crate) fn ensure_xact_callbacks_registered() {
    XACT_CB_REGISTERED.with(|reg| {
        if reg.get() {
            return;
        }
        reg.set(true);

        // PreCommit: drain dirty entries and persist each one. We
        // intentionally use `PreCommit` (not `Commit`) so the
        // relfile rewrite lands in the user's transaction — that
        // buys us WAL correctness for free and lets `ereport(ERROR)`
        // cleanly roll the user's transaction back if persistence
        // fails. The matching `Abort` callback below then evicts
        // the still-dirty entries.
        register_xact_callback(PgXactCallbackEvent::PreCommit, || {
            XACT_CB_REGISTERED.with(|r| r.set(false));
            let dirty = cache::drain_dirty();
            if dirty.is_empty() {
                return;
            }
            // PreCommit fires after the executor has popped the
            // active snapshot. The relfile path uses raw buffer-
            // manager calls (no SPI) so we don't need to push a
            // snapshot here, but pushing one is harmless and keeps
            // the hook compatible with any future SPI work that
            // might land inside `flush_to_relfile`.
            unsafe {
                pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
            }
            // C-NEW-1 fix (multi-index partial-flush corruption): a
            // guard `ereport(ERROR)` while flushing the SECOND dirty
            // index would longjmp AFTER the FIRST index's relfile pages
            // were already physically written (GenericXLog is not rolled
            // back on abort), leaving index A with phantom CTIDs for
            // never-committed rows and index B missing rows. So do a
            // PURE validation pass over ALL dirty snapshots FIRST (no
            // I/O, no page writes): the row-count drift + in-memory
            // id-0/duplicate bijection are exactly the conditions a
            // well-formed transaction could trip. If any entry is bad,
            // abort here with ZERO physical writes done, so no index is
            // left half-flushed. (The on-disk-dup guard inside
            // reconcile_and_write_flush stays too, but it fires on
            // pre-existing corruption, not on this txn's snapshot.)
            for d in &dirty {
                let guard = d.index.read();
                validate_flush_snapshot(d.key.rel_oid, &guard, &d.persist);
                drop(guard);
            }
            for d in &dirty {
                let guard = d.index.read();
                unsafe {
                    flush_to_relfile(d.key.rel_oid, &*guard, &d.persist);
                }
                drop(guard);
                cache::clear_dirty(d.key);
            }
            unsafe {
                pg_sys::PopActiveSnapshot();
            }
        });

        // Abort: invalidate every dirty entry so the next access in
        // this backend reloads committed state from the relfile
        // pages. We don't journal undo — clone-on-write would have
        // made rollback cheap but the per-insert clone cost on
        // hundred-MiB indexes was unacceptable, so we trade a
        // post-rollback reload for a fast hot path.
        register_xact_callback(PgXactCallbackEvent::Abort, || {
            XACT_CB_REGISTERED.with(|r| r.set(false));
            cache::invalidate_dirty();
        });

        // M-NEW-4: SAVEPOINT / subtransaction rollback. aminsert mutates
        // the cached in-memory IdMapIndex + touched_ids + dirty flag
        // eagerly; a `ROLLBACK TO SAVEPOINT` undoes the heap rows but
        // nothing here, so without this the rolled-back rows would be
        // spliced onto disk at top-level PreCommit (index bloat + silent
        // recall loss). We don't journal per-subxact undo (the
        // per-insert clone cost is unacceptable, same rationale as the
        // Abort path), so on ANY subxact abort we conservatively
        // invalidate the whole dirty set — the next access in this
        // backend reloads committed state from the relfile. Correct
        // (never persists a rolled-back row) at the cost of re-reading
        // the index after a savepoint rollback, which is rare on the
        // bulk-insert hot path.
        register_subxact_callback(PgSubXactCallbackEvent::AbortSub, |_my, _parent| {
            cache::invalidate_dirty();
        });

        // Parallel-worker and 2PC paths fall through unhandled
        // (`amcanparallel = false` already prevents the former;
        // PREPARE TRANSACTION is rare for OLTP-style bulk-insert
        // workloads). Documented as a follow-up.
    });
}
