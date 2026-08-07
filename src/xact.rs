//! Transaction-callback registration for the deferred-commit
//! `aminsert` path.
//!
//! See `src/index/insert.rs` for the strategy: mutate cached
//! `IdMapIndex` under `RwLock`, mark dirty, defer the relfile
//! rewrite to `PreCommit`. This module owns the once-per-
//! transaction callback wiring.

use std::cell::Cell;

use pgrx::callbacks::{PgXactCallbackEvent, register_xact_callback};
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
    // SINGLE SOURCE OF TRUTH for the persisted row count: the
    // in-memory `IdMapIndex`'s own `slot_to_id` length, NOT the
    // separately-incremented `PersistState.n_vectors`. See
    // [`reconciled_row_count`] for the full rationale. This aborts
    // the transaction (the matching Abort callback then evicts the
    // dirty cache entry) rather than silently persisting a
    // mismatched meta page.
    let n_vectors = reconciled_row_count(state.n_vectors, idx.slot_to_id().len()).unwrap_or_else(
        |(mirror, authoritative)| {
            pgrx::error!(
                "turbovec persist: row-count drift detected \
                 (PersistState.n_vectors = {mirror}, slot_to_id.len() = {authoritative}); \
                 aborting to avoid writing a corrupt relfile. This is an \
                 internal invariant violation; please report it."
            )
        },
    );
    crate::index::relfile::write_full_with_prepared(
        rel,
        state.bit_width as u8,
        state.dim as u32,
        n_vectors,
        idx.packed_codes(),
        idx.scales(),
        idx.slot_to_id(),
        state.version as u32,
        {
            // Pre-bake the codebook + rotation so backends opening
            // the post-commit relfile don't pay the per-backend
            // ~5–8 s Lloyd-Max compute / QR. Phase P; mirrors the
            // ambuild path. Phase Q-0 (v7) no longer persists the
            // SIMD-blocked chain — it's recomputed from the packed
            // codes at index-open — so we don't hand it over here.
            idx.prepare_eager();
            let rotation = idx.rotation();
            crate::index::relfile::PreparedParts {
                centroids: idx.centroids(),
                boundaries: idx.boundaries(),
                rotation,
            }
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

        // Parallel-worker and 2PC paths fall through unhandled
        // (`amcanparallel = false` already prevents the former;
        // PREPARE TRANSACTION is rare for OLTP-style bulk-insert
        // workloads). Documented as a follow-up.
    });
}
