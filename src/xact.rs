//! Transaction-callback registration for the deferred-commit
//! `aminsert` path.
//!
//! See `src/index/insert.rs` for the strategy: mutate cached
//! `IdMapIndex` under `RwLock`, mark dirty, defer the
//! `persist::save` SPI to `PreCommit`. This module owns the
//! once-per-transaction callback wiring.

use std::cell::Cell;

use pgrx::callbacks::{register_xact_callback, PgXactCallbackEvent};
use pgrx::pg_sys;

use crate::cache;
use crate::index::persist;

/// PreCommit flush sink for the relfile path: re-opens the index
/// relation by oid (the original `Relation` from aminsert was
/// dropped at end of the executor's tuple loop), writes the cached
/// `IdMapIndex` out as relfile pages, then closes. WAL-logged via
/// the `GenericXLog` path inside `relfile::write_full`.
#[cfg(feature = "relfile_storage")]
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
    crate::index::relfile::write_full_with_prepared(
        rel,
        state.bit_width as u8,
        state.dim as u32,
        state.n_vectors as u64,
        idx.packed_codes(),
        idx.scales(),
        idx.slot_to_id(),
        state.version as u32,
        {
            // Pre-bake the SIMD-blocked layout and codebook so
            // backends opening the post-commit relfile in the
            // future don't pay the per-backend ~12–15 s
            // `pack::repack` and ~5–8 s Lloyd-Max compute.
            // Phase P; mirrors the ambuild path. Single-row
            // aminserts pay this on every commit — in the
            // existing rewrite-everything model that's an
            // acceptable cost (we already rewrite all chains
            // here), and the deferred-commit batching in
            // cache.rs amortises it across all the rows in one
            // transaction.
            idx.prepare_eager();
            crate::index::relfile::PreparedParts {
                blocked_codes: idx.blocked_codes(),
                n_blocks: idx.n_blocks() as u32,
                centroids: idx.centroids(),
                boundaries: idx.boundaries(),
            }
        },
    );
    // Mirror n_vectors into the side-table so `am_storage` queries
    // (used by `docs/RECALL.md` reproduction scripts and a couple
    // of legacy tests) keep working post-Phase-N-C.
    persist::save_empty_with_count(
        indexrelid,
        state.bit_width,
        state.dim,
        state.n_vectors,
    );
    pg_sys::index_close(rel, pg_sys::RowExclusiveLock as i32);
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
        // `persist::save` SPI lands in the user's transaction —
        // that buys us WAL correctness for free and lets
        // `ereport(ERROR)` cleanly roll the user's transaction back
        // if persistence fails. The matching `Abort` callback below
        // then evicts the still-dirty entries.
        register_xact_callback(PgXactCallbackEvent::PreCommit, || {
            XACT_CB_REGISTERED.with(|r| r.set(false));
            let dirty = cache::drain_dirty();
            if dirty.is_empty() {
                return;
            }
            // PreCommit fires after the executor has popped the
            // active snapshot, so SPI — which expects one — errors
            // out with "cannot execute SQL without an outer snapshot
            // or portal" unless we re-establish one. Push the
            // current transaction's snapshot for the duration of
            // the persist work and pop it before returning.
            unsafe {
                pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
            }
            for d in &dirty {
                let guard = d.index.read();
                #[cfg(feature = "relfile_storage")]
                {
                    unsafe { flush_to_relfile(d.key.rel_oid, &*guard, &d.persist); }
                }
                #[cfg(not(feature = "relfile_storage"))]
                {
                    persist::save(
                        d.key.rel_oid,
                        d.persist.bit_width,
                        d.persist.dim,
                        d.persist.n_vectors,
                        &*guard,
                        d.persist.version,
                        &d.persist.live_ids,
                    );
                }
                drop(guard);
                cache::clear_dirty(d.key);
            }
            unsafe {
                pg_sys::PopActiveSnapshot();
            }
        });

        // Abort: invalidate every dirty entry so the next access in
        // this backend reloads committed state from `am_storage`.
        // We don't journal undo — clone-on-write would have made
        // rollback cheap but the per-insert clone cost on
        // hundred-MiB indexes was unacceptable, so we trade a
        // post-rollback reload for a fast hot path.
        register_xact_callback(PgXactCallbackEvent::Abort, || {
            XACT_CB_REGISTERED.with(|r| r.set(false));
            cache::invalidate_dirty();
        });

        // Parallel-worker and 2PC paths fall through unhandled in
        // v1.1 (`amcanparallel = false` already prevents the
        // former; PREPARE TRANSACTION is rare for OLTP-style
        // bulk-insert workloads). Documented as a follow-up.
    });
}
