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
                persist::save(
                    d.key.rel_oid,
                    d.persist.bit_width,
                    d.persist.dim,
                    d.persist.n_vectors,
                    &*guard,
                    d.persist.version,
                    &d.persist.live_ids,
                );
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
