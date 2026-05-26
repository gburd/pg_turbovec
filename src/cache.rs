//! Backend-local cache of materialised `turbovec::IdMapIndex`
//! instances, used by both `turbovec.knn()` and the index AM
//! scan path.
//!
//! Cache keys are `(rel_oid, attnum_or_zero, bit_width, dim)`.
//! `attnum = 0` is reserved for the index AM path (the index relation
//! owns a single attribute and we don't disambiguate further);
//! positive values are heap attnums from `turbovec.knn()`.
//!
//! Invalidation is best-effort:
//! * Each entry stores the relation's `pg_class.relfilenode` and
//!   either `count(*)` (knn path) or the relfile meta page's
//!   `am_version` (AM path) at load time. Relfile rewrites
//!   (CLUSTER, VACUUM FULL, REINDEX, TRUNCATE) bump the
//!   relfilenode, and ordinary DML changes the row count / bumps
//!   the version; either mismatch forces a rebuild on the next
//!   lookup.
//! * Total cache size capped at `turbovec.cache_size_mb`. When the
//!   cap is exceeded the LRU entry is evicted.
//!
//! ## Mutation (AM path)
//!
//! `aminsert` mutates the cached `IdMapIndex` in place under a
//! `parking_lot::RwLock` write guard, then marks the entry dirty
//! and bumps a per-entry `PersistState` mirror that tracks the
//! relfile-meta-page fields (`bit_width`, `dim`, `n_vectors`,
//! `version`, `live_ids`). A transaction `PreCommit` callback
//! drains every dirty entry and runs a single relfile rewrite
//! per index, then clears the dirty flag and updates the
//! freshness slot to match the new on-disk `am_version`.
//!
//! Concurrency: PostgreSQL backends are single-threaded and our AM
//! advertises `amcanparallel = false`, so the RwLock never sees
//! contention in practice. The lock exists to satisfy `Send + Sync`
//! for the global cache and to keep the read/write paths obviously
//! correct should pgrx ever introduce in-process parallelism.
//!
//! Rollback: on `XACT_EVENT_ABORT` the dirty entries are evicted
//! from the cache so the next access reloads the committed state
//! from the relfile pages. We do not journal undo information.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use parking_lot::{Mutex, RwLock};
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::guc;
use crate::index::mmap_static::StaticRegionsMap;

/// Composite cache key. `attnum = 0` is reserved for the index AM
/// path; positive values are heap attribute numbers from the
/// function-driven path.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct CacheKey {
    pub rel_oid: pg_sys::Oid,
    pub attnum: i16,
    pub bit_width: u8,
    pub dim: u32,
}

/// Mutable mirror of relfile meta-page state alongside an AM-path
/// cache entry. Maintained by `aminsert` and flushed by the
/// `PreCommit` xact callback. `None` for the knn path (read-only
/// snapshots).
#[derive(Clone)]
pub struct PersistState {
    pub bit_width: i32,
    pub dim: i32,
    pub n_vectors: i64,
    pub version: i32,
    pub live_ids: Vec<u64>,
}

struct Entry {
    /// Lazily-shared, mutable index. Multiple concurrent readers see
    /// the same `Arc`; mutators take the write guard. Within a
    /// single backend the lock is uncontended (Postgres backends
    /// are single-threaded and we don't run inside parallel
    /// workers).
    index: Arc<RwLock<IdMapIndex>>,
    /// Phase R-3: optional `Mmap` of the relfile's static regions.
    /// `Some(_)` when the entry was installed via the mmap-load
    /// path; `None` when it came from a pure `read_full` fall-back
    /// (e.g. `mmap_static_blocked = off`, non-default tablespace,
    /// or the brief mid-REINDEX rename window where opening the
    /// relfile by path returns `ENOENT`). The `IdMapIndex` does
    /// not literally borrow into this map today (we copy the
    /// chains off it once at cache-fill time), but holding the
    /// `Mmap` here pins the lifetime contract for the
    /// `from_*_with_prepared_borrowed` constructor and lets a
    /// future zero-copy follow-up be additive without touching
    /// the cache machinery again. The `Mmap` is dropped (and the
    /// kernel mapping torn down) when the cache entry is.
    #[allow(dead_code)] // retained for Drop ordering / lifetime contract
    mmap: Option<StaticRegionsMap>,
    /// Approximate bytes the entry occupies. Used for the LRU cap.
    bytes: usize,
    /// `pg_class.relfilenode` snapshot. Zero means we didn't track it
    /// (treated as "always stale" so the next lookup rebuilds).
    relfilenode: u32,
    /// Freshness signal. For the knn path this is `count(*)`; for
    /// the AM path this is the relfile meta page's `am_version`
    /// at load time, advanced to `persist.version` after a
    /// successful commit-time persist.
    n_rows: i64,
    /// Insertion order for LRU eviction. Higher = more recently used.
    seq: u64,
    /// Set by `aminsert` once the in-memory index has been mutated
    /// past the persisted snapshot. Cleared by the `PreCommit` hook
    /// after the relfile rewrite succeeds, or by `invalidate_dirty`
    /// after `XACT_EVENT_ABORT`.
    dirty: bool,
    /// AM-path mirror of the relfile meta-page fields. `None` for
    /// entries installed by the read-only knn path.
    persist: Option<PersistState>,
}

static CACHE: LazyLock<Mutex<HashMap<CacheKey, Entry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SEQ: LazyLock<Mutex<u64>> = LazyLock::new(|| Mutex::new(0));

fn next_seq() -> u64 {
    let mut s = SEQ.lock();
    *s += 1;
    *s
}

/// Look up the entry for `key`, validating it against the current
/// `(relfilenode, freshness)`. On hit, returns the cached
/// `Arc<RwLock<IdMapIndex>>`. On miss the caller must call
/// [`insert`] (knn path) or [`am_install`] (AM path) with a freshly
/// built index.
pub fn lookup(
    key: CacheKey,
    expected_relfile: u32,
    expected_n_rows: i64,
) -> Option<Arc<RwLock<IdMapIndex>>> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile || entry.n_rows != expected_n_rows {
        // Don't evict if we have unflushed mutations — the on-disk
        // version is intentionally behind the in-memory state until
        // the xact commits. The mutating backend is the only one
        // that sees a stale-looking version while dirty.
        if entry.dirty {
            entry.seq = next_seq();
            return Some(entry.index.clone());
        }
        g.remove(&key);
        return None;
    }
    entry.seq = next_seq();
    Some(entry.index.clone())
}

/// AM-mutation lookup: returns the cached entry whenever the
/// `relfilenode` matches, regardless of the version freshness slot.
/// `aminsert` uses this so a bulk insert doesn't pay a meta-page
/// version read per row — the in-backend cache is the
/// authoritative copy for the duration of the transaction. The
/// scan path keeps using [`lookup`] so cross-session committed
/// inserts are visible to other backends.
///
/// Returns `None` when the entry is absent, lacks a persist mirror,
/// or when the relation has been rewritten (CLUSTER / VACUUM FULL
/// / REINDEX / TRUNCATE) since the entry was installed.
pub fn am_lookup_for_mutation(
    key: CacheKey,
    expected_relfile: u32,
) -> Option<Arc<RwLock<IdMapIndex>>> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile {
        if entry.dirty {
            // Dirty + relfile mismatch is impossible in practice
            // (we don't reindex our own index mid-aminsert), but be
            // conservative and keep the dirty entry rather than
            // silently dropping unflushed mutations.
            entry.seq = next_seq();
            return Some(entry.index.clone());
        }
        g.remove(&key);
        return None;
    }
    if entry.persist.is_none() {
        // The entry was installed by the read-only knn path and
        // lacks the persist mirror aminsert needs. Drop it so the
        // caller reloads via `am_install`.
        g.remove(&key);
        return None;
    }
    entry.seq = next_seq();
    Some(entry.index.clone())
}

/// AM-scan visibility lookup: find the dirty AM-path cache entry
/// for `rel_oid` with `attnum = 0`, regardless of `bit_width` or
/// `dim`. Used by the scan path when the relfile meta page is
/// the `(dim = 0, n_vectors = 0)` sentinel written by
/// `ambuildempty` — the in-memory mirror has the truthful
/// `(bit_width, dim, n_vectors, version)` tuple. Returns the cache
/// key and a snapshot of the persist-state mirror alongside the
/// shared index, so the caller can install a freshness signal that
/// matches what the next `aminsert` would see.
pub fn am_find_dirty_by_rel(
    rel_oid: pg_sys::Oid,
) -> Option<(CacheKey, Arc<RwLock<IdMapIndex>>, PersistState)> {
    let g = CACHE.lock();
    for (k, e) in g.iter() {
        if k.rel_oid == rel_oid && k.attnum == 0 && e.persist.is_some() {
            let p = e.persist.as_ref().unwrap().clone();
            return Some((*k, e.index.clone(), p));
        }
    }
    None
}

/// knn-path install: insert or replace the entry for `key` with no
/// persistence-state mirror attached. The cached index is treated
/// as read-only by the knn callers.
pub fn insert(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> Arc<RwLock<IdMapIndex>> {
    insert_with_mmap(key, index, bytes, relfilenode, n_rows, None)
}

/// Phase R-3 variant of [`insert`] that also takes an optional
/// [`StaticRegionsMap`]. The `Mmap` is colocated on the cache
/// entry so it lives for at least as long as the
/// `Arc<RwLock<IdMapIndex>>` returned to the caller. Drop
/// ordering on `Entry` is `Drop::drop(self)` -> drops `index`
/// (the `Arc`; if this was the last reference, the inner
/// `RwLock<IdMapIndex>` is freed) -> drops `mmap` (`munmap(2)`).
///
/// Crate-private because `StaticRegionsMap` is itself
/// `pub(crate)`; external callers use [`insert`].
pub(crate) fn insert_with_mmap(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
    mmap: Option<StaticRegionsMap>,
) -> Arc<RwLock<IdMapIndex>> {
    let arc = Arc::new(RwLock::new(index));
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: arc.clone(),
            mmap,
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
            dirty: false,
            persist: None,
        },
    );
    enforce_cap(&mut g);
    arc
}

/// AM-path install: insert or replace the entry for `key` and
/// attach the supplied `PersistState` mirror so subsequent
/// `aminsert` calls can mutate the in-memory index and defer the
/// relfile rewrite to commit time.
pub fn am_install(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    freshness: i64,
    persist: PersistState,
) -> Arc<RwLock<IdMapIndex>> {
    let arc = Arc::new(RwLock::new(index));
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: arc.clone(),
            mmap: None,
            bytes,
            relfilenode,
            n_rows: freshness,
            seq: next_seq(),
            dirty: false,
            persist: Some(persist),
        },
    );
    enforce_cap(&mut g);
    arc
}

/// Mutate the AM-path persist mirror under the cache mutex. Returns
/// the `CacheKey` if the entry exists and has a persist state,
/// otherwise `None` (caller must install a fresh entry).
///
/// The closure is invoked with `&mut PersistState` and is
/// responsible for advancing `n_vectors`, `version`, and
/// `live_ids`. The `dirty` flag is set after the closure returns.
pub fn am_mark_dirty<F: FnOnce(&mut PersistState)>(key: CacheKey, f: F) -> bool {
    let mut g = CACHE.lock();
    let Some(entry) = g.get_mut(&key) else {
        return false;
    };
    let Some(p) = entry.persist.as_mut() else {
        return false;
    };
    f(p);
    entry.dirty = true;
    true
}

/// Snapshot of a dirty AM-path entry that the `PreCommit` xact
/// callback can flush to the relfile main fork. We hand the caller
/// the `Arc<RwLock<IdMapIndex>>` so it can take a read guard for
/// the duration of the relfile rewrite without holding the cache
/// mutex.
pub struct DirtyEntry {
    pub key: CacheKey,
    pub index: Arc<RwLock<IdMapIndex>>,
    pub persist: PersistState,
}

/// Snapshot every currently-dirty AM-path entry. Does **not**
/// clear the dirty flag — call [`clear_dirty`] after each
/// relfile rewrite succeeds, so a panic mid-flush leaves the
/// remaining entries dirty for the matching `Abort` callback to
/// invalidate.
pub fn drain_dirty() -> Vec<DirtyEntry> {
    let g = CACHE.lock();
    let mut out = Vec::new();
    for (k, e) in g.iter() {
        if !e.dirty {
            continue;
        }
        let Some(p) = e.persist.as_ref() else {
            continue;
        };
        out.push(DirtyEntry {
            key: *k,
            index: e.index.clone(),
            persist: p.clone(),
        });
    }
    out
}

/// Mark `key`'s entry clean and advance its freshness slot to the
/// current `persist.version`, so subsequent in-backend lookups hit
/// without forcing another reload. Called after the relfile
/// rewrite succeeds.
pub fn clear_dirty(key: CacheKey) {
    let mut g = CACHE.lock();
    if let Some(entry) = g.get_mut(&key) {
        entry.dirty = false;
        if let Some(p) = entry.persist.as_ref() {
            entry.n_rows = p.version as i64;
        }
    }
}

/// Drop every dirty AM-path entry. Called from the `Abort` xact
/// callback so a rolled-back transaction cannot leave in-memory
/// mutations visible to the next scan in this backend.
pub fn invalidate_dirty() {
    let mut g = CACHE.lock();
    g.retain(|_, e| !e.dirty);
}

/// Drop every entry referencing `rel_oid`. Called from index/table
/// DROP paths; harmless to call unconditionally.
#[allow(dead_code)]
pub fn invalidate(rel_oid: pg_sys::Oid) {
    let mut g = CACHE.lock();
    g.retain(|k, _| k.rel_oid != rel_oid);
}

/// Drop the entire cache. Used by tests.
#[allow(dead_code)]
pub fn invalidate_all() {
    CACHE.lock().clear();
}

/// Number of cached entries. Test/diagnostic only.
#[allow(dead_code)]
pub fn len() -> usize {
    CACHE.lock().len()
}

fn enforce_cap(map: &mut HashMap<CacheKey, Entry>) {
    let cap_mb = guc::CACHE_SIZE_MB.get();
    if cap_mb <= 0 {
        // GUC = 0 disables caching entirely. Don't drop dirty
        // entries — flushing them is the PreCommit hook's job.
        map.retain(|_, e| e.dirty);
        return;
    }
    let cap = (cap_mb as usize).saturating_mul(1024 * 1024);
    let mut total: usize = map.values().map(|e| e.bytes).sum();
    while total > cap && map.len() > 1 {
        // Find LRU entry by lowest `seq`. Skip dirty entries — they
        // hold un-persisted mutations and can only be evicted via
        // the xact-end callbacks.
        let lru_key = map
            .iter()
            .filter(|(_, e)| !e.dirty)
            .min_by_key(|(_, e)| e.seq)
            .map(|(k, _)| *k);
        match lru_key {
            Some(k) => {
                if let Some(e) = map.remove(&k) {
                    total = total.saturating_sub(e.bytes);
                }
            }
            None => break,
        }
    }
}

/// Look up the relation's current `relfilenode` via `pg_class`.
/// Returns 0 on lookup failure (callers treat that as "unknown" — a
/// `0 != stored.relfilenode` comparison forces a rebuild).
pub fn current_relfilenode(rel_oid: pg_sys::Oid) -> u32 {
    let v: Option<i64> = Spi::get_one_with_args(
        "SELECT (relfilenode)::int8 FROM pg_class WHERE oid = $1",
        &[rel_oid.into()],
    )
    .ok()
    .flatten();
    v.unwrap_or(0) as u32
}

/// Pull the current relfilenode straight off the in-memory
/// `Relation` struct without an SPI roundtrip. The field name
/// changed between PG 15 and PG 16 (`rd_node` -> `rd_locator`,
/// `relNode` -> `relNumber`); both encode the same `Oid` /
/// `RelFileNumber` value as `u32`.
///
/// # Safety
///
/// Caller must pass a non-null `Relation` pointer that's pinned
/// in the relcache for the duration of the call (true for any
/// `Relation` Postgres hands an AM callback).
#[allow(dead_code)]
pub unsafe fn relfilenode_from_relation(rel: pg_sys::Relation) -> u32 {
    if rel.is_null() {
        return 0;
    }
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
    {
        // pg13/14/15: `rd_node.relNode` is an `Oid`.
        let oid: pg_sys::Oid = (*rel).rd_node.relNode;
        oid.to_u32()
    }
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    {
        // pg16+: `rd_locator.relNumber` is a `RelFileNumber`, which
        // is a typedef for `Oid`. Use `Oid::to_u32` for the
        // conversion — `as u32` doesn't work on the newtype.
        let oid: pg_sys::Oid = (*rel).rd_locator.relNumber;
        oid.to_u32()
    }
}
