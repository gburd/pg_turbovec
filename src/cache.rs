//! Backend-local cache of materialised `turbovec::IdMapIndex`
//! instances, used by both `turbovec.knn()` and (when the
//! `experimental_index_am` feature is on) the index AM scan path.
//!
//! Cache keys are `(rel_oid, attnum_or_zero, bit_width, dim)`.
//! `attnum = 0` is reserved for the index AM path (the index relation
//! owns a single attribute and we don't disambiguate further);
//! positive values are heap attnums from `turbovec.knn()`.
//!
//! Invalidation is best-effort:
//! * Each entry stores the relation's `pg_class.relfilenode` and
//!   `count(*)` at load time. Relfile rewrites (CLUSTER, VACUUM
//!   FULL, REINDEX, TRUNCATE) bump the relfilenode, and ordinary
//!   DML changes the row count; either mismatch forces a rebuild
//!   on the next lookup.
//! * Total cache size capped at `turbovec.cache_size_mb`. When the
//!   cap is exceeded the LRU entry is evicted.
//!
//! This is intentionally simpler than registering a relcache
//! invalidation callback. We pay an extra `pg_class` lookup on each
//! cache hit in exchange for not having to manage callback
//! registration across `_PG_init` shutdowns.

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::guc;

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

struct Entry {
    /// Lazily-shared index. Multiple concurrent readers see the same
    /// `Arc`; only the loader pays the build cost. `IdMapIndex::search`
    /// is `&self`-thread-safe per the upstream crate's `OnceLock`
    /// design.
    index: Arc<IdMapIndex>,
    /// Approximate bytes the entry occupies. Used for the LRU cap.
    bytes: usize,
    /// `pg_class.relfilenode` snapshot. Zero means we didn't track it
    /// (treated as "always stale" so the next lookup rebuilds).
    relfilenode: u32,
    /// `count(*)` snapshot.
    n_rows: i64,
    /// Insertion order for LRU eviction. Higher = more recently used.
    seq: u64,
}

static CACHE: Lazy<Mutex<HashMap<CacheKey, Entry>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static SEQ: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(0));

fn next_seq() -> u64 {
    let mut s = SEQ.lock();
    *s += 1;
    *s
}

/// Look up the entry for `key`, validating it against the current
/// `(relfilenode, n_rows)`. On hit, returns the cached
/// `Arc<IdMapIndex>`. On miss the caller must call [`insert`] with a
/// freshly built index.
pub fn lookup(
    key: CacheKey,
    expected_relfile: u32,
    expected_n_rows: i64,
) -> Option<Arc<IdMapIndex>> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile || entry.n_rows != expected_n_rows {
        g.remove(&key);
        return None;
    }
    entry.seq = next_seq();
    Some(entry.index.clone())
}

/// Insert or replace the entry for `key`.
pub fn insert(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> Arc<IdMapIndex> {
    let arc = Arc::new(index);
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: arc.clone(),
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
        },
    );
    enforce_cap(&mut g);
    arc
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
        // GUC = 0 disables caching entirely.
        map.clear();
        return;
    }
    let cap = (cap_mb as usize).saturating_mul(1024 * 1024);
    let mut total: usize = map.values().map(|e| e.bytes).sum();
    while total > cap && map.len() > 1 {
        // Find LRU entry by lowest `seq`.
        let lru_key = map
            .iter()
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
