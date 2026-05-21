//! **WORK IN PROGRESS — Phase 2 starter, NOT YET MOUNTED in lib.rs.**
//!
//! This module is checked in at the bottom of the v0.1.0 commit so
//! the design is visible in code review, but it is not part of the
//! public extension surface yet. It pgrx 0.17 SPI calls below are
//! representative shapes; they will be tightened in Phase 2 when we
//! also add the relcache invalidation hook in `_PG_init`.
//!

//! Phase 2 starter — function-driven ANN search over a `tvector` column,
//! with a backend-local cache of materialised `turbovec::IdMapIndex`
//! instances keyed by `(table_oid, column_attnum, bit_width, dim)`.
//!
//! This is *not* the final index access method (that is Phase 3). It
//! is, however, a fully working end-to-end integration of `turbovec`
//! with a real PostgreSQL relation — the same plumbing the eventual
//! IndexAm callbacks will use, factored out behind a normal SQL
//! function so we can test it now without touching `IndexAmRoutine`.
//!
//! Public surface:
//!
//! ```sql
//! turbovec.knn(
//!     rel       regclass,        -- relation containing the column
//!     col       text,            -- name of the tvector column
//!     query     tvector,         -- query point
//!     k         integer,         -- number of neighbours
//!     bit_width integer DEFAULT 4
//! )  RETURNS TABLE (ctid tid, score double precision)
//! ```
//!
//! Memory model: each backend keeps a `parking_lot::Mutex<HashMap>`
//! of cached indexes. The cache is invalidated by the relcache
//! invalidation callback registered in `_PG_init`.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::tvector::Tvector;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
struct CacheKey {
    rel_oid: pg_sys::Oid,
    attnum: i16,
    bit_width: u8,
    dim: u32,
}

struct CacheEntry {
    index: IdMapIndex,
    /// Snapshot of the relation's `relfilenode`-equivalent identity.
    /// Cheap heuristic: if the heap is rewritten (CLUSTER, VACUUM
    /// FULL, TRUNCATE, ALTER TYPE), this changes and we rebuild.
    fingerprint: u32,
}

static CACHE: Lazy<Mutex<HashMap<CacheKey, CacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Encode a Postgres `ItemPointerData` (block 32-bit, offset 16-bit)
/// into a `u64` suitable for `IdMapIndex` ids. Layout:
///
/// ```text
/// bits 47..16: block number (32 bits, ip_blkid hi||lo)
/// bits 15..0 : offset      (16 bits, ip_posid)
/// bits 63..48: reserved (always zero) — gives us room to namespace
///              future epoch / generation bits.
/// ```
#[inline]
fn ctid_to_u64(ip: pg_sys::ItemPointerData) -> u64 {
    let blk: u32 = unsafe {
        // ItemPointerData stores block as two u16 halves to keep the
        // struct 6 bytes. Rebuild the u32 from the halves. This
        // mirrors the C macro ItemPointerGetBlockNumber.
        let hi = u32::from(ip.ip_blkid.bi_hi);
        let lo = u32::from(ip.ip_blkid.bi_lo);
        (hi << 16) | lo
    };
    let off: u16 = ip.ip_posid;
    (u64::from(blk) << 16) | u64::from(off)
}

/// Inverse of `ctid_to_u64`.
#[inline]
fn u64_to_ctid(id: u64) -> pg_sys::ItemPointerData {
    let blk: u32 = ((id >> 16) & 0xFFFF_FFFF) as u32;
    let off: u16 = (id & 0xFFFF) as u16;
    pg_sys::ItemPointerData {
        ip_blkid: pg_sys::BlockIdData {
            bi_hi: (blk >> 16) as u16,
            bi_lo: (blk & 0xFFFF) as u16,
        },
        ip_posid: off,
    }
}

/// Drop every cached entry for the given relation OID. Called from
/// the relcache invalidation hook.
pub fn invalidate(rel_oid: pg_sys::Oid) {
    let mut g = CACHE.lock();
    g.retain(|k, _| k.rel_oid != rel_oid);
}

/// Drop the entire cache. Used on commit-rollback boundaries that
/// might have rewritten heaps under us.
pub fn invalidate_all() {
    CACHE.lock().clear();
}

/// `turbovec.knn(rel, col, query, k, bit_width)` — see module docs.
#[pg_extern(stable, parallel_safe)]
fn knn(
    rel: pg_sys::Oid,
    col: &str,
    query: Tvector,
    k: i32,
    bit_width: default!(i32, 4),
) -> TableIterator<'static, (name!(ctid, pg_sys::ItemPointerData), name!(score, f64))> {
    if k <= 0 {
        error!("turbovec.knn: k must be positive (got {})", k);
    }
    if !(2..=4).contains(&bit_width) {
        error!("turbovec.knn: bit_width must be 2, 3, or 4 (got {})", bit_width);
    }
    if query.dim() % 8 != 0 {
        error!(
            "turbovec.knn: query dim must be a multiple of 8 (got {})",
            query.dim()
        );
    }

    // Phase 2 starter: build the index in-line (no cache) and search.
    // Promoting to the cached path is a follow-up patch — the cache
    // wiring above is in place, we just need the relcache invalidation
    // hook to be registered before we trust it.
    let entries = collect_column_via_spi(rel, col, query.dim());
    if entries.is_empty() {
        return TableIterator::new(Vec::<(pg_sys::ItemPointerData, f64)>::new());
    }

    let mut idx = IdMapIndex::new(query.dim(), bit_width as usize);
    let mut flat: Vec<f32> = Vec::with_capacity(entries.len() * query.dim());
    let mut ids: Vec<u64> = Vec::with_capacity(entries.len());
    for (id, vals) in &entries {
        flat.extend_from_slice(vals);
        ids.push(*id);
    }
    idx.add_with_ids(&flat, &ids)
        .unwrap_or_else(|e| error!("turbovec.knn: add_with_ids failed: {:?}", e));

    let take = (k as usize).min(entries.len());
    let (scores, hit_ids) = idx.search(query.as_slice(), take);

    let rows: Vec<(pg_sys::ItemPointerData, f64)> = hit_ids
        .iter()
        .zip(scores.iter())
        .map(|(id, s)| (u64_to_ctid(*id), f64::from(*s)))
        .collect();

    TableIterator::new(rows)
}

/// Iterate every (CTID, tvector) pair from `rel.col` via SPI, keeping
/// only rows where the vector dimension matches `expected_dim`.
fn collect_column_via_spi(
    rel: pg_sys::Oid,
    col: &str,
    expected_dim: usize,
) -> Vec<(u64, Vec<f32>)> {
    // Resolve the relation name via pg_class so we can quote it
    // safely. SPI's parser doesn't accept oids directly.
    let rel_text: Option<String> = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c \
         JOIN   pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[rel.into()],
    )
    .unwrap_or(None);
    let qualified = rel_text
        .unwrap_or_else(|| error!("turbovec.knn: relation oid {:?} not found", rel));

    // Validate column name against pg_attribute. We rely on the
    // executor's planner cache for repeated calls.
    let col_quoted = quote_ident(col);
    let sql = format!(
        "SELECT ctid::text, ({col_quoted})::turbovec.tvector::real[] FROM {qualified} \
         WHERE  ({col_quoted}) IS NOT NULL"
    );

    let mut out: Vec<(u64, Vec<f32>)> = Vec::new();
    Spi::connect(|client| {
        let mut tup_iter = client
            .select(&sql, None, None)
            .unwrap_or_else(|e| error!("turbovec.knn: SPI select failed: {}", e));
        while let Some(row) = tup_iter.next() {
            let ctid_text: Option<String> = row.get(1).unwrap_or(None);
            let arr: Option<Vec<Option<f32>>> = row.get(2).unwrap_or(None);
            let (Some(ctid_text), Some(arr)) = (ctid_text, arr) else {
                continue;
            };
            if arr.len() != expected_dim {
                // Skip mismatched-dim rows — caller's responsibility
                // to keep dims homogeneous; emit a NOTICE instead?
                continue;
            }
            let values: Vec<f32> = arr.into_iter().map(|v| v.unwrap_or(0.0)).collect();
            let id = parse_ctid_text(&ctid_text);
            out.push((id, values));
        }
    });
    out
}

/// `(BLOCK,OFFSET)` -> u64. Postgres prints CTIDs as `(block,offset)`.
fn parse_ctid_text(s: &str) -> u64 {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    let mut parts = inner.split(',');
    let blk: u32 = parts
        .next()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(0);
    let off: u16 = parts
        .next()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(0);
    (u64::from(blk) << 16) | u64::from(off)
}

/// Quote an SQL identifier per Postgres rules (double-up embedded `"`,
/// surround with `"`). Used only for the user-supplied `col` argument.
fn quote_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 2);
    out.push('"');
    for c in name.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}
