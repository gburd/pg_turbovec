//! SPI-backed persistence for `turbovec.am_storage`.
//!
//! We deliberately use SPI rather than reaching into the index
//! relation's main fork: SPI gives us WAL-correctness for free, and
//! Phase 4 can ship a working AM without writing a single page-format
//! callback. Phase 5 will replace this with relfile-resident pages.
//!
//! All functions assume they are called from within a PostgreSQL
//! backend with a valid memory context.
//!
//! Phase L: under `--features relfile_storage` most of this module
//! is dead code (the AM uses `relfile.rs` instead). We keep it
//! compileable so the side-table marker writes (`save_empty`,
//! `save_empty_with_count`) remain available, and squelch the
//! per-fn dead-code warnings with a single `#[allow]`.
#![cfg_attr(feature = "relfile_storage", allow(dead_code))]

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

/// Persisted state for a single `turbovec` index.
pub(crate) struct StoredIndex {
    pub bit_width: i32,
    pub dim: i32,
    pub n_vectors: i64,
    pub index: IdMapIndex,
    pub version: i32,
    /// Every u64 id currently in `index`. Maintained as a parallel
    /// structure so `ambulkdelete` can enumerate live ids without
    /// reaching into `IdMapIndex`'s private `slot_to_id`. Persisted
    /// in the `live_ids bytea` column of `turbovec.am_storage`.
    pub live_ids: Vec<u64>,
}

/// Cheap metadata-only fetch: enough to build a cache key and
/// compute a freshness signal without paying the cost of dragging
/// the full `payload bytea` (which can be hundreds of MiB on big
/// indexes) over SPI. Used on the AM scan hot path.
pub(crate) struct StoredMeta {
    pub bit_width: i32,
    pub dim: i32,
    pub n_vectors: i64,
    pub version: i32,
}

/// Read just `(bit_width, dim, n_vectors, version)` for `indexrelid`.
/// Returns `None` when no side-table row exists yet (e.g. between
/// `CREATE INDEX` and the first `ambuildempty`).
pub(crate) fn load_meta(indexrelid: pg_sys::Oid) -> Option<StoredMeta> {
    let row: Option<(i32, i32, i64, i32)> = Spi::connect(|client| {
        let sql = "SELECT bit_width, dim, n_vectors, version \
                   FROM turbovec.am_storage WHERE indexrelid = $1";
        let mut iter = match client.select(sql, Some(1), &[indexrelid.into()]) {
            Ok(t) => t,
            Err(_) => return None,
        };
        let row = iter.next()?;
        let bw: Option<i32> = row.get(1).ok().flatten();
        let dim: Option<i32> = row.get(2).ok().flatten();
        let nv: Option<i64> = row.get(3).ok().flatten();
        let ver: Option<i32> = row.get(4).ok().flatten();
        match (bw, dim, nv, ver) {
            (Some(bw), Some(dim), Some(nv), Some(ver)) => Some((bw, dim, nv, ver)),
            _ => None,
        }
    });
    let (bit_width, dim, n_vectors, version) = row?;
    Some(StoredMeta {
        bit_width,
        dim,
        n_vectors,
        version,
    })
}

/// Read the current payload for `indexrelid`. Returns `None` if no
/// row exists (typical immediately after `CREATE INDEX` before
/// `ambuildempty` has run).
pub(crate) fn load(indexrelid: pg_sys::Oid) -> Option<StoredIndex> {
    let row: Option<(i32, i32, i64, Vec<u8>, i32, Option<Vec<u8>>)> = Spi::connect(|client| {
        let sql = "SELECT bit_width, dim, n_vectors, payload, version, live_ids \
                   FROM turbovec.am_storage WHERE indexrelid = $1";
        let mut iter = match client.select(sql, Some(1), &[indexrelid.into()]) {
            Ok(t) => t,
            Err(_) => return None,
        };
        let row = iter.next()?;
        let bw: Option<i32> = row.get(1).ok().flatten();
        let dim: Option<i32> = row.get(2).ok().flatten();
        let nv: Option<i64> = row.get(3).ok().flatten();
        let payload: Option<Vec<u8>> = row.get(4).ok().flatten();
        let ver: Option<i32> = row.get(5).ok().flatten();
        let live: Option<Vec<u8>> = row.get(6).ok().flatten();
        match (bw, dim, nv, payload, ver) {
            (Some(bw), Some(dim), Some(nv), Some(payload), Some(ver)) => {
                Some((bw, dim, nv, payload, ver, live))
            }
            _ => None,
        }
    });
    let (bit_width, dim, n_vectors, payload, version, live_blob) = row?;

    if payload.is_empty() {
        return None;
    }

    let mut cursor = std::io::Cursor::new(payload);
    let index = match read_idmap_from(&mut cursor) {
        Ok(idx) => idx,
        Err(e) => error!(
            "turbovec.am_storage: failed to deserialise IdMapIndex for {:?}: {}",
            indexrelid, e
        ),
    };
    let live_ids = decode_live_ids(live_blob.as_deref().unwrap_or(&[]));
    Some(StoredIndex {
        bit_width,
        dim,
        n_vectors,
        index,
        version,
        live_ids,
    })
}

/// Pack `&[u64]` into little-endian bytes for the `live_ids bytea` column.
fn encode_live_ids(ids: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

/// Inverse of `encode_live_ids`.
fn decode_live_ids(bytes: &[u8]) -> Vec<u64> {
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(u64::from_le_bytes(buf));
    }
    out
}

/// Read an `IdMapIndex` from an arbitrary `Read`. Avoids the
/// tmpfile dance by going directly through the upstream
/// `IdMapIndex::load_from_reader` API surfaced in our vendor patch.
fn read_idmap_from<R: std::io::Read>(r: &mut R) -> Result<IdMapIndex, String> {
    IdMapIndex::load_from_reader(r).map_err(|e| e.to_string())
}

/// Serialise an `IdMapIndex` to bytes. Uses the upstream
/// `IdMapIndex::write_to_writer` API surfaced in our vendor patch —
/// no tmpfile, no syscall, no /tmp churn.
fn write_idmap_to_bytes(idx: &IdMapIndex) -> Result<Vec<u8>, String> {
    let mut buf = Vec::with_capacity(estimate_idmap_size(idx));
    idx.write_to_writer(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Estimate the serialised size of an `IdMapIndex` so we can presize
/// the destination buffer. Slightly over-allocating beats reallocating
/// during the write.
fn estimate_idmap_size(idx: &IdMapIndex) -> usize {
    // 4 (magic) + 1 (version) + 9 (core header)
    //   + (dim/8) * bit_width * n_vectors  (packed codes)
    //   + n_vectors * 4                     (scales: f32)
    //   + n_vectors * 8                     (slot_to_id: u64)
    let n = idx.len();
    let dim = idx.dim();
    let bw = idx.bit_width();
    let codes = (dim / 8) * bw * n;
    let scales = n * 4;
    let ids = n * 8;
    14 + codes + scales + ids
}

/// Insert or update the side-table row for this index.
pub(crate) fn save(
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: i32,
    n_vectors: i64,
    idx: &IdMapIndex,
    new_version: i32,
    live_ids: &[u64],
) {
    let payload = match write_idmap_to_bytes(idx) {
        Ok(b) => b,
        Err(e) => error!(
            "turbovec.am_storage: failed to serialise IdMapIndex for {:?}: {}",
            indexrelid, e
        ),
    };
    let live_blob = encode_live_ids(live_ids);
    Spi::connect_mut(|client| {
        let sql = "INSERT INTO turbovec.am_storage \
                       (indexrelid, bit_width, dim, n_vectors, payload, version, live_ids, updated_at) \
                   VALUES ($1, $2, $3, $4, $5, $6, $7, now()) \
                   ON CONFLICT (indexrelid) DO UPDATE SET \
                       bit_width = EXCLUDED.bit_width, \
                       dim       = EXCLUDED.dim, \
                       n_vectors = EXCLUDED.n_vectors, \
                       payload   = EXCLUDED.payload, \
                       version   = EXCLUDED.version, \
                       live_ids  = EXCLUDED.live_ids, \
                       updated_at = EXCLUDED.updated_at";
        let _ = client.update(
            sql,
            None,
            &[
                indexrelid.into(),
                bit_width.into(),
                dim.into(),
                n_vectors.into(),
                payload.into(),
                new_version.into(),
                live_blob.into(),
            ],
        );
    });
}

/// Insert an empty marker row used by `ambuildempty`.
pub(crate) fn save_empty(indexrelid: pg_sys::Oid, bit_width: i32, dim: i32) {
    Spi::connect_mut(|client| {
        let sql = "INSERT INTO turbovec.am_storage \
                       (indexrelid, bit_width, dim, n_vectors, payload, version, updated_at) \
                   VALUES ($1, $2, $3, 0, ''::bytea, 1, now()) \
                   ON CONFLICT (indexrelid) DO NOTHING";
        let _ = client.update(
            sql,
            None,
            &[indexrelid.into(), bit_width.into(), dim.into()],
        );
    });
}

/// Phase L marker row: same shape as [`save_empty`] but records
/// the current row count so existing tests that read `n_vectors`
/// from the side-table keep working under the relfile path. The
/// payload column stays empty (relfile path doesn't use it). Hard
/// to be too clever here — the side-table is going away in v1.1.
#[cfg(feature = "relfile_storage")]
pub(crate) fn save_empty_with_count(
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: i32,
    n_vectors: i64,
) {
    Spi::connect_mut(|client| {
        let sql = "INSERT INTO turbovec.am_storage \
                       (indexrelid, bit_width, dim, n_vectors, payload, version, live_ids, updated_at) \
                   VALUES ($1, $2, $3, $4, ''::bytea, 1, ''::bytea, now()) \
                   ON CONFLICT (indexrelid) DO UPDATE SET \
                       bit_width  = EXCLUDED.bit_width, \
                       dim        = EXCLUDED.dim, \
                       n_vectors  = EXCLUDED.n_vectors, \
                       payload    = EXCLUDED.payload, \
                       version    = EXCLUDED.version, \
                       live_ids   = EXCLUDED.live_ids, \
                       updated_at = EXCLUDED.updated_at";
        let _ = client.update(
            sql,
            None,
            &[
                indexrelid.into(),
                bit_width.into(),
                dim.into(),
                n_vectors.into(),
            ],
        );
    });
}

/// Remove the side-table row when the index is dropped. Hooked into
/// the relcache invalidation callback in Phase 5.
#[allow(dead_code)]
pub(crate) fn drop_row(indexrelid: pg_sys::Oid) {
    Spi::connect_mut(|client| {
        let _ = client.update(
            "DELETE FROM turbovec.am_storage WHERE indexrelid = $1",
            None,
            &[indexrelid.into()],
        );
    });
}
