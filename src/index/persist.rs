//! SPI-backed persistence for `turbovec.am_storage`.
//!
//! We deliberately use SPI rather than reaching into the index
//! relation's main fork: SPI gives us WAL-correctness for free, and
//! Phase 4 can ship a working AM without writing a single page-format
//! callback. Phase 5 will replace this with relfile-resident pages.
//!
//! All functions assume they are called from within a PostgreSQL
//! backend with a valid memory context.

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
}

/// Read the current payload for `indexrelid`. Returns `None` if no
/// row exists (typical immediately after `CREATE INDEX` before
/// `ambuildempty` has run).
pub(crate) fn load(indexrelid: pg_sys::Oid) -> Option<StoredIndex> {
    let row: Option<(i32, i32, i64, Vec<u8>, i32)> = Spi::connect(|client| {
        let sql = "SELECT bit_width, dim, n_vectors, payload, version \
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
        match (bw, dim, nv, payload, ver) {
            (Some(bw), Some(dim), Some(nv), Some(payload), Some(ver)) => {
                Some((bw, dim, nv, payload, ver))
            }
            _ => None,
        }
    });
    let (bit_width, dim, n_vectors, payload, version) = row?;

    if payload.is_empty() {
        // Empty marker — caller should treat as "index not yet built".
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
    Some(StoredIndex {
        bit_width,
        dim,
        n_vectors,
        index,
        version,
    })
}

/// Read an `IdMapIndex` from an arbitrary `Read`. The upstream crate
/// only exposes `IdMapIndex::load(path)` which reads from disk; we
/// dump to a temp file and call into it.  This is genuinely slow and
/// is the most obvious target for a Phase-5 in-memory deserialiser
/// upstream.
fn read_idmap_from<R: std::io::Read>(r: &mut R) -> Result<IdMapIndex, String> {
    use std::io::Write;
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    // Write to a tmpfile so we can call IdMapIndex::load(path). This
    // is unfortunate; v0.5 should reach upstream for an in-memory
    // load. We use mkstemp via PgMemoryContexts to keep cleanup
    // deterministic even on backend abort.
    let dir = std::env::temp_dir();
    let pid = unsafe { libc::getpid() };
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let path = dir.join(format!("turbovec-load-{}-{}.tvim", pid, nonce));
    std::fs::write(&path, &buf).map_err(|e| e.to_string())?;
    let idx = IdMapIndex::load(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(idx)
}

/// Serialise an `IdMapIndex` to bytes via the same temp-file dance.
fn write_idmap_to_bytes(idx: &IdMapIndex) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir();
    let pid = unsafe { libc::getpid() };
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let path = dir.join(format!("turbovec-store-{}-{}.tvim", pid, nonce));
    idx.write(&path).map_err(|e| e.to_string())?;
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(bytes)
}

/// Insert or update the side-table row for this index.
pub(crate) fn save(
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: i32,
    n_vectors: i64,
    idx: &IdMapIndex,
    new_version: i32,
) {
    let payload = match write_idmap_to_bytes(idx) {
        Ok(b) => b,
        Err(e) => error!(
            "turbovec.am_storage: failed to serialise IdMapIndex for {:?}: {}",
            indexrelid, e
        ),
    };
    Spi::connect_mut(|client| {
        let sql = "INSERT INTO turbovec.am_storage \
                       (indexrelid, bit_width, dim, n_vectors, payload, version, updated_at) \
                   VALUES ($1, $2, $3, $4, $5, $6, now()) \
                   ON CONFLICT (indexrelid) DO UPDATE SET \
                       bit_width = EXCLUDED.bit_width, \
                       dim       = EXCLUDED.dim, \
                       n_vectors = EXCLUDED.n_vectors, \
                       payload   = EXCLUDED.payload, \
                       version   = EXCLUDED.version, \
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

/// Remove the side-table row when the index is dropped. Hooked into
/// the relcache invalidation callback in Phase 5.
pub(crate) fn drop_row(indexrelid: pg_sys::Oid) {
    Spi::connect_mut(|client| {
        let _ = client.update(
            "DELETE FROM turbovec.am_storage WHERE indexrelid = $1",
            None,
            &[indexrelid.into()],
        );
    });
}
