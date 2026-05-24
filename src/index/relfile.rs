//! Phase L — relfile-resident I/O for the `turbovec` AM.
//!
//! Wires [`crate::index::page`] to PostgreSQL's buffer manager.
//! All page reads / writes go through `ReadBufferExtended`,
//! `LockBuffer`, `BufferGetPage`, `MarkBufferDirty`,
//! `UnlockReleaseBuffer`. shared_buffers caches the pages
//! cluster-wide; cold scans pay one set of buffer-pool reads
//! instead of an SPI fetch + TOAST detoast + IdMapIndex parse.
//!
//! ## Concurrency
//!
//! Phase L stub takes a per-page exclusive lock on each write and a
//! shared lock on each read, mirroring the standard index-AM
//! protocol. There is no `aminsert` deferred-commit batching yet
//! (Phase K hook); each insert serialises the whole index, like the
//! SPI path. Cold-path reads, the headline win, work today.
//!
//! ## WAL
//!
//! **Not implemented.** Pages are written via `MarkBufferDirty` but
//! we never call `log_newpage_buffer`. After an immediate-shutdown
//! crash the index relfile may diverge from the WAL stream, in
//! which case PG will silently scan an empty / partial index. The
//! handoff doc tracks this as the Phase L follow-up.

#![cfg(feature = "relfile_storage")]

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::page::{MetaPageData, BLCKSZ, META_BLKNO, PAGE_HEADER_BYTES};

/// Convenience wrapper around `ReadBufferExtended` that
/// `LockBuffer`s and returns both the buffer and the data-region
/// slice of the page. The data region is the bytes immediately
/// after the standard `PageHeaderData` (24 bytes); we never use
/// line pointers.
///
/// `mode` is one of `pg_sys::ReadBufferMode::RBM_NORMAL`,
/// `RBM_ZERO_AND_LOCK`, etc. When passing `RBM_ZERO_AND_LOCK` the
/// buffer comes back already exclusive-locked, so we skip the
/// secondary `LockBuffer`.
///
/// # Safety
///
/// Caller must hold a relation reference; the returned buffer is
/// pinned + locked and must be released via `UnlockReleaseBuffer`.
pub(crate) unsafe fn read_block(
    rel: pg_sys::Relation,
    blkno: u32,
    exclusive: bool,
) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        rel,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        blkno,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    if buf == pg_sys::InvalidBuffer as pg_sys::Buffer {
        error!(
            "turbovec relfile: ReadBufferExtended returned InvalidBuffer for blk {}",
            blkno
        );
    }
    let mode = if exclusive {
        pg_sys::BUFFER_LOCK_EXCLUSIVE
    } else {
        pg_sys::BUFFER_LOCK_SHARE
    };
    pg_sys::LockBuffer(buf, mode as i32);
    buf
}

/// Borrow a page's data region (bytes after the 24-byte PG page
/// header) immutably.
///
/// # Safety
///
/// `buf` must be pinned and locked.
pub(crate) unsafe fn page_data(buf: pg_sys::Buffer) -> *const u8 {
    let page = pg_sys::BufferGetPage(buf);
    page.cast::<u8>().add(PAGE_HEADER_BYTES) as *const u8
}

/// Mutable counterpart of [`page_data`].
///
/// # Safety
///
/// `buf` must be pinned and exclusive-locked.
pub(crate) unsafe fn page_data_mut(buf: pg_sys::Buffer) -> *mut u8 {
    let page = pg_sys::BufferGetPage(buf);
    page.cast::<u8>().add(PAGE_HEADER_BYTES)
}

/// Initialise a freshly-extended page with `PageInit` so the
/// standard PG header is well-formed. We never call `PageAddItem`;
/// the data region starting at byte 24 is private to us.
///
/// # Safety
///
/// `buf` must be pinned and exclusive-locked.
pub(crate) unsafe fn page_init(buf: pg_sys::Buffer) {
    let page = pg_sys::BufferGetPage(buf);
    pg_sys::PageInit(page, BLCKSZ, 0);
}

/// Extend the relation by one block. Returns a pinned + exclusive-
/// locked buffer initialised by `PageInit`.
///
/// Implementation note: pgrx-pg-sys exposes `ExtendBufferedRel` for
/// pg16+, but `ReadBufferExtended(rel, MAIN_FORKNUM, P_NEW=InvalidBlockNumber, RBM_NORMAL, NULL)`
/// is portable across pg13..pg18 and is what pgvector uses, so we
/// stick with it for the stub.
///
/// # Safety
///
/// Caller must hold a relation reference. The returned buffer must
/// be released via `MarkBufferDirty` + `UnlockReleaseBuffer`.
pub(crate) unsafe fn extend_block(rel: pg_sys::Relation) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        rel,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        pg_sys::InvalidBlockNumber,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    if buf == pg_sys::InvalidBuffer as pg_sys::Buffer {
        error!("turbovec relfile: ReadBufferExtended(P_NEW) returned InvalidBuffer");
    }
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    page_init(buf);
    buf
}

/// Number of blocks currently in the index relation's main fork.
pub(crate) unsafe fn nblocks(rel: pg_sys::Relation) -> u32 {
    pg_sys::RelationGetNumberOfBlocksInFork(rel, pg_sys::ForkNumber::MAIN_FORKNUM)
}

/// Read the meta page (block 0). Returns `None` for an empty
/// relfile, e.g. immediately after `RelationCreateStorage` and
/// before our `ambuildempty` ran.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_meta(rel: pg_sys::Relation) -> Option<MetaPageData> {
    if nblocks(rel) == 0 {
        return None;
    }
    let buf = read_block(rel, META_BLKNO, /*exclusive=*/ false);
    let data = page_data(buf);
    let slice = std::slice::from_raw_parts(data, BLCKSZ - PAGE_HEADER_BYTES);
    let meta = MetaPageData::decode(slice);
    pg_sys::UnlockReleaseBuffer(buf);
    match meta {
        Ok(m) => Some(m),
        Err(e) => error!("turbovec relfile: corrupt meta page: {}", e),
    }
}

/// Write or rewrite the meta page (block 0). Extends the relation
/// if it's empty; otherwise updates the existing block in place.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock or otherwise
/// guarantee no concurrent writers (ambuild always runs alone;
/// aminsert serialises via the heap-tuple lock).
pub(crate) unsafe fn write_meta(rel: pg_sys::Relation, meta: &MetaPageData) {
    let buf = if nblocks(rel) == 0 {
        extend_block(rel)
    } else {
        let buf = read_block(rel, META_BLKNO, /*exclusive=*/ true);
        page_init(buf); // re-init so partial old contents don't leak
        buf
    };
    let dst = page_data_mut(buf);
    let encoded = meta.encode();
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), dst, encoded.len());
    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// Append `chain_bytes` worth of payload across freshly-extended
/// pages, `rows_per_page * stride` bytes per page (last page may be
/// short). Returns the first block number of the chain.
///
/// Currently unused by the in-place rewrite path; kept for the
/// Phase L follow-up that lazily extends instead of upfront-extends.
///
/// # Safety
///
/// `chain_bytes.len()` must equal `n_vectors * stride`. Caller must
/// hold an exclusive relation lock.
#[allow(dead_code)]
pub(crate) unsafe fn write_chain(
    rel: pg_sys::Relation,
    chain_bytes: &[u8],
    stride: u32,
    rows_per_page: u32,
    n_vectors: u64,
) -> u32 {
    if n_vectors == 0 || rows_per_page == 0 {
        return pg_sys::InvalidBlockNumber;
    }
    debug_assert_eq!(chain_bytes.len() as u64, n_vectors * u64::from(stride));

    let bytes_per_full_page = (rows_per_page as usize) * (stride as usize);
    let mut written = 0usize;
    let mut first_blkno: u32 = pg_sys::InvalidBlockNumber;

    while written < chain_bytes.len() {
        let buf = extend_block(rel);
        let blkno = pg_sys::BufferGetBlockNumber(buf);
        if first_blkno == pg_sys::InvalidBlockNumber {
            first_blkno = blkno;
        }
        let take = bytes_per_full_page.min(chain_bytes.len() - written);
        let src = chain_bytes.as_ptr().add(written);
        std::ptr::copy_nonoverlapping(src, page_data_mut(buf), take);
        pg_sys::MarkBufferDirty(buf);
        pg_sys::UnlockReleaseBuffer(buf);
        written += take;
    }
    first_blkno
}

/// Write `chain_bytes` to a fixed range of blocks starting at
/// `start_blkno`. Each block is reinitialised via `PageInit` so any
/// stale contents from an earlier layout get wiped. The caller is
/// responsible for having extended the relation to at least
/// `start_blkno + pages_needed` blocks beforehand (see
/// [`extend_to`]).
///
/// # Safety
///
/// `chain_bytes.len()` must equal `n_vectors * stride`. Caller must
/// hold an exclusive relation lock.
pub(crate) unsafe fn write_chain_at(
    rel: pg_sys::Relation,
    start_blkno: u32,
    chain_bytes: &[u8],
    stride: u32,
    rows_per_page: u32,
    n_vectors: u64,
) {
    if n_vectors == 0 || rows_per_page == 0 {
        return;
    }
    debug_assert_eq!(chain_bytes.len() as u64, n_vectors * u64::from(stride));

    let bytes_per_full_page = (rows_per_page as usize) * (stride as usize);
    let mut written = 0usize;
    let mut blkno = start_blkno;

    while written < chain_bytes.len() {
        let buf = read_block(rel, blkno, /*exclusive=*/ true);
        page_init(buf);
        let take = bytes_per_full_page.min(chain_bytes.len() - written);
        let src = chain_bytes.as_ptr().add(written);
        std::ptr::copy_nonoverlapping(src, page_data_mut(buf), take);
        pg_sys::MarkBufferDirty(buf);
        pg_sys::UnlockReleaseBuffer(buf);
        written += take;
        blkno += 1;
    }
}

/// Extend the relation until `nblocks(rel) >= target`. New blocks
/// are PageInit'd and immediately released; subsequent writers can
/// pick them up via `read_block`.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock.
pub(crate) unsafe fn extend_to(rel: pg_sys::Relation, target: u32) {
    while nblocks(rel) < target {
        let buf = extend_block(rel);
        pg_sys::MarkBufferDirty(buf);
        pg_sys::UnlockReleaseBuffer(buf);
    }
}

/// Concatenate every page in a chain into a contiguous `Vec<u8>` of
/// `n_vectors * stride` bytes. Used by `ambeginscan` to reconstruct
/// the codes / scales / ids arrays.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_chain(
    rel: pg_sys::Relation,
    first_blkno: u32,
    stride: u32,
    rows_per_page: u32,
    n_vectors: u64,
) -> Vec<u8> {
    let total_bytes = (n_vectors as usize) * (stride as usize);
    let mut out = Vec::<u8>::with_capacity(total_bytes);
    if total_bytes == 0 {
        return out;
    }
    let bytes_per_full_page = (rows_per_page as usize) * (stride as usize);
    let mut remaining = total_bytes;
    let mut blkno = first_blkno;

    while remaining > 0 {
        let buf = read_block(rel, blkno, /*exclusive=*/ false);
        let take = bytes_per_full_page.min(remaining);
        let src = page_data(buf);
        let pos = out.len();
        out.set_len(pos + take);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(pos), take);
        pg_sys::UnlockReleaseBuffer(buf);
        remaining -= take;
        blkno += 1;
    }
    out
}

/// High-level helper: write a fully-built `IdMapIndex` to the
/// relation, replacing any existing pages. Implemented as
/// "truncate + extend" — we read no old pages, just append the
/// new ones starting at block 0 (meta) and 1.. (chains). When the
/// relation already has more blocks than we need, the trailing
/// blocks are left as orphans for now (Phase L follow-up will call
/// `RelationTruncate`).
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured
/// the index isn't being concurrently scanned. ambuild satisfies
/// both; aminsert and ambulkdelete take the heap-tuple lock,
/// which is sufficient for the existing single-writer SPI path
/// and remains so here.
pub(crate) unsafe fn write_full(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
) {
    assert_eq!(slot_to_id.len() as u64, n_vectors);
    assert_eq!(scales.len() as u64, n_vectors);

    // If the relfile already has >0 blocks, we're rewriting (REINDEX
    // / aminsert / ambulkdelete). Truncate to zero first so block
    // numbers in the new meta are stable.
    let existing = nblocks(rel);
    let _ = existing;
    // Phase L follow-up: call RelationTruncate to release pages
    // that fall past `meta.total_blocks()` in the new layout. The
    // current rewrite strategy reuses leading blocks in place;
    // any tail blocks from a larger prior layout become orphans
    // until REINDEX.

    let meta = MetaPageData::plan(bit_width, dim, n_vectors, am_version);

    // Phase L stub: rewrite-in-place strategy.
    //
    //   1. Extend the relation up to `meta.total_blocks()` so every
    //      block in the new layout exists physically.
    //   2. Overwrite block 0 with the new meta header.
    //   3. Overwrite the codes / scales / ids chain blocks at the
    //      offsets the meta page advertises.
    //
    // Trailing blocks left over from a larger previous layout are
    // *not* truncated. They become orphans — unreachable through
    // the meta header, but still occupy disk pages until the next
    // REINDEX. Phase L follow-up should call `RelationTruncate`
    // here to release them. For an experimental feature on the
    // common monotonic-grow path (ambuild + aminsert) this is
    // harmless.
    extend_to(rel, meta.total_blocks().max(1));

    write_meta(rel, &meta);

    if n_vectors > 0 {
        write_chain_at(
            rel,
            meta.codes_first,
            packed_codes,
            meta.stride_bytes,
            meta.rows_per_codes_page,
            n_vectors,
        );

        // Scales chain. Reinterpret f32 slice as raw bytes.
        let scales_bytes: &[u8] =
            std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), std::mem::size_of_val(scales));
        write_chain_at(
            rel,
            meta.scales_first,
            scales_bytes,
            std::mem::size_of::<f32>() as u32,
            meta.rows_per_scales_page,
            n_vectors,
        );

        // Ids chain.
        let ids_bytes: &[u8] = std::slice::from_raw_parts(
            slot_to_id.as_ptr().cast::<u8>(),
            std::mem::size_of_val(slot_to_id),
        );
        write_chain_at(
            rel,
            meta.ids_first,
            ids_bytes,
            std::mem::size_of::<u64>() as u32,
            meta.rows_per_ids_page,
            n_vectors,
        );
    }
}

/// Read every chain back into Rust-owned buffers. Returns
/// `(packed_codes, scales, slot_to_id)`.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_full(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> (Vec<u8>, Vec<f32>, Vec<u64>) {
    if meta.n_vectors == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let codes = read_chain(
        rel,
        meta.codes_first,
        meta.stride_bytes,
        meta.rows_per_codes_page,
        meta.n_vectors,
    );
    let scales_bytes = read_chain(
        rel,
        meta.scales_first,
        std::mem::size_of::<f32>() as u32,
        meta.rows_per_scales_page,
        meta.n_vectors,
    );
    let ids_bytes = read_chain(
        rel,
        meta.ids_first,
        std::mem::size_of::<u64>() as u32,
        meta.rows_per_ids_page,
        meta.n_vectors,
    );
    let scales: Vec<f32> = scales_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ids: Vec<u64> = ids_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    (codes, scales, ids)
}
