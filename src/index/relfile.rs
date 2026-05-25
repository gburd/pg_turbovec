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
//! Phase L hardening (item 1): every page write is wrapped in a
//! `GenericXLogStart` / `GenericXLogRegisterBuffer` /
//! `GenericXLogFinish` triplet, which is the recommended pattern
//! for custom AMs that don't define their own resource manager
//! (see pgvector's `hnswbuild.c`). Pages are registered with
//! `GENERIC_XLOG_FULL_IMAGE` because each write rewrites the
//! entire 8 KB page from our private byte format — there's no
//! useful diff, and the full-image flag skips the (otherwise
//! pointless) delta computation. For `RELPERSISTENCE_PERMANENT`
//! relations this emits an `XLOG_GENERIC` record; for unlogged /
//! temp relations `GenericXLogFinish` skips WAL but still writes
//! the modified page back to the buffer and marks it dirty.
//!
//! `GenericXLog` allows up to `MAX_GENERIC_XLOG_PAGES` (= 4)
//! buffers per state, so chain writes are batched in groups of 4
//! per record.
//!
//! ## Init fork (unlogged indexes)
//!
//! [`write_meta_in_fork`] accepts a `ForkNumber`. `ambuildempty`
//! calls it with `INIT_FORKNUM` so PG can copy the init fork over
//! the main fork after a crash, restoring the empty meta page
//! (Phase L hardening item 2).
//!
//! ## Truncate after rebuild
//!
//! [`write_full`] calls `RelationTruncate` when the new layout is
//! smaller than the old one (Phase L hardening item 3), so a
//! shrinking REINDEX or `ambulkdelete` consolidation actually
//! frees disk pages instead of leaving them as orphans.

#![cfg(feature = "relfile_storage")]

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::page::{MetaPageData, BLCKSZ, META_BLKNO, PAGE_HEADER_BYTES};

/// Maximum number of buffers we register with one `GenericXLog`
/// state — PG hard-codes this at `XLR_NORMAL_MAX_BLOCK_ID = 4`.
const GENERIC_XLOG_BATCH: usize = pg_sys::MAX_GENERIC_XLOG_PAGES as usize;

/// True when the relation's persistence is `'p'` (PERMANENT). For
/// unlogged / temp indexes `GenericXLogFinish` skips the WAL
/// record but still writes the modified page back — we don't need
/// to branch on this ourselves.
#[allow(dead_code)]
unsafe fn rel_needs_wal(rel: pg_sys::Relation) -> bool {
    let rd_rel = (*rel).rd_rel;
    !rd_rel.is_null()
        && (*rd_rel).relpersistence as u8 == pg_sys::RELPERSISTENCE_PERMANENT
}

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
    read_block_in_fork(rel, pg_sys::ForkNumber::MAIN_FORKNUM, blkno, exclusive)
}

/// Same as [`read_block`] but takes a fork. Internal helper used by
/// the init-fork path in `ambuildempty`.
unsafe fn read_block_in_fork(
    rel: pg_sys::Relation,
    fork: pg_sys::ForkNumber::Type,
    blkno: u32,
    exclusive: bool,
) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        rel,
        fork,
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

/// Mutable counterpart of [`page_data`]. Currently unused by
/// live code — all writers go through `GenericXLog`-managed
/// workspace pages — but kept for future direct-write callers.
///
/// # Safety
///
/// `buf` must be pinned and exclusive-locked.
#[allow(dead_code)]
pub(crate) unsafe fn page_data_mut(buf: pg_sys::Buffer) -> *mut u8 {
    let page = pg_sys::BufferGetPage(buf);
    page.cast::<u8>().add(PAGE_HEADER_BYTES)
}

/// PageInit + flag the entire data region as "used" so
/// `GenericXLogFinish` doesn't zero it as a hole. PostgreSQL's
/// generic-xlog machinery treats `[pd_lower, pd_upper)` as a free-
/// space hole (`memset 0` on apply, omitted from WAL deltas).
/// Standard AMs that use line pointers + tuples maintain
/// `pd_lower` correctly as they grow; we use the data region as a
/// private byte buffer, so we set `pd_lower = pd_upper` (the page
/// has no free space — every byte is meaningful). SP-GiST does
/// the same trick for its meta page; see `spgutils.c::SpGistInitMetapage`.
unsafe fn page_init_no_hole(page: pg_sys::Page) {
    pg_sys::PageInit(page, BLCKSZ, 0);
    let hdr = page.cast::<pg_sys::PageHeaderData>();
    // After PageInit: pd_lower = SizeOfPageHeaderData, pd_upper =
    // pd_special = BLCKSZ. Lift pd_lower up to pd_upper so there
    // is no hole.
    (*hdr).pd_lower = (*hdr).pd_upper;
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
    extend_block_in_fork(rel, pg_sys::ForkNumber::MAIN_FORKNUM)
}

/// Fork-aware extension. The returned buffer is pinned and
/// exclusive-locked; the page contents are PG-`PageInit`'d but the
/// caller is expected to overwrite them under a `GenericXLog`
/// state to get crash-recovery for free.
unsafe fn extend_block_in_fork(
    rel: pg_sys::Relation,
    fork: pg_sys::ForkNumber::Type,
) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        rel,
        fork,
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

/// Number of blocks in an arbitrary fork. Used by the init-fork
/// helpers in `ambuildempty`.
unsafe fn nblocks_in_fork(rel: pg_sys::Relation, fork: pg_sys::ForkNumber::Type) -> u32 {
    pg_sys::RelationGetNumberOfBlocksInFork(rel, fork)
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

/// Write or rewrite the meta page (block 0) of the main fork.
/// Extends the relation if it's empty; otherwise updates the
/// existing block in place. Wrapped in `GenericXLog` so the write
/// is durable across crashes.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock or otherwise
/// guarantee no concurrent writers (ambuild always runs alone;
/// aminsert serialises via the heap-tuple lock).
pub(crate) unsafe fn write_meta(rel: pg_sys::Relation, meta: &MetaPageData) {
    write_meta_in_fork(rel, pg_sys::ForkNumber::MAIN_FORKNUM, meta);
}

/// Fork-aware variant of [`write_meta`]. Phase L hardening (item
/// 2): `ambuildempty` calls this with `INIT_FORKNUM` so unlogged
/// indexes can be reset from the init fork after a crash.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock.
pub(crate) unsafe fn write_meta_in_fork(
    rel: pg_sys::Relation,
    fork: pg_sys::ForkNumber::Type,
    meta: &MetaPageData,
) {
    let buf = if nblocks_in_fork(rel, fork) == 0 {
        extend_block_in_fork(rel, fork)
    } else {
        read_block_in_fork(rel, fork, META_BLKNO, /*exclusive=*/ true)
    };

    let state = pg_sys::GenericXLogStart(rel);
    let page = pg_sys::GenericXLogRegisterBuffer(
        state,
        buf,
        pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
    );
    // Re-init on the GenericXLog workspace page so partial old
    // contents (e.g. a previous larger meta layout) don't leak.
    // page_init_no_hole flags the data region as "used" so
    // GenericXLogFinish doesn't zero it.
    page_init_no_hole(page);
    let dst = page.cast::<u8>().add(PAGE_HEADER_BYTES);
    let encoded = meta.encode();
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), dst, encoded.len());
    pg_sys::GenericXLogFinish(state);
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
    let total = chain_bytes.len();
    let mut written = 0usize;
    let mut blkno = start_blkno;
    let existing = nblocks(rel);

    // Process up to GENERIC_XLOG_BATCH pages per WAL record so we
    // amortise the record-header overhead while staying under PG's
    // hard cap on registered buffers per `GenericXLog` state.
    while written < total {
        let mut bufs: [pg_sys::Buffer; GENERIC_XLOG_BATCH] =
            [pg_sys::InvalidBuffer as pg_sys::Buffer; GENERIC_XLOG_BATCH];
        let state = pg_sys::GenericXLogStart(rel);
        let mut n_in_batch = 0usize;

        while n_in_batch < GENERIC_XLOG_BATCH && written < total {
            // Fresh-extend trailing blocks; rewrite leading blocks
            // in place. Both paths land us with a pinned +
            // exclusive-locked buffer ready for GenericXLog.
            let buf = if blkno < existing {
                read_block(rel, blkno, /*exclusive=*/ true)
            } else {
                extend_block(rel)
            };
            let page = pg_sys::GenericXLogRegisterBuffer(
                state,
                buf,
                pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
            );
            // Re-init on the workspace page; old contents (from a
            // bigger prior layout) don't leak through. page_init_no_hole
            // flags the data region as "used" so GenericXLogFinish
            // doesn't zero it.
            page_init_no_hole(page);
            let take = bytes_per_full_page.min(total - written);
            let src = chain_bytes.as_ptr().add(written);
            let dst = page.cast::<u8>().add(PAGE_HEADER_BYTES);
            std::ptr::copy_nonoverlapping(src, dst, take);

            bufs[n_in_batch] = buf;
            n_in_batch += 1;
            written += take;
            blkno += 1;
        }

        pg_sys::GenericXLogFinish(state);
        for buf in &bufs[..n_in_batch] {
            pg_sys::UnlockReleaseBuffer(*buf);
        }
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
        // Wrap the extension in GenericXLog so a crash between
        // extend and the eventual write_chain_at doesn't leave us
        // with un-WAL'd zero pages on disk.
        let buf = extend_block(rel);
        let state = pg_sys::GenericXLogStart(rel);
        let page = pg_sys::GenericXLogRegisterBuffer(
            state,
            buf,
            pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
        );
        page_init_no_hole(page);
        pg_sys::GenericXLogFinish(state);
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
/// relation, replacing any existing pages.
///
/// Strategy: rewrite leading blocks in place, fresh-extend the
/// tail, and `RelationTruncate` if the new layout is smaller than
/// the existing one. Every page write is wrapped in `GenericXLog`
/// for crash recovery.
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

    let meta = MetaPageData::plan(bit_width, dim, n_vectors, am_version);
    let new_total = meta.total_blocks().max(1);

    // Step 1: ensure the relation is at least `new_total` blocks
    // long. write_chain_at will rewrite leading blocks and extend
    // the tail itself, but the meta page (block 0) has to exist
    // before we call write_meta when the relation is empty —
    // write_meta handles that case via extend_block_in_fork.
    //
    // For the in-place rewrite path (`existing >= new_total`) we
    // skip extend_to entirely and rely on write_chain_at's
    // in-place branch.
    let existing_before = nblocks(rel);
    if existing_before < new_total {
        // Pre-extend so the chain offsets in the meta page are
        // valid the moment readers see them.
        extend_to(rel, new_total);
    }

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

    // Step 3 — Phase L hardening (item 3): release any trailing
    // blocks left over from a larger previous layout. This
    // matters after a shrinking REINDEX or `ambulkdelete`
    // consolidation; without it, dropped rows linger on disk as
    // orphan pages until the next REINDEX. `RelationTruncate`
    // emits its own WAL record (`XLOG_SMGR_TRUNCATE`) and is
    // crash-safe.
    let existing_after = nblocks(rel);
    if existing_after > new_total {
        pg_sys::RelationTruncate(rel, new_total);
    }
}

/// In-place swap-remove on a single chain: copy the row at slot
/// `from` to slot `to`, leaving the contents of slot `from` as
/// dead-but-still-on-disk bytes. The caller is expected to lower
/// `n_vectors` (in the meta page) so the now-trailing slot is no
/// longer addressed.
///
/// Both same-page (one buffer, intra-page byte copy) and cross-
/// page (one source-read + one destination-write) cases are
/// handled. The destination write is wrapped in a `GenericXLog`
/// state so the page change is WAL-logged. The source page is not
/// dirtied — its bytes at `from_slot` are simply ignored once the
/// meta page commits the smaller `n_vectors`.
///
/// `from_slot` and `to_slot` are 0-based slot indices over the
/// whole chain (page index = slot / rows_per_page,
/// intra-page offset = (slot mod rows_per_page) * stride).
/// `from_slot == to_slot` is a no-op.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured
/// the chain extends at least up to `max(from_slot, to_slot) + 1`
/// rows. Both slots must be within the existing chain length.
pub(crate) unsafe fn copy_slot_in_chain(
    rel: pg_sys::Relation,
    first_blkno: u32,
    stride: u32,
    rows_per_page: u32,
    from_slot: u64,
    to_slot: u64,
) {
    if from_slot == to_slot {
        return;
    }
    debug_assert!(rows_per_page > 0);
    debug_assert!(stride > 0);

    let rpp = u64::from(rows_per_page);
    let stride_us = stride as usize;

    let src_page_idx = (from_slot / rpp) as u32;
    let dst_page_idx = (to_slot / rpp) as u32;
    let src_off = ((from_slot % rpp) as usize) * stride_us;
    let dst_off = ((to_slot % rpp) as usize) * stride_us;
    let src_blkno = first_blkno + src_page_idx;
    let dst_blkno = first_blkno + dst_page_idx;

    if src_blkno == dst_blkno {
        // Same page: register once, ptr::copy (handles overlap),
        // finish. Cheaper than a separate read.
        let buf = read_block(rel, src_blkno, /*exclusive=*/ true);
        let state = pg_sys::GenericXLogStart(rel);
        let page = pg_sys::GenericXLogRegisterBuffer(
            state,
            buf,
            pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
        );
        let data = page.cast::<u8>().add(PAGE_HEADER_BYTES);
        std::ptr::copy(data.add(src_off), data.add(dst_off), stride_us);
        pg_sys::GenericXLogFinish(state);
        pg_sys::UnlockReleaseBuffer(buf);
        return;
    }

    // Different pages: copy out of the source under a shared
    // lock (no modification, no WAL), then register the
    // destination under an exclusive lock and overwrite the
    // single row. We deliberately avoid registering both buffers
    // in a single GenericXLog state because (a) the source isn't
    // dirty and would force an unnecessary FPW, (b) it keeps the
    // batch slot count comfortably under MAX_GENERIC_XLOG_PAGES.
    let mut row = vec![0u8; stride_us];
    let src_buf = read_block(rel, src_blkno, /*exclusive=*/ false);
    let src_data = page_data(src_buf);
    std::ptr::copy_nonoverlapping(src_data.add(src_off), row.as_mut_ptr(), stride_us);
    pg_sys::UnlockReleaseBuffer(src_buf);

    let dst_buf = read_block(rel, dst_blkno, /*exclusive=*/ true);
    let state = pg_sys::GenericXLogStart(rel);
    let page = pg_sys::GenericXLogRegisterBuffer(
        state,
        dst_buf,
        pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
    );
    // NOTE: do NOT call page_init_no_hole here. The destination
    // page was previously written via write_chain_at, which set
    // pd_lower = pd_upper. PageInit would reset that and zero
    // every byte we're trying to preserve.
    let dst_data = page.cast::<u8>().add(PAGE_HEADER_BYTES);
    std::ptr::copy_nonoverlapping(row.as_ptr(), dst_data.add(dst_off), stride_us);
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(dst_buf);
}

/// Read the ids chain (only) into a `Vec<u64>`. Cheaper than
/// `read_full` for the ambulkdelete callback walk, which only
/// needs ids to feed into the dead-tuple callback. ~8 MiB on a
/// 1 M-row index versus ~200 MiB for the full read.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_ids_only(rel: pg_sys::Relation, meta: &MetaPageData) -> Vec<u64> {
    if meta.n_vectors == 0 {
        return Vec::new();
    }
    let ids_bytes = read_chain(
        rel,
        meta.ids_first,
        std::mem::size_of::<u64>() as u32,
        meta.rows_per_ids_page,
        meta.n_vectors,
    );
    ids_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// Rewrite the meta page in place with a smaller `n_vectors` and
/// a bumped `am_version`. Used by `ambulkdelete` after walking the
/// page chains and swap-removing dead rows. The chain offsets
/// (`codes_first`, `scales_first`, `ids_first`, `rows_per_*_page`,
/// `stride_bytes`) are preserved so existing on-disk row positions
/// remain valid for survivors.
///
/// `*_count` fields are recomputed from the new `n_vectors` for
/// consistency, but readers ignore them — they walk the chain by
/// `n_vectors` only. The next `write_full` (build / aminsert
/// commit-time flush) re-plans from scratch, packing the chains
/// tightly.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock.
pub(crate) unsafe fn write_meta_shrink_in_place(
    rel: pg_sys::Relation,
    old: &MetaPageData,
    new_n_vectors: u64,
    new_am_version: u32,
) {
    let new_meta = MetaPageData {
        n_vectors: new_n_vectors,
        am_version: new_am_version,
        codes_count: MetaPageData::pages_needed(new_n_vectors, old.rows_per_codes_page),
        scales_count: MetaPageData::pages_needed(new_n_vectors, old.rows_per_scales_page),
        ids_count: MetaPageData::pages_needed(new_n_vectors, old.rows_per_ids_page),
        ..*old
    };
    write_meta(rel, &new_meta);
}

/// Truncate trailing pages of the ids chain after a shrink. The
/// ids chain is the last chain in the file layout (block order:
/// meta, codes, scales, ids), so any pages past the new ids tail
/// are pure garbage and can be safely released.
///
/// Pages between the now-shorter codes / scales chains and the
/// ids chain remain in place — moving the ids chain backward
/// would defeat the whole "don't rewrite the file" point. They'll
/// be reclaimed at the next `write_full` (which re-packs).
///
/// # Safety
///
/// Caller must hold an exclusive relation lock.
pub(crate) unsafe fn truncate_ids_tail(rel: pg_sys::Relation, meta: &MetaPageData) {
    if meta.n_vectors == 0 && meta.ids_first == 0 {
        return;
    }
    let new_ids_count = MetaPageData::pages_needed(meta.n_vectors, meta.rows_per_ids_page);
    let new_total = meta.ids_first + new_ids_count;
    let cur = nblocks(rel);
    if cur > new_total {
        pg_sys::RelationTruncate(rel, new_total);
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
