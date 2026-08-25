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

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::index::page::{BLCKSZ, META_BLKNO, MetaPageData, PAGE_HEADER_BYTES};

/// `LockBuffer`'s `mode` parameter type changed across PG majors:
/// `int` (i32) through PG18, and `BufferLockMode::Type` (u32) in PG19.
/// The `BUFFER_LOCK_*` constants are `u32` either way; this helper
/// coerces to whatever the current `LockBuffer` binding expects so the
/// call sites stay version-agnostic.
#[cfg(not(feature = "pg19"))]
#[inline]
pub(crate) fn lock_buffer_mode(m: u32) -> i32 {
    m as i32
}
#[cfg(feature = "pg19")]
#[inline]
pub(crate) fn lock_buffer_mode(m: u32) -> u32 {
    m
}

/// Maximum number of buffers we register with one `GenericXLog`
/// state — PG hard-codes this at `XLR_NORMAL_MAX_BLOCK_ID = 4`.
const GENERIC_XLOG_BATCH: usize = pg_sys::MAX_GENERIC_XLOG_PAGES as usize;

/// # Relfile-rewrite serialization (v1.29.1 corruption fix)
///
/// The deferred-commit `aminsert` flush ([`crate::xact::flush_to_relfile`])
/// and VACUUM's in-place shrink both REWRITE the whole relfile (meta
/// page + codes/scales/ids/rotation chains) under only a
/// `RowExclusiveLock` / `ShareUpdateExclusiveLock` heap lock — locks
/// deliberately chosen so writers don't block scanning readers
/// (who hold `AccessShareLock`). But the rewrite touches many pages
/// across several `GenericXLog` records, holding only a *per-page*
/// buffer content lock at a time; a concurrent [`read_full`] walks
/// the same chains page-by-page under per-page SHARE locks. With
/// nothing serializing the two *whole-chain* operations, a reader
/// could observe a TORN ids chain — some pages already rewritten,
/// some still old, some `PageInit`'d-to-zero mid-write — surfacing
/// as "duplicate ids (id N appears in more than one slot)" (XX001)
/// or, on the writer's own racing cold read, a `swap_remove`
/// `assert!` SIGABRT. The on-disk relfile was never corrupt (a
/// serialized `turbovec_check` always read it clean); the fault was
/// purely a read/write interleaving, which is why REINDEX "didn't
/// hold" and only quiescing the workload appeared to fix it.
///
/// We serialize the two whole-chain operations with a heavyweight
/// page lock on the META page (block 0), used purely as a rewrite
/// mutex, NOT via buffer content locks (which would deadlock against
/// the per-page locks `write_meta` / `write_chain_at` take, and
/// aren't held across the whole multi-page op). `ShareLock` (many
/// concurrent readers) conflicts with `ExclusiveLock` (one
/// rewriter) in PG's lock matrix, and same-transaction locks never
/// self-conflict — so a writer's own cold [`read_full`] (ShareLock)
/// followed by its rewrite (ExclusiveLock) in the same xact is
/// fine. The lock is heap-independent, deadlock-detected, and
/// released explicitly at the end of each operation.
///
/// This changes NO on-disk bytes (no wire-format change, no
/// REINDEX): it only adds an in-memory lock around existing reads
/// and writes.
const RELFILE_REWRITE_LOCK_BLK: pg_sys::BlockNumber = META_BLKNO;

// v1.29.4 corruption-fix regression harness (test-only). Two knobs let a
// single-backend `#[pg_test]` prove the meta-LAST write ordering:
//   * WRITE_META_FIRST_FOR_TEST: restore the PRE-FIX order (meta written
//     BEFORE the chains) so the test can demonstrate the fail-before.
//   * INJECT_TEAR_BEFORE_IDS: simulate a `pg_terminate_backend`/cancel
//     longjmp mid-rewrite by returning from write_full_inner right after
//     the codes/scales chains but BEFORE the ids chain + final meta write
//     (the exact window the real interrupt hits).
#[cfg(any(test, feature = "pg_test"))]
thread_local! {
    pub(crate) static WRITE_META_FIRST_FOR_TEST: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    pub(crate) static INJECT_TEAR_BEFORE_IDS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Take the shared side of the relfile-rewrite lock (a reader
/// reconstructing the in-memory index from the chains). Released
/// by [`unlock_relfile_read`].
///
/// # Safety
/// Caller must hold a relation reference and MUST pair this with
/// exactly one [`unlock_relfile_read`] on all paths (including
/// error unwinds — see [`read_full`], which brackets its body).
#[inline]

pub(crate) unsafe fn lock_relfile_read(rel: pg_sys::Relation) {
    pg_sys::LockPage(
        rel,
        RELFILE_REWRITE_LOCK_BLK,
        pg_sys::ShareLock as pg_sys::LOCKMODE,
    );
}

/// Release the shared side taken by [`lock_relfile_read`].
#[inline]
pub(crate) unsafe fn unlock_relfile_read(rel: pg_sys::Relation) {
    pg_sys::UnlockPage(
        rel,
        RELFILE_REWRITE_LOCK_BLK,
        pg_sys::ShareLock as pg_sys::LOCKMODE,
    );
}

/// Take the exclusive side of the relfile-rewrite lock (a whole-
/// relfile rewrite: build / insert-flush / vacuum-shrink). Blocks
/// until every in-flight reader has finished its [`read_full`], and
/// blocks new readers until [`unlock_relfile_write`]. Idempotent-
/// safe within one transaction (same-xact locks don't self-
/// conflict), so a caller that already holds it (nested rewrite in
/// one tx) does not deadlock.
///
/// # Safety
/// Caller must hold the appropriate heap lock (RowExclusive for the
/// insert flush, ShareUpdateExclusive for VACUUM, AccessExclusive
/// for build/REINDEX) and MUST pair this with exactly one
/// [`unlock_relfile_write`].
#[inline]
pub(crate) unsafe fn lock_relfile_write(rel: pg_sys::Relation) {
    pg_sys::LockPage(
        rel,
        RELFILE_REWRITE_LOCK_BLK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
}

/// Release the exclusive side taken by [`lock_relfile_write`].
#[inline]
pub(crate) unsafe fn unlock_relfile_write(rel: pg_sys::Relation) {
    pg_sys::UnlockPage(
        rel,
        RELFILE_REWRITE_LOCK_BLK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
}

/// True when the relation's persistence is `'p'` (PERMANENT). For
/// unlogged / temp indexes `GenericXLogFinish` skips the WAL
/// record but still writes the modified page back — we don't need
/// to branch on this ourselves.
#[allow(dead_code)]
unsafe fn rel_needs_wal(rel: pg_sys::Relation) -> bool {
    let rd_rel = (*rel).rd_rel;
    !rd_rel.is_null() && (*rd_rel).relpersistence as u8 == pg_sys::RELPERSISTENCE_PERMANENT
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
    pg_sys::LockBuffer(buf, lock_buffer_mode(mode));
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
    pg_sys::LockBuffer(buf, lock_buffer_mode(pg_sys::BUFFER_LOCK_EXCLUSIVE));
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
    let page =
        pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
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
        // P0 (managed-PG readiness): a large in-place relfile rewrite
        // (e.g. an aminsert PreCommit flush of a big index) is many
        // GenericXLog batches of pure CPU + I/O; without polling it is
        // uninterruptible for seconds, so statement_timeout /
        // pg_cancel_backend / platform admin signals are ignored. Poll
        // at the TOP of the outer batch loop, before GenericXLogStart
        // — no buffer content lock is held here, so a cancel unwinds
        // cleanly. (Never poll between RegisterBuffer and Finish.)
        pg_sys::check_for_interrupts!();
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
        let page =
            pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
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
    // H-NEW-2/H-NEW-3 guard: a corrupt/torn/bit-flipped meta page (bad
    // n_vectors, first_blkno, or rows_per_page) would otherwise walk
    // blkno past EOF (ReadBufferExtended(RBM_NORMAL) extends the
    // relation or errors deep in the buffer manager) AND over-allocate
    // `total_bytes` (a palloc bomb sized by the corrupt n_vectors).
    // Bound the chain against the physical relation length up front and
    // raise a clean, actionable ERROR instead. `rows_per_page == 0`
    // would divide-by-zero below, so treat it as corrupt too.
    if rows_per_page == 0 {
        pgrx::ereport!(
            pgrx::PgLogLevel::ERROR,
            pgrx::PgSqlErrorCode::ERRCODE_DATA_CORRUPTED,
            "turbovec: relfile chain has rows_per_page=0 (corrupt meta page)",
            "REINDEX INDEX to rebuild from the heap."
        );
    }
    let pages_needed = (n_vectors as usize).div_ceil(rows_per_page as usize);
    let nblk =
        pg_sys::RelationGetNumberOfBlocksInFork(rel, pg_sys::ForkNumber::MAIN_FORKNUM) as u64;
    let last_needed = first_blkno as u64 + pages_needed as u64;
    if last_needed > nblk {
        pgrx::ereport!(
            pgrx::PgLogLevel::ERROR,
            pgrx::PgSqlErrorCode::ERRCODE_DATA_CORRUPTED,
            format!(
                "turbovec: relfile chain [blk {first_blkno}, +{pages_needed} pages] runs past the relation end ({nblk} blocks) — the meta page's n_vectors/offsets are inconsistent with the physical file (corrupt or truncated index)"
            ),
            "REINDEX INDEX to rebuild from the heap."
        );
    }
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
/// relation, replacing any existing pages. Skips the prepared
/// layout (no blocked chain, no inline codebook) — callers that
/// need cold-scan acceleration should go through
/// [`write_full_with_prepared`] instead.
///
/// Used today only by ambulkdelete's bookkeeping path; the build
/// + commit-flush paths route through `write_full_with_prepared`
/// so backends opening the index skip the per-backend
/// `pack::repack` and Lloyd-Max compute (Phase P).
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured
/// the index isn't being concurrently scanned. ambuild satisfies
/// both; aminsert and ambulkdelete take the heap-tuple lock,
/// which is sufficient for the existing single-writer SPI path
/// and remains so here.
#[allow(clippy::too_many_arguments)]
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
    write_full_inner(
        rel,
        bit_width,
        dim,
        n_vectors,
        packed_codes,
        scales,
        slot_to_id,
        am_version,
        None,
        None,
        None,
    );
}

/// Prepared parts of an [`IdMapIndex`] required for Phase R-2's
/// persisted rotation + Lloyd-Max codebook.
///
/// Phase Q-0 (v7) note: the SIMD-blocked chain is NO LONGER persisted
/// — it's a pure function of the packed codes (`pack::repack`) and is
/// recomputed once per backend at index-open (see
/// `scan::install_whole_index`). So this struct no longer carries
/// `blocked_codes` / `n_blocks`.
///
/// wire v8 (turbovec 1.0.0): the Lloyd-Max codebook AND the rotation
/// are now deterministic functions of `(bit_width, dim)` and no longer
/// persisted — turbovec 1.0.0 derives them at index-open. The only
/// per-index codebook state that must round-trip is the TQ+ calibration
/// pair (`tqplus_shift`, `tqplus_scale`, each `dim` f32), persisted in
/// the repurposed v3 "rotation" chain as `shift ++ scale`. Both come
/// straight off `IdMapIndex` after `prepare()` via the TQ+ getters.
/// (The meta page's inline codebook slot is still stamped with the
/// DERIVED codebook — see `write_full_inner` — for forward-compat /
/// debugging, but is NOT read back.)
pub(crate) struct PreparedParts<'a> {
    /// TQ+ per-coordinate shift (`dim` f32, or empty = identity).
    pub tqplus_shift: &'a [f32],
    /// TQ+ per-coordinate scale (`dim` f32, or empty = identity).
    pub tqplus_scale: &'a [f32],
}

impl PreparedParts<'_> {
    /// Byte length of the TQ+ chain this will persist (`shift ++
    /// scale`, `2*dim*4`), or 0 when TQ+ is identity/absent (either
    /// array empty). Drives `plan_with_blocked`'s chain-sizing arg.
    pub(crate) fn tqplus_chain_len(&self) -> u64 {
        if self.tqplus_shift.is_empty() || self.tqplus_scale.is_empty() {
            return 0;
        }
        (std::mem::size_of_val(self.tqplus_shift) + std::mem::size_of_val(self.tqplus_scale)) as u64
    }

    /// Concatenated little-endian TQ+ chain bytes (`shift ++ scale`),
    /// or empty when identity/absent. Matches [`read_tqplus`]'s layout.
    pub(crate) fn tqplus_chain_bytes(&self) -> Vec<u8> {
        if self.tqplus_shift.is_empty() || self.tqplus_scale.is_empty() {
            return Vec::new();
        }
        let mut out =
            Vec::with_capacity(self.tqplus_chain_len() as usize);
        for &v in self.tqplus_shift.iter().chain(self.tqplus_scale.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// v1.29.1 corruption fix #2 (deferred-flush lost-update): compute
/// the reconciled flush image — the CURRENT on-disk state with this
/// transaction's upserts (`touched_ids`) spliced in — as owned
/// `(codes, scales, ids)` byte/f32/u64 arrays.
///
/// This is the pure core of `reconcile_and_write_flush`, factored out
/// so it is unit-testable without a live cluster. It does NOT touch
/// the relation; the caller reads `disk_*` under the exclusive rewrite
/// lock, calls this, then writes the result back under the same lock.
///
/// ## Why this exists
///
/// The pre-fix flush blindly rewrote the WHOLE relfile from the
/// mutating backend's in-memory `IdMapIndex`, which is a snapshot of
/// the on-disk state *at first-mutation time* plus this backend's
/// adds. If a concurrent VACUUM shrank the relfile (deleted dead
/// rows) between that snapshot and this commit, the blind rewrite
/// RESURRECTED the deleted rows (lost update) — the true source of
/// the "id 0 / duplicate id appears in more than one slot"
/// corruption. Reconciling against the CURRENT disk state at flush
/// preserves VACUUM's deletes AND other committed writers' adds,
/// while still landing this txn's upserts.
///
/// ## Algorithm
///
/// Start from the fresh on-disk arrays. For each `id` in
/// `touched_ids` (deduplicated, last-write-wins) look up its codes /
/// scale in the committing backend's snapshot (`snap_*`, indexed by
/// `snap_id_to_slot`):
/// * if the id is already present on disk (`disk_id_to_slot`),
///   overwrite that slot's codes + scale in place (an UPDATE);
/// * otherwise append a new slot (an INSERT).
///
/// The quantized codes are byte-identical whether encoded against the
/// disk index or the snapshot, because the Lloyd-Max codebook +
/// rotation are a deterministic pure function of `(bit_width, dim)`
/// (and the fixed `ROTATION_SEED`) — both indexes share them — so
/// splicing the snapshot's stored code bytes needs no re-encode.
///
/// `stride` is bytes-per-vector (`dim * bit_width / 8`). Returns
/// `(codes, scales, ids)` sized to the reconciled row count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconcile_flush_image(
    stride: usize,
    disk_codes: &[u8],
    disk_scales: &[f32],
    disk_ids: &[u64],
    snap_codes: &[u8],
    snap_scales: &[f32],
    snap_ids: &[u64],
    touched_ids: &[u64],
) -> (Vec<u8>, Vec<f32>, Vec<u64>) {
    use std::collections::HashMap;

    let mut out_codes = disk_codes.to_vec();
    let mut out_scales = disk_scales.to_vec();
    let mut out_ids = disk_ids.to_vec();

    // disk id -> slot (position in out_*). Mutated as we append.
    let mut disk_id_to_slot: HashMap<u64, usize> = HashMap::with_capacity(disk_ids.len());
    for (slot, &id) in disk_ids.iter().enumerate() {
        disk_id_to_slot.insert(id, slot);
    }
    // snapshot id -> slot (source of the code bytes to splice).
    let mut snap_id_to_slot: HashMap<u64, usize> = HashMap::with_capacity(snap_ids.len());
    for (slot, &id) in snap_ids.iter().enumerate() {
        snap_id_to_slot.insert(id, slot);
    }

    // Dedup touched ids, last occurrence wins (matches the in-memory
    // index's final state for a re-touched id).
    let mut seen: HashMap<u64, ()> = HashMap::with_capacity(touched_ids.len());
    // Iterate in reverse so the FIRST time we see an id is its LAST
    // upsert; skip earlier duplicates.
    let mut ordered: Vec<u64> = Vec::with_capacity(touched_ids.len());
    for &id in touched_ids.iter().rev() {
        if seen.insert(id, ()).is_none() {
            ordered.push(id);
        }
    }

    for &id in &ordered {
        let Some(&snap_slot) = snap_id_to_slot.get(&id) else {
            // The id was touched but is no longer in the snapshot
            // (e.g. removed later in the same txn). Nothing to splice.
            continue;
        };
        let src_codes = &snap_codes[snap_slot * stride..(snap_slot + 1) * stride];
        let src_scale = snap_scales[snap_slot];
        match disk_id_to_slot.get(&id).copied() {
            Some(dst) => {
                // UPDATE in place.
                out_codes[dst * stride..(dst + 1) * stride].copy_from_slice(src_codes);
                out_scales[dst] = src_scale;
                // out_ids[dst] already == id.
            }
            None => {
                // INSERT: append a new slot.
                let dst = out_ids.len();
                out_codes.extend_from_slice(src_codes);
                out_scales.push(src_scale);
                out_ids.push(id);
                disk_id_to_slot.insert(id, dst);
            }
        }
    }

    (out_codes, out_scales, out_ids)
}

/// v1.29.1 corruption fix #2: the reconcile-on-flush write path used
/// by the deferred-commit `aminsert` PreCommit callback
/// (`xact::flush_to_relfile`). Takes the EXCLUSIVE relfile-rewrite
/// lock, RE-READS the current on-disk `(codes, scales, ids)` under
/// it, splices this transaction's upserts (`touched_ids`, sourced
/// from the committing backend's snapshot arrays) onto that fresh
/// state via [`reconcile_flush_image`], and writes the reconciled
/// image back — all under the one held lock so no concurrent VACUUM
/// / flush can slip a change between the re-read and the write.
///
/// This replaces the pre-fix blind whole-relfile overwrite from the
/// backend's stale snapshot, which lost concurrent VACUUM deletes.
///
/// Falls back to a plain full write (from the snapshot) ONLY when the
/// index has NO existing on-disk rows to reconcile against (first
/// build from empty / `n_vectors == 0`) or when the on-disk meta is
/// missing — there is nothing to lose in those cases.
///
/// # Safety
/// Caller holds `RowExclusiveLock` on the heap-side relation (the
/// PreCommit path does). Takes + releases the page-level exclusive
/// rewrite lock internally.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn reconcile_and_write_flush(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    snap_codes: &[u8],
    snap_scales: &[f32],
    snap_ids: &[u64],
    touched_ids: &[u64],
    am_version: u32,
    prepared: PreparedParts<'_>,
) {
    // Take the exclusive rewrite lock for the WHOLE reconcile (read
    // current disk state -> splice -> write). Same-xact re-entry
    // (write_full_inner takes it again) does not self-conflict. The
    // lock manager auto-releases at xact end if anything below
    // longjmps.
    lock_relfile_write(rel);

    let disk_meta = read_meta(rel);
    let stride = (dim as usize) * (bit_width as usize) / 8;

    let (n_vectors, out_codes, out_scales, out_ids) = match disk_meta {
        Some(m) if m.n_vectors > 0 => {
            // Re-read current on-disk chains under the held lock.
            let codes = read_chain(
                rel,
                m.codes_first,
                m.stride_bytes,
                m.rows_per_codes_page,
                m.n_vectors,
            );
            let scales_bytes = read_chain(
                rel,
                m.scales_first,
                std::mem::size_of::<f32>() as u32,
                m.rows_per_scales_page,
                m.n_vectors,
            );
            let ids_bytes = read_chain(
                rel,
                m.ids_first,
                std::mem::size_of::<u64>() as u32,
                m.rows_per_ids_page,
                m.n_vectors,
            );
            let disk_scales: Vec<f32> = scales_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let disk_ids: Vec<u64> = ids_bytes
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            // v1.29.4 corruption fix (deferred-flush GUARD, belt-and-
            // suspenders): before we splice this txn's upserts onto the
            // CURRENT on-disk ids and re-persist them, verify the on-disk
            // ids we just read are a clean bijection. If a PRE-FIX binary
            // (or any other cause) already left an id-0 / duplicate slot on
            // disk, reconciling onto it would ENTRENCH the corruption
            // (re-persist the bad slots as "real" data). Aborting the flush
            // here with an actionable REINDEX hint is retryable; silently
            // re-persisting on-disk corruption is not. The meta-LAST
            // ordering above prevents this binary from CREATING the hole;
            // this guard stops it from PROPAGATING a pre-existing one.
            // GATED to the bijective flat/single kind (`lists == 0`):
            // an IVF index (`lists > 0`) legitimately repeats an external
            // id across cells (soft-assignment to the 2nd..Mth nearest
            // cell), so `first_duplicate_id` would FALSE-POSITIVE on
            // every real IVF index and reject the first INSERT into it.
            // (The in-memory reload degrades IVF->flat on insert, but the
            // on-disk `disk_ids` we read here is still the soft-assigned
            // IVF image, so it must NOT be dup-checked.) Matches the
            // `meta.lists == 0` gate on the insert/read paths.
            if m.lists == 0 {
                if let Some(bad) = crate::index::scan::first_duplicate_id(&disk_ids) {
                    unlock_relfile_write(rel);
                    let idx_name = crate::index::scan::index_relname(rel);
                    pgrx::ereport!(
                        pgrx::PgLogLevel::ERROR,
                        pgrx::PgSqlErrorCode::ERRCODE_DATA_CORRUPTED,
                        format!(
                            "turbovec deferred-flush: refusing to reconcile onto a corrupt .tvim id table in index \"{idx_name}\" (id {bad} appears in more than one slot on disk); aborting this transaction to avoid entrenching the corruption"
                        ),
                        format!(
                            "Run `REINDEX INDEX {idx_name};` to rebuild it from the heap. This on-disk state was left by an older binary (pre-v1.29.4) whose in-place relfile rewrite could tear on `pg_terminate_backend`/cancel; v1.29.4 writes the meta page LAST so new flushes can no longer create it."
                        )
                    );
                }
            }
            let (oc, os, oi) = reconcile_flush_image(
                stride,
                &codes,
                &disk_scales,
                &disk_ids,
                snap_codes,
                snap_scales,
                snap_ids,
                touched_ids,
            );
            let n = oi.len() as u64;
            (n, oc, os, oi)
        }
        _ => {
            // No existing on-disk rows (fresh index) or no meta:
            // nothing to reconcile against, write the snapshot as-is.
            (
                snap_ids.len() as u64,
                snap_codes.to_vec(),
                snap_scales.to_vec(),
                snap_ids.to_vec(),
            )
        }
    };

    write_full_inner(
        rel,
        bit_width,
        dim,
        n_vectors,
        &out_codes,
        &out_scales,
        &out_ids,
        am_version,
        Some(prepared),
        None,
        None,
    );

    unlock_relfile_write(rel);
}

/// IVF coarse-quantizer parts persisted in v4 (IVF-1). Written by
/// [`write_full_with_prepared_ivf`] when an index is built
/// `WITH (lists > 0)`.
///
/// `lists` is `nlist`; `coarse_centroids` is the row-major
/// `lists * dim` f32 coarse codebook in the **rotated** space
/// (cells live in the same space the fine quantizer and query
/// use); `cell_dir_bytes` is the packed cell directory
/// (`lists * CellEntry::ENCODED_BYTES` little-endian bytes,
/// produced by `ivf::CellDirectory::encode`).
pub(crate) struct IvfParts<'a> {
    pub lists: u32,
    pub coarse_centroids: &'a [f32],
    pub cell_dir_bytes: &'a [u8],
    /// Phase F-2: stamp the meta page as a ColBERT / multivector
    /// token index (`mark_colbert()`, wire v5) when `true`. A normal
    /// single-vector IVF build leaves this `false`, so the meta stays
    /// byte-identical to the v4 wire format.
    pub colbert: bool,
}

/// Phase G-2a graph parts, persisted by
/// [`write_full_with_prepared_graph`] when an index is built
/// `WITH (graph = true)`.
///
/// `offsets_bytes` / `neighbors_bytes` are the two concatenated CSR
/// sub-chains produced by `graph::GraphAdjacency::encode_offsets` /
/// `encode_neighbors`; `entry_point` is the slot id greedy search
/// starts from (`graph::build_vamana`'s second return value).
pub(crate) struct GraphParts<'a> {
    pub offsets_bytes: &'a [u8],
    pub neighbors_bytes: &'a [u8],
    pub entry_point: u32,
}

/// High-level helper: write a fully-built `IdMapIndex` plus the
/// prepared SIMD-blocked layout and inline codebook to the
/// relation. Backends opening the resulting v2 index skip both
/// `pack::repack` and the Lloyd-Max codebook compute on first
/// scan — the headline Phase P win.
///
/// # Safety
///
/// Same constraints as [`write_full`]: caller holds an exclusive
/// relation lock.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn write_full_with_prepared(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
    prepared: PreparedParts<'_>,
) {
    write_full_inner(
        rel,
        bit_width,
        dim,
        n_vectors,
        packed_codes,
        scales,
        slot_to_id,
        am_version,
        Some(prepared),
        None,
        None,
    );
}

/// High-level helper: same as [`write_full_with_prepared`] but also
/// persists the v4 IVF chains (coarse centroids + cell directory).
/// The caller must have already reordered `packed_codes` / `scales`
/// / `slot_to_id` into cell-contiguous order matching `ivf`'s cell
/// directory (IVF-1 build path in `build.rs`).
///
/// # Safety
///
/// Same constraints as [`write_full`]: caller holds an exclusive
/// relation lock.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn write_full_with_prepared_ivf(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
    prepared: PreparedParts<'_>,
    ivf: IvfParts<'_>,
) {
    write_full_inner(
        rel,
        bit_width,
        dim,
        n_vectors,
        packed_codes,
        scales,
        slot_to_id,
        am_version,
        Some(prepared),
        Some(ivf),
        None,
    );
}

/// High-level helper: same as [`write_full_with_prepared`] but also
/// persists the v6 graph adjacency chain (Phase G-2a). The caller
/// (the graph build path in `build.rs`) has already built the
/// `IdMapIndex`/codes/scales/ids exactly as a flat build would (a
/// graph node's vector storage is identical to a flat index's row)
/// and separately run `graph::build_vamana` over the same corpus to
/// get the adjacency + entry point.
///
/// # Safety
///
/// Same constraints as [`write_full`]: caller holds an exclusive
/// relation lock.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn write_full_with_prepared_graph(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
    prepared: PreparedParts<'_>,
    graph: GraphParts<'_>,
) {
    write_full_inner(
        rel,
        bit_width,
        dim,
        n_vectors,
        packed_codes,
        scales,
        slot_to_id,
        am_version,
        Some(prepared),
        None,
        Some(graph),
    );
}

/// Layout state handed from [`write_packed_phase`] to
/// [`write_blocked_phase_and_meta`]: the planned chain offsets and
/// page counts for the row-major code / scales / ids chains, plus
/// the build-time configuration that phase 2 needs to re-plan with
/// the blocked + rotation byte sizes filled in.
///
/// **Phase W-2 (v1.7.0) introduced this split. It was reverted in
/// v1.7.1** — for the regression
/// analysis. The struct and its two helper functions
/// (`write_packed_phase`, `write_blocked_phase_and_meta`) are
/// kept as parked dead-code APIs because a future Phase W-3
/// attempt (e.g. streaming `pack::repack`) may want to revisit a
/// split-write design from a different angle. They have no
/// callers in pg_turbovec after the revert.
#[allow(dead_code)]
pub(crate) struct PackedPhaseLayout {
    /// `bit_width` from the build config; phase 2 re-plans with it.
    pub bit_width: u8,
    /// `dim` from the build config; phase 2 re-plans with it.
    pub dim: u32,
    /// `n_vectors` from the build config; phase 2 re-plans with it.
    pub n_vectors: u64,
    /// `am_version` to stamp on the final meta page.
    pub am_version: u32,
    /// First block number of the row-major codes chain (always 1
    /// today; carried explicitly so the assertion in phase 2 has
    /// something to check against).
    pub codes_first_blkno: pg_sys::BlockNumber,
    /// Number of pages in the codes chain.
    pub codes_n_pages: u32,
    /// First block number of the per-vector scales chain.
    pub scales_first_blkno: pg_sys::BlockNumber,
    /// Number of pages in the scales chain.
    pub scales_n_pages: u32,
    /// First block number of the slot-to-id chain.
    pub ids_first_blkno: pg_sys::BlockNumber,
    /// Number of pages in the ids chain.
    pub ids_n_pages: u32,
}

/// Phase 1 of the split write: serialise `(packed_codes, scales,
/// slot_to_id)` into relfile pages. The meta page is **not** yet
/// written — by deferring the meta to phase 2 we keep the
/// “meta page is the atomic-complete signal” invariant: a crash
/// between phases leaves a relfile whose block 0 is either
/// zero-filled (fresh build) or carries the previous index’s meta
/// (REINDEX); either way `ambeginscan` either treats it as empty
/// or sees the old layout. The new chains, while persisted, are
/// unreferenced until phase 2 commits the new meta.
///
/// Returns a [`PackedPhaseLayout`] handle that phase 2 must be
/// called with. Between the two calls the caller may safely drop
/// the in-memory `packed_codes` Vec (e.g. via
/// `IdMapIndex::take_packed_codes()`).
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured
/// the index isn't being concurrently scanned (same constraint as
/// [`write_full`] and [`write_full_with_prepared`]).
///
/// **Parked dead code as of v1.7.1** — Phase W-2's split-write
/// design was reverted; see the [`PackedPhaseLayout`] doc comment.
#[allow(dead_code)]
pub(crate) unsafe fn write_packed_phase(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
) -> PackedPhaseLayout {
    assert_eq!(slot_to_id.len() as u64, n_vectors);
    assert_eq!(scales.len() as u64, n_vectors);
    if n_vectors > 0 {
        let stride = MetaPageData::codes_stride(bit_width, dim) as usize;
        debug_assert_eq!(
            packed_codes.len(),
            (n_vectors as usize) * stride,
            "packed_codes length must equal n_vectors * (dim/8)*bit_width",
        );
    }

    // Plan the codes/scales/ids chains. blocked/rotation are not
    // known yet (their byte sizes depend on prepare_eager output);
    // pass 0/0/0 so phase 1 doesn't reserve space for them. The
    // resulting codes_first / scales_first / ids_first are the
    // same in this plan as in the final phase-2 plan because
    // `MetaPageData::plan_with_blocked` lays out blocked/rotation
    // AFTER ids — the row-major chain offsets are stable across
    // re-planning.
    let plan = MetaPageData::plan_with_blocked(bit_width, dim, n_vectors, am_version, 0, 0, 0);
    let layout = PackedPhaseLayout {
        bit_width,
        dim,
        n_vectors,
        am_version,
        codes_first_blkno: plan.codes_first,
        codes_n_pages: plan.codes_count,
        scales_first_blkno: plan.scales_first,
        scales_n_pages: plan.scales_count,
        ids_first_blkno: plan.ids_first,
        ids_n_pages: plan.ids_count,
    };

    if n_vectors == 0 {
        // No row chains to write. Phase 2 will plan the empty
        // layout, write the meta page, and (if needed) extend
        // block 0.
        return layout;
    }

    // Pre-extend so the chain offsets we just planned are valid
    // pages on disk before any reader could see them. Phase 2 may
    // extend further (for blocked + rotation) but never shrinks.
    let row_chain_total = 1 + layout.codes_n_pages + layout.scales_n_pages + layout.ids_n_pages;
    let existing_before = nblocks(rel);
    if existing_before < row_chain_total {
        extend_to(rel, row_chain_total);
    }

    write_chain_at(
        rel,
        layout.codes_first_blkno,
        packed_codes,
        plan.stride_bytes,
        plan.rows_per_codes_page,
        n_vectors,
    );

    // Scales chain. Reinterpret f32 slice as raw bytes.
    let scales_bytes: &[u8] =
        std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), std::mem::size_of_val(scales));
    write_chain_at(
        rel,
        layout.scales_first_blkno,
        scales_bytes,
        std::mem::size_of::<f32>() as u32,
        plan.rows_per_scales_page,
        n_vectors,
    );

    // Ids chain.
    let ids_bytes: &[u8] = std::slice::from_raw_parts(
        slot_to_id.as_ptr().cast::<u8>(),
        std::mem::size_of_val(slot_to_id),
    );
    write_chain_at(
        rel,
        layout.ids_first_blkno,
        ids_bytes,
        std::mem::size_of::<u64>() as u32,
        plan.rows_per_ids_page,
        n_vectors,
    );

    layout
}

/// Phase 2 of the split write: materialise the SIMD-blocked
/// layout, the persisted rotation matrix, and the inline
/// codebook, then write the meta page LAST. Caller must have
/// invoked [`write_packed_phase`] previously and dropped its
/// in-memory `packed_codes` Vec (or kept it; the bytes on disk
/// are already independent of the in-memory copy). After this
/// call returns the relfile is complete and visible to
/// `ambeginscan`.
///
/// Writing the meta page LAST gives us the same crash-safety
/// guarantee the standard PG index AMs (hash, gist) rely on:
/// a crash anywhere before the meta-page WAL record commits
/// leaves block 0 in its previous state (zero-filled for a
/// fresh build, the previous meta for REINDEX). Readers see
/// the index as either empty/legacy (zero magic) or as the
/// previous version, never as a half-written new layout.
///
/// The phase-2 plan re-derives the chain offsets with
/// `MetaPageData::plan_with_blocked` filled in for the actual
/// blocked/rotation byte sizes. We assert the codes / scales /
/// ids offsets match phase 1's plan; they must, because
/// `plan_with_blocked` always lays the row-major chains out
/// first, before any blocked/rotation chain.
///
/// # Safety
///
/// Same constraints as [`write_full`].
///
/// **Parked dead code as of v1.7.1** — Phase W-2's split-write
/// design was reverted; see the [`PackedPhaseLayout`] doc comment.
#[allow(dead_code)]
pub(crate) unsafe fn write_blocked_phase_and_meta(
    rel: pg_sys::Relation,
    layout: PackedPhaseLayout,
    prepared: Option<PreparedParts<'_>>,
) {
    let (_n_blocks_blocked, rotation_bytes) = match &prepared {
        Some(p) => (0u32, p.tqplus_chain_len()),
        None => (0, 0),
    };
    let mut meta = MetaPageData::plan_with_blocked(
        layout.bit_width,
        layout.dim,
        layout.n_vectors,
        layout.am_version,
        /* blocked_bytes = */ 0,
        /* n_blocks_blocked = */ 0,
        rotation_bytes,
    );
    if let Some(p) = &prepared {
        // wire v8: the inline codebook slot is stamped with the DERIVED
        // codebook (a pure fn of (bit_width, dim)); it is forward-compat
        // /debug only and is NOT read back (the reader derives it too).
        let _ = &p; // p carries only TQ+ now
        let (deriv_boundaries, deriv_centroids) =
            turbovec::expected_codebook(layout.bit_width as usize, layout.dim as usize);
        meta.set_codebook(&deriv_centroids, &deriv_boundaries);
    }

    if layout.n_vectors > 0 {
        // The phase-1 layout must agree with what plan_with_blocked
        // computes now, modulo the new blocked/rotation chains. If
        // this ever fails, plan_with_blocked has changed its chain
        // ordering and we'd be writing blocked bytes into stale
        // codes/scales/ids ranges — catch it here, not on a corrupt
        // index three months from now.
        debug_assert_eq!(meta.codes_first, layout.codes_first_blkno);
        debug_assert_eq!(meta.codes_count, layout.codes_n_pages);
        debug_assert_eq!(meta.scales_first, layout.scales_first_blkno);
        debug_assert_eq!(meta.scales_count, layout.scales_n_pages);
        debug_assert_eq!(meta.ids_first, layout.ids_first_blkno);
        debug_assert_eq!(meta.ids_count, layout.ids_n_pages);
    }

    let new_total = meta.total_blocks().max(1);
    let existing_before = nblocks(rel);
    if existing_before < new_total {
        extend_to(rel, new_total);
    }

    if layout.n_vectors > 0 {
        if let Some(p) = &prepared {
            // Phase Q-0 (v7): the SIMD-blocked chain is no longer
            // persisted. wire v8: the (repurposed) chain now carries
            // the TQ+ arrays (shift ++ scale), not a rotation matrix.
            let tqplus_buf = p.tqplus_chain_bytes();
            if !tqplus_buf.is_empty() {
                write_chain_at(
                    rel,
                    meta.rotation_first,
                    &tqplus_buf,
                    1,
                    crate::index::page::PAYLOAD_BYTES as u32,
                    tqplus_buf.len() as u64,
                );
            }
        }
    }

    // Meta page LAST: this is the atomic-complete signal. A crash
    // before this WAL record commits leaves block 0 in its
    // previous state.
    write_meta(rel, &meta);

    // Phase L hardening (item 3): release any trailing blocks left
    // over from a larger previous layout. After a shrinking
    // REINDEX or `ambulkdelete` consolidation, without this the
    // dropped rows linger as orphan pages until the next REINDEX.
    let existing_after = nblocks(rel);
    if existing_after > new_total {
        pg_sys::RelationTruncate(rel, new_total);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn write_full_inner(
    rel: pg_sys::Relation,
    bit_width: u8,
    dim: u32,
    n_vectors: u64,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
    am_version: u32,
    prepared: Option<PreparedParts<'_>>,
    ivf: Option<IvfParts<'_>>,
    graph: Option<GraphParts<'_>>,
) {
    // v1.7.1 revert: restored to v1.6.0's single-pass batched-
    // GenericXLog flow. Phase W-2 (v1.7.0) split this into
    // `write_packed_phase` + `write_blocked_phase_and_meta` so
    // ambuild could drop `packed_codes` between the two halves.
    // Validation on `meh` at 10 M × 1536-d showed the split made
    // the build 53% slower (5052 → 7748 s) and used 2.7 GiB of
    // swap (vs 0 in v1.6.0) without actually lowering peak RSS
    // (the "freed" heap pages just migrated to pinned shared
    // buffers, which `ps -o rss` still counts). See
    // `benches/results/phase_w_2_validate_meh_10m_2026_05_27.json`
    // and  for the full analysis.
    // HARD PERSIST-SITE GUARD (v1.28.4): `slot_to_id` is the SINGLE
    // SOURCE OF TRUTH for the on-disk row count. These are `assert_eq!`
    // (NOT `debug_assert_eq!`) on purpose — they run in release builds
    // too, so a row-count drift (`n_vectors` disagreeing with the
    // actual ids/scales arrays) ALWAYS aborts the transaction here
    // rather than persisting a meta page that over-reads the ids chain
    // into zeroed trailing slots (which reloads as the reported
    // "duplicate ids (id 0 in more than one slot)" corruption). Do NOT
    // downgrade these to `debug_assert_eq!`. The insert-flush path
    // (`xact::flush_to_relfile`) now derives `n_vectors` from
    // `slot_to_id.len()` so it can't trip these; the guard stays as
    // defense in depth for the build/IVF/graph callers.
    assert_eq!(
        slot_to_id.len() as u64,
        n_vectors,
        "turbovec relfile write: slot_to_id.len() must equal n_vectors"
    );
    assert_eq!(
        scales.len() as u64,
        n_vectors,
        "turbovec relfile write: scales.len() must equal n_vectors"
    );

    // v1.29.1 corruption fix #4 (LAST-LINE-OF-DEFENSE persist guard):
    // for the bijective flat/single kind (NOT IVF, whose soft-assign
    // legitimately repeats ids across cells; NOT graph, whose
    // slot_to_id is 0..n synthetic slot ids), a duplicate external id
    // OR an id-0 slot is exactly the reported corruption signature
    // ("id N appears in more than one slot"; a real CTID never encodes
    // to 0, so an id-0 slot is always zeroed trailing bytes). If any
    // upstream race were EVER to hand a bad slot_to_id to the persist
    // path, abort the transaction HERE rather than write it to disk
    // and serve it. This is a `error!` (release-active) not a
    // `debug_assert!` on purpose — it must hold in production. It is
    // O(n) over the ids (a HashSet build) and runs once per whole-
    // relfile rewrite, dwarfed by the page I/O it guards.
    // ponytail: O(n) HashSet per flush; acceptable next to the ~200MB
    // rewrite it guards. Skip for IVF/graph where dups/synthetic ids
    // are legal.
    if ivf.is_none() && graph.is_none() && n_vectors > 0 {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::with_capacity(slot_to_id.len());
        for (slot, &id) in slot_to_id.iter().enumerate() {
            if id == 0 {
                error!(
                    "turbovec relfile write: refusing to persist id 0 at slot {slot} \
                     (a real CTID never encodes to 0 — this is the zeroed-trailing-slot \
                     corruption signature). Aborting the transaction to protect the index."
                );
            }
            if !seen.insert(id) {
                error!(
                    "turbovec relfile write: refusing to persist DUPLICATE id {id} \
                     (appears in more than one slot; slot {slot}) for a bijective \
                     flat/single index. Aborting the transaction to protect the index."
                );
            }
        }
    }
    // v1.29.1 corruption fix: take the EXCLUSIVE side of the relfile-
    // rewrite lock for the ENTIRE multi-page rewrite below. This
    // blocks any concurrent reader ([`read_full`] / [`read_ids_only`],
    // which take the ShareLock) from observing the meta page + chains
    // in a half-rewritten (torn) state. Released at the bottom of
    // this function on success, and auto-released by the lock manager
    // if anything below `error!`s (longjmp) or the tx aborts. Same-
    // xact re-entry (e.g. VACUUM already holds it, or a build) does
    // not self-conflict. See `lock_relfile_write`.
    lock_relfile_write(rel);

    // Phase Q-0 (v7): the SIMD-blocked chain is no longer persisted
    // (it's recomputed from the packed codes at index-open), so we
    // always plan a ZERO-length blocked chain regardless of
    // `prepared`. wire v8: the (repurposed v3) chain now carries the
    // TQ+ arrays (shift ++ scale), NOT a rotation matrix — the
    // rotation is derived from (bit_width, dim) by turbovec 1.0.0.
    let rotation_bytes = match &prepared {
        Some(p) => p.tqplus_chain_len(),
        None => 0,
    };
    let mut meta = MetaPageData::plan_with_blocked(
        bit_width,
        dim,
        n_vectors,
        am_version,
        /* blocked_bytes = */ 0,
        /* n_blocks_blocked = */ 0,
        rotation_bytes,
    );
    if let Some(p) = &prepared {
        // wire v8: the inline codebook slot is stamped with the DERIVED
        // codebook (a pure fn of (bit_width, dim)); it is forward-compat
        // /debug only and is NOT read back (the reader derives it too).
        let _ = &p; // p carries only TQ+ now
        let (deriv_boundaries, deriv_centroids) =
            turbovec::expected_codebook(bit_width as usize, dim as usize);
        meta.set_codebook(&deriv_centroids, &deriv_boundaries);
    }
    // v4 IVF: lay the coarse-centroid + cell-directory chains out
    // after the rotation chain and stamp `lists`. Must happen before
    // `total_blocks()` / `write_meta` so the planned offsets are on
    // the meta page readers see. A `lists == 0` (or `None`) build is
    // a no-op here, leaving the meta byte-identical to the v3 flat
    // layout modulo the version byte.
    if let Some(iv) = &ivf {
        let coarse_bytes = std::mem::size_of_val(iv.coarse_centroids) as u64;
        let cell_dir_bytes = iv.cell_dir_bytes.len() as u64;
        meta.set_ivf_chains(iv.lists, coarse_bytes, cell_dir_bytes);
        // Phase F-2: stamp the ColBERT / multivector kind (wire v5).
        // Done AFTER set_ivf_chains so the chain layout is identical
        // to a single-vector IVF index — only the discriminator byte
        // changes. A non-colbert build leaves the meta at v4.
        if iv.colbert {
            meta.mark_colbert();
        }
    }
    // v6 graph (Phase G-2a): lay the adjacency chain out after EVERY
    // prior chain (including the IVF chains above, though a graph
    // build never sets `lists > 0` in practice — `set_graph_chain`
    // computes its offset from whatever chains preceded it, so this
    // is correct either way) and stamp `kind = KIND_GRAPH` + bump
    // `version` to 6. A `None` graph (the ordinary flat/IVF/ColBERT
    // build) is a no-op here, leaving the meta unchanged.
    if let Some(g) = &graph {
        meta.set_graph_chain(
            g.offsets_bytes.len() as u64,
            g.neighbors_bytes.len() as u64,
            g.entry_point,
        );
    }
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

    // TEST-ONLY: reproduce the PRE-FIX meta-first ordering on demand so a
    // #[pg_test] can show the fail-before (see WRITE_META_FIRST_FOR_TEST).
    #[cfg(any(test, feature = "pg_test"))]
    let _meta_first_for_test = WRITE_META_FIRST_FOR_TEST.with(|c| c.get());
    #[cfg(any(test, feature = "pg_test"))]
    if _meta_first_for_test {
        write_meta(rel, &meta);
    }

    // v1.29.4 corruption fix (VACUUM-INDEPENDENT torn-write on interrupt):
    // Write the row chains FIRST and the meta page LAST. This mirrors the
    // build path's `write_blocked_phase_and_meta` crash-safety invariant.
    //
    // The bug this fixes: `write_chain_at` polls `check_for_interrupts!()`
    // at the top of every GenericXLog batch, so a `pg_terminate_backend`
    // (SIGTERM -> ProcDiePending) or a query-cancel during a deferred
    // PreCommit flush can longjmp (FATAL) out of the chain write. The
    // per-batch `GenericXLogFinish` page changes are PHYSICAL (WAL-logged,
    // NOT rolled back on the transaction abort). With the OLD order
    // (write_meta first, then chains), an interrupt after the meta commit
    // but before the ids chain finished left `meta.n_vectors = N` pointing
    // at an ids chain whose appended (or in-place) region was still the
    // zero-initialised page bytes -> "id 0 appears in more than one slot"
    // (a contiguous zeroed run in the middle/tail of the ids chain, with
    // `count_matches` still true because the chain is physically N slots
    // long). No VACUUM required. Evidence: a 730-slot zeroed run == the
    // tail of one ids page, appearing immediately after the first
    // writer-restart (`pg_terminate_backend`) in a 1.84M-row upsert repro.
    //
    // By writing every chain first and the meta LAST, an interrupt during
    // the chain writes leaves block 0 (the meta) UNCHANGED, so readers /
    // the next reconcile-flush still observe the previous, fully-valid
    // `n_vectors` + chains. Any pages we extended/partially wrote past the
    // old count are simply unreferenced until the next full rewrite (or
    // the `RelationTruncate` below). The meta write itself is a single
    // GenericXLog page op with no interrupt poll, so it cannot tear.
    // Truncate stays AFTER the meta so a shrink never leaves the meta
    // referencing a page past EOF.
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

        // TEST-ONLY: simulate a pg_terminate_backend/cancel longjmp landing
        // AFTER codes+scales but BEFORE the ids chain (the real interrupt
        // window). With meta-LAST the meta page is still the pre-flush one,
        // so a reload observes the previous valid n_vectors + ids; with the
        // pre-fix meta-FIRST order the meta already advanced -> torn ids.
        #[cfg(any(test, feature = "pg_test"))]
        if INJECT_TEAR_BEFORE_IDS.with(|c| c.get()) {
            unlock_relfile_write(rel);
            return;
        }

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

        // v3: persisted rotation matrix. Stored as a flat-byte
        // chain (stride = 1, rows_per_page = PAYLOAD_BYTES). The
        // consumer
        // (`turbovec::IdMapIndex::from_id_map_parts_with_prepared`)
        // pre-fills the rotation `OnceLock` from these bytes and
        // skips the per-backend QR decomposition that dominates
        // warm-scan latency at large dim.
        //
        // Phase Q-0 (v7): the SIMD-blocked chain is NO LONGER
        // written here — it's recomputed from the packed codes at
        // index-open via `pack::repack` (halving the on-disk
        // footprint). wire v8: the (repurposed v3) chain now carries
        // the TQ+ arrays (shift ++ scale), NOT a rotation matrix; the
        // codebook + rotation are derived from (bit_width, dim) at
        // index-open by turbovec 1.0.0. The inline codebook slot is
        // still stamped (set_codebook above) for forward-compat.
        if let Some(p) = &prepared {
            let tqplus_buf = p.tqplus_chain_bytes();
            if !tqplus_buf.is_empty() {
                write_chain_at(
                    rel,
                    meta.rotation_first,
                    &tqplus_buf,
                    1,
                    crate::index::page::PAYLOAD_BYTES as u32,
                    tqplus_buf.len() as u64,
                );
            }
        }

        // v4 IVF: coarse centroids (f32, rotated space) + cell
        // directory. Same flat-byte chain shape as blocked/rotation.
        // Only written when lists > 0 (the chains were planned by
        // set_ivf_chains above).
        if let Some(iv) = &ivf {
            if iv.lists > 0 && !iv.coarse_centroids.is_empty() {
                let coarse_buf: &[u8] = std::slice::from_raw_parts(
                    iv.coarse_centroids.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(iv.coarse_centroids),
                );
                write_chain_at(
                    rel,
                    meta.coarse_first,
                    coarse_buf,
                    1,
                    crate::index::page::PAYLOAD_BYTES as u32,
                    coarse_buf.len() as u64,
                );
            }
            if iv.lists > 0 && !iv.cell_dir_bytes.is_empty() {
                write_chain_at(
                    rel,
                    meta.cell_dir_first,
                    iv.cell_dir_bytes,
                    1,
                    crate::index::page::PAYLOAD_BYTES as u32,
                    iv.cell_dir_bytes.len() as u64,
                );
            }
        }

        // v6 graph (Phase G-2a): adjacency chain, two concatenated
        // flat byte sub-chains (offsets then neighbors). Only written
        // when this is a graph build (the chain was planned by
        // set_graph_chain above).
        if let Some(g) = &graph {
            if !g.offsets_bytes.is_empty() || !g.neighbors_bytes.is_empty() {
                let mut combined =
                    Vec::with_capacity(g.offsets_bytes.len() + g.neighbors_bytes.len());
                combined.extend_from_slice(g.offsets_bytes);
                combined.extend_from_slice(g.neighbors_bytes);
                write_chain_at(
                    rel,
                    meta.graph_first,
                    &combined,
                    1,
                    crate::index::page::PAYLOAD_BYTES as u32,
                    combined.len() as u64,
                );
            }
        }
    }

    // Step 3 — Phase L hardening (item 3): release any trailing
    // blocks left over from a larger previous layout. This
    // matters after a shrinking REINDEX or `ambulkdelete`
    // consolidation; without it, dropped rows linger on disk as
    // orphan pages until the next REINDEX. `RelationTruncate`
    // emits its own WAL record (`XLOG_SMGR_TRUNCATE`) and is
    // crash-safe.
    // v1.29.4 corruption fix (meta-LAST): the row chains are fully
    // written; commit the meta page NOW so `n_vectors` only ever advances
    // to N once the chain bytes backing all N slots are durably in place.
    write_meta(rel, &meta);

    let existing_after = nblocks(rel);
    if existing_after > new_total {
        pg_sys::RelationTruncate(rel, new_total);
    }

    // v1.29.1 corruption fix: the whole-relfile rewrite is complete
    // and internally consistent; release the exclusive rewrite lock
    // so waiting readers proceed against the fully-written chains.
    unlock_relfile_write(rel);
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
        let page =
            pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
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
    let page =
        pg_sys::GenericXLogRegisterBuffer(state, dst_buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
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
    // v1.29.1 corruption fix: same whole-chain read/rewrite
    // serialization as `read_full`. `read_ids_only` is a reader too
    // (turbovec_check, scan-path visibility). VACUUM's call already
    // holds the exclusive side (same xact => no self-conflict). The
    // ShareLock auto-releases if `read_chain` `error!`s or the tx
    // aborts, and is released explicitly below on success.
    lock_relfile_read(rel);
    // v1.29.1 corruption fix (stale-meta race): RE-READ the meta page
    // UNDER the lock so the ids-chain offsets / n_vectors we read
    // with are the CURRENT relfile's, not the caller's pre-lock
    // snapshot. A concurrent flush/vacuum completing between the
    // caller's `read_meta` and this lock left `turbovec_check`
    // (and any other read_ids_only caller) reading the NEW chain
    // with STALE offsets/count — an over-read into zeroed trailing
    // slots that surfaced as a transient "id 0 appears in more than
    // one slot" false positive. Fall back to the caller's meta only
    // if the meta page vanished (relation being dropped).
    let cur = read_meta(rel).unwrap_or(*meta);
    if cur.n_vectors == 0 {
        unlock_relfile_read(rel);
        return Vec::new();
    }
    let ids_bytes = read_chain(
        rel,
        cur.ids_first,
        std::mem::size_of::<u64>() as u32,
        cur.rows_per_ids_page,
        cur.n_vectors,
    );
    unlock_relfile_read(rel);
    ids_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// Gather a set of contiguous slot ranges out of the codes chain
/// into one compact, gapless `Vec<u8>` THROUGH THE BUFFER MANAGER,
/// reading ONLY the pages backing the requested ranges. This is the
/// out-of-core (Phase B-1) gather fallback used when the relfile
/// can't be mmapped (e.g. a freshly-built index in the same
/// transaction before its dirty buffers are flushed to the segment
/// file). It reads each range's slots page-by-page via
/// `ReadBufferExtended`, so the resident codes are bounded by the
/// gathered ranges (`probes * cell_size`), not the whole chain.
///
/// `ranges` are `(slot_start, slot_count)` pairs into the chain.
/// `stride` is the per-slot byte width; `rows_per_page` the number
/// of slots per chain page. The output is the concatenation of each
/// range's bytes in the given order, length `sum(count) * stride`.
///
/// # Safety
///
/// Caller must hold a relation reference for the duration.
pub(crate) unsafe fn gather_codes_ranges(
    rel: pg_sys::Relation,
    first_blkno: u32,
    stride: u32,
    rows_per_page: u32,
    ranges: &[(u64, u64)],
) -> Vec<u8> {
    let stride_us = stride as usize;
    let rpp = rows_per_page as u64;
    if stride_us == 0 || rpp == 0 {
        return Vec::new();
    }
    let total_slots: u64 = ranges.iter().map(|&(_, c)| c).sum();
    let mut out = Vec::<u8>::with_capacity(total_slots as usize * stride_us);
    for &(start, count) in ranges {
        let end = start + count;
        let mut slot = start;
        while slot < end {
            let page_idx = slot / rpp;
            let in_page = (slot % rpp) as usize;
            let blkno = first_blkno + page_idx as u32;
            let slots_left_on_page = rpp - slot % rpp;
            let take_slots = slots_left_on_page.min(end - slot) as usize;
            let buf = read_block(rel, blkno, /*exclusive=*/ false);
            let src = page_data(buf).add(in_page * stride_us);
            let pos = out.len();
            out.set_len(pos + take_slots * stride_us);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(pos), take_slots * stride_us);
            pg_sys::UnlockReleaseBuffer(buf);
            slot += take_slots as u64;
        }
    }
    out
}

/// Read the scales chain (only) into a `Vec<f32>`. The out-of-core
/// IVF path (Phase B-1) keeps scales resident (4 B/vec, small) so
/// the per-query gather pulls only the codes off the mmap; reading
/// scales alone avoids slurping the O(n) codes chain that
/// `read_full` would.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_scales_only(rel: pg_sys::Relation, meta: &MetaPageData) -> Vec<f32> {
    if meta.n_vectors == 0 {
        return Vec::new();
    }
    let scales_bytes = read_chain(
        rel,
        meta.scales_first,
        std::mem::size_of::<f32>() as u32,
        meta.rows_per_scales_page,
        meta.n_vectors,
    );
    scales_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Rewrite the meta page in place with a smaller `n_vectors` and
/// a bumped `am_version`. Used by `ambulkdelete`'s FLAT path after
/// walking the page chains and swap-removing dead rows. The chain
/// offsets (`codes_first`, `scales_first`, `ids_first`,
/// `rows_per_*_page`, `stride_bytes`) are preserved so existing
/// on-disk row positions remain valid for survivors.
///
/// **IVF indexes do NOT use this path.** Swap-remove moves the global
/// last slot into a deleted hole, which crosses cell boundaries and
/// breaks the cell directory's contiguity invariant — that is the
/// Phase E-2 degradation landmine. The IVF VACUUM path instead
/// tombstones dead slots via [`write_tombstones_and_meta`], leaving
/// `n_vectors` and the cell layout untouched. This function is kept
/// for the flat (`lists == 0`) path and as a documented last-resort
/// safety net; if ever called on an IVF index it preserves `lists`
/// and the IVF chain offsets and flips `ivf_degraded` so the
/// degradation is observable (`is_degraded()` true) rather than
/// silent.
///
/// `*_count` fields are recomputed from the new `n_vectors` for
/// consistency, but readers ignore them — they walk the chain by
/// `n_vectors` only. The next `write_full` (build / aminsert
/// commit-time flush) re-plans from scratch, packing the chains
/// tightly.
///
/// **Phase P / v2 invariant:** the prepared SIMD-blocked chain
/// (`blocked_first` / `blocked_count` / `blocked_bytes`) and the
/// inline codebook (`codebook_n_levels` + `centroids` +
/// `boundaries`) are *not* updated by the swap-remove pass; the
/// blocked layout was derived from the old `packed_codes` and no
/// longer matches the post-swap order. Leaving it on disk would
/// give readers wrong search results until the next `REINDEX`.
///
/// We blank out the prepared-layout fields here so
/// `MetaPageData::has_prepared_layout()` returns false and
/// readers fall back to per-backend `pack::repack`. The on-disk
/// blocked-chain pages are left where they are (mid-file gaps)
/// to avoid having to re-extend the relation; the next
/// `write_full_with_prepared` re-packs everything tight.
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
        // Phase P: invalidate the prepared layout. The blocked
        // chain on disk is now stale (old slot order); readers
        // see has_prepared_layout() == false and fall back to
        // per-backend repack until the next full rewrite. We
        // bump version to VERSION so the meta page itself stays
        // current (it's still v3-shaped on the wire).
        blocked_first: 0,
        blocked_count: 0,
        blocked_bytes: 0,
        n_blocks_blocked: 0,
        codebook_n_levels: 0,
        centroids: [0.0; crate::index::page::MAX_CODEBOOK_LEVELS],
        boundaries: [0.0; crate::index::page::MAX_CODEBOOK_LEVELS - 1],
        // Phase R-2: the rotation chain is a deterministic
        // function of `(dim, ROTATION_SEED)` and survives row
        // shuffles, so we keep the existing rotation_first /
        // rotation_count / rotation_dim intact. Readers still
        // get the persisted rotation back even after a vacuum.
        //
        // IVF (Phase E-2): the swap-remove path only runs for FLAT
        // indexes now (`lists == 0`); the IVF path tombstones instead
        // of swap-removing, preserving cell contiguity (see
        // `vacuum.rs`). If this is ever called on an IVF index as a
        // last-resort safety net, we DELIBERATELY KEEP `lists` and the
        // coarse/cell chain offsets so the degradation is observable
        // (`index_was_ivf()` stays true) and flip `ivf_degraded` so
        // `is_degraded()` reports the latency cliff to the operator,
        // rather than silently blanking the IVF identity. The scan
        // path treats `ivf_degraded` as "take the flat fallback".
        ivf_degraded: old.index_was_ivf(),
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
/// `(meta, packed_codes, scales, slot_to_id)` where `meta` is a
/// snapshot of the meta page RE-READ atomically under the shared
/// relfile-rewrite lock alongside the chains.
///
/// v1.29.1 corruption fix: the passed-in `meta_hint` is used ONLY
/// for the `n_vectors == 0` fast path. All chain reads — AND the
/// `meta` returned to the caller — come from a snapshot re-read
/// under the lock. This closes the stale-meta race: a caller reads
/// the meta page separately (in `ambeginscan` / `aminsert`) and
/// then calls `read_full`; without re-reading under the lock, a
/// concurrent rewrite completing in that window would leave the
/// caller reading the NEW chains with STALE offsets / `n_vectors`,
/// over-reading into zeroed or adjacent-chain pages (surfacing as
/// the reported "duplicate ids ... id 0 appears in more than one
/// slot"). Callers MUST size the reconstructed index from the
/// RETURNED `meta`, not their pre-read hint.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_full_consistent(
    rel: pg_sys::Relation,
    meta_hint: &MetaPageData,
) -> (MetaPageData, Vec<u8>, Vec<f32>, Vec<u64>) {
    if meta_hint.n_vectors == 0 {
        return (*meta_hint, Vec::new(), Vec::new(), Vec::new());
    }
    // v1.29.1 corruption fix: serialize the whole (meta + chains)
    // read against a concurrent whole-relfile rewrite (insert-flush
    // / vacuum-shrink). The heavyweight ShareLock is released
    // explicitly on every return path below, and auto-released by
    // the lock manager if any read `error!`s (longjmp) or the tx
    // aborts — so a longjmp cannot leak it. See `lock_relfile_read`.
    lock_relfile_read(rel);
    // Re-read the meta page UNDER the lock so offsets + n_vectors we
    // read the chains with are the CURRENT relfile's, not the
    // caller's pre-lock snapshot.
    let meta = match read_meta(rel) {
        Some(m) => m,
        None => {
            unlock_relfile_read(rel);
            return (*meta_hint, Vec::new(), Vec::new(), Vec::new());
        }
    };
    if meta.n_vectors == 0 {
        unlock_relfile_read(rel);
        return (meta, Vec::new(), Vec::new(), Vec::new());
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
    unlock_relfile_read(rel);
    let scales: Vec<f32> = scales_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ids: Vec<u64> = ids_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    (meta, codes, scales, ids)
}

/// Back-compat wrapper: [`read_full_consistent`] but dropping the
/// returned meta, for the handful of callers (tests, colbert,
/// ambulkdelete bookkeeping) that only need the buffers and whose
/// own `meta` snapshot is already taken under a lock that excludes
/// a concurrent rewrite (AccessExclusive/ShareUpdateExclusive). New
/// scan/insert code should prefer `read_full_consistent` and size
/// the index from the RETURNED meta.
///
/// # Safety
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_full(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> (Vec<u8>, Vec<f32>, Vec<u64>) {
    let (_m, codes, scales, ids) = read_full_consistent(rel, meta);
    (codes, scales, ids)
}

/// Read the prepared SIMD-blocked layout from a pre-v7 blocked
/// chain. Returns the raw byte buffer; caller is expected to know
/// `n_blocks` from the meta page.
///
/// Returns an empty vector when no blocked chain is present
/// (signalled by `meta.blocked_bytes == 0`). Phase Q-0 (v7) NO
/// LONGER persists the blocked chain — the install path recomputes
/// it via `pack::repack` — so on a v7 index this always returns
/// empty. Kept (`#[allow(dead_code)]`) for the pre-v7 round-trip
/// tests and as documentation of the retired chain shape.
///
/// # Safety
///
/// Caller must hold a relation reference.
#[allow(dead_code)]
pub(crate) unsafe fn read_blocked(rel: pg_sys::Relation, meta: &MetaPageData) -> Vec<u8> {
    if meta.blocked_bytes == 0 || meta.blocked_first == 0 {
        return Vec::new();
    }
    read_chain(
        rel,
        meta.blocked_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        meta.blocked_bytes,
    )
}

/// Read the persisted TQ+ calibration arrays from the (repurposed
/// v3 "rotation") chain. Returns `(tqplus_shift, tqplus_scale)`, each
/// a `dim`-length `Vec<f32>`, ready to feed straight into
/// `turbovec::TurboQuantIndex::from_parts` (or the IdMapIndex twin).
///
/// wire v8 (turbovec 1.0.0): the codebook + rotation are derived from
/// `(bit_width, dim)` and no longer persisted; the only per-index
/// codebook state that must round-trip is the TQ+ pair, laid out in
/// this chain as `tqplus_shift (dim f32) ++ tqplus_scale (dim f32)`
/// = `2*dim*4` bytes. `meta.rotation_dim` is the `dim` the pair was
/// built for (`rotation_count`/`rotation_first` locate the chain).
///
/// Returns `(empty, empty)` for an empty index or one persisted with
/// identity/absent TQ+ (`meta.rotation_count == 0`); in that case the
/// index behaves as un-calibrated TQ (from_parts fills identity TQ+).
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_tqplus(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> (Vec<f32>, Vec<f32>) {
    if meta.rotation_count == 0 || meta.rotation_first == 0 || meta.rotation_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let dim = meta.rotation_dim as usize;
    let n_elems = 2 * dim; // shift ++ scale
    let n_bytes = (n_elems * std::mem::size_of::<f32>()) as u64;
    let bytes = read_chain(
        rel,
        meta.rotation_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        n_bytes,
    );
    debug_assert_eq!(bytes.len(), n_bytes as usize);
    let all: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (shift, scale) = all.split_at(dim);
    (shift.to_vec(), scale.to_vec())
}

/// Read the v4 coarse-centroid chain into a row-major `lists * dim`
/// `Vec<f32>` (rotated space). Returns an empty vector for flat
/// (v3, or v4 `lists == 0`) indexes.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_coarse_centroids(rel: pg_sys::Relation, meta: &MetaPageData) -> Vec<f32> {
    if meta.lists == 0 || meta.coarse_first == 0 || meta.coarse_count == 0 {
        return Vec::new();
    }
    let n_elems = (meta.lists as usize) * (meta.dim as usize);
    let n_bytes = (n_elems * std::mem::size_of::<f32>()) as u64;
    let bytes = read_chain(
        rel,
        meta.coarse_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        n_bytes,
    );
    debug_assert_eq!(bytes.len(), n_bytes as usize);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read the v4 cell directory into an [`crate::index::ivf::CellDirectory`].
/// Returns `None` for flat (v3, or v4 `lists == 0`) indexes.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_cell_directory(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> Option<crate::index::ivf::CellDirectory> {
    use crate::index::ivf::{CellDirectory, CellEntry};
    if meta.lists == 0 || meta.cell_dir_first == 0 || meta.cell_dir_count == 0 {
        return None;
    }
    let lists = meta.lists as usize;
    let n_bytes = (lists * CellEntry::ENCODED_BYTES) as u64;
    let bytes = read_chain(
        rel,
        meta.cell_dir_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        n_bytes,
    );
    debug_assert_eq!(bytes.len(), n_bytes as usize);
    Some(CellDirectory::decode(&bytes, lists))
}

/// Read the v6 graph adjacency chain (Phase G-2a) into a
/// [`crate::index::graph::GraphAdjacency`]. Returns `None` for a
/// non-graph index (`kind != KIND_GRAPH`) or an empty (0-row) graph
/// build.
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_graph_adjacency(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> Option<crate::index::graph::GraphAdjacency> {
    use crate::index::graph::GraphAdjacency;
    if !meta.has_graph() {
        return None;
    }
    let n = meta.n_vectors as usize;
    let total_bytes = meta.graph_offsets_bytes + meta.graph_neighbors_bytes;
    let combined = read_chain(
        rel,
        meta.graph_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        total_bytes,
    );
    debug_assert_eq!(combined.len(), total_bytes as usize);
    let split = meta.graph_offsets_bytes as usize;
    if split > combined.len() {
        error!(
            "turbovec relfile: corrupt graph adjacency chain (offsets_bytes exceeds chain length)"
        );
    }
    let (offsets_bytes, neighbors_bytes) = combined.split_at(split);
    match GraphAdjacency::decode(offsets_bytes, neighbors_bytes, n) {
        Ok(g) => Some(g),
        Err(e) => error!("turbovec relfile: corrupt graph adjacency chain: {}", e),
    }
}

/// Read the v4 E-2 tombstone bitmap (one bit per slot, LSB-first;
/// bit set ⇒ slot is dead). Returns an empty vector when the index
/// has no tombstone chain (`tombstone_bytes == 0`), i.e. nothing has
/// been deleted yet — callers treat that as "all slots live".
///
/// # Safety
///
/// Caller must hold a relation reference.
pub(crate) unsafe fn read_tombstones(rel: pg_sys::Relation, meta: &MetaPageData) -> Vec<u8> {
    if meta.tombstone_bytes == 0 || meta.tombstone_first == 0 || meta.tombstone_count == 0 {
        return Vec::new();
    }
    read_chain(
        rel,
        meta.tombstone_first,
        1,
        crate::index::page::PAYLOAD_BYTES as u32,
        meta.tombstone_bytes,
    )
}

/// Append-or-rewrite the per-slot tombstone bitmap chain at the END
/// of the relation, then rewrite the meta page LAST so it points at
/// the new chain. Used by the IVF VACUUM path (`vacuum.rs`) to mark
/// dead slots WITHOUT moving any rows, so the cell-contiguous layout
/// and the cell directory stay valid and the index keeps serving IVF
/// scans (`has_ivf()` stays true).
///
/// `bitmap` is `ceil(n_vectors / 8)` bytes. The chain is laid out
/// after every existing chain (it's the last thing in the file), so
/// it never disturbs the codes/scales/ids/coarse/cell-dir chains.
/// On a repeat vacuum we reuse the existing chain blocks in place
/// when they're big enough, extending only if the bitmap grew (it
/// never does for a fixed `n_vectors`, but a re-cluster could change
/// it). `new_am_version` drives the per-backend cache freshness
/// check so the next scan re-reads the bitmap.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock.
pub(crate) unsafe fn write_tombstones_and_meta(
    rel: pg_sys::Relation,
    old: &MetaPageData,
    bitmap: &[u8],
    new_am_version: u32,
) {
    let tombstone_bytes = bitmap.len() as u64;
    let tombstone_count = MetaPageData::byte_pages_needed(tombstone_bytes);
    // The tombstone chain is the last chain in the file. Its first
    // block is whatever the existing tombstone chain already used
    // (reuse in place) or, on the first vacuum, the first free block
    // after every other chain -- INCLUDING the graph adjacency chain
    // (Phase G-2b: a graph index's tombstone chain must be placed
    // AFTER `graph_count`, or its computed offset collides with the
    // already-persisted graph chain, silently corrupting it on the
    // very next incremental `aminsert` -- a real bug found and fixed
    // during G-2b's own test-writing; `graph_count` is `0` for
    // every non-graph kind, so this is a no-op for flat/IVF/ColBERT).
    let after_all_other_chains = 1
        + old.codes_count
        + old.scales_count
        + old.ids_count
        + old.blocked_count
        + old.rotation_count
        + old.coarse_count
        + old.cell_dir_count
        + old.graph_count;
    let tombstone_first = if old.tombstone_first != 0 {
        old.tombstone_first
    } else {
        after_all_other_chains
    };

    // Ensure the relation has room for the chain before we write the
    // meta page that references it.
    let needed_total = tombstone_first + tombstone_count;
    if nblocks(rel) < needed_total {
        extend_to(rel, needed_total);
    }

    // Write the bitmap as a flat byte chain (stride 1, PAYLOAD_BYTES
    // rows/page), same shape as the blocked / rotation / coarse
    // chains.
    if tombstone_bytes > 0 {
        write_chain_at(
            rel,
            tombstone_first,
            bitmap,
            1,
            crate::index::page::PAYLOAD_BYTES as u32,
            tombstone_bytes,
        );
    }

    // Meta page LAST (atomic-complete signal). We KEEP every IVF
    // field intact (lists, coarse/cell chains) so the index stays
    // IVF; we only stamp the tombstone chain offsets + the bumped
    // am_version. ivf_degraded stays false: a tombstoned IVF index is
    // healthy, just carrying some dead space until the next REINDEX.
    let new_meta = MetaPageData {
        am_version: new_am_version,
        tombstone_first,
        tombstone_count,
        tombstone_bytes,
        ivf_degraded: false,
        ..*old
    };
    write_meta(rel, &new_meta);
}

/// Test-only helper: forge the on-disk meta-page version byte to
/// `version`, leaving every other byte untouched. Used by the
/// upgrade-path #[pg_test]s in `src/lib.rs` to simulate an index
/// built under a pre-Phase-R-2 wire format without having to keep
/// old binaries around.
///
/// Unlike a real REINDEX, this leaves the v2/v3 chain offsets on
/// disk; that's intentional, because the legacy detection path in
/// `ambeginscan` only inspects `MetaPageData::version`, and we
/// want the test to exercise that exact predicate.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured
/// the relation has at least one block (the meta page). Only
/// compiled into the cargo-test / pgrx-test build.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) unsafe fn force_meta_version(rel: pg_sys::Relation, version: u8) {
    assert!(
        nblocks(rel) > 0,
        "force_meta_version: relation has no blocks"
    );
    let buf = read_block(rel, META_BLKNO, /*exclusive=*/ true);
    let state = pg_sys::GenericXLogStart(rel);
    let page =
        pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
    // Byte layout: PG_HEADER (24) + magic (4) + version (1).
    // Patch only the version byte; the surrounding chain offsets
    // and codebook stay valid so the decoder accepts the page.
    let version_byte = page.cast::<u8>().add(PAGE_HEADER_BYTES + 4);
    *version_byte = version;
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// Test-only helper: blank the v4 IVF meta fields (`lists`,
/// `coarse_first/count`, `cell_dir_first/count`) in place, leaving
/// the version byte at 4 and every other field untouched. This
/// simulates a vacuum-degraded IVF index: `write_meta_shrink_in_place`
/// blanks exactly these fields after a swap-remove (cell contiguity
/// breaks), so the index keeps its v4 codes/scales/ids but reports
/// `has_ivf() == false` and must be scanned via the flat fallback.
///
/// The five u32 fields are packed contiguously at payload offset
/// `v4_base = 224` (see `MetaPageData::decode`); on the page they
/// start at `PAGE_HEADER_BYTES + 224`. Zeroing 5 * 4 = 20 bytes
/// clears all of them.
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured the
/// relation has at least one block. Only compiled into the test
/// build.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) unsafe fn force_meta_blank_ivf(rel: pg_sys::Relation) {
    assert!(
        nblocks(rel) > 0,
        "force_meta_blank_ivf: relation has no blocks"
    );
    let buf = read_block(rel, META_BLKNO, /*exclusive=*/ true);
    let state = pg_sys::GenericXLogStart(rel);
    let page =
        pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
    // v4 IVF fields: payload offset 224, 5 contiguous u32s = 20 bytes.
    const V4_BASE: usize = 224;
    let ivf_fields = page.cast::<u8>().add(PAGE_HEADER_BYTES + V4_BASE);
    std::ptr::write_bytes(ivf_fields, 0u8, 20);
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// Test-only helper: set the v4 E-2 `ivf_degraded` flag byte to 1 in
/// place, KEEPING `lists` and every IVF chain offset intact. This
/// simulates the degrade-to-flat *safety net* (the new
/// `write_meta_shrink_in_place` behaviour for an IVF index): the
/// index still reports `index_was_ivf() == true` and `lists > 0`,
/// but `is_degraded()` is now true and the scan takes the flat
/// fallback while emitting the throttled WARNING. The byte lives at
/// payload offset `e2_base = 244` (page offset `PAGE_HEADER_BYTES +
/// 244`).
///
/// # Safety
///
/// Caller must hold an exclusive relation lock and have ensured the
/// relation has at least one block. Only compiled into the test build.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) unsafe fn force_meta_set_degraded(rel: pg_sys::Relation) {
    assert!(
        nblocks(rel) > 0,
        "force_meta_set_degraded: relation has no blocks"
    );
    let buf = read_block(rel, META_BLKNO, /*exclusive=*/ true);
    let state = pg_sys::GenericXLogStart(rel);
    let page =
        pg_sys::GenericXLogRegisterBuffer(state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);
    // E-2 fields begin at payload offset 244 (ivf_degraded is the
    // first byte).
    const E2_BASE: usize = 244;
    let degraded_byte = page.cast::<u8>().add(PAGE_HEADER_BYTES + E2_BASE);
    *degraded_byte = 1u8;
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
}

#[cfg(test)]
mod reconcile_tests {
    use super::reconcile_flush_image;

    // v1.29.1 corruption fix #2 (deferred-flush lost-update) gate.
    // These run under plain `cargo test --lib` (no PostgreSQL cluster).
    // They assert the pure reconcile core preserves a concurrent
    // VACUUM's deletes while landing this txn's upserts. A blind
    // whole-snapshot overwrite (the pre-fix behaviour) fails
    // `vacuum_delete_is_not_resurrected`.

    // 1 byte/vec stride for these tiny fixtures (codes are opaque here).
    fn v(id: u64) -> u8 {
        // deterministic "code" for an id, so we can assert which slot
        // holds which id's data.
        (id & 0xff) as u8
    }

    #[test]
    fn upsert_new_id_appends_onto_current_disk() {
        // Disk currently has ids [10, 11]. Txn inserted id 12.
        let disk_codes = vec![v(10), v(11)];
        let disk_scales = vec![10.0f32, 11.0];
        let disk_ids = vec![10u64, 11];
        // Snapshot (this backend's index) has [10, 11, 12].
        let snap_codes = vec![v(10), v(11), v(12)];
        let snap_scales = vec![10.0f32, 11.0, 12.0];
        let snap_ids = vec![10u64, 11, 12];
        let touched = vec![12u64];
        let (c, s, i) = reconcile_flush_image(
            1,
            &disk_codes,
            &disk_scales,
            &disk_ids,
            &snap_codes,
            &snap_scales,
            &snap_ids,
            &touched,
        );
        assert_eq!(i, vec![10, 11, 12]);
        assert_eq!(c, vec![v(10), v(11), v(12)]);
        assert_eq!(s, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn vacuum_delete_is_not_resurrected() {
        // THE regression. This backend loaded disk when it had
        // [10, 11, 12, 13] (its snapshot), then inserted 99. Meanwhile
        // a concurrent VACUUM deleted 11 and 12, so CURRENT disk is
        // [10, 13]. Reconcile must produce [10, 13, 99] — NOT resurrect
        // 11/12. A blind snapshot overwrite would write
        // [10, 11, 12, 13, 99], resurrecting the deleted rows.
        let disk_codes = vec![v(10), v(13)];
        let disk_scales = vec![10.0f32, 13.0];
        let disk_ids = vec![10u64, 13];
        let snap_codes = vec![v(10), v(11), v(12), v(13), v(99)];
        let snap_scales = vec![10.0f32, 11.0, 12.0, 13.0, 99.0];
        let snap_ids = vec![10u64, 11, 12, 13, 99];
        let touched = vec![99u64];
        let (c, s, i) = reconcile_flush_image(
            1,
            &disk_codes,
            &disk_scales,
            &disk_ids,
            &snap_codes,
            &snap_scales,
            &snap_ids,
            &touched,
        );
        assert_eq!(
            i,
            vec![10, 13, 99],
            "vacuum-deleted 11 and 12 must stay deleted"
        );
        assert_eq!(c, vec![v(10), v(13), v(99)]);
        assert_eq!(s, vec![10.0, 13.0, 99.0]);
        // No duplicate ids, no resurrection.
        assert!(!i.contains(&11));
        assert!(!i.contains(&12));
    }

    #[test]
    fn on_conflict_update_overwrites_in_place_no_dup() {
        // Disk has [10, 11]. Txn ON-CONFLICT-updated id 11 (new codes)
        // and inserted id 12. Reconcile: 11 updated in place (no dup),
        // 12 appended.
        let disk_codes = vec![v(10), v(11)];
        let disk_scales = vec![10.0f32, 11.0];
        let disk_ids = vec![10u64, 11];
        // Snapshot's slot for 11 carries NEW code 0xAB, scale 111.0.
        let snap_codes = vec![v(10), 0xAB, v(12)];
        let snap_scales = vec![10.0f32, 111.0, 12.0];
        let snap_ids = vec![10u64, 11, 12];
        let touched = vec![11u64, 12u64];
        let (c, s, i) = reconcile_flush_image(
            1,
            &disk_codes,
            &disk_scales,
            &disk_ids,
            &snap_codes,
            &snap_scales,
            &snap_ids,
            &touched,
        );
        assert_eq!(i, vec![10, 11, 12]);
        assert_eq!(
            c,
            vec![v(10), 0xAB, v(12)],
            "id 11's codes updated in place"
        );
        assert_eq!(s, vec![10.0, 111.0, 12.0]);
        // exactly one slot for id 11 (no duplicate).
        assert_eq!(i.iter().filter(|&&x| x == 11).count(), 1);
    }

    #[test]
    fn duplicate_touched_id_last_write_wins() {
        // id 12 touched twice in the txn (insert then re-add). The
        // snapshot holds its final state; reconcile must land exactly
        // one slot for it.
        let disk_codes: Vec<u8> = vec![v(10)];
        let disk_scales = vec![10.0f32];
        let disk_ids = vec![10u64];
        let snap_codes = vec![v(10), 0xCD];
        let snap_scales = vec![10.0f32, 12.5];
        let snap_ids = vec![10u64, 12];
        let touched = vec![12u64, 12u64, 12u64];
        let (c, s, i) = reconcile_flush_image(
            1,
            &disk_codes,
            &disk_scales,
            &disk_ids,
            &snap_codes,
            &snap_scales,
            &snap_ids,
            &touched,
        );
        assert_eq!(i, vec![10, 12]);
        assert_eq!(c, vec![v(10), 0xCD]);
        assert_eq!(s, vec![10.0, 12.5]);
        assert_eq!(i.iter().filter(|&&x| x == 12).count(), 1);
    }
}
