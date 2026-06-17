//! Phase R-3 \u2014 mmap-based reads of the relfile's static regions.
//!
//! ## Why
//!
//! v1.4.0's warm-scan profile (`docs/RECALL.md \u00a72.5`) showed PG's
//! buffer manager dominating warm-scan time on the dbpedia-1M
//! corpus: `ReadBufferExtended` 37%, `WaitReadBuffers` 29%,
//! `mdreadv` / `pread` 28%. The 1.5 GB on-disk index doesn't fit
//! in the bench's 512 MiB `shared_buffers`, so each cache miss
//! re-pulls touched pages through `pread` -> `_copy_to_iter` ->
//! `__memmove_avx_unaligned_erms` into a buffer-pool slot.
//!
//! Mmap-`MAP_PRIVATE` of the relfile lets us read those bytes
//! straight from the OS page cache without going through PG's
//! buffer manager. The OS handles caching for us; we skip the
//! `BufTableLookup` / pin / lock / `pread`-into-shared_buffers
//! triplet on every page touch.
//!
//! ## What we mmap
//!
//! Only the **deterministic-after-`ambuild`** regions:
//!
//! * **Persisted SIMD-blocked codes chain** (\u2248 768 MiB at
//!   1 M \u00d7 1536-d \u00d7 4-bit) \u2014 the bulk of the I/O.
//! * **Persisted rotation matrix chain** (\u2248 9 MiB at dim 1536) \u2014
//!   was the lazy QR hotspot pre-Phase-R-2; reading from mmap
//!   keeps the win.
//! * **Inline codebook** in the meta page \u2014 64 bytes; reading
//!   via mmap or via the buffer manager is the same cost, but
//!   doing both off the same map keeps the API uniform.
//!
//! Codes / scales / ids chains stay on the `ReadBufferExtended`
//! path. Phase O-1's `ambulkdelete` swap-removes those chains
//! in-place under `ShareUpdateExclusiveLock`; the buffer manager
//! is the canonical reader on the same rows. Mmap'ing them would
//! introduce torn-page semantics we don't need: the static
//! regions above are written exactly once at `ambuild` (or fully
//! rewritten at the next `aminsert` commit-flush, which bumps
//! `am_version` and invalidates every backend's mmap'd cache
//! entry).
//!
//! ## Layout reality
//!
//! Each chain page has the standard 24-byte PG `PageHeaderData`
//! prefix followed by `PAYLOAD_BYTES = 8168` bytes of useful
//! payload. The chain bytes are therefore **not contiguous** in
//! the file; an mmap'd view has 24-byte gaps every 8168 bytes.
//! Since `turbovec::search` wants a contiguous `&[u8]`, we walk
//! the chain pages and copy the payload into a `Vec<u8>` once
//! per cache-fill. The win over the buffer-manager path is:
//!
//! 1. No `BufTableLookup` / pin / lock per page.
//! 2. No extra memcpy from OS-page-cache -> shared_buffers slot
//!    -> chain `Vec` (mmap reads go OS-page-cache -> chain `Vec`
//!    in one copy).
//! 3. No shared_buffers pressure: the static-region pages don't
//!    get cached twice (once in OS page cache, once in
//!    shared_buffers) on hosts where the index is bigger than
//!    `shared_buffers`.
//!
//! Going truly zero-copy from the SIMD kernel would require
//! either dropping the page headers from the on-disk static
//! chains (wire-format change) or teaching `turbovec::search`
//! to walk a per-page iterator. The Cow-based borrowed API on
//! the fork (`from_id_map_parts_with_prepared_borrowed`) is
//! wired up and ready for that follow-up; for now it accepts
//! `Cow::Owned` for the chains and reserves the borrowed
//! variant for embedders whose on-disk layout is already
//! contiguous.
//!
//! ## Isolation contract
//!
//! Reading from mmap with relaxed consistency vs PG's buffer
//! manager is correct because the index AM's contract with the
//! executor is approximate-by-design:
//!
//! 1. **Heap visibility is the source of truth.** We return
//!    TIDs; the executor calls `heap_fetch` which checks
//!    transaction visibility. An mmap'd codes chain that lags a
//!    just-committed update can return TIDs for rows that have
//!    been deleted in the heap; the visibility filter rejects
//!    them. New rows visible to the snapshot but not yet in the
//!    mmap'd image are missed; the next scan's `am_version` bump
//!    invalidates the cache and re-mmaps.
//! 2. **`xs_recheckorderby = true`** stays asserted in
//!    `amgettuple`. The executor recomputes the ORDER BY
//!    expression from the heap tuple, correcting any ranking
//!    error introduced by reading from a slightly stale mmap.
//! 3. **Locking.** Scans hold `AccessShareLock`, aminsert holds
//!    `RowExclusiveLock`, ambulkdelete holds
//!    `ShareUpdateExclusiveLock`. AccessShare is compatible
//!    with both writers; the mmap path adds no new locking.
//! 4. **Cache invalidation.** The cache key includes the
//!    relation's `relfilenode`; REINDEX bumps it and forces a
//!    fresh mmap. Within one relfile, every commit that mutated
//!    the index bumps `am_version`; the next `cache::lookup`
//!    sees the mismatch and re-installs the entry, dropping
//!    the old `Mmap` handle.
//!
//! See `docs/ARCHITECTURE.md` \u00a7 \"Index AM \u00b7 mmap isolation
//! contract\" for the full argument and worked examples.

use std::fs::File;
use std::path::PathBuf;

use memmap2::{Mmap, MmapOptions};
use pgrx::pg_sys;

use crate::index::page::{
    MetaPageData, BLCKSZ, MAX_CODEBOOK_LEVELS, META_BLKNO, PAGE_HEADER_BYTES, PAYLOAD_BYTES,
};

/// Owns the relfile mapping for a backend-cache entry's
/// lifetime. Stored alongside the `Arc<RwLock<IdMapIndex>>` in
/// `cache.rs::Entry::mmap` so dropping the entry unmaps after
/// the index that borrowed from it has been freed.
///
/// Currently we copy the chain bytes off the map at construction
/// (see `read_static_chains`), so the index does not literally
/// borrow into `inner`. The `Mmap` is still kept alive for two
/// reasons:
///
/// 1. **Future zero-copy work.** The borrowed-cache constructor
///    on the fork is wired to take `Cow::Borrowed(&'static [u8])`
///    pointing at this mapping, once the on-disk static-region
///    chains are made contiguous.
/// 2. **Isolation marker.** Holding a snapshot of the relfile
///    bytes from cache-fill time makes the "this entry's view of
///    the index is from `am_version = N`" contract concrete; if
///    a writer rewrites the relfile, our `Mmap` keeps pointing
///    at the unlinked inode (or the pre-rewrite block contents)
///    and any future borrowed-path reads see a consistent
///    snapshot until the cache invalidates.
#[allow(dead_code)] // `inner` retained for the lifetime contract; see doc.
pub(crate) struct StaticRegionsMap {
    /// File mapping. `Drop` unmaps via `munmap(2)` when the
    /// cache entry is evicted.
    inner: Mmap,
}

impl StaticRegionsMap {
    /// Open the relfile's segment-0 file (`base/<dbid>/<relfilenode>`)
    /// read-only and `mmap` it `MAP_PRIVATE`. Returns `None` if
    /// the file isn't accessible (relation not yet flushed by
    /// PG's smgr layer, mid-REINDEX rename race, etc.) so the
    /// caller can fall back to the buffer-manager read path.
    ///
    /// # Safety
    ///
    /// Caller must hold a relation reference for the duration
    /// of the call; the mapping itself is independent of any PG
    /// buffer pin (we open our own RO fd). The returned
    /// `StaticRegionsMap` may outlive the original relation
    /// reference, but PG's relfile lifecycle (kept until no
    /// longer referenced by any snapshot) plus our own cache
    /// invalidation on `relfilenode` change keeps the mapping
    /// pointing at valid data until it's dropped.
    unsafe fn open(rel: pg_sys::Relation) -> Option<Self> {
        let path = relfile_path(rel)?;
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => return None,
        };
        // memmap2 defaults to MAP_PRIVATE for the read-only
        // mapping returned by `MmapOptions::map`. That matches
        // the brief: we don't want write-back semantics. We do
        // not request MAP_POPULATE \u2014 memmap2 doesn't expose it
        // portably and the kernel will fault pages in lazily as
        // the SIMD kernel touches them, which is what we want.
        let mmap = match MmapOptions::new().map(&file) {
            Ok(m) => m,
            Err(_) => return None,
        };
        Some(Self { inner: mmap })
    }

    /// Public (crate) opener for the out-of-core IVF path: map the
    /// relfile RO so [`Self::gather_slot_ranges`] can fault the
    /// probed cells' code pages on demand. Same `MAP_PRIVATE`
    /// read-only mapping as the static-regions path; the caller
    /// colocates the returned map inside the `OocIvfIndex` so it
    /// lives for the cache entry's lifetime.
    ///
    /// # Safety
    ///
    /// Caller must hold a relation reference for the duration of the
    /// call; the mapping is independent of any PG buffer pin.
    pub(crate) unsafe fn open_for_ooc(rel: pg_sys::Relation) -> Option<Self> {
        Self::open(rel)
    }

    /// Return the data region of block `blkno`, public for the OOC
    /// path's sanity re-decode of the meta page off the mapping.
    pub(crate) fn page_data_pub(&self, blkno: u32) -> Option<&[u8]> {
        self.page_data(blkno)
    }

    /// Return the data region of block `blkno` (the bytes after
    /// the 24-byte PG page header). Returns `None` if the block
    /// is past the mapped file's tail.
    fn page_data(&self, blkno: u32) -> Option<&[u8]> {
        let off = (blkno as usize)
            .checked_mul(BLCKSZ)?
            .checked_add(PAGE_HEADER_BYTES)?;
        let end = (blkno as usize).checked_mul(BLCKSZ)?.checked_add(BLCKSZ)?;
        if end > self.inner.len() {
            return None;
        }
        Some(&self.inner[off..off + PAYLOAD_BYTES])
    }

    /// Gather a set of contiguous slot ranges out of a
    /// uniform-stride chain into one compact, gapless `Vec<u8>`,
    /// copying ONLY the requested slots' bytes off the mmap. This
    /// is the out-of-core (Phase B-1) primitive: instead of
    /// reading the whole `n_vectors * stride` chain (`read_chain`),
    /// it touches only the pages backing the probed cells, so the
    /// resident set is bounded by the gathered ranges (the OS page
    /// cache holds the faulted-in pages; cold ones fault on demand).
    ///
    /// `ranges` are `(slot_start, slot_count)` pairs into the chain
    /// (a cell's contiguous `[code_offset, code_offset + n_vectors)`
    /// range from the cell directory). `stride` is the per-slot byte
    /// width; `rows_per_page` the number of slots per chain page.
    /// The output is the concatenation of each range's bytes in the
    /// order given, length `sum(count) * stride`. Returns `None` if
    /// any range runs off the end of the mapping (corrupt index or
    /// post-truncate race) so the caller can fall back to the
    /// whole-index load path.
    pub(crate) fn gather_slot_ranges(
        &self,
        first_blkno: u32,
        stride: u32,
        rows_per_page: u32,
        ranges: &[(u64, u64)],
    ) -> Option<Vec<u8>> {
        let stride = stride as usize;
        let rpp = rows_per_page as usize;
        if stride == 0 || rpp == 0 {
            return Some(Vec::new());
        }
        let total_slots: u64 = ranges.iter().map(|&(_, c)| c).sum();
        let mut out = Vec::<u8>::with_capacity((total_slots as usize).checked_mul(stride)?);
        for &(start, count) in ranges {
            let mut slot = start;
            let end = start.checked_add(count)?;
            while slot < end {
                // Page that holds this slot, and the slot's offset
                // within that page's payload.
                let page_idx = slot / rpp as u64;
                let in_page = (slot % rpp as u64) as usize;
                let blkno = first_blkno.checked_add(u32::try_from(page_idx).ok()?)?;
                let payload = self.page_data(blkno)?;
                // How many slots remain on this page (don't run past
                // its `rows_per_page` window or the requested range).
                let slots_left_on_page = (rpp - in_page) as u64;
                let take_slots = slots_left_on_page.min(end - slot) as usize;
                let byte_off = in_page * stride;
                let byte_len = take_slots * stride;
                if byte_off + byte_len > payload.len() {
                    return None;
                }
                out.extend_from_slice(&payload[byte_off..byte_off + byte_len]);
                slot += take_slots as u64;
            }
        }
        Some(out)
    }

    /// Walk a chain starting at `first_blkno` with payload
    /// `bytes_per_full_page = rows_per_page * stride`, copying
    /// the payload into a freshly-allocated contiguous `Vec<u8>`
    /// of length `n_vectors * stride`. Returns `None` if the
    /// chain runs off the end of the mapping (corrupt index or
    /// post-truncate race).
    fn read_chain(
        &self,
        first_blkno: u32,
        stride: u32,
        rows_per_page: u32,
        n_vectors: u64,
    ) -> Option<Vec<u8>> {
        let total_bytes = (n_vectors as usize).checked_mul(stride as usize)?;
        let mut out = Vec::<u8>::with_capacity(total_bytes);
        if total_bytes == 0 {
            return Some(out);
        }
        let bytes_per_full_page = (rows_per_page as usize) * (stride as usize);
        let mut remaining = total_bytes;
        let mut blkno = first_blkno;
        while remaining > 0 {
            let payload = self.page_data(blkno)?;
            let take = bytes_per_full_page.min(remaining);
            out.extend_from_slice(&payload[..take]);
            remaining -= take;
            blkno = blkno.checked_add(1)?;
        }
        Some(out)
    }
}

/// Materialised static-region bytes copied off an mmap. Owned
/// so they can be moved into the borrowed-cache constructor as
/// `Cow::Owned`. The accompanying `StaticRegionsMap` is stored
/// separately on the cache entry to enforce drop ordering.
pub(crate) struct StaticRegionsBytes {
    pub blocked_codes: Vec<u8>,
    pub n_blocks: usize,
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub rotation: Vec<f32>,
}

/// Load the deterministic-after-`ambuild` regions through a
/// single `mmap` of the relfile. Returns `None` if mmap isn't
/// available (e.g. on a tablespace whose backing FS doesn't
/// support shared mappings, or if the relfile has been unlinked
/// mid-call) so the caller can fall back to the buffer-manager
/// read path.
///
/// The returned `StaticRegionsMap` is the lifetime owner; the
/// returned `StaticRegionsBytes` are independent owned `Vec`s
/// (one-time copy off the mapping, see module doc for why).
/// Both are stored on the cache entry so they live as long as
/// the `Arc<RwLock<IdMapIndex>>` borrowing from them.
///
/// # Safety
///
/// `rel` must be a valid relation pointer; `meta` must come
/// from `relfile::read_meta(rel)` on the same relation in the
/// same scan. Caller holds at least `AccessShareLock` on the
/// relation.
pub(crate) unsafe fn load_static_regions(
    rel: pg_sys::Relation,
    meta: &MetaPageData,
) -> Option<(StaticRegionsMap, StaticRegionsBytes)> {
    if !meta.has_prepared_layout() {
        return None;
    }
    let map = StaticRegionsMap::open(rel)?;

    // Sanity: the meta page on the mapping should match the one
    // we already decoded via the buffer manager. If they
    // disagree (concurrent rewrite raced ahead of our
    // AccessShareLock release at meta-read time), bail and let
    // the caller fall back \u2014 the buffer-manager path will
    // re-read consistently under the SUE-blocking lock the
    // writer just released.
    let mapped_meta_bytes = map.page_data(META_BLKNO)?;
    let mapped_meta = match MetaPageData::decode(mapped_meta_bytes) {
        Ok(m) => m,
        Err(_) => return None,
    };
    if mapped_meta.am_version != meta.am_version
        || mapped_meta.blocked_first != meta.blocked_first
        || mapped_meta.rotation_first != meta.rotation_first
        || mapped_meta.n_vectors != meta.n_vectors
    {
        return None;
    }

    // Blocked codes chain: stride=1, rows_per_page=PAYLOAD_BYTES.
    let blocked_codes = map.read_chain(
        meta.blocked_first,
        1,
        PAYLOAD_BYTES as u32,
        meta.blocked_bytes,
    )?;

    // Rotation chain: stride=1, rows_per_page=PAYLOAD_BYTES.
    let rot_n_elems = (meta.rotation_dim as usize).checked_mul(meta.rotation_dim as usize)?;
    let rot_n_bytes = rot_n_elems.checked_mul(std::mem::size_of::<f32>())?;
    let rotation_bytes = map.read_chain(
        meta.rotation_first,
        1,
        PAYLOAD_BYTES as u32,
        rot_n_bytes as u64,
    )?;
    if rotation_bytes.len() != rot_n_bytes {
        return None;
    }
    let mut rotation = Vec::<f32>::with_capacity(rot_n_elems);
    for chunk in rotation_bytes.chunks_exact(4) {
        rotation.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // Centroids / boundaries are inlined in the meta page
    // (already decoded into `meta`). Copy out the meaningful
    // prefix; this is at most MAX_CODEBOOK_LEVELS f32s.
    let n_levels = (meta.codebook_n_levels as usize).min(MAX_CODEBOOK_LEVELS);
    let centroids = meta.centroids[..n_levels].to_vec();
    let boundaries = if n_levels >= 2 {
        meta.boundaries[..n_levels - 1].to_vec()
    } else {
        Vec::new()
    };

    Some((
        map,
        StaticRegionsBytes {
            blocked_codes,
            n_blocks: meta.n_blocks_blocked as usize,
            centroids,
            boundaries,
            rotation,
        },
    ))
}

/// Compute the on-disk path of the relation's segment-0 file.
/// Mirrors `GetRelationPath` in PG's `relpath.c` for the main
/// fork, segment 0. Returns `None` if any of the required
/// fields are zero (signalling we can't determine the path \u2014
/// fall back to the buffer manager).
///
/// We don't use `RelationGetSmgr` + `smgrnblocks` + path-from-smgr
/// because pgrx exposes that surface inconsistently across pg13..18.
/// The `base/<dbid>/<relnumber>` layout has been stable since pg9.x
/// for the default tablespace; for non-default tablespaces we'd
/// need `pg_tblspc/<spcid>/<TABLESPACE_VERSION_DIRECTORY>/<dbid>/...`,
/// which we handle via `(*rd_locator).spcOid` when present.
///
/// # Safety
///
/// `rel` must be a valid relation pointer.
unsafe fn relfile_path(rel: pg_sys::Relation) -> Option<PathBuf> {
    if rel.is_null() {
        return None;
    }
    let data_dir = data_directory()?;

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
    let (db_oid, rel_oid, tbs_oid) = {
        let node = (*rel).rd_node;
        (node.dbNode.to_u32(), node.relNode.to_u32(), node.spcNode.to_u32())
    };
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    let (db_oid, rel_oid, tbs_oid) = {
        let loc = (*rel).rd_locator;
        (
            loc.dbOid.to_u32(),
            loc.relNumber.to_u32(),
            loc.spcOid.to_u32(),
        )
    };

    if rel_oid == 0 {
        return None;
    }

    // GLOBALTABLESPACE_OID = 1664; DEFAULTTABLESPACE_OID = 1663.
    // For default tablespace (or oid=0 sentinel): base/<dbid>/<rel>.
    // For shared catalogs (db_oid=0) under GLOBAL: global/<rel>.
    // Otherwise: pg_tblspc/<spc>/<TABLESPACE_VERSION_DIRECTORY>/<db>/<rel>.
    let mut p = data_dir;
    if tbs_oid == 0 || tbs_oid == pg_sys::DEFAULTTABLESPACE_OID.to_u32() {
        if db_oid == 0 {
            p.push("global");
        } else {
            p.push("base");
            p.push(db_oid.to_string());
        }
    } else if tbs_oid == pg_sys::GLOBALTABLESPACE_OID.to_u32() {
        p.push("global");
    } else {
        // Non-default tablespace: defer to the buffer-manager
        // path. Computing TABLESPACE_VERSION_DIRECTORY portably
        // across pg13..18 is fiddly and the common case (default
        // tablespace) is what the bench corpus uses. Returning
        // None makes the caller fall through.
        return None;
    }
    p.push(rel_oid.to_string());
    Some(p)
}

/// Resolve PG's `data_directory` GUC string into a `PathBuf`.
/// Returns `None` if the GUC isn't set (shouldn't happen inside
/// a backend, but be defensive).
fn data_directory() -> Option<PathBuf> {
    // SAFETY: `DataDir` is a `*const c_char` global PG variable
    // initialised at server start; reading it from any backend
    // context is safe.
    unsafe {
        let p = pg_sys::DataDir;
        if p.is_null() {
            return None;
        }
        let cstr = std::ffi::CStr::from_ptr(p);
        Some(PathBuf::from(cstr.to_string_lossy().into_owned()))
    }
}
