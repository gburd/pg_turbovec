//! Phase L - relfile-resident page layouts for the `turbovec`
//! access method.
//!
//! **Status: STUB** - only the layout types and pure-bytes
//! (de)serialisers live here. The `relfile.rs` sibling wires them
//! to PostgreSQL's buffer manager.
//!
//! ## Wire format (little-endian throughout)
//!
//! Block 0 is always the [`MetaPage`]:
//!
//! ```text
//!  off  size  field
//!    0    24  PageHeader (PostgreSQL standard)        \   pd_lower / pd_upper
//!   24     4  magic = "TVRM"                          |   left at SizeOfPageHeaderData
//!   28     1  version = 3                             |   / BLCKSZ; we don't use
//!   29     1  bit_width                               |   line pointers and our
//!   30     2  reserved (zero)                         |   data lives in the data
//!   32     4  dim          (u32)                      |   region as private bytes.
//!   36     8  n_vectors    (u64)                      |
//!   44     4  codes_first  (BlockNumber)              |
//!   48     4  codes_count  (u32)                      |
//!   52     4  scales_first (BlockNumber)              |
//!   56     4  scales_count (u32)                      |
//!   60     4  ids_first    (BlockNumber)              |
//!   64     4  ids_count    (u32)                      |
//!   68     4  rows_per_codes_page (u32)               |
//!   72     4  rows_per_scales_page (u32)              |
//!   76     4  rows_per_ids_page (u32)                 |
//!   80     4  stride_bytes (u32)  = (dim/8)*bit_width |
//!   84     4  am_version   (u32)  bumped on commit    |
//!   88     4  blocked_first (BlockNumber)             |  v2+
//!   92     4  blocked_count (u32)                     |
//!   96     8  blocked_bytes (u64)                     |
//!  104     4  n_blocks_blocked (u32)                  |
//!  108     4  codebook_n_levels (u32) = 1 << bit_width|
//!  112    64  centroids[16] (f32, zero-padded tail)   |
//!  176    60  boundaries[15] (f32, zero-padded tail)  |
//!  236     4  rotation_first (BlockNumber)            |  v3+
//!  240     4  rotation_count (u32)                    |
//!  244     4  rotation_dim (u32)                      |
//!  248     4  lists (u32)  nlist; 0 = flat            |  v4+
//!  252     4  coarse_first  (BlockNumber)             |
//!  256     4  coarse_count  (u32)                     |
//!  260     4  cell_dir_first (BlockNumber)            |
//!  264     4  cell_dir_count (u32)                    |
//!  268     1  ivf_degraded (u8) 1 = was-IVF, now flat |  v4 (E-2)
//!  269     3  reserved (zero)                         |
//!  272     4  tombstone_first (BlockNumber)           |  v4 (E-2)
//!  276     4  tombstone_count (u32)                   |
//!  280     8  tombstone_bytes (u64)                   |
//!  288   ...  reserved (zero)                         /
//! ```
//!
//! After the meta block come three contiguous page chains for
//! the row-major codes / scales / ids, followed in v2 by a
//! fourth chain holding the prepared SIMD-blocked layout, in v3
//! by a fifth chain holding the persisted random orthogonal
//! rotation matrix, and in v4 by (when `lists > 0`) a sixth
//! chain holding the coarse centroids (`lists * dim` f32, rotated
//! space) and a seventh holding the cell directory (`lists`
//! `(code_offset: u64, n_vectors: u32)` entries). v4 with
//! `lists = 0` is byte-identical to v3 modulo the version byte
//! and the zeroed v4 fields — the IVF feature is strictly opt-in,
//! so existing v3 indexes need no REINDEX. The blocked, rotation,
//! `(dim, ROTATION_SEED)` whose lazy QR-on-first-search was
//! the warm-scan hotspot Phase R diagnosed). The blocked,
//! rotation, coarse-centroid, and cell-directory chains are flat
//! byte chains (no fixed row stride): every page after the page
//! header is `PAYLOAD_BYTES` of raw bytes, with the last page
//! holding the residual tail.
//!
//! ## Why no PageAddItem / line pointers?
//!
//! Our rows are fixed-stride and we never need to delete one in
//! place; `aminsert` appends, `ambulkdelete` rebuilds. Line
//! pointers add 4 bytes/row of overhead and force us to decode an
//! item-id lookup table on every read. The flat layout matches the
//! existing TVIM byte stream exactly, so reading a page and
//! `slice::from_raw_parts`-ing the data area gives us the same
//! view the SPI loader sees today.
//!
//! ## What this module does *not* know
//!
//! - PostgreSQL FFI. All functions take `&[u8]` / `&mut [u8]`.
//! - WAL. `relfile.rs` will (eventually) wrap dirty-page writes in
//!   `log_newpage_buffer`. Phase L stub skips WAL entirely; the
//!   handoff doc lists this as a known gap.

use core::mem::size_of;

/// 4-byte file magic. "TurboVec RelMain".
pub const MAGIC: [u8; 4] = *b"TVRM";

/// On-disk format version.
///
/// `1` - Phase L: meta + 3 chains (codes / scales / ids).
/// `2` - Phase P: meta + 4 chains, with the prepared SIMD-blocked
///       layout persisted in the new `blocked` chain and the
///       Lloyd-Max codebook stored inline on the meta page.
///       Backends opening a v2 index skip the per-backend
///       `pack::repack` (~12-15 s on 1 M × 1536-d) and Lloyd-Max
///       compute (~5-8 s).
/// `3` - Phase R-2: meta + 5 chains, adding the persisted random
///       orthogonal rotation matrix in a new `rotation` chain.
///       Backends opening a v3 index skip the per-backend QR
///       decomposition (`rotation::make_rotation_matrix`), which
///       at `dim = 1536` is ~64% self time on the warm-scan
///       profile (see `docs/PROFILING.md`).
/// `4` - IVF-1: meta + (when `lists > 0`) 7 chains, adding the
///       coarse centroids + cell directory for the inverted-file
///       layer (`docs/IVF_PLAN.md`). IVF is opt-in via
///       `WITH (lists = N)`; `lists = 0` (the default) is
///       byte-identical to v3 modulo the version byte, so the v3
///       flat decode path stays valid and existing v3 indexes
///       need no REINDEX. The scan path is still FLAT in IVF-1
///       (cells are persisted but not yet probed); cell-restricted
///       search is IVF-2.
pub const VERSION: u8 = 4;

/// The on-disk version we read **and** write today. Decode
/// accepts strictly older versions for migration-HINT purposes
/// (callers detect them via [`MetaPageData::version`]) but cannot
/// upgrade them in place - a REINDEX rewrites the relation under
/// the current `VERSION`.
pub const MIN_DECODE_VERSION: u8 = 1;

/// Maximum centroids the inline codebook slot in the meta page
/// can hold. `bit_width = 4` is the largest supported width and
/// produces `1 << 4 = 16` centroids; smaller widths leave the
/// trailing slots zero.
pub const MAX_CODEBOOK_LEVELS: usize = 16;

/// Standard PostgreSQL page size. We assert at runtime that the
/// running cluster matches; an 8KB-vs-32KB BLCKSZ mismatch would
/// silently produce unreadable indexes otherwise.
pub const BLCKSZ: usize = 8192;

/// Bytes consumed by the PostgreSQL `PageHeaderData` struct
/// (`offsetof(PageHeaderData, pd_linp)`). We store our private
/// payload immediately after the page header.
pub const PAGE_HEADER_BYTES: usize = 24;

/// Useful payload bytes per page. The top of the page is left
/// untouched (`pd_upper = BLCKSZ`) because we never call
/// `PageAddItem`.
pub const PAYLOAD_BYTES: usize = BLCKSZ - PAGE_HEADER_BYTES;

/// Block number convention. Block 0 is always the meta page.
pub const META_BLKNO: u32 = 0;

/// Decoded view of the meta page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetaPageData {
    /// On-disk format version. `1` = Phase L (no prepared layout),
    /// `2` = Phase P (prepared layout + inline codebook).
    /// `ambeginscan` consults this to emit the migration HINT for
    /// indexes built before the prepared-layout work landed.
    pub version: u8,
    pub bit_width: u8,
    pub dim: u32,
    pub n_vectors: u64,
    pub codes_first: u32,
    pub codes_count: u32,
    pub scales_first: u32,
    pub scales_count: u32,
    pub ids_first: u32,
    pub ids_count: u32,
    pub rows_per_codes_page: u32,
    pub rows_per_scales_page: u32,
    pub rows_per_ids_page: u32,
    pub stride_bytes: u32,
    /// Bumped on every commit (ambuild / aminsert / ambulkdelete).
    /// Drives the shared `cache.rs` freshness check.
    pub am_version: u32,

    // ---- v2 fields ----
    /// First block of the prepared SIMD-blocked codes chain.
    /// Zero on v1 indexes (and on empty v2 indexes).
    pub blocked_first: u32,
    /// Number of pages in the blocked chain.
    pub blocked_count: u32,
    /// Total byte length of the blocked layout. Drives how many
    /// bytes the reader pulls off the chain.
    pub blocked_bytes: u64,
    /// `n_blocks` count from `pack::repack`. Needed by
    /// `turbovec::search` so we don't recompute it on the read
    /// side.
    pub n_blocks_blocked: u32,
    /// `1 << bit_width`. `0` on v1 indexes and on empty v2
    /// indexes (then no codebook is persisted).
    pub codebook_n_levels: u32,
    /// Lloyd-Max centroids, zero-padded to `MAX_CODEBOOK_LEVELS`.
    /// Only the first `codebook_n_levels` entries are meaningful.
    pub centroids: [f32; MAX_CODEBOOK_LEVELS],
    /// Lloyd-Max decision boundaries, zero-padded to
    /// `MAX_CODEBOOK_LEVELS - 1`. Only the first
    /// `codebook_n_levels.saturating_sub(1)` entries are
    /// meaningful.
    pub boundaries: [f32; MAX_CODEBOOK_LEVELS - 1],

    // ---- v3 fields ----
    /// First block of the persisted rotation chain. Zero on v1
    /// or v2 indexes (and on empty v3 indexes).
    pub rotation_first: u32,
    /// Number of pages in the rotation chain.
    pub rotation_count: u32,
    /// Dimensionality the matrix was built for. Stored
    /// explicitly (rather than derived from `self.dim`) so a
    /// future ALTER-style dim change can be detected; today it
    /// always equals `self.dim`.
    pub rotation_dim: u32,

    // ---- v4 fields (IVF) ----
    /// IVF coarse-cell count (`nlist`). `0` = flat (v3-equivalent;
    /// the default). When `> 0` the codes/scales/ids chains are
    /// stored in cell-contiguous order and the coarse-centroid +
    /// cell-directory chains below are populated.
    pub lists: u32,
    /// First block of the coarse-centroid chain (`lists * dim` f32,
    /// rotated space). `0` when `lists == 0` or empty.
    pub coarse_first: u32,
    /// Number of pages in the coarse-centroid chain.
    pub coarse_count: u32,
    /// First block of the cell-directory chain (`lists`
    /// `(code_offset: u64, n_vectors: u32)` entries). `0` when
    /// `lists == 0` or empty.
    pub cell_dir_first: u32,
    /// Number of pages in the cell-directory chain.
    pub cell_dir_count: u32,

    // ---- v4 E-2 fields (VACUUM survival / observability) ----
    /// Set to `true` when an index that was BUILT `WITH (lists > 0)`
    /// has degraded to a flat scan (its coarse/cell metadata was
    /// invalidated). With the tombstone-vacuum path this should never
    /// trip for IVF; it remains a loud, queryable safety-net signal
    /// for any path that still blanks the IVF chains (and for the flat
    /// `lists == 0` swap-remove path it is always `false`). Default
    /// `false` on every pre-E-2 v4 index, which reads as "not
    /// degraded" — the backward-compat path.
    pub ivf_degraded: bool,
    /// First block of the per-slot tombstone bitmap chain. `0` when no
    /// rows have been deleted from an IVF index yet (or for flat
    /// indexes, which swap-remove instead of tombstone). One bit per
    /// slot, LSB-first; bit set ⇒ slot is dead and excluded from the
    /// scan mask.
    pub tombstone_first: u32,
    /// Number of pages in the tombstone chain.
    pub tombstone_count: u32,
    /// Byte length of the tombstone bitmap (`ceil(n_vectors / 8)`).
    /// `0` ⇒ no tombstones (all slots live).
    pub tombstone_bytes: u64,
}

impl MetaPageData {
    /// Stride in bytes of one packed code row.
    pub fn codes_stride(bit_width: u8, dim: u32) -> u32 {
        (dim / 8) * u32::from(bit_width)
    }

    /// Compute rows-per-page for a uniform-stride chain.
    pub fn rows_per_page(stride: u32) -> u32 {
        match stride {
            0 => 0,
            s => (PAYLOAD_BYTES as u32) / s,
        }
    }

    /// Total number of pages required for `n_vectors` rows at
    /// `rows_per_page`. Returns 0 when n_vectors == 0.
    pub fn pages_needed(n_vectors: u64, rows_per_page: u32) -> u32 {
        if n_vectors == 0 || rows_per_page == 0 {
            return 0;
        }
        let rpp = u64::from(rows_per_page);
        u32::try_from(n_vectors.div_ceil(rpp)).unwrap_or(u32::MAX)
    }

    /// Number of live rows on page `page_idx` of a chain.
    #[allow(dead_code)] // exercised by tests + future ambulkdelete
    pub fn rows_on_page(n_vectors: u64, rows_per_page: u32, page_idx: u32) -> u32 {
        let rpp = u64::from(rows_per_page);
        if rows_per_page == 0 || n_vectors == 0 {
            return 0;
        }
        let total_pages = Self::pages_needed(n_vectors, rows_per_page);
        match (page_idx + 1).cmp(&total_pages) {
            std::cmp::Ordering::Less => rows_per_page,
            std::cmp::Ordering::Equal => {
                let rem = n_vectors % rpp;
                if rem == 0 {
                    rows_per_page
                } else {
                    rem as u32
                }
            }
            std::cmp::Ordering::Greater => 0,
        }
    }

    /// Plan a layout for `n_vectors`, `dim`, `bit_width`. Block 0
    /// is the meta page; codes follow at block 1, then scales,
    /// then ids, then the prepared blocked chain, then the
    /// rotation chain.
    ///
    /// `blocked_bytes` is the total size of the prepared SIMD-
    /// blocked layout (output of `turbovec::pack::repack`). Pass
    /// `0` for an empty index or when the prepared layout isn't
    /// being persisted (which gives a v3 meta with an empty
    /// blocked chain — readers fall back to per-backend repack).
    /// `n_blocks_blocked` is the matching `n_blocks` count from
    /// `pack::repack`. `rotation_bytes` is the byte size of the
    /// row-major `dim*dim` `f32` rotation matrix; pass `0` when
    /// no rotation is being persisted (lazy QR on first search).
    pub fn plan_with_blocked(
        bit_width: u8,
        dim: u32,
        n_vectors: u64,
        am_version: u32,
        blocked_bytes: u64,
        n_blocks_blocked: u32,
        rotation_bytes: u64,
    ) -> Self {
        assert_eq!(dim % 8, 0, "dim must be a multiple of 8");
        let stride_bytes = Self::codes_stride(bit_width, dim);
        let rows_per_codes_page = Self::rows_per_page(stride_bytes);
        let rows_per_scales_page = Self::rows_per_page(size_of::<f32>() as u32);
        let rows_per_ids_page = Self::rows_per_page(size_of::<u64>() as u32);

        let codes_count = Self::pages_needed(n_vectors, rows_per_codes_page);
        let scales_count = Self::pages_needed(n_vectors, rows_per_scales_page);
        let ids_count = Self::pages_needed(n_vectors, rows_per_ids_page);
        let blocked_count = Self::byte_pages_needed(blocked_bytes);
        let rotation_count = Self::byte_pages_needed(rotation_bytes);

        let codes_first = 1;
        let scales_first = codes_first + codes_count;
        let ids_first = scales_first + scales_count;
        let blocked_first_blkno = ids_first + ids_count;
        let rotation_first_blkno = blocked_first_blkno + blocked_count;

        Self {
            version: VERSION,
            bit_width,
            dim,
            n_vectors,
            codes_first,
            codes_count,
            scales_first,
            scales_count,
            ids_first,
            ids_count,
            rows_per_codes_page,
            rows_per_scales_page,
            rows_per_ids_page,
            stride_bytes,
            am_version,
            blocked_first: if blocked_bytes > 0 { blocked_first_blkno } else { 0 },
            blocked_count,
            blocked_bytes,
            n_blocks_blocked,
            codebook_n_levels: 0,
            centroids: [0.0; MAX_CODEBOOK_LEVELS],
            boundaries: [0.0; MAX_CODEBOOK_LEVELS - 1],
            rotation_first: if rotation_bytes > 0 { rotation_first_blkno } else { 0 },
            rotation_count,
            rotation_dim: if rotation_bytes > 0 { dim } else { 0 },
            // v4 IVF fields default to flat; set_ivf_chains() fills
            // them in (and lays the coarse + cell-dir chains out
            // after the rotation chain) when lists > 0.
            lists: 0,
            coarse_first: 0,
            coarse_count: 0,
            cell_dir_first: 0,
            cell_dir_count: 0,
            ivf_degraded: false,
            tombstone_first: 0,
            tombstone_count: 0,
            tombstone_bytes: 0,
        }
    }

    /// Lay out the v4 IVF chains (coarse centroids + cell directory)
    /// AFTER the rotation chain, and stamp `lists`. Must be called on
    /// a meta already planned by [`Self::plan_with_blocked`] (so the
    /// row-major + blocked + rotation chain offsets are fixed).
    ///
    /// `coarse_bytes` is the byte length of the `lists * dim` f32
    /// coarse-centroid buffer; `cell_dir_bytes` is the byte length of
    /// the packed cell-directory (`lists * CellEntry::ENCODED_BYTES`).
    /// Pass `lists = 0` (the default after `plan_with_blocked`) to
    /// leave the index flat — in that case this is a no-op.
    pub fn set_ivf_chains(
        &mut self,
        lists: u32,
        coarse_bytes: u64,
        cell_dir_bytes: u64,
    ) {
        self.lists = lists;
        if lists == 0 {
            self.coarse_first = 0;
            self.coarse_count = 0;
            self.cell_dir_first = 0;
            self.cell_dir_count = 0;
            return;
        }
        // The IVF chains follow the rotation chain. Compute the first
        // free block after every prior chain.
        let after_rotation = 1
            + self.codes_count
            + self.scales_count
            + self.ids_count
            + self.blocked_count
            + self.rotation_count;
        let coarse_count = Self::byte_pages_needed(coarse_bytes);
        let cell_dir_count = Self::byte_pages_needed(cell_dir_bytes);
        self.coarse_count = coarse_count;
        self.cell_dir_count = cell_dir_count;
        self.coarse_first = if coarse_bytes > 0 { after_rotation } else { 0 };
        self.cell_dir_first = if cell_dir_bytes > 0 {
            after_rotation + coarse_count
        } else {
            0
        };
    }

    /// Plan a layout without a prepared blocked chain or
    /// rotation. Equivalent to `plan_with_blocked(…,
    /// blocked_bytes = 0, n_blocks_blocked = 0,
    /// rotation_bytes = 0)`. Used by `aminsert` paths that
    /// rewrite the relfile incrementally and don't have the
    /// prepared layout handy. Readers fall back to per-backend
    /// `pack::repack` and lazy QR for these indexes.
    pub fn plan(bit_width: u8, dim: u32, n_vectors: u64, am_version: u32) -> Self {
        Self::plan_with_blocked(bit_width, dim, n_vectors, am_version, 0, 0, 0)
    }

    /// Set the inline codebook fields. `centroids` must have
    /// length `1 << bit_width` (≤ [`MAX_CODEBOOK_LEVELS`]) and
    /// `boundaries` must have length `centroids.len() - 1`.
    /// Anything beyond those lengths is zero-padded.
    pub fn set_codebook(&mut self, centroids: &[f32], boundaries: &[f32]) {
        let n = centroids.len();
        assert!(
            n <= MAX_CODEBOOK_LEVELS,
            "codebook has {} levels; max is {}",
            n,
            MAX_CODEBOOK_LEVELS,
        );
        assert_eq!(
            boundaries.len() + 1,
            n,
            "boundaries.len() must be centroids.len() - 1",
        );
        self.codebook_n_levels = n as u32;
        self.centroids = [0.0; MAX_CODEBOOK_LEVELS];
        self.boundaries = [0.0; MAX_CODEBOOK_LEVELS - 1];
        self.centroids[..n].copy_from_slice(centroids);
        self.boundaries[..n - 1].copy_from_slice(boundaries);
    }

    /// Pages needed to hold `n_bytes` of opaque payload, packing
    /// `PAYLOAD_BYTES` per full page.
    pub fn byte_pages_needed(n_bytes: u64) -> u32 {
        if n_bytes == 0 {
            return 0;
        }
        u32::try_from(n_bytes.div_ceil(PAYLOAD_BYTES as u64)).unwrap_or(u32::MAX)
    }

    /// Total number of blocks (including meta) required for this
    /// layout.
    #[allow(dead_code)] // exercised by tests; not yet read by relfile.rs
    pub fn total_blocks(&self) -> u32 {
        1 + self.codes_count
            + self.scales_count
            + self.ids_count
            + self.blocked_count
            + self.rotation_count
            + self.coarse_count
            + self.cell_dir_count
            + self.tombstone_count
    }

    /// Serialise the meta header (no PG page header) to a
    /// `PAYLOAD_BYTES`-sized buffer suitable for memcpy into the
    /// data area of block 0.
    pub fn encode(&self) -> [u8; PAYLOAD_BYTES] {
        let mut out = [0u8; PAYLOAD_BYTES];
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = VERSION;
        out[5] = self.bit_width;
        // out[6..8] reserved
        out[8..12].copy_from_slice(&self.dim.to_le_bytes());
        out[12..20].copy_from_slice(&self.n_vectors.to_le_bytes());
        out[20..24].copy_from_slice(&self.codes_first.to_le_bytes());
        out[24..28].copy_from_slice(&self.codes_count.to_le_bytes());
        out[28..32].copy_from_slice(&self.scales_first.to_le_bytes());
        out[32..36].copy_from_slice(&self.scales_count.to_le_bytes());
        out[36..40].copy_from_slice(&self.ids_first.to_le_bytes());
        out[40..44].copy_from_slice(&self.ids_count.to_le_bytes());
        out[44..48].copy_from_slice(&self.rows_per_codes_page.to_le_bytes());
        out[48..52].copy_from_slice(&self.rows_per_scales_page.to_le_bytes());
        out[52..56].copy_from_slice(&self.rows_per_ids_page.to_le_bytes());
        out[56..60].copy_from_slice(&self.stride_bytes.to_le_bytes());
        out[60..64].copy_from_slice(&self.am_version.to_le_bytes());
        // v2 fields
        out[64..68].copy_from_slice(&self.blocked_first.to_le_bytes());
        out[68..72].copy_from_slice(&self.blocked_count.to_le_bytes());
        out[72..80].copy_from_slice(&self.blocked_bytes.to_le_bytes());
        out[80..84].copy_from_slice(&self.n_blocks_blocked.to_le_bytes());
        out[84..88].copy_from_slice(&self.codebook_n_levels.to_le_bytes());
        for (i, c) in self.centroids.iter().enumerate() {
            let off = 88 + i * 4;
            out[off..off + 4].copy_from_slice(&c.to_le_bytes());
        }
        let bound_base = 88 + MAX_CODEBOOK_LEVELS * 4; // = 152
        for (i, b) in self.boundaries.iter().enumerate() {
            let off = bound_base + i * 4;
            out[off..off + 4].copy_from_slice(&b.to_le_bytes());
        }
        // v3 fields begin at bound_base + (MAX_CODEBOOK_LEVELS - 1) * 4 = 212.
        let v3_base = bound_base + (MAX_CODEBOOK_LEVELS - 1) * 4;
        out[v3_base..v3_base + 4].copy_from_slice(&self.rotation_first.to_le_bytes());
        out[v3_base + 4..v3_base + 8].copy_from_slice(&self.rotation_count.to_le_bytes());
        out[v3_base + 8..v3_base + 12].copy_from_slice(&self.rotation_dim.to_le_bytes());
        // v4 IVF fields begin at v3_base + 12 = 224 (data region),
        // i.e. page offset 248 (24-byte PG header + 224).
        let v4_base = v3_base + 12;
        out[v4_base..v4_base + 4].copy_from_slice(&self.lists.to_le_bytes());
        out[v4_base + 4..v4_base + 8].copy_from_slice(&self.coarse_first.to_le_bytes());
        out[v4_base + 8..v4_base + 12].copy_from_slice(&self.coarse_count.to_le_bytes());
        out[v4_base + 12..v4_base + 16].copy_from_slice(&self.cell_dir_first.to_le_bytes());
        out[v4_base + 16..v4_base + 20].copy_from_slice(&self.cell_dir_count.to_le_bytes());
        // v4 E-2 fields begin at v4_base + 20 = 244 (page offset 268):
        // ivf_degraded (1 byte) + 3 reserved + tombstone first/count
        // (u32 each) + tombstone_bytes (u64).
        let e2_base = v4_base + 20;
        out[e2_base] = u8::from(self.ivf_degraded);
        // out[e2_base + 1 .. e2_base + 4] reserved (zero)
        out[e2_base + 4..e2_base + 8].copy_from_slice(&self.tombstone_first.to_le_bytes());
        out[e2_base + 8..e2_base + 12].copy_from_slice(&self.tombstone_count.to_le_bytes());
        out[e2_base + 12..e2_base + 20].copy_from_slice(&self.tombstone_bytes.to_le_bytes());
        // Trailing bytes reserved (zero).
        out
    }

    /// Inverse of [`Self::encode`]. Input must be the page's data
    /// region (no PG page header) of at least 64 bytes; longer is
    /// fine. Accepts both v1 (Phase L) and v2 (Phase P) layouts -
    /// the v1 path leaves the new fields zeroed so callers can
    /// detect an unmigrated index via `version < VERSION`.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 64 {
            return Err("meta page data region too short");
        }
        if bytes[0..4] != MAGIC {
            return Err("bad magic on meta page");
        }
        let version = bytes[4];
        if version < MIN_DECODE_VERSION || version > VERSION {
            return Err("unsupported meta page version");
        }
        let bit_width = bytes[5];
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let n_vectors = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let codes_first = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let codes_count = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let scales_first = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let scales_count = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
        let ids_first = u32::from_le_bytes(bytes[36..40].try_into().unwrap());
        let ids_count = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let rows_per_codes_page = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
        let rows_per_scales_page = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
        let rows_per_ids_page = u32::from_le_bytes(bytes[52..56].try_into().unwrap());
        let stride_bytes = u32::from_le_bytes(bytes[56..60].try_into().unwrap());
        let am_version = u32::from_le_bytes(bytes[60..64].try_into().unwrap());

        let mut me = Self {
            version,
            bit_width,
            dim,
            n_vectors,
            codes_first,
            codes_count,
            scales_first,
            scales_count,
            ids_first,
            ids_count,
            rows_per_codes_page,
            rows_per_scales_page,
            rows_per_ids_page,
            stride_bytes,
            am_version,
            blocked_first: 0,
            blocked_count: 0,
            blocked_bytes: 0,
            n_blocks_blocked: 0,
            codebook_n_levels: 0,
            centroids: [0.0; MAX_CODEBOOK_LEVELS],
            boundaries: [0.0; MAX_CODEBOOK_LEVELS - 1],
            rotation_first: 0,
            rotation_count: 0,
            rotation_dim: 0,
            lists: 0,
            coarse_first: 0,
            coarse_count: 0,
            cell_dir_first: 0,
            cell_dir_count: 0,
            ivf_degraded: false,
            tombstone_first: 0,
            tombstone_count: 0,
            tombstone_bytes: 0,
        };

        if version >= 2 {
            // v2 needs at least 88 + 16*4 + 15*4 = 212 bytes.
            if bytes.len() < 212 {
                return Err("v2 meta page data region too short");
            }
            me.blocked_first = u32::from_le_bytes(bytes[64..68].try_into().unwrap());
            me.blocked_count = u32::from_le_bytes(bytes[68..72].try_into().unwrap());
            me.blocked_bytes = u64::from_le_bytes(bytes[72..80].try_into().unwrap());
            me.n_blocks_blocked = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
            me.codebook_n_levels = u32::from_le_bytes(bytes[84..88].try_into().unwrap());
            if me.codebook_n_levels as usize > MAX_CODEBOOK_LEVELS {
                return Err("codebook_n_levels exceeds maximum");
            }
            for i in 0..MAX_CODEBOOK_LEVELS {
                let off = 88 + i * 4;
                me.centroids[i] =
                    f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            }
            let bound_base = 88 + MAX_CODEBOOK_LEVELS * 4;
            for i in 0..MAX_CODEBOOK_LEVELS - 1 {
                let off = bound_base + i * 4;
                me.boundaries[i] =
                    f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            }
        }

        if version >= 3 {
            // v3 needs at least 224 bytes (212 + 12 for rotation
            // first/count/dim).
            if bytes.len() < 224 {
                return Err("v3 meta page data region too short");
            }
            let v3_base = 212;
            me.rotation_first =
                u32::from_le_bytes(bytes[v3_base..v3_base + 4].try_into().unwrap());
            me.rotation_count =
                u32::from_le_bytes(bytes[v3_base + 4..v3_base + 8].try_into().unwrap());
            me.rotation_dim =
                u32::from_le_bytes(bytes[v3_base + 8..v3_base + 12].try_into().unwrap());
        }

        if version >= 4 {
            // v4 needs at least 244 bytes (224 + 20 for the five IVF
            // u32 fields). A v3 meta read by a v4 binary leaves these
            // zeroed (lists = 0 => flat), which is exactly the
            // backward-compat path: no REINDEX for v3 indexes.
            if bytes.len() < 244 {
                return Err("v4 meta page data region too short");
            }
            let v4_base = 224;
            me.lists = u32::from_le_bytes(bytes[v4_base..v4_base + 4].try_into().unwrap());
            me.coarse_first =
                u32::from_le_bytes(bytes[v4_base + 4..v4_base + 8].try_into().unwrap());
            me.coarse_count =
                u32::from_le_bytes(bytes[v4_base + 8..v4_base + 12].try_into().unwrap());
            me.cell_dir_first =
                u32::from_le_bytes(bytes[v4_base + 12..v4_base + 16].try_into().unwrap());
            me.cell_dir_count =
                u32::from_le_bytes(bytes[v4_base + 16..v4_base + 20].try_into().unwrap());
            // v4 E-2 fields (offset 244..264). Additive: a pre-E-2 v4
            // meta has these bytes zeroed (the page is always
            // PAYLOAD_BYTES long), which reads as ivf_degraded=false +
            // no tombstones — the backward-compat path, no REINDEX. We
            // only decode them when the buffer is long enough; a short
            // test buffer (>=244, <264) leaves them at the struct
            // default of 0/false.
            let e2_base = v4_base + 20; // 244
            if bytes.len() >= e2_base + 20 {
                me.ivf_degraded = bytes[e2_base] != 0;
                me.tombstone_first =
                    u32::from_le_bytes(bytes[e2_base + 4..e2_base + 8].try_into().unwrap());
                me.tombstone_count =
                    u32::from_le_bytes(bytes[e2_base + 8..e2_base + 12].try_into().unwrap());
                me.tombstone_bytes =
                    u64::from_le_bytes(bytes[e2_base + 12..e2_base + 20].try_into().unwrap());
            }
        }

        Ok(me)
    }

    /// Slice of meaningful centroids, as decoded from the inline
    /// codebook. Returns an empty slice on v1 indexes (where the
    /// codebook was never persisted).
    pub fn centroids_slice(&self) -> &[f32] {
        let n = self.codebook_n_levels as usize;
        let n = n.min(MAX_CODEBOOK_LEVELS);
        &self.centroids[..n]
    }

    /// Slice of meaningful boundaries.
    pub fn boundaries_slice(&self) -> &[f32] {
        let n = self.codebook_n_levels as usize;
        if n < 2 {
            return &[];
        }
        let n = n.min(MAX_CODEBOOK_LEVELS);
        &self.boundaries[..n - 1]
    }

    /// Returns `true` when this meta page describes an index
    /// built under a wire format with a prepared blocked layout
    /// AND a persisted rotation matrix actually present. v1/v2
    /// indexes and empty (no-rows) v3/v4 indexes return `false`.
    ///
    /// IVF-1 note: this checks `version >= 3` (NOT `>= VERSION`)
    /// so a v3 index opened by the v4 binary still reports its
    /// prepared layout and scans flat with no REINDEX. A v4
    /// `lists = 0` index is structurally a v3 layout and reports
    /// the same.
    pub fn has_prepared_layout(&self) -> bool {
        self.version >= 3
            && self.blocked_bytes > 0
            && self.codebook_n_levels > 0
            && self.rotation_count > 0
    }

    /// Returns `true` when the meta page is in the older v1 wire
    /// format (Phase L preview, pre-v1.3.0). `ambeginscan` uses
    /// this to emit the migration `ERROR` directing the user to
    /// `REINDEX INDEX <name>;`.
    pub fn is_legacy_v1(&self) -> bool {
        self.version < 2
    }

    /// Returns `true` when the meta page is in the v2 wire
    /// format (v1.3.x; Phase P prepared layout but no persisted
    /// rotation matrix). v1.4.0+ binaries refuse to scan these
    /// because the rotation chain offsets don't exist on disk
    /// and the lazy QR was the warm-scan hotspot Phase R-2 fixed.
    /// `ambeginscan` uses this to emit the migration `ERROR`.
    pub fn is_legacy_v2(&self) -> bool {
        self.version < 3
    }

    /// Returns `true` when the meta page is in a wire format the
    /// IVF-1 (v4) binary cannot read.
    ///
    /// **Deliberately always `false`.** The whole point of IVF-1's
    /// opt-in design is that v3 (flat) indexes remain fully readable
    /// and writable under the v4 binary with NO REINDEX: a v3 meta
    /// decodes with `lists = 0` (flat), and the v4 flat decode path
    /// is byte-compatible with v3. There is no pre-v4 format that a
    /// v4 binary genuinely cannot read (v1/v2 are already caught by
    /// `is_legacy_v1`/`is_legacy_v2`). The predicate exists to
    /// satisfy the `AGENTS.md` migration contract (every wire bump
    /// ships an `is_legacy_v{N}()`) and to give a future v5 a place
    /// to flag a genuinely-unreadable v4-and-earlier format. Today
    /// it never trips, so `ambeginscan` never errors on a v3 or v4
    /// index. See `docs/UPGRADING.md`.
    #[allow(dead_code)]
    pub fn is_legacy_v3(&self) -> bool {
        false
    }

    /// Returns `true` when this index uses the IVF layout
    /// (`lists > 0`, v4+). Flat indexes (v3, or v4 with
    /// `lists = 0`) return `false`. The IVF-1 scan path ignores
    /// this (it stays flat regardless); it's here for IVF-2 and
    /// for the round-trip tests.
    pub fn has_ivf(&self) -> bool {
        self.version >= 4 && self.lists > 0
    }

    /// Returns `true` when this index was BUILT as an IVF index
    /// (`lists > 0` recorded on the meta page), regardless of whether
    /// it is currently serving IVF scans. With the tombstone-vacuum
    /// path (Phase E-2) an IVF index keeps `lists > 0` and stays IVF
    /// across a VACUUM, so this equals [`Self::has_ivf`] in the
    /// healthy case. It diverges only when a degrade-to-flat safety
    /// net fired (then `lists > 0` but [`Self::is_degraded`] is
    /// `true`). Used by the scan-time degradation WARNING and the
    /// `turbovec.index_is_degraded()` operator function.
    pub fn index_was_ivf(&self) -> bool {
        self.version >= 4 && self.lists > 0
    }

    /// Returns `true` when an index built `WITH (lists > 0)` is no
    /// longer serving IVF scans — i.e. it has silently degraded to a
    /// flat O(n) scan. This is the queryable signal for the
    /// production latency-cliff landmine: an operator can detect a
    /// degraded index and `REINDEX` it. With tombstone vacuums this
    /// is always `false` for a healthy IVF index; it only trips if a
    /// fallback path explicitly set `ivf_degraded`.
    pub fn is_degraded(&self) -> bool {
        self.index_was_ivf() && (self.ivf_degraded || !self.has_ivf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_meta_v4_flat() {
        let dim: u32 = 384;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta =
            MetaPageData::plan_with_blocked(4, dim, 1_000_000, 7, 12_345_678, 31_250, rotation_bytes);
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert_eq!(back.version, 4);
        assert!(back.has_prepared_layout());
        assert_eq!(back.centroids_slice(), centroids.as_slice());
        assert_eq!(back.boundaries_slice(), boundaries.as_slice());
        assert_eq!(back.rotation_dim, dim);
        assert!(back.rotation_first > 0);
        assert!(back.rotation_count > 0);
        assert!(!back.is_legacy_v1());
        assert!(!back.is_legacy_v2());
        assert!(!back.is_legacy_v3());
        // lists = 0 => flat, not IVF.
        assert_eq!(back.lists, 0);
        assert!(!back.has_ivf());
        assert_eq!(back.coarse_first, 0);
        assert_eq!(back.cell_dir_first, 0);
    }

    #[test]
    fn round_trip_meta_v4_ivf() {
        let dim: u32 = 64;
        let lists: u32 = 16;
        let n: u64 = 4096;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta =
            MetaPageData::plan_with_blocked(4, dim, n, 9, 200_000, 781, rotation_bytes);
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        let coarse_bytes = u64::from(lists) * u64::from(dim) * 4;
        let cell_dir_bytes = u64::from(lists) * 12;
        meta.set_ivf_chains(lists, coarse_bytes, cell_dir_bytes);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert_eq!(back.version, 4);
        assert!(back.has_ivf());
        assert_eq!(back.lists, lists);
        // Coarse + cell-dir chains laid out after rotation, no overlap.
        assert!(back.coarse_first > 0);
        assert!(back.coarse_count > 0);
        assert!(back.cell_dir_first > back.coarse_first);
        assert_eq!(
            back.cell_dir_first,
            back.coarse_first + back.coarse_count,
            "cell dir must immediately follow the coarse chain"
        );
        // The coarse chain must follow the rotation chain.
        assert_eq!(
            back.coarse_first,
            1 + back.codes_count
                + back.scales_count
                + back.ids_count
                + back.blocked_count
                + back.rotation_count,
        );
    }

    #[test]
    fn round_trip_meta_v4_e2_tombstone_and_degraded() {
        // E-2 additive fields (ivf_degraded + tombstone chain) must
        // round-trip and default to a healthy state.
        let dim: u32 = 64;
        let lists: u32 = 16;
        let n: u64 = 4096;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta =
            MetaPageData::plan_with_blocked(4, dim, n, 9, 200_000, 781, rotation_bytes);
        let coarse_bytes = u64::from(lists) * u64::from(dim) * 4;
        let cell_dir_bytes = u64::from(lists) * 12;
        meta.set_ivf_chains(lists, coarse_bytes, cell_dir_bytes);
        // Defaults: healthy, no tombstones.
        assert!(!meta.ivf_degraded);
        assert_eq!(meta.tombstone_bytes, 0);
        assert!(!meta.is_degraded());
        assert!(meta.index_was_ivf());
        // Now stamp a tombstone chain + the degraded flag and
        // round-trip them.
        meta.tombstone_first = meta.total_blocks();
        meta.tombstone_bytes = n.div_ceil(8);
        meta.tombstone_count = MetaPageData::byte_pages_needed(meta.tombstone_bytes);
        meta.ivf_degraded = true;
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert!(back.ivf_degraded);
        assert_eq!(back.tombstone_first, meta.tombstone_first);
        assert_eq!(back.tombstone_bytes, n.div_ceil(8));
        assert_eq!(back.tombstone_count, meta.tombstone_count);
        // is_degraded combines "was IVF" with the flag.
        assert!(back.is_degraded());
    }

    #[test]
    fn flat_index_is_never_degraded() {
        // A flat (lists = 0) index can't "degrade" — it was never IVF.
        let mut meta = MetaPageData::plan(4, 64, 1000, 1);
        meta.ivf_degraded = true; // even with the flag forced
        assert!(!meta.index_was_ivf());
        assert!(!meta.is_degraded());
    }

    #[test]
    fn degraded_via_blanked_lists_reads_not_degraded() {
        // If a path blanks lists to 0 (the pre-E-2 behaviour), the
        // index reads as flat (not IVF, not degraded) — still correct,
        // just no longer recognisable as IVF-built. The E-2 safety net
        // KEEPS lists + sets ivf_degraded so is_degraded() is true.
        let dim: u32 = 64;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta =
            MetaPageData::plan_with_blocked(4, dim, 4096, 9, 200_000, 781, rotation_bytes);
        meta.set_ivf_chains(16, u64::from(16u32) * u64::from(dim) * 4, 16 * 12);
        assert!(meta.is_degraded() == false && meta.has_ivf());
        // Blank lists (pre-E-2): flat, not degraded.
        meta.lists = 0;
        assert!(!meta.index_was_ivf());
        assert!(!meta.is_degraded());
    }

    #[test]
    fn plan_layout_for_million_384d_4bit_with_blocked() {
        // 384/8 * 4 = 192 bytes per row -> floor(8168/192) = 42 rows/page.
        let meta = MetaPageData::plan_with_blocked(4, 384, 1_000_000, 1, 0, 0, 0);
        assert_eq!(meta.stride_bytes, 192);
        assert_eq!(meta.rows_per_codes_page, 42);
        assert_eq!(meta.codes_count, 23810);
        assert_eq!(meta.rows_per_scales_page, 2042);
        assert_eq!(meta.scales_count, 490);
        assert_eq!(meta.rows_per_ids_page, 1021);
        assert_eq!(meta.ids_count, 980);
        // Empty blocked / rotation chains when the byte sizes are 0.
        assert_eq!(meta.blocked_count, 0);
        assert_eq!(meta.blocked_first, 0);
        assert_eq!(meta.rotation_count, 0);
        assert_eq!(meta.rotation_first, 0);
        // chain layout: 1 (meta) + 23810 + 490 + 980 = 25281
        assert_eq!(meta.total_blocks(), 25281);

        // Now plan with a real blocked layout: 1M * 384/2 = ~192 MB
        // and the matching 384x384 rotation matrix (~590 KB).
        let dim: u32 = 384;
        let rot_bytes = u64::from(dim) * u64::from(dim) * 4;
        let with_prepared =
            MetaPageData::plan_with_blocked(4, dim, 1_000_000, 1, 192_000_000, 31_250, rot_bytes);
        let blocked_pages = MetaPageData::byte_pages_needed(192_000_000);
        let rotation_pages = MetaPageData::byte_pages_needed(rot_bytes);
        assert_eq!(with_prepared.blocked_count, blocked_pages);
        assert_eq!(with_prepared.blocked_first, 1 + 23810 + 490 + 980);
        assert_eq!(with_prepared.rotation_count, rotation_pages);
        assert_eq!(
            with_prepared.rotation_first,
            1 + 23810 + 490 + 980 + blocked_pages,
        );
        assert_eq!(
            with_prepared.total_blocks(),
            25281 + blocked_pages + rotation_pages,
        );
    }

    #[test]
    fn rows_on_page_partial_last_page() {
        // 100 rows, 42 per page -> 3 pages: 42, 42, 16
        assert_eq!(MetaPageData::rows_on_page(100, 42, 0), 42);
        assert_eq!(MetaPageData::rows_on_page(100, 42, 1), 42);
        assert_eq!(MetaPageData::rows_on_page(100, 42, 2), 16);
        assert_eq!(MetaPageData::rows_on_page(100, 42, 3), 0);
        // exact multiple: 84 = 2*42, last page is full
        assert_eq!(MetaPageData::rows_on_page(84, 42, 1), 42);
    }

    #[test]
    fn empty_index_has_no_data_pages() {
        let meta = MetaPageData::plan(4, 384, 0, 1);
        assert_eq!(meta.codes_count, 0);
        assert_eq!(meta.scales_count, 0);
        assert_eq!(meta.ids_count, 0);
        assert_eq!(meta.blocked_count, 0);
        assert_eq!(meta.total_blocks(), 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = MetaPageData::plan(4, 8, 0, 1).encode();
        buf[0] = b'X';
        let err = MetaPageData::decode(&buf).unwrap_err();
        assert!(err.contains("magic"));
    }

    #[test]
    fn decodes_legacy_v1_meta_with_zero_blocked_fields() {
        // Hand-craft a v1 meta page (only the first 64 bytes are
        // meaningful; everything else stays zero).
        let mut buf = [0u8; PAYLOAD_BYTES];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4] = 1; // v1
        buf[5] = 4; // bit_width
        buf[8..12].copy_from_slice(&384u32.to_le_bytes());
        buf[12..20].copy_from_slice(&100u64.to_le_bytes());
        buf[20..24].copy_from_slice(&1u32.to_le_bytes());
        buf[24..28].copy_from_slice(&3u32.to_le_bytes());
        buf[28..32].copy_from_slice(&4u32.to_le_bytes());
        buf[32..36].copy_from_slice(&1u32.to_le_bytes());
        buf[36..40].copy_from_slice(&5u32.to_le_bytes());
        buf[40..44].copy_from_slice(&1u32.to_le_bytes());
        buf[44..48].copy_from_slice(&42u32.to_le_bytes());
        buf[48..52].copy_from_slice(&2042u32.to_le_bytes());
        buf[52..56].copy_from_slice(&1021u32.to_le_bytes());
        buf[56..60].copy_from_slice(&192u32.to_le_bytes());
        buf[60..64].copy_from_slice(&3u32.to_le_bytes());

        let meta = MetaPageData::decode(&buf).expect("v1 decode");
        assert_eq!(meta.version, 1);
        assert_eq!(meta.bit_width, 4);
        assert_eq!(meta.n_vectors, 100);
        assert_eq!(meta.am_version, 3);
        // v2/v3 fields zeroed:
        assert_eq!(meta.blocked_first, 0);
        assert_eq!(meta.blocked_bytes, 0);
        assert_eq!(meta.codebook_n_levels, 0);
        assert_eq!(meta.rotation_first, 0);
        assert_eq!(meta.rotation_count, 0);
        assert!(meta.is_legacy_v1());
        assert!(meta.is_legacy_v2());
        assert!(!meta.has_prepared_layout());
    }

    #[test]
    fn decodes_legacy_v2_meta_with_zero_rotation_fields() {
        // Hand-craft a v2 meta page — first 212 bytes are
        // meaningful, the v3 rotation fields stay zero.
        let mut buf = [0u8; PAYLOAD_BYTES];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4] = 2; // v2
        buf[5] = 4;
        buf[8..12].copy_from_slice(&384u32.to_le_bytes());
        buf[12..20].copy_from_slice(&100u64.to_le_bytes());
        buf[20..24].copy_from_slice(&1u32.to_le_bytes());
        buf[24..28].copy_from_slice(&3u32.to_le_bytes());
        buf[28..32].copy_from_slice(&4u32.to_le_bytes());
        buf[32..36].copy_from_slice(&1u32.to_le_bytes());
        buf[36..40].copy_from_slice(&5u32.to_le_bytes());
        buf[40..44].copy_from_slice(&1u32.to_le_bytes());
        buf[44..48].copy_from_slice(&42u32.to_le_bytes());
        buf[48..52].copy_from_slice(&2042u32.to_le_bytes());
        buf[52..56].copy_from_slice(&1021u32.to_le_bytes());
        buf[56..60].copy_from_slice(&192u32.to_le_bytes());
        buf[60..64].copy_from_slice(&3u32.to_le_bytes());
        // Plausible v2 prepared-layout fields.
        buf[64..68].copy_from_slice(&7u32.to_le_bytes()); // blocked_first
        buf[68..72].copy_from_slice(&2u32.to_le_bytes()); // blocked_count
        buf[72..80].copy_from_slice(&12_000u64.to_le_bytes()); // blocked_bytes
        buf[80..84].copy_from_slice(&5u32.to_le_bytes()); // n_blocks_blocked
        buf[84..88].copy_from_slice(&16u32.to_le_bytes()); // codebook_n_levels
        // Centroid/boundary tables stay zero — fine for the decoder.

        let meta = MetaPageData::decode(&buf).expect("v2 decode");
        assert_eq!(meta.version, 2);
        assert_eq!(meta.blocked_first, 7);
        assert_eq!(meta.blocked_bytes, 12_000);
        assert_eq!(meta.codebook_n_levels, 16);
        // v3 fields zeroed:
        assert_eq!(meta.rotation_first, 0);
        assert_eq!(meta.rotation_count, 0);
        assert_eq!(meta.rotation_dim, 0);
        assert!(!meta.is_legacy_v1());
        assert!(meta.is_legacy_v2(), "v2 must trip the legacy_v2 flag");
        // v2 has no rotation chain so has_prepared_layout is false.
        assert!(!meta.has_prepared_layout());
    }

    #[test]
    fn decodes_legacy_v3_meta_as_flat_under_v4() {
        // A v3 meta page (no v4 IVF fields) must decode under the v4
        // binary as a flat index (lists = 0), readable with NO
        // REINDEX. We forge a v3 page: first 224 bytes meaningful,
        // the v4 fields stay zero.
        let mut buf = [0u8; PAYLOAD_BYTES];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4] = 3; // v3
        buf[5] = 4;
        buf[8..12].copy_from_slice(&384u32.to_le_bytes());
        buf[12..20].copy_from_slice(&100u64.to_le_bytes());
        buf[20..24].copy_from_slice(&1u32.to_le_bytes());
        buf[24..28].copy_from_slice(&3u32.to_le_bytes());
        buf[28..32].copy_from_slice(&4u32.to_le_bytes());
        buf[32..36].copy_from_slice(&1u32.to_le_bytes());
        buf[36..40].copy_from_slice(&5u32.to_le_bytes());
        buf[40..44].copy_from_slice(&1u32.to_le_bytes());
        buf[44..48].copy_from_slice(&42u32.to_le_bytes());
        buf[48..52].copy_from_slice(&2042u32.to_le_bytes());
        buf[52..56].copy_from_slice(&1021u32.to_le_bytes());
        buf[56..60].copy_from_slice(&192u32.to_le_bytes());
        buf[60..64].copy_from_slice(&3u32.to_le_bytes());
        // v2 prepared-layout fields.
        buf[64..68].copy_from_slice(&7u32.to_le_bytes());
        buf[68..72].copy_from_slice(&2u32.to_le_bytes());
        buf[72..80].copy_from_slice(&12_000u64.to_le_bytes());
        buf[80..84].copy_from_slice(&5u32.to_le_bytes());
        buf[84..88].copy_from_slice(&16u32.to_le_bytes());
        // v3 rotation fields (data offset 212..224).
        buf[212..216].copy_from_slice(&9u32.to_le_bytes()); // rotation_first
        buf[216..220].copy_from_slice(&72u32.to_le_bytes()); // rotation_count
        buf[220..224].copy_from_slice(&384u32.to_le_bytes()); // rotation_dim

        let meta = MetaPageData::decode(&buf).expect("v3 decode under v4 binary");
        assert_eq!(meta.version, 3);
        assert_eq!(meta.rotation_first, 9);
        assert_eq!(meta.rotation_dim, 384);
        // v4 IVF fields zeroed => flat, no IVF.
        assert_eq!(meta.lists, 0);
        assert!(!meta.has_ivf());
        assert_eq!(meta.coarse_first, 0);
        assert_eq!(meta.cell_dir_first, 0);
        // A v3 index is NOT legacy under v4 — it scans flat, no REINDEX.
        assert!(!meta.is_legacy_v1());
        assert!(!meta.is_legacy_v2());
        assert!(!meta.is_legacy_v3());
        // It still reports its prepared layout so the flat scan path
        // works.
        assert!(meta.has_prepared_layout());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = MetaPageData::plan(4, 8, 0, 1).encode();
        buf[4] = 99; // bogus future version
        let err = MetaPageData::decode(&buf).unwrap_err();
        assert!(err.contains("version"));
    }
}
