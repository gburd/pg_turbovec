//! Phase L — relfile-resident page layouts for the `turbovec`
//! access method.
//!
//! **Status: STUB** — only the layout types and pure-bytes
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
//!   28     1  version = 2                             |   / BLCKSZ; we don't use
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
//!   88     4  blocked_first (BlockNumber)             |  v2 only
//!   92     4  blocked_count (u32)                     |
//!   96     8  blocked_bytes (u64)                     |
//!  104     4  n_blocks_blocked (u32)                  |
//!  108     4  codebook_n_levels (u32) = 1 << bit_width|
//!  112    64  centroids[16] (f32, zero-padded tail)   |
//!  176    60  boundaries[15] (f32, zero-padded tail)  |
//!  236     4  reserved (zero)                         |
//!  240   ...  reserved (zero)                         /
//! ```
//!
//! After the meta block come three contiguous page chains for
//! the row-major codes / scales / ids, followed in v2 by a
//! fourth chain holding the prepared SIMD-blocked layout. The
//! blocked chain is a flat byte chain (no fixed row stride):
//! every page after the page header is `PAYLOAD_BYTES` of raw
//! blocked bytes, with the last page holding the residual tail.
//!
//! Reading: walk the chain, concatenate `[24..PAGE_END]` bytes
//! per page until `blocked_bytes` has been consumed.
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
/// `1` — Phase L: meta + 3 chains (codes / scales / ids).
/// `2` — Phase P: meta + 4 chains, with the prepared SIMD-blocked
///       layout persisted in the new `blocked` chain and the
///       Lloyd-Max codebook stored inline on the meta page.
///       Backends opening a v2 index skip the per-backend
///       `pack::repack` (~12–15 s on 1 M × 1536-d) and Lloyd-Max
///       compute (~5–8 s).
pub const VERSION: u8 = 2;

/// The on-disk version we read **and** write today. Decode
/// accepts strictly older versions for migration-HINT purposes
/// (callers detect them via [`MetaPageData::version`]) but cannot
/// upgrade them in place — a REINDEX rewrites the relation under
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
    /// then ids, then the prepared blocked chain.
    ///
    /// `blocked_bytes` is the total size of the prepared SIMD-
    /// blocked layout (output of `turbovec::pack::repack`). Pass
    /// `0` for an empty index or when the prepared layout isn't
    /// being persisted (which gives a v2 meta with an empty
    /// blocked chain — readers fall back to per-backend repack).
    /// `n_blocks_blocked` is the matching `n_blocks` count from
    /// `pack::repack`.
    pub fn plan_with_blocked(
        bit_width: u8,
        dim: u32,
        n_vectors: u64,
        am_version: u32,
        blocked_bytes: u64,
        n_blocks_blocked: u32,
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

        let codes_first = 1;
        let scales_first = codes_first + codes_count;
        let ids_first = scales_first + scales_count;
        let blocked_first = ids_first + ids_count;

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
            blocked_first: if blocked_bytes > 0 { blocked_first } else { 0 },
            blocked_count,
            blocked_bytes,
            n_blocks_blocked,
            codebook_n_levels: 0,
            centroids: [0.0; MAX_CODEBOOK_LEVELS],
            boundaries: [0.0; MAX_CODEBOOK_LEVELS - 1],
        }
    }

    /// Plan a layout without a prepared blocked chain. Equivalent
    /// to `plan_with_blocked(…, blocked_bytes = 0,
    /// n_blocks_blocked = 0)`. Used by `aminsert` paths that
    /// rewrite the relfile incrementally and don't have the
    /// prepared layout handy. Readers fall back to per-backend
    /// `pack::repack` for these indexes — same as v1.
    pub fn plan(bit_width: u8, dim: u32, n_vectors: u64, am_version: u32) -> Self {
        Self::plan_with_blocked(bit_width, dim, n_vectors, am_version, 0, 0)
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
        1 + self.codes_count + self.scales_count + self.ids_count + self.blocked_count
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
        // Trailing bytes reserved (zero).
        out
    }

    /// Inverse of [`Self::encode`]. Input must be the page's data
    /// region (no PG page header) of at least 64 bytes; longer is
    /// fine. Accepts both v1 (Phase L) and v2 (Phase P) layouts —
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
    /// built under the current (Phase P) wire format with a
    /// prepared blocked layout actually present. v1 indexes and
    /// empty v2 indexes (no rows yet) return `false`.
    pub fn has_prepared_layout(&self) -> bool {
        self.version >= VERSION && self.blocked_bytes > 0 && self.codebook_n_levels > 0
    }

    /// Returns `true` when the meta page is in the older v1 wire
    /// format. `ambeginscan` uses this to emit the migration
    /// HINT directing the user to `REINDEX INDEX <name>;`.
    pub fn is_legacy_v1(&self) -> bool {
        self.version < VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_meta_v2() {
        let mut meta =
            MetaPageData::plan_with_blocked(4, 384, 1_000_000, 7, 12_345_678, 31_250);
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert_eq!(back.version, 2);
        assert!(back.has_prepared_layout());
        assert_eq!(back.centroids_slice(), centroids.as_slice());
        assert_eq!(back.boundaries_slice(), boundaries.as_slice());
    }

    #[test]
    fn plan_layout_for_million_384d_4bit_with_blocked() {
        // 384/8 * 4 = 192 bytes per row -> floor(8168/192) = 42 rows/page.
        let meta = MetaPageData::plan_with_blocked(4, 384, 1_000_000, 1, 0, 0);
        assert_eq!(meta.stride_bytes, 192);
        assert_eq!(meta.rows_per_codes_page, 42);
        assert_eq!(meta.codes_count, 23810);
        assert_eq!(meta.rows_per_scales_page, 2042);
        assert_eq!(meta.scales_count, 490);
        assert_eq!(meta.rows_per_ids_page, 1021);
        assert_eq!(meta.ids_count, 980);
        // Empty blocked chain when blocked_bytes = 0.
        assert_eq!(meta.blocked_count, 0);
        assert_eq!(meta.blocked_first, 0);
        // chain layout: 1 (meta) + 23810 + 490 + 980 = 25281
        assert_eq!(meta.total_blocks(), 25281);

        // Now plan with a real blocked layout: 1M * 384/2 = ~192 MB.
        // (codes_per_byte = 2 at 4-bit, n_byte_groups = dim/2 = 192,
        //  n_blocks = ceil(1M/32) = 31250, blocked_bytes = 31250*192*32 = 192_000_000.)
        let with_blocked = MetaPageData::plan_with_blocked(4, 384, 1_000_000, 1, 192_000_000, 31_250);
        let blocked_pages = MetaPageData::byte_pages_needed(192_000_000);
        assert_eq!(with_blocked.blocked_count, blocked_pages);
        assert_eq!(with_blocked.blocked_first, 1 + 23810 + 490 + 980);
        assert_eq!(
            with_blocked.total_blocks(),
            25281 + blocked_pages,
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
        // v2 fields zeroed:
        assert_eq!(meta.blocked_first, 0);
        assert_eq!(meta.blocked_bytes, 0);
        assert_eq!(meta.codebook_n_levels, 0);
        assert!(meta.is_legacy_v1());
        assert!(!meta.has_prepared_layout());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = MetaPageData::plan(4, 8, 0, 1).encode();
        buf[4] = 99; // bogus future version
        let err = MetaPageData::decode(&buf).unwrap_err();
        assert!(err.contains("version"));
    }
}
