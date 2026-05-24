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
//!   28     1  version = 1                             |   / BLCKSZ; we don't use
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
//!   88    40  reserved (zero)                         /
//! ```
//!
//! After the meta block come three contiguous page chains. Layout
//! within each non-meta page:
//!
//! ```text
//!  off  size  field
//!    0    24  PageHeader (PostgreSQL standard)
//!   24     ?  packed rows: stride_bytes per row, rows_per_*_page rows max
//! ```
//!
//! There is **no per-page header** beyond the standard PG header —
//! the meta page tells the reader how many rows live on each page,
//! which is uniform except for the *last* page in each chain (live
//! row count = `n_vectors mod rows_per_page`, or `rows_per_page`
//! when divisible).
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

#![cfg(feature = "relfile_storage")]

use core::mem::size_of;

/// 4-byte file magic. "TurboVec RelMain".
pub const MAGIC: [u8; 4] = *b"TVRM";

/// On-disk format version.
pub const VERSION: u8 = 1;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaPageData {
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
    /// then ids.
    pub fn plan(bit_width: u8, dim: u32, n_vectors: u64, am_version: u32) -> Self {
        assert_eq!(dim % 8, 0, "dim must be a multiple of 8");
        let stride_bytes = Self::codes_stride(bit_width, dim);
        let rows_per_codes_page = Self::rows_per_page(stride_bytes);
        let rows_per_scales_page = Self::rows_per_page(size_of::<f32>() as u32);
        let rows_per_ids_page = Self::rows_per_page(size_of::<u64>() as u32);

        let codes_count = Self::pages_needed(n_vectors, rows_per_codes_page);
        let scales_count = Self::pages_needed(n_vectors, rows_per_scales_page);
        let ids_count = Self::pages_needed(n_vectors, rows_per_ids_page);

        let codes_first = 1;
        let scales_first = codes_first + codes_count;
        let ids_first = scales_first + scales_count;

        Self {
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
        }
    }

    /// Total number of blocks (including meta) required for this
    /// layout. This is the value we extend the relation to.
    #[allow(dead_code)] // exercised by tests; not yet read by relfile.rs
    pub fn total_blocks(&self) -> u32 {
        1 + self.codes_count + self.scales_count + self.ids_count
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
        // out[64..] reserved
        out
    }

    /// Inverse of [`Self::encode`]. Input must be the page's data
    /// region (no PG page header) of at least 64 bytes; longer is
    /// fine.
    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 64 {
            return Err("meta page data region too short");
        }
        if bytes[0..4] != MAGIC {
            return Err("bad magic on meta page");
        }
        if bytes[4] != VERSION {
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
        Ok(Self {
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_meta() {
        let meta = MetaPageData::plan(4, 384, 1_000_000, 7);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
    }

    #[test]
    fn plan_layout_for_million_384d_4bit() {
        // 384/8 * 4 = 192 bytes per row -> floor(8168/192) = 42 rows/page.
        let meta = MetaPageData::plan(4, 384, 1_000_000, 1);
        assert_eq!(meta.stride_bytes, 192);
        assert_eq!(meta.rows_per_codes_page, 42);
        // ceil(1e6 / 42) = 23810
        assert_eq!(meta.codes_count, 23810);
        // floor(8168/4) = 2042 -> ceil(1e6/2042) = 490
        assert_eq!(meta.rows_per_scales_page, 2042);
        assert_eq!(meta.scales_count, 490);
        // floor(8168/8) = 1021 -> ceil(1e6/1021) = 980
        assert_eq!(meta.rows_per_ids_page, 1021);
        assert_eq!(meta.ids_count, 980);
        // chain layout: 1 (meta) + 23810 + 490 + 980 = 25281
        assert_eq!(meta.total_blocks(), 25281);
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
        assert_eq!(meta.total_blocks(), 1);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = MetaPageData::plan(4, 8, 0, 1).encode();
        buf[0] = b'X';
        let err = MetaPageData::decode(&buf).unwrap_err();
        assert!(err.contains("magic"));
    }
}
