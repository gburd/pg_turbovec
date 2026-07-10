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
//!   30     1  kind = 0 single-vec / 1 colbert (v5)    |   data lives in the data
//!   31     1  reserved (zero)                         |
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
//!  288     4  graph_first    (BlockNumber)            |  v6 (Phase G-2a)
//!  292     4  graph_count    (u32)                    |
//!  296     8  graph_offsets_bytes (u64)                |
//!  304     8  graph_neighbors_bytes (u64)              |
//!  312     4  graph_entry_point (u32)                 |
//!  316   ...  reserved (zero)                         /
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
//! `kind = KIND_GRAPH` (v6, Phase G-2a) is a DIFFERENT index shape
//! from `KIND_SINGLE`/`KIND_COLBERT`: the codes/scales/ids chains
//! are laid out exactly like a flat single-vector index (same
//! TurboQuant row storage — a graph node's vector is stored
//! identically to a flat row), but `lists` stays 0 (a graph index
//! is not IVF) and an EIGHTH chain (the adjacency chain, CSR-style:
//! a flat `u32` offsets array of length `n_vectors + 1` immediately
//! followed by a flat `u32` neighbor-id array) is appended after
//! whichever prior chains this kind's meta fields point past
//! (blocked + rotation, since a graph index reuses the prepared-
//! layout path). `graph_entry_point` is the slot id the greedy
//! beam search starts from. A single-vector (`KIND_SINGLE`) or
//! ColBERT (`KIND_COLBERT`) meta page has the graph fields zeroed,
//! which decodes as "no adjacency chain" — the additive
//! backward-compat path; see [`VERSION`]'s doc comment.
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
///       profile (see an internal design note).
/// `4` - IVF-1: meta + (when `lists > 0`) 7 chains, adding the
///       coarse centroids + cell directory for the inverted-file
///       layer (an internal design note). IVF is opt-in via
///       `WITH (lists = N)`; `lists = 0` (the default) is
///       byte-identical to v3 modulo the version byte, so the v3
///       flat decode path stays valid and existing v3 indexes
///       need no REINDEX. The scan path is still FLAT in IVF-1
///       (cells are persisted but not yet probed); cell-restricted
///       search is IVF-2.
/// `5` - Phase F-2: ADDITIVE multivector/ColBERT index kind. The
///       new `kind` byte (page offset 30, formerly reserved) is `1`
///       for a ColBERT token index and `0` for the single-vector
///       index. **A single-vector index still emits version=4** —
///       `encode` only writes 5 when `kind == KIND_COLBERT`. So a
///       v4 single-vector relfile is BYTE-IDENTICAL under the v5
///       binary (the kind byte was already a zeroed reserved byte),
///       and existing v4 indexes need no REINDEX. A ColBERT index
///       is a brand-new on-disk shape (token slots, doc-id repeated
///       in the ids chain) that only a v5 binary produces; there is
///       no in-place migration of a v4 index into a v5 ColBERT one.
/// `6` - Phase G-2a: ADDITIVE Vamana-graph index kind
///       (an internal design note).
///       `kind = KIND_GRAPH` (`2`) marks a `WITH (graph = true)`
///       build: the codes/scales/ids chains are stored exactly like
///       a flat single-vector index (same TurboQuant row storage),
///       plus a new adjacency chain (CSR: `u32` offsets of length
///       `n_vectors + 1` followed by a flat `u32` neighbor-id array)
///       and a `graph_entry_point` slot id. **A single-vector or
///       ColBERT index still emits version 4/5 respectively** —
///       `encode` only writes 6 when `kind == KIND_GRAPH`, so v4 and
///       v5 relfiles are BYTE-IDENTICAL under the v6 binary (the new
///       graph fields were already zeroed-reserved bytes on v4/v5),
///       and existing v4/v5 indexes need no REINDEX. A graph index is
///       a brand-new on-disk shape that only a v6 binary produces;
///       there is no in-place migration of a v4/v5 index into a v6
///       graph one — it is built fresh via `WITH (graph = true)`.
/// `7` - Phase Q-0: de-duplicate the on-disk codes storage. Prior
///       versions persisted the quantized codes TWICE — the
///       row-major bit-plane `packed_codes` chain AND the
///       SIMD-`blocked` chain (`pack::repack(packed_codes, …)`),
///       doubling the dominant O(n) storage term. Since the blocked
///       layout is a PURE FUNCTION of the packed codes, v7 drops the
///       blocked chain entirely and recomputes it once per backend at
///       index-open via `pack::repack` (the same one-time compute a
///       pre-v2 index already paid on first scan). This roughly halves
///       the per-vector on-disk footprint (e.g. 768d/4-bit: 384 B
///       codes stored once, not twice). **This is NOT additive** — a
///       v7 relfile has no blocked chain, so it is NOT byte-compatible
///       with any prior version FOR ANY KIND (single-vector, ColBERT,
///       or graph). Unlike the v4→v5→v6 additive bumps, EVERY kind now
///       emits wire version 7 (the `kind` byte still discriminates
///       single/colbert/graph). A pre-v7 index is detected by
///       [`MetaPageData::is_legacy_v6`] (`version < 7`) and REINDEXed;
///       there is no in-place migration. The maintainer OK'd this
///       REINDEX for the 100M-in-40GB storage win — see
///       `docs/UPGRADING.md`.
pub const VERSION: u8 = 7;

/// Wire version a ColBERT (`KIND_COLBERT`) index emits. As of Phase
/// Q-0 (v7) this equals [`VERSION`]: the codes-dedup change is not
/// additive, so a ColBERT index emits v7 like every other kind (the
/// `kind` byte is the discriminator). [`MetaPageData::mark_colbert`]
/// stamps `kind = KIND_COLBERT` without changing the version.
const COLBERT_VERSION: u8 = VERSION;

/// Index kind discriminator (page offset 30, formerly a reserved
/// byte). `0` = single-vector (the v1..v4 default; a `vector` column
/// with `vec_*_ops`). `1` = ColBERT/multivector token index (a
/// `turbovec.vector[]` column with `vec_colbert_ops`, Phase F-2). `2`
/// = Vamana graph index (a `vector` column built `WITH (graph =
/// true)`, Phase G-2a). A v4 or v5 meta page has this byte zeroed or
/// set to `KIND_COLBERT` respectively, never `KIND_GRAPH`, so it
/// never misdecodes as a graph index — the additive backward-compat
/// path.
pub const KIND_SINGLE: u8 = 0;
/// ColBERT/multivector token index kind. See [`KIND_SINGLE`].
pub const KIND_COLBERT: u8 = 1;
/// Vamana graph index kind (Phase G-2a). See [`KIND_SINGLE`].
pub const KIND_GRAPH: u8 = 2;

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
    /// Index kind: [`KIND_SINGLE`] (0, single-vector),
    /// [`KIND_COLBERT`] (1, multivector token index), or
    /// [`KIND_GRAPH`] (2, Vamana graph index, Phase G-2a). A v4/v5
    /// meta page has the kind byte zeroed or set to `KIND_COLBERT`,
    /// never `KIND_GRAPH`, so it decodes as single-vector/colbert.
    pub kind: u8,
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

    // ---- v6 fields (Phase G-2a graph index) ----
    /// First block of the graph adjacency chain (CSR: a flat `u32`
    /// offsets array of length `n_vectors + 1`, immediately followed
    /// by a flat `u32` neighbor-id array). `0` when this is not a
    /// graph index (`kind != KIND_GRAPH`) or the graph is empty.
    pub graph_first: u32,
    /// Number of pages in the adjacency chain.
    pub graph_count: u32,
    /// Byte length of the CSR offsets sub-chain (`(n_vectors + 1) *
    /// 4`). Needed to find where the neighbor-id sub-chain starts
    /// within the flat byte chain.
    pub graph_offsets_bytes: u64,
    /// Byte length of the CSR neighbor-id sub-chain.
    pub graph_neighbors_bytes: u64,
    /// Slot id of the graph's entry point (where greedy beam search
    /// starts). `0` when this is not a graph index or the graph is
    /// empty.
    pub graph_entry_point: u32,
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
            // Phase Q-0 (v7): the codes-dedup change is not additive, so
            // EVERY kind emits wire version 7 (a v7 relfile has no
            // blocked chain and is not byte-compatible with any prior
            // version). `kind` still discriminates single/colbert/graph;
            // mark_colbert() / set_graph_chain() flip only `kind`, not
            // the version.
            version: VERSION,
            bit_width,
            // plan_with_blocked always plans a SINGLE-vector layout;
            // the colbert build calls mark_colbert() after planning
            // (mirroring how set_ivf_chains flips `lists`).
            kind: KIND_SINGLE,
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
            blocked_first: if blocked_bytes > 0 {
                blocked_first_blkno
            } else {
                0
            },
            blocked_count,
            blocked_bytes,
            n_blocks_blocked,
            codebook_n_levels: 0,
            centroids: [0.0; MAX_CODEBOOK_LEVELS],
            boundaries: [0.0; MAX_CODEBOOK_LEVELS - 1],
            rotation_first: if rotation_bytes > 0 {
                rotation_first_blkno
            } else {
                0
            },
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
            // v6 graph fields default to "no adjacency chain";
            // set_graph_chain() fills them in (after every prior
            // chain) when the build opts into kind = KIND_GRAPH.
            graph_first: 0,
            graph_count: 0,
            graph_offsets_bytes: 0,
            graph_neighbors_bytes: 0,
            graph_entry_point: 0,
        }
    }

    /// Lay out the v6 graph adjacency chain (Phase G-2a) AFTER every
    /// prior chain (row-major codes/scales/ids, blocked, rotation,
    /// and — harmlessly, since a graph build never sets `lists` —
    /// the IVF coarse/cell-dir/tombstone chains), and stamp `kind =
    /// KIND_GRAPH` + bump `version` to [`VERSION`] (6) in lock-step,
    /// mirroring [`Self::mark_colbert`]'s pattern. Must be called on
    /// a meta already planned by [`Self::plan_with_blocked`] (so
    /// every prior chain's offsets are fixed).
    ///
    /// `offsets_bytes` is the byte length of the CSR offsets
    /// sub-chain (`(n_vectors + 1) * 4`); `neighbors_bytes` is the
    /// byte length of the flat neighbor-id sub-chain. The two
    /// sub-chains are concatenated into ONE flat byte chain (offsets
    /// first, then neighbors) starting at `graph_first`. Pass `0` for
    /// both on an empty (0-row) graph build, which leaves this a
    /// no-op (both chain fields stay 0).
    pub fn set_graph_chain(&mut self, offsets_bytes: u64, neighbors_bytes: u64, entry_point: u32) {
        if offsets_bytes == 0 && neighbors_bytes == 0 {
            self.graph_first = 0;
            self.graph_count = 0;
            self.graph_offsets_bytes = 0;
            self.graph_neighbors_bytes = 0;
            self.graph_entry_point = 0;
            return;
        }
        let after_every_prior_chain = 1
            + self.codes_count
            + self.scales_count
            + self.ids_count
            + self.blocked_count
            + self.rotation_count
            + self.coarse_count
            + self.cell_dir_count
            + self.tombstone_count;
        let total_bytes = offsets_bytes + neighbors_bytes;
        self.graph_first = after_every_prior_chain;
        self.graph_count = Self::byte_pages_needed(total_bytes);
        self.graph_offsets_bytes = offsets_bytes;
        self.graph_neighbors_bytes = neighbors_bytes;
        self.graph_entry_point = entry_point;
        self.kind = KIND_GRAPH;
        // Phase Q-0 (v7): version stays at VERSION (7) for every kind;
        // only `kind` discriminates. (Pre-v7 this also bumped version.)
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
    pub fn set_ivf_chains(&mut self, lists: u32, coarse_bytes: u64, cell_dir_bytes: u64) {
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

    /// Mark this meta page as a ColBERT / multivector token index
    /// (Phase F-2). Flips `kind` to [`KIND_COLBERT`] and bumps
    /// `version` to `COLBERT_VERSION` (5) in lock-step, so `encode`
    /// emits the v5 wire format. Call AFTER `plan_with_blocked` /
    /// `set_ivf_chains` (which always plan a single-vector v4 layout);
    /// the chain layout is otherwise identical to a single-vector IVF
    /// index — only the discriminator changes. A single-vector index
    /// never calls this, so it stays byte-identical to v1.16.0.
    pub fn mark_colbert(&mut self) {
        self.kind = KIND_COLBERT;
        // Phase Q-0 (v7): COLBERT_VERSION == VERSION, so this is a
        // no-op on the version; kept explicit to document intent.
        self.version = COLBERT_VERSION;
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
            + self.graph_count
    }

    /// Serialise the meta header (no PG page header) to a
    /// `PAYLOAD_BYTES`-sized buffer suitable for memcpy into the
    /// data area of block 0.
    pub fn encode(&self) -> [u8; PAYLOAD_BYTES] {
        let mut out = [0u8; PAYLOAD_BYTES];
        out[0..4].copy_from_slice(&MAGIC);
        // Wire-version is Phase Q-0 (v7) and NO LONGER additive-per-
        // kind: every kind emits version 7 (the codes-dedup change
        // dropped the blocked chain, so a v7 relfile is not
        // byte-compatible with any prior version, for any kind). The
        // `kind` byte (offset 6) is the sole kind discriminator; the
        // version byte is the belt-and-braces signal a pre-v7 binary
        // uses to refuse the index outright (is_legacy_v6).
        debug_assert!(
            self.version == VERSION && matches!(self.kind, KIND_SINGLE | KIND_COLBERT | KIND_GRAPH),
            "version/kind out of sync: kind={} version={}",
            self.kind,
            self.version,
        );
        out[4] = self.version;
        out[5] = self.bit_width;
        out[6] = self.kind;
        // out[7] reserved
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
        // v6 fields begin at e2_base + 20 = 264 (page offset 288):
        // graph_first + graph_count (u32 each) + graph_offsets_bytes +
        // graph_neighbors_bytes (u64 each) + graph_entry_point (u32).
        let v6_base = e2_base + 20;
        out[v6_base..v6_base + 4].copy_from_slice(&self.graph_first.to_le_bytes());
        out[v6_base + 4..v6_base + 8].copy_from_slice(&self.graph_count.to_le_bytes());
        out[v6_base + 8..v6_base + 16].copy_from_slice(&self.graph_offsets_bytes.to_le_bytes());
        out[v6_base + 16..v6_base + 24].copy_from_slice(&self.graph_neighbors_bytes.to_le_bytes());
        out[v6_base + 24..v6_base + 28].copy_from_slice(&self.graph_entry_point.to_le_bytes());
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
        // Kind byte (offset 6). On a v1..v4 meta page this is a zeroed
        // reserved byte, so it reads as KIND_SINGLE — the additive
        // backward-compat path. Only a v5 ColBERT index sets it to
        // KIND_COLBERT.
        let kind = bytes[6];
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
            kind,
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
            graph_first: 0,
            graph_count: 0,
            graph_offsets_bytes: 0,
            graph_neighbors_bytes: 0,
            graph_entry_point: 0,
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
                me.centroids[i] = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            }
            let bound_base = 88 + MAX_CODEBOOK_LEVELS * 4;
            for i in 0..MAX_CODEBOOK_LEVELS - 1 {
                let off = bound_base + i * 4;
                me.boundaries[i] = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            }
        }

        if version >= 3 {
            // v3 needs at least 224 bytes (212 + 12 for rotation
            // first/count/dim).
            if bytes.len() < 224 {
                return Err("v3 meta page data region too short");
            }
            let v3_base = 212;
            me.rotation_first = u32::from_le_bytes(bytes[v3_base..v3_base + 4].try_into().unwrap());
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
            // v6 graph fields (data offset 264..292, page offset
            // 288..316). Additive: a pre-G-2a v4/v5 meta has these
            // bytes zeroed ("no adjacency chain"), the backward-compat
            // path, no REINDEX. Only decoded when the buffer is long
            // enough; a short test buffer (>=264, <292) leaves them at
            // the struct default of 0.
            let v6_base = e2_base + 20; // 264
            if bytes.len() >= v6_base + 28 {
                me.graph_first =
                    u32::from_le_bytes(bytes[v6_base..v6_base + 4].try_into().unwrap());
                me.graph_count =
                    u32::from_le_bytes(bytes[v6_base + 4..v6_base + 8].try_into().unwrap());
                me.graph_offsets_bytes =
                    u64::from_le_bytes(bytes[v6_base + 8..v6_base + 16].try_into().unwrap());
                me.graph_neighbors_bytes =
                    u64::from_le_bytes(bytes[v6_base + 16..v6_base + 24].try_into().unwrap());
                me.graph_entry_point =
                    u32::from_le_bytes(bytes[v6_base + 24..v6_base + 28].try_into().unwrap());
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
    /// built under a wire format with the prepared caches actually
    /// present: a persisted Lloyd-Max codebook AND a persisted
    /// rotation matrix. v1/v2 indexes and empty (no-rows) v3+ indexes
    /// return `false`.
    ///
    /// Phase Q-0 (v7) note: v7 no longer persists the SIMD-blocked
    /// chain (it's recomputed once per backend at index-open via
    /// `pack::repack`), so this NO LONGER requires `blocked_bytes >
    /// 0`. It checks `version >= 3` for the historical (pre-v7)
    /// path where the blocked chain WAS persisted — for those the
    /// install path reads it — but note ambeginscan errors out on
    /// any `version < 7` index before install runs, so at runtime
    /// this only ever returns true for v7 indexes with a non-empty
    /// codebook + rotation. The install path recomputes the blocked
    /// layout from the packed codes regardless.
    pub fn has_prepared_layout(&self) -> bool {
        self.version >= 3 && self.codebook_n_levels > 0 && self.rotation_count > 0
    }

    /// Returns `true` when the meta page is in the older v1 wire
    /// format (Phase L preview, pre-v1.3.0). `ambeginscan` uses
    /// this to emit the migration `ERROR` directing the user to
    /// `REINDEX INDEX <name>;`.
    ///
    /// Phase Q-0 (v7): superseded at the scan gate by
    /// [`Self::is_legacy_v6`] (which fires on any `version < 7`), but
    /// kept for its specific historical meaning and the round-trip
    /// tests. `#[allow(dead_code)]` because the runtime path now uses
    /// the broader v6 predicate.
    #[allow(dead_code)]
    pub fn is_legacy_v1(&self) -> bool {
        self.version < 2
    }

    /// Returns `true` when the meta page is in the v2 wire
    /// format (v1.3.x; Phase P prepared layout but no persisted
    /// rotation matrix). v1.4.0+ binaries refuse to scan these
    /// because the rotation chain offsets don't exist on disk
    /// and the lazy QR was the warm-scan hotspot Phase R-2 fixed.
    /// `ambeginscan` uses this to emit the migration `ERROR`.
    ///
    /// Phase Q-0 (v7): superseded at the scan gate by
    /// [`Self::is_legacy_v6`]; kept for its historical meaning and
    /// the round-trip tests.
    #[allow(dead_code)]
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

    /// Returns `true` when the meta page is in a wire format the
    /// Phase F-2 (v5) binary cannot read.
    ///
    /// **Deliberately always `false`.** v5 is purely additive: a v4
    /// single-vector meta page decodes under the v5 binary as
    /// `kind = KIND_SINGLE` (the kind byte was a zeroed reserved byte
    /// on v4), and the single-vector flat/IVF decode path is
    /// byte-compatible with v4. So a v4 index remains fully readable
    /// and writable under v5 with NO REINDEX. There is no pre-v5
    /// format a v5 binary genuinely cannot read (v1/v2 are already
    /// caught by `is_legacy_v1`/`is_legacy_v2`; v3/v4 read fine). The
    /// predicate exists to satisfy the `AGENTS.md` migration contract
    /// (every wire bump ships an `is_legacy_v{N}()`) and gives a
    /// future v6 a place to flag a genuinely-unreadable v5-and-earlier
    /// format. Today it never trips. See `docs/UPGRADING.md`.
    #[allow(dead_code)]
    pub fn is_legacy_v4(&self) -> bool {
        false
    }

    /// Returns `true` when the meta page is in a wire format the
    /// Phase G-2a (v6) binary cannot read.
    ///
    /// **Deliberately always `false`.** v6 is purely additive: a v4
    /// single-vector or v5 ColBERT meta page decodes under the v6
    /// binary with the graph fields zeroed (`kind` stays
    /// `KIND_SINGLE`/`KIND_COLBERT`), and the existing decode paths
    /// are byte-compatible with v4/v5. So a v4/v5 index remains fully
    /// readable and writable under v6 with NO REINDEX. There is no
    /// pre-v6 format a v6 binary genuinely cannot read (v1/v2 are
    /// already caught by `is_legacy_v1`/`is_legacy_v2`; v3/v4/v5 read
    /// fine). The predicate exists to satisfy the `AGENTS.md`
    /// migration contract (every wire bump ships an `is_legacy_v{N}()`)
    /// and gives a future v7 a place to flag a genuinely-unreadable
    /// v6-and-earlier format. Today it never trips. See
    /// `docs/UPGRADING.md`.
    #[allow(dead_code)]
    pub fn is_legacy_v5(&self) -> bool {
        false
    }

    /// Returns `true` when the meta page is in a wire format the
    /// Phase Q-0 (v7) binary cannot read — i.e. ANY pre-v7 index
    /// (v1..v6).
    ///
    /// **Unlike the deliberately-always-`false` v3/v4/v5 predicates,
    /// this one genuinely trips.** Phase Q-0 de-duplicated the on-disk
    /// codes storage by dropping the persisted SIMD-blocked chain,
    /// which every prior version (v4 single-vector, v5 ColBERT, v6
    /// graph) DID persist. A v7 relfile is therefore NOT
    /// byte-compatible with any pre-v7 index for any kind, so a pre-v7
    /// index must be REINDEXed. `ambeginscan` (and `amgettuple`'s
    /// first fetch) uses this to emit a clear `ERROR` naming the index
    /// with a `HINT: REINDEX INDEX <name>;`. See `docs/UPGRADING.md`.
    pub fn is_legacy_v6(&self) -> bool {
        self.version < VERSION
    }

    /// Returns `true` when this meta page describes a ColBERT /
    /// multivector token index (Phase F-2, `kind = KIND_COLBERT`). A
    /// single-vector index returns `false`.
    ///
    /// `scan.rs` consults this to REJECT an `ORDER BY` scan against a
    /// ColBERT index (it has no single-vector orderby semantics —
    /// query it with `turbovec.colbert_search`). `colbert_search`
    /// consults it to find the persistent token index.
    pub fn is_colbert(&self) -> bool {
        self.kind == KIND_COLBERT
    }

    /// Returns `true` when this meta page describes a Vamana graph
    /// index (Phase G-2a, `kind = KIND_GRAPH`, wire version 6). A
    /// single-vector or ColBERT index returns `false`.
    ///
    /// Unlike ColBERT, a graph index DOES support `ORDER BY <=>`
    /// scans (the whole point is to match HNSW-style ANN latency), so
    /// `scan.rs` does not reject it here — it dispatches to the graph
    /// scan path instead.
    pub fn is_graph(&self) -> bool {
        self.kind == KIND_GRAPH
    }

    /// Returns `true` when this graph index (`is_graph()`) actually
    /// has a persisted, non-empty adjacency chain to navigate. `false`
    /// for a non-graph index or an empty (0-row) graph build.
    pub fn has_graph(&self) -> bool {
        self.is_graph() && self.graph_first != 0 && self.n_vectors > 0
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
        let mut meta = MetaPageData::plan_with_blocked(
            4,
            dim,
            1_000_000,
            7,
            12_345_678,
            31_250,
            rotation_bytes,
        );
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert_eq!(back.version, 7);
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
        let mut meta = MetaPageData::plan_with_blocked(4, dim, n, 9, 200_000, 781, rotation_bytes);
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        let coarse_bytes = u64::from(lists) * u64::from(dim) * 4;
        let cell_dir_bytes = u64::from(lists) * 12;
        meta.set_ivf_chains(lists, coarse_bytes, cell_dir_bytes);
        let buf = meta.encode();
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(meta, back);
        assert_eq!(back.version, 7);
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
        let mut meta = MetaPageData::plan_with_blocked(4, dim, n, 9, 200_000, 781, rotation_bytes);
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
    fn decodes_legacy_v3_meta_is_rejected_under_v7() {
        // Phase Q-0 (v7): a v3 meta page (or any pre-v7 version) is
        // now LEGACY — the codes-dedup change is not additive, so a v7
        // binary cannot read it and `ambeginscan` errors with a
        // REINDEX hint. (Before Q-0 a v3 index decoded as a flat v4
        // index and scanned with no REINDEX.) We forge a v3 page:
        // first 224 bytes meaningful, the v4+ fields stay zero.
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

        let meta = MetaPageData::decode(&buf).expect("v3 still decodes under v7");
        assert_eq!(meta.version, 3);
        assert_eq!(meta.rotation_first, 9);
        assert_eq!(meta.rotation_dim, 384);
        // A v3 index IS legacy under v7 — REINDEX required.
        assert!(meta.is_legacy_v6(), "a v3 index is legacy under v7");
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = MetaPageData::plan(4, 8, 0, 1).encode();
        buf[4] = 99; // bogus future version
        let err = MetaPageData::decode(&buf).unwrap_err();
        assert!(err.contains("version"));
    }

    /// INVARIANT #1 guard (Phase F-2): a single-vector index must
    /// INVARIANT (Phase Q-0 / v7): every kind now emits wire version
    /// 7 (the codes-dedup change is not additive). A single-vector
    /// index has a ZEROED kind byte; the version byte is 7, NOT 4.
    #[test]
    fn single_vector_emits_v7_bytes() {
        let dim: u32 = 384;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta = MetaPageData::plan_with_blocked(4, dim, 1000, 7, 0, 0, rotation_bytes);
        let centroids: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let boundaries: Vec<f32> = (0..15).map(|i| i as f32 * 0.05 - 0.5).collect();
        meta.set_codebook(&centroids, &boundaries);
        assert_eq!(meta.kind, KIND_SINGLE);
        let buf = meta.encode();
        assert_eq!(buf[4], 7, "single-vector index must emit wire version 7");
        assert_eq!(buf[6], 0, "single-vector kind byte must be zero");
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(back.version, 7);
        assert_eq!(back.kind, KIND_SINGLE);
        assert!(!back.is_colbert());
        assert!(!back.is_legacy_v6());
        // v7 no longer persists a blocked chain.
        assert_eq!(back.blocked_bytes, 0);
        assert_eq!(back.blocked_first, 0);
        assert_eq!(meta, back);
    }

    /// A ColBERT (multivector) index round-trips at wire version 7
    /// with kind = KIND_COLBERT (Phase Q-0 dropped the additive
    /// per-kind versioning; the `kind` byte discriminates).
    #[test]
    fn colbert_index_emits_v7_with_kind() {
        let dim: u32 = 64;
        let lists: u32 = 16;
        let n: u64 = 4096; // token slots
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta = MetaPageData::plan_with_blocked(4, dim, n, 9, 0, 0, rotation_bytes);
        meta.set_codebook(
            &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &(0..15).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
        );
        meta.set_ivf_chains(
            lists,
            u64::from(lists) * u64::from(dim) * 4,
            u64::from(lists) * 12,
        );
        // A colbert build flips kind AFTER planning.
        meta.mark_colbert();
        let buf = meta.encode();
        assert_eq!(buf[4], 7, "colbert index must emit wire version 7");
        assert_eq!(buf[6], KIND_COLBERT);
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(back.version, 7);
        assert!(back.is_colbert());
        assert!(back.has_ivf(), "colbert index is IVF-backed");
        assert!(!back.is_legacy_v6());
        assert_eq!(meta, back);
    }

    /// A forged v4 (genuine pre-Q-0) meta page IS legacy under the v7
    /// binary — REINDEX required. (Before Q-0 a v4 index scanned with
    /// no REINDEX; the codes-dedup wire break ended that.)
    #[test]
    fn forged_v4_meta_is_legacy_under_v7() {
        let mut buf =
            MetaPageData::plan_with_blocked(4, 384, 100, 3, 0, 0, u64::from(384u32) * 384 * 4)
                .encode();
        // Force the version byte to 4 (a genuine pre-Q-0 index).
        buf[4] = 4;
        buf[6] = 0;
        let meta = MetaPageData::decode(&buf).expect("v4 still decodes under v7");
        assert_eq!(meta.version, 4);
        assert_eq!(meta.kind, KIND_SINGLE);
        assert!(!meta.is_colbert());
        assert!(meta.is_legacy_v6(), "a v4 index is legacy under v7");
    }

    /// A Vamana graph index (Phase G-2a) round-trips at wire version 7
    /// with kind = KIND_GRAPH, and its adjacency chain + entry point
    /// survive encode/decode.
    #[test]
    fn graph_index_emits_v7_with_kind_and_adjacency_chain() {
        let dim: u32 = 64;
        let n: u64 = 500;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta = MetaPageData::plan_with_blocked(4, dim, n, 3, 0, 0, rotation_bytes);
        meta.set_codebook(
            &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &(0..15).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
        );
        // CSR: (n+1) u32 offsets + some flat u32 neighbor ids.
        let offsets_bytes = (n + 1) * 4;
        let neighbors_bytes = n * 16 * 4; // pretend degree ~16
        meta.set_graph_chain(offsets_bytes, neighbors_bytes, 42);
        let buf = meta.encode();
        assert_eq!(buf[4], 7, "graph index must emit wire version 7");
        assert_eq!(buf[6], KIND_GRAPH);
        let back = MetaPageData::decode(&buf).expect("decode");
        assert_eq!(back.version, 7);
        assert!(back.is_graph());
        assert!(back.has_graph());
        assert!(!back.is_colbert());
        assert!(!back.has_ivf(), "a graph index is not IVF");
        assert!(!back.is_legacy_v6());
        assert_eq!(back.graph_entry_point, 42);
        assert_eq!(back.graph_offsets_bytes, offsets_bytes);
        assert_eq!(back.graph_neighbors_bytes, neighbors_bytes);
        assert!(back.graph_first > 0);
        assert!(back.graph_count > 0);
        // The graph chain must follow every prior chain (rotation,
        // since this build has lists = 0; blocked_count is 0 in v7).
        assert_eq!(
            back.graph_first,
            1 + back.codes_count
                + back.scales_count
                + back.ids_count
                + back.blocked_count
                + back.rotation_count,
        );
        assert_eq!(meta, back);
    }

    /// A forged v5 ColBERT meta page IS legacy under the v7 binary —
    /// REINDEX required. (Before Q-0 a v5 ColBERT index scanned with
    /// no REINDEX; the codes-dedup wire break ended that.)
    #[test]
    fn forged_v5_colbert_meta_is_legacy_under_v7() {
        let dim: u32 = 64;
        let lists: u32 = 16;
        let n: u64 = 4096;
        let rotation_bytes = u64::from(dim) * u64::from(dim) * 4;
        let mut meta = MetaPageData::plan_with_blocked(4, dim, n, 9, 0, 0, rotation_bytes);
        meta.set_ivf_chains(
            lists,
            u64::from(lists) * u64::from(dim) * 4,
            u64::from(lists) * 12,
        );
        meta.mark_colbert();
        let mut buf = meta.encode();
        // Force the version byte to 5 (a genuine pre-Q-0 ColBERT index).
        buf[4] = 5;
        let back = MetaPageData::decode(&buf).expect("v5 still decodes under v7");
        assert_eq!(back.version, 5);
        assert!(back.is_colbert());
        assert!(!back.is_graph());
        assert!(back.is_legacy_v6(), "a v5 colbert index is legacy under v7");
    }
}
