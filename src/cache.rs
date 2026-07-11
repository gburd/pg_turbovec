//! Backend-local cache of materialised `turbovec::IdMapIndex`
//! instances, used by both `turbovec.knn()` and the index AM
//! scan path.
//!
//! Cache keys are `(rel_oid, attnum_or_zero, bit_width, dim)`.
//! `attnum = 0` is reserved for the index AM path (the index relation
//! owns a single attribute and we don't disambiguate further);
//! positive values are heap attnums from `turbovec.knn()`.
//!
//! Invalidation is best-effort:
//! * Each entry stores the relation's `pg_class.relfilenode` and
//!   either `count(*)` (knn path) or the relfile meta page's
//!   `am_version` (AM path) at load time. Relfile rewrites
//!   (CLUSTER, VACUUM FULL, REINDEX, TRUNCATE) bump the
//!   relfilenode, and ordinary DML changes the row count / bumps
//!   the version; either mismatch forces a rebuild on the next
//!   lookup.
//! * Total cache size capped at `turbovec.cache_size_mb`. When the
//!   cap is exceeded the LRU entry is evicted.
//!
//! ## Mutation (AM path)
//!
//! `aminsert` mutates the cached `IdMapIndex` in place under a
//! `parking_lot::RwLock` write guard, then marks the entry dirty
//! and bumps a per-entry `PersistState` mirror that tracks the
//! relfile-meta-page fields (`bit_width`, `dim`, `n_vectors`,
//! `version`, `live_ids`). A transaction `PreCommit` callback
//! drains every dirty entry and runs a single relfile rewrite
//! per index, then clears the dirty flag and updates the
//! freshness slot to match the new on-disk `am_version`.
//!
//! Concurrency: PostgreSQL backends are single-threaded and our AM
//! advertises `amcanparallel = false`, so the RwLock never sees
//! contention in practice. The lock exists to satisfy `Send + Sync`
//! for the global cache and to keep the read/write paths obviously
//! correct should pgrx ever introduce in-process parallelism.
//!
//! Rollback: on `XACT_EVENT_ABORT` the dirty entries are evicted
//! from the cache so the next access reloads the committed state
//! from the relfile pages. We do not journal undo information.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use parking_lot::{Mutex, RwLock};
use pgrx::prelude::*;
use turbovec::{IdMapIndex, SearchResults, TurboQuantIndex};

use crate::guc;
use crate::index::ivf::{coarse_probe_dispatch, rotate_query, CellDirectory};

/// Read-only materialisation of a turbovec index for the index-AM
/// scan path. Unlike [`IdMapIndex`] it stores **only** the inner
/// positional [`TurboQuantIndex`] plus the `slot -> id` table; it
/// does **not** build the `id -> slot` `HashMap`.
///
/// ## Why this exists (cold-scan latency, parity gap #3)
///
/// `IdMapIndex::from_id_map_parts*` eagerly materialises an
/// `id_to_slot: HashMap<u64, usize>` in `finalise_from_inner`. On a
/// 1 M-row index that single allocation + insert loop is the
/// dominant per-backend cold-scan term (profiled at ~50 ms in a
/// debug build at 200 k rows; it scales linearly with `n`). The
/// scan path never needs `id_to_slot`: `IdMapIndex::search` with
/// `allowlist = None` only ever reads `slot_to_id[slot]` (a `Vec`
/// index), so the `HashMap` is pure dead weight on a read-only
/// scan.
///
/// A read-only backend that only ever scans therefore skips the
/// `HashMap` build entirely. The map is only needed by `aminsert`
/// / `remove` (mutation), which rebuild a full [`IdMapIndex`] via
/// [`am_install`] on the first mutation in a transaction — so the
/// `HashMap` build is *deferred* to the first mutation, exactly
/// when it's actually needed. See `am_lookup_for_mutation`, which
/// returns `None` for a read-only entry and forces that rebuild.
pub struct ReadOnlyIndex {
    inner: TurboQuantIndex,
    /// `slot_to_id[i]` is the external id of the vector in slot `i`.
    /// The scan path translates the kernel's slot indices back to
    /// CTIDs through this `Vec` (an O(1) index, not a hash lookup).
    slot_to_id: Vec<u64>,
}

impl ReadOnlyIndex {
    /// Build a read-only index from owned parts + the prepared
    /// SIMD-blocked layout / Lloyd-Max codebook / rotation. This is
    /// the buffer-manager fall-back twin of
    /// [`IdMapIndex::from_id_map_parts_with_prepared`], minus the
    /// `id_to_slot` `HashMap` build.
    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_parts(
        bit_width: usize,
        dim: usize,
        n_vectors: usize,
        packed_codes: Vec<u8>,
        scales: Vec<f32>,
        slot_to_id: Vec<u64>,
        blocked_codes: Vec<u8>,
        n_blocks: usize,
        centroids: Vec<f32>,
        boundaries: Vec<f32>,
        rotation: Option<Vec<f32>>,
    ) -> Self {
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        let inner = TurboQuantIndex::from_parts_with_prepared(
            dim_opt,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
            blocked_codes,
            n_blocks,
            centroids,
            boundaries,
            rotation,
        );
        Self { inner, slot_to_id }
    }

    /// Borrowed-cache twin of [`Self::from_prepared_parts`] for the
    /// mmap fast path. Accepts `Cow` for the heavy buffers so the
    /// caller can hand off owned `Vec`s (today) or zero-copy
    /// borrowed slices into a `memmap2::Mmap` (a future follow-up).
    /// `slot_to_id` stays owned because we keep it as the
    /// translation table.
    pub fn from_prepared_parts_borrowed<'a>(
        bit_width: usize,
        dim: usize,
        n_vectors: usize,
        packed_codes: std::borrow::Cow<'a, [u8]>,
        scales: std::borrow::Cow<'a, [f32]>,
        slot_to_id: Vec<u64>,
        prepared: turbovec::PreparedCachesBorrowed<'a>,
    ) -> Self {
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        let inner = TurboQuantIndex::from_parts_with_prepared_borrowed(
            dim_opt,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
            prepared,
        );
        Self { inner, slot_to_id }
    }

    /// Build a read-only index from raw parts with no prepared
    /// caches (legacy fall-through; the inner index lazily computes
    /// the blocked layout / codebook on first search). Mirrors
    /// [`IdMapIndex::from_id_map_parts`] minus the `HashMap`.
    pub fn from_parts(
        bit_width: usize,
        dim: usize,
        n_vectors: usize,
        packed_codes: Vec<u8>,
        scales: Vec<f32>,
        slot_to_id: Vec<u64>,
    ) -> Self {
        let dim_opt = if dim == 0 { None } else { Some(dim) };
        let inner = TurboQuantIndex::from_parts(
            dim_opt,
            bit_width,
            n_vectors,
            packed_codes,
            scales,
            Vec::new(),
            Vec::new(),
        );
        Self { inner, slot_to_id }
    }

    /// Number of live vectors. Mirrors [`IdMapIndex::len`].
    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    /// True if the index has no live vectors.
    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    /// Build a per-query [`GraphScorer`] over this index's public
    /// TurboQuant parts (rotation, codebook centroids, packed
    /// bit-plane codes, per-vector scales). The scorer amortises the
    /// query rotation + LUT build ONCE, then scores an arbitrary
    /// candidate slot batch in `O(batch * dim)` by touching only
    /// those rows' codes — the Phase G-2c per-hop scoring win over
    /// [`Self::score_slots`], which re-rotates + rebuilds the LUT and
    /// scans the whole `O(n)` blocked buffer on every hop. See
    /// [`GraphScorer`] for the equivalence / determinism contract.
    pub(crate) fn graph_scorer(&self, query: &[f32]) -> GraphScorer {
        GraphScorer::new(
            query,
            self.inner.dim(),
            self.inner.bit_width(),
            self.inner.rotation(),
            self.inner.centroids(),
            self.inner.packed_codes(),
            self.inner.scales(),
        )
    }

    /// Top-`k` search returning `(scores, ids)`, byte-for-byte the
    /// same shape and ordering as [`IdMapIndex::search`] for the
    /// `allowlist = None` case (the only case the scan path uses).
    ///
    /// The inner kernel returns `i64` slot indices; we translate
    /// each through `slot_to_id`. Slot indices are always in-bounds
    /// for a valid index (the kernel never returns a negative or
    /// out-of-range slot), so the `Vec` index can't panic in
    /// practice; an out-of-range slot would be a corrupt-index bug,
    /// and the bounds check makes that crash-loud rather than
    /// silently wrong.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        let res: SearchResults = self.inner.search(queries, k);
        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            ids.push(self.slot_to_id[slot as usize]);
        }
        (res.scores, ids)
    }

    /// IVF cell-restricted top-`k` search: identical to [`Self::search`]
    /// but only slots whose `mask` entry is `true` contribute to the
    /// top-`k`. `mask.len()` must equal [`Self::len`].
    ///
    /// turbovec's blocked kernel short-circuits whole 32-vector blocks
    /// whose mask window is all-zero (see `block_has_allowed` in
    /// turbovec/src/search.rs), so when the `true` slots are clustered
    /// into a few contiguous cell ranges (as the IVF build lays them
    /// out) the masked-out blocks skip their LUT scoring work entirely
    /// — this is the IVF latency win, not just a result filter.
    pub fn search_masked(&self, queries: &[f32], k: usize, mask: &[bool]) -> (Vec<f32>, Vec<u64>) {
        let res: SearchResults = self.inner.search_with_mask(queries, k, Some(mask));
        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            ids.push(self.slot_to_id[slot as usize]);
        }
        (res.scores, ids)
    }

    /// Build a by-slot allowlist mask from a set of EXTERNAL ids
    /// (Phase C operator-path allowlist). `allow_slot[i] == true` iff
    /// slot `i`'s external id (`slot_to_id[i]`) is in `allowed`. The
    /// scan ANDs this into the IVF probe/tombstone mask (or uses it
    /// alone on the flat path) before [`Self::search_masked`], so
    /// turbovec's blocked kernel skips 32-vector blocks with no
    /// allowed slot — the same in-kernel block-skip
    /// `turbovec.knn(..., allowed)` gets, now on the ORDER BY path.
    /// O(n) bools, built only when an allowlist is active.
    pub fn allow_slot_mask(&self, allowed: &HashSet<u64>) -> Vec<bool> {
        self.slot_to_id
            .iter()
            .map(|id| allowed.contains(id))
            .collect()
    }

    /// External id stored at slot `slot`. Used by the graph scan path
    /// (Phase G-2a) to translate the entry point / adjacency's slot
    /// ids to CTIDs.
    pub fn id_at_slot(&self, slot: usize) -> u64 {
        self.slot_to_id[slot]
    }

    /// Score exactly the slots in `ids` against `query`, returning
    /// their scores in THE SAME ORDER as `ids` (turbovec's native
    /// higher-is-closer convention, matching
    /// `TurboQuantIndex::search`'s `SearchResults.scores`).
    ///
    /// Phase G-2a's graph scan path uses this as the per-hop batch
    /// scoring oracle `graph::graph_search` needs: reuses the
    /// EXISTING masked-search kernel (`search_with_mask`) rather than
    /// hand-rolling a new distance kernel — turbovec's blocked kernel
    /// already skips 32-vector blocks with no allowed slot, so a
    /// small `ids` batch (a graph node's out-degree, typically ~32)
    /// costs close to `O(n/32)` block-skip checks, not a full `O(n)`
    /// scan, even though the mask itself is `O(n)` to allocate. A
    /// SIMD-aware, allocation-free per-hop scorer is exactly the kind
    /// of optimisation  scopes
    /// to G-2c, not this correctness-first sub-phase.
    ///
    /// `ids` must be non-empty and every id must be `< self.len()`;
    /// duplicates are tolerated (the mask is idempotent) but the
    /// output would then have fewer entries than `ids.len()` for the
    /// duplicated id, which the graph search never does (each hop's
    /// candidate batch is already deduplicated via the `visited`
    /// bitmap in `graph::graph_search`).
    pub fn score_slots(&self, query: &[f32], ids: &[u32]) -> Vec<f32> {
        let n = self.len();
        let mut mask = vec![false; n];
        for &id in ids {
            mask[id as usize] = true;
        }
        let res: SearchResults = self.inner.search_with_mask(query, ids.len(), Some(&mask));
        let mut by_slot: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::with_capacity(res.indices.len());
        for (&slot, &score) in res.indices.iter().zip(res.scores.iter()) {
            by_slot.insert(slot as u32, score);
        }
        ids.iter()
            .map(|id| {
                *by_slot
                    .get(id)
                    .expect("score_slots: masked search dropped a requested slot")
            })
            .collect()
    }
}

/// Out-of-core (Phase B-1/B-2) cell-scoped IVF index. Unlike
/// [`ReadOnlyIndex`], which holds the WHOLE blocked-codes buffer
/// resident (per-backend `O(n)`), this variant keeps only the
/// **bounded** index metadata resident — the coarse centroids, the
/// cell directory, the rotation matrix, the Lloyd-Max codebook, and
/// the small per-slot tables (`scales` 4 B/vec, `slot_to_id`
/// 8 B/vec). The big `O(n)` codes buffer is **never** materialised
/// whole.
///
/// Per query (see [`Self::search_ooc`]) it coarse-probes the cached
/// centroids, then gathers ONLY the probed cells' contiguous packed
/// code ranges through PostgreSQL's buffer manager
/// (`relfile::gather_codes_ranges`) into a compact, gapless buffer,
/// builds a throwaway [`TurboQuantIndex`] over just those rows, and
/// runs an unmasked top-`k` search on it. The resident set is
/// therefore `O(probes * cell_size)`, not `O(n)`: only the probed
/// cells' pages are read (the buffer manager + OS cache hold hot
/// pages; cold pages are read on demand and evict under memory
/// pressure). **This is the out-of-core serving path** — an IVF
/// index can exceed RAM as long as the working set (hot cells) fits.
///
/// Only IVF indexes (`lists > 0`, live cell directory) get this
/// path; flat indexes have no cells to scope and keep the
/// whole-index [`ReadOnlyIndex`] load. The compact sub-index is
/// built with identity TQ+ calibration, matching the relfile codes
/// (which were encoded under identity TQ+, exactly as the
/// whole-index [`ReadOnlyIndex`] path assumes).
pub(crate) struct OocIvfIndex {
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    /// Codes chain layout for the per-query buffer-manager gather.
    codes_first: u32,
    codes_stride: u32,
    rows_per_codes_page: u32,
    /// Coarse centroids (row-major `lists * dim`, rotated space).
    coarse_centroids: Vec<f32>,
    lists: usize,
    /// Phase G-1: an in-memory (never persisted) navigable graph over
    /// `coarse_centroids`, built once here at cache-install time from
    /// the already-persisted centroids. `None` when
    /// `turbovec.coarse_graph` gates it off (small `lists` under
    /// `auto`, or `off`) — [`Self::coarse_probe_cells`] then falls
    /// back to the exact linear scan via `coarse_probe_dispatch`.
    graph: Option<crate::index::ivf::CentroidGraph>,
    /// Rotation matrix (row-major `dim * dim`) for the coarse probe.
    rotation: Vec<f32>,
    /// Cell directory: each cell's `[code_offset, +n_vectors)` range.
    directory: CellDirectory,
    /// Lloyd-Max codebook for the compact sub-index search caches.
    codebook_centroids: Vec<f32>,
    codebook_boundaries: Vec<f32>,
    /// Per-slot scale (4 B/vec; small, kept resident — gathered per
    /// query into the compact sub-index).
    scales: Vec<f32>,
    /// Slot -> external id (8 B/vec; small, kept resident). Compact
    /// sub-index slots are remapped to global slots, then to ids.
    slot_to_id: Vec<u64>,
}

/// Phase G-2c: per-query graph traversal scorer.
///
/// Reuses TurboQuant's LUT-based table-lookup scoring PRIMITIVE (the
/// one 
/// graph path because a graph node's codes are stored in the SAME
/// per-coordinate bit-plane layout as a flat/IVF row), but restructured
/// for graph TRAVERSAL rather than a whole-corpus scan:
///
/// - The query rotation (`q_rot = query @ rotation^T`) and the
///   per-coordinate query LUT (`qlut[d*C + c] = q_rot[d] *
///   centroids[c]`, `C = 2^bit_width`) are built **once per query** in
///   [`Self::new`], then reused across every hop of the beam search.
///   `ReadOnlyIndex::score_slots` (the G-2a correctness-first path)
///   rebuilt both on *every* hop via `search_with_mask`.
/// - [`Self::score_batch`] scores a hop's candidate slot batch by
///   touching ONLY those rows' code bytes — `O(batch * dim)` — instead
///   of scanning the whole `O(n)` blocked-codes buffer per hop (which
///   is what `search_with_mask` does, even with the block-skip mask,
///   because a graph node's ~R neighbors are scattered across the
///   corpus so most 32-vector blocks are still visited). This is the
///   "SIMD-aware, allocation-free per-hop scorer" G-2c was scoped to
///   deliver (see the `score_slots` doc comment).
///
/// ## Score identity with the SIMD kernel
///
/// For an identity-TQ+ index (which every pg_turbovec index is — the
/// relfile codes are encoded under identity calibration, per the
/// `ReadOnlyIndex` construction comments), turbovec's SIMD kernel
/// computes, for slot `i`:
///   `score(i) = vec_scale[i] * ( bias + Σ_d q_rot[d] * centroids[code_d] )`
/// where `bias` folds in the per-sub-table minima the kernel
/// subtracts and re-adds. This scorer computes the SAME quantity
/// directly and unquantized (the SIMD kernel quantises the LUT to a
/// u8 [0,127] range for its integer accumulators; this scorer keeps
/// f32 throughout). The two therefore rank IDENTICALLY except on
/// exact near-ties where the u8 LUT rounding could reorder two
/// candidates — vanishingly rare on real data, and when it happens
/// the f32 path is the MORE faithful proxy of the true distance, not
/// less. The graph scan is a beam search whose result SET is what
/// matters (reranked exactly by `xs_recheckorderby` downstream
/// anyway); `graph_scan_simd_matches_scalar_result_set` asserts the
/// two paths return the same top-k set on a real corpus.
///
/// ## Determinism
///
/// Purely serial and allocation-deterministic: a fixed query yields a
/// fixed `q_rot`/`qlut` (fixed-order f32 reductions), and
/// `score_batch` maps candidate ids to scores in input order with no
/// threading. Same score for the same (query, slot) on every call.
pub(crate) struct GraphScorer {
    dim: usize,
    bit_width: usize,
    /// `2^bit_width` — number of codebook centroids / LUT columns.
    n_levels: usize,
    /// Per-coordinate query LUT: `qlut[d * n_levels + c]` is
    /// `q_rot[d] * centroids[c]`. Built once per query.
    qlut: Vec<f32>,
    /// Borrowed packed bit-plane codes (`n * bit_width * dim/8`).
    packed_codes: *const u8,
    packed_len: usize,
    /// Borrowed per-vector scales (`n` f32).
    scales: *const f32,
    scales_len: usize,
    bytes_per_plane: usize,
    bytes_per_row: usize,
}

impl GraphScorer {
    fn new(
        query: &[f32],
        dim: usize,
        bit_width: usize,
        rotation: &[f32],
        centroids: &[f32],
        packed_codes: &[u8],
        scales: &[f32],
    ) -> Self {
        let n_levels = 1usize << bit_width;
        // q_rot[d] = Σ_j query[j] * rotation[d*dim + j]  (== query @ R^T,
        // matching turbovec search.rs's batched GEMM). If rotation is
        // empty (a lazy/degenerate index), fall back to identity so the
        // scorer still produces a consistent ranking.
        let mut q_rot = vec![0.0f32; dim];
        if rotation.len() == dim * dim {
            for (d, qr) in q_rot.iter_mut().enumerate() {
                let rrow = &rotation[d * dim..(d + 1) * dim];
                let mut acc = 0.0f32;
                for (j, &qv) in query.iter().enumerate().take(dim) {
                    acc += qv * rrow[j];
                }
                *qr = acc;
            }
        } else {
            q_rot[..dim.min(query.len())].copy_from_slice(&query[..dim.min(query.len())]);
        }
        // Per-coordinate LUT: qlut[d][c] = q_rot[d] * centroids[c].
        let mut qlut = vec![0.0f32; dim * n_levels];
        for d in 0..dim {
            let qd = q_rot[d];
            let base = d * n_levels;
            for c in 0..n_levels {
                qlut[base + c] = qd * centroids[c];
            }
        }
        let bytes_per_plane = dim / 8;
        let bytes_per_row = bit_width * bytes_per_plane;
        Self {
            dim,
            bit_width,
            n_levels,
            qlut,
            packed_codes: packed_codes.as_ptr(),
            packed_len: packed_codes.len(),
            scales: scales.as_ptr(),
            scales_len: scales.len(),
            bytes_per_plane,
            bytes_per_row,
        }
    }

    /// Extract the `dim`-length code sequence for slot `i` from the
    /// bit-plane packed layout and sum the query LUT, then scale by
    /// the per-vector scale. Higher = closer (turbovec's native
    /// convention). Exactly reproduces the kernel's identity-TQ+
    /// score formula (see the struct doc). `i` must be `< n`.
    #[inline]
    fn score_one(&self, i: usize, packed: &[u8], scales: &[f32]) -> f32 {
        let row_base = i * self.bytes_per_row;
        let mut acc = 0.0f32;
        // For each coordinate, gather its `bit_width`-bit code from the
        // bit-planes and add qlut[d][code]. Matches
        // `encode.rs::fused_quantize_scale_pack`'s packing exactly:
        // plane `p` bit for coord `d` lives at
        // `row_base + p*bytes_per_plane + d/8`, bit `7 - (d%8)`.
        for d in 0..self.dim {
            let byte_pos = d / 8;
            let bit_pos = 7 - (d % 8);
            let mut code = 0usize;
            for p in 0..self.bit_width {
                let byte = packed[row_base + p * self.bytes_per_plane + byte_pos];
                code |= (((byte >> bit_pos) & 1) as usize) << p;
            }
            acc += self.qlut[d * self.n_levels + code];
        }
        acc * scales[i]
    }

    /// Score a batch of candidate slot ids, returning scores in the
    /// SAME order as `ids` (matching
    /// `graph::graph_search`'s `score_batch` contract). `O(ids * dim)`
    /// — touches only the requested rows' codes.
    pub(crate) fn score_batch(&self, ids: &[u32]) -> Vec<f32> {
        // SAFETY: the borrowed slices outlive `self` (they point into
        // the `Arc<ReadOnlyIndex>` the owning `GraphIndex` holds for
        // the whole scan; `graph_scorer` is only called with a live
        // `&self`). Reconstituted here so `score_one` can index them
        // without a lifetime on the struct (kept raw to avoid
        // threading a borrow through `graph_search`'s `FnMut`).
        let packed = unsafe { std::slice::from_raw_parts(self.packed_codes, self.packed_len) };
        let scales = unsafe { std::slice::from_raw_parts(self.scales, self.scales_len) };
        ids.iter()
            .map(|&id| self.score_one(id as usize, packed, scales))
            .collect()
    }
}

/// Phase G-2a: RAM-resident Vamana graph index. Wraps a
/// [`ReadOnlyIndex`] (the graph's vector storage is IDENTICAL to a
/// flat index's — same codes/scales/ids, same TurboQuant encode
/// path) with the persisted adjacency chain + entry point.
///
/// Per query ([`Self::search`]) it runs `graph::graph_search`,
/// navigating the adjacency via a per-query [`GraphScorer`] (Phase
/// G-2c): the query rotation + LUT are built ONCE and reused across
/// every hop, and each hop scores only its candidate rows'
/// bit-plane codes (`O(batch * dim)`), instead of re-rotating,
/// rebuilding the LUT and scanning the whole `O(n)` blocked buffer
/// per hop the way the G-2a `ReadOnlyIndex::score_slots` path did.
/// The result SET is unchanged (a speed change, not a recall change —
/// see `GraphScorer`'s score-identity note and
/// `graph_scan_simd_matches_scalar_result_set`).
pub(crate) struct GraphIndex {
    inner: std::sync::Arc<ReadOnlyIndex>,
    adjacency: crate::index::graph::GraphAdjacency,
    entry_point: u32,
    /// Phase G-2b: per-slot tombstone bitmap (LSB-first, bit set ⇒
    /// dead), read ONCE at cache-install time
    /// (`scan.rs::install_graph_index`) exactly like the IVF path
    /// reads it once per scan-open. Freshness is enforced by the
    /// SAME mechanism as every other cached field here: VACUUM bumps
    /// `am_version`, which busts this whole `GraphIndex` out of the
    /// cache (`scan_lookup`'s relfilenode/n_rows freshness check), so
    /// a stale bitmap can never outlive the vacuum that produced it.
    /// Empty when nothing has been deleted.
    tombstones: Vec<u8>,
}

impl GraphIndex {
    pub(crate) fn new(
        inner: std::sync::Arc<ReadOnlyIndex>,
        adjacency: crate::index::graph::GraphAdjacency,
        entry_point: u32,
        tombstones: Vec<u8>,
    ) -> Self {
        Self {
            inner,
            adjacency,
            entry_point,
            tombstones,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    /// Top-`k` greedy beam search over the graph. Returns `(scores,
    /// ids)` in the SAME shape as [`ReadOnlyIndex::search`] (scores
    /// descending, translated to external ids) so the AM scan path's
    /// existing `populate_batch` / `xs_recheckorderby` machinery needs
    /// no graph-specific branch.
    pub(crate) fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        if self.len() == 0 || k == 0 {
            return (Vec::new(), Vec::new());
        }
        // Phase G-2c: build the per-query LUT-based scorer ONCE, reuse
        // it across every hop. The scorer touches only each hop's
        // candidate rows' codes (`O(batch * dim)`), not the whole
        // `O(n)` blocked buffer the old `score_slots` re-scanned per
        // hop.
        let scorer = self.inner.graph_scorer(query);
        let hits = crate::index::graph::graph_search(
            &self.adjacency,
            self.entry_point,
            k,
            &self.tombstones,
            |ids| scorer.score_batch(ids),
        );
        let mut scores = Vec::with_capacity(hits.len());
        let mut ids = Vec::with_capacity(hits.len());
        for (s, slot) in hits {
            scores.push(s);
            ids.push(self.inner.id_at_slot(slot as usize));
        }
        (scores, ids)
    }
}

impl OocIvfIndex {
    /// Build an OOC IVF index. The caller (the scan path) has
    /// already read the meta page and the (bounded) static regions;
    /// this just moves them into the cache-resident container.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        bit_width: usize,
        dim: usize,
        n_vectors: usize,
        codes_first: u32,
        codes_stride: u32,
        rows_per_codes_page: u32,
        coarse_centroids: Vec<f32>,
        lists: usize,
        rotation: Vec<f32>,
        directory: CellDirectory,
        codebook_centroids: Vec<f32>,
        codebook_boundaries: Vec<f32>,
        scales: Vec<f32>,
        slot_to_id: Vec<u64>,
    ) -> Self {
        // Phase G-1: build the centroid graph once here, in-memory,
        // from the already-persisted centroids (never written back
        // to the relfile — purely additive). Gated by
        // `turbovec.coarse_graph` (auto/on/off); see
        // `guc::coarse_graph_enabled` and `ivf::GRAPH_MIN_LISTS`.
        let graph = if crate::guc::coarse_graph_enabled(lists) {
            Some(crate::index::ivf::build_centroid_graph(
                &coarse_centroids,
                lists,
                dim,
            ))
        } else {
            None
        };
        Self {
            bit_width,
            dim,
            n_vectors,
            codes_first,
            codes_stride,
            rows_per_codes_page,
            coarse_centroids,
            lists,
            graph,
            rotation,
            directory,
            codebook_centroids,
            codebook_boundaries,
            scales,
            slot_to_id,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.n_vectors
    }

    pub(crate) fn lists(&self) -> usize {
        self.lists
    }

    /// Coarse-probe the cached centroids for the `probes` nearest
    /// cells to `query` (already normalised by the caller). Mirrors
    /// the whole-index path's `ivf_setup_and_search` so OOC results
    /// match whole-load exactly. Returns the probed cell ids
    /// (ascending distance, deterministic tie-break).
    pub(crate) fn coarse_probe_cells(&self, query_unit: &[f32], probes: usize) -> Vec<u32> {
        let q_rot = rotate_query(&self.rotation, query_unit, self.dim);
        let probes = probes.clamp(1, self.lists.max(1));
        coarse_probe_dispatch(
            &self.coarse_centroids,
            self.lists,
            self.dim,
            &q_rot,
            probes,
            self.graph.as_ref(),
        )
    }

    /// Cell-scoped top-`k` search over the `probed` cells. Copies
    /// ONLY those cells' contiguous code ranges off the mmap into a
    /// compact gapless buffer, builds a throwaway [`TurboQuantIndex`]
    /// over just those rows, runs an unmasked top-`k` search, and
    /// remaps the compact slot indices back to global slots and then
    /// to external ids. `dead` is the per-slot tombstone bitmap
    /// (LSB-first, bit set ⇒ slot dead, Phase E-2); tombstoned slots
    /// are skipped during the gather so they never enter the compact
    /// index. Returns `None` if the gather runs off the mapping
    /// (corrupt index / post-truncate race) so the caller falls back
    /// to the whole-index load path.
    pub(crate) fn search_ooc(
        &self,
        rel: pg_sys::Relation,
        query: &[f32],
        k: usize,
        probed: &[u32],
        dead: &[u8],
        allow: Option<&HashSet<u64>>,
    ) -> Option<(Vec<f32>, Vec<u64>)> {
        // Collect the probed cells' contiguous slot ranges, in cell
        // order, deduping cell ids. Track the global slot of each
        // compact slot for the remap. We expand tombstoned slots out
        // here so the compact index never scores a dead row.
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(probed.len());
        let mut seen = vec![false; self.lists];
        // global_slots[compact_slot] = global slot index.
        let mut global_slots: Vec<u32> = Vec::new();
        let tombstoned = |slot: usize| -> bool {
            if dead.is_empty() {
                return false;
            }
            let byte = slot / 8;
            byte < dead.len() && (dead[byte] >> (slot % 8)) & 1 != 0
        };
        for &c in probed {
            let c = c as usize;
            if c >= self.lists || seen[c] {
                continue;
            }
            seen[c] = true;
            let e = self.directory.entries[c];
            let start = e.code_offset;
            let end = (e.code_offset + u64::from(e.n_vectors)).min(self.n_vectors as u64);
            // The gather copies code bytes for [start, end); the
            // tombstone skip is applied to the compact slot list so
            // dead rows are dropped after the gather (the gather is
            // contiguous and cheap; filtering bytes mid-gather would
            // fragment it). We record the live global slots and, when
            // there ARE tombstones in the range, gather per-live-run
            // instead of the whole cell.
            if dead.is_empty() {
                if end > start {
                    ranges.push((start, end - start));
                    for s in start..end {
                        global_slots.push(s as u32);
                    }
                }
            } else {
                // Build contiguous live runs within the cell so the
                // gather only copies live slots' bytes.
                let mut run_start: Option<u64> = None;
                let mut s = start;
                while s < end {
                    if tombstoned(s as usize) {
                        if let Some(rs) = run_start.take() {
                            ranges.push((rs, s - rs));
                        }
                    } else {
                        if run_start.is_none() {
                            run_start = Some(s);
                        }
                        global_slots.push(s as u32);
                    }
                    s += 1;
                }
                if let Some(rs) = run_start.take() {
                    ranges.push((rs, end - rs));
                }
            }
        }
        let n_compact = global_slots.len();
        if n_compact == 0 {
            return Some((Vec::new(), Vec::new()));
        }

        // Gather the compact packed codes for the probed cells
        // through the buffer manager: only the probed cells' pages
        // are read (cell-scoped), so the resident codes stay bounded
        // at O(probes * cell_size), not O(n). All index data goes
        // through `ReadBufferExtended` — see
        // docs/BUFFER_CACHE_ONLY_DESIGN.md.
        //
        // SAFETY: `rel` is a live index relation reference held by
        // the scan for the call's duration.
        let compact_codes = unsafe {
            crate::index::relfile::gather_codes_ranges(
                rel,
                self.codes_first,
                self.codes_stride,
                self.rows_per_codes_page,
                &ranges,
            )
        };
        debug_assert_eq!(compact_codes.len(), n_compact * self.codes_stride as usize);

        // Gather the matching scales (resident; cheap).
        let mut compact_scales = Vec::<f32>::with_capacity(n_compact);
        for &gs in &global_slots {
            compact_scales.push(self.scales[gs as usize]);
        }

        // Fine-scan the compact rows for the top-`k`. At high dim /
        // high probes this is the dominant per-query cost and is
        // embarrassingly parallel across disjoint row ranges (item #2
        // of the IVF-scaling work). `compact_codes` is a plain owned
        // `Vec<u8>` of contiguous rows and `compact_scales` a
        // `Vec<f32>`; splitting them into `T` row chunks and scanning
        // each in a bounded rayon pool is PURE COMPUTE over owned
        // bytes — no buffer-manager / catalog / `pg_sys` access
        // happens inside the threads (the gather above already ran on
        // this backend thread). The T local top-`k` heaps merge into
        // the global top-`k`; the union of per-chunk top-`k` lists
        // contains the true global top-`k` (a top-`k` row is always in
        // its own chunk's top-`k`), so the returned SET matches a
        // serial scan. Tie order at the k-th boundary is immaterial:
        // the executor re-ranks by exact distance (xs_recheckorderby).
        //
        // Phase C operator-path allowlist: a compact-slot mask
        // (allow_compact[cslot] = that compact slot's id is allowed),
        // pushed into the blocked kernel per chunk so the 32-vector
        // block-skip applies to the OOC path too. Only built when an
        // allowlist is active.
        let allow_compact: Option<Vec<bool>> = allow.map(|set| {
            global_slots
                .iter()
                .map(|&gs| set.contains(&self.slot_to_id[gs as usize]))
                .collect()
        });

        let t = guc::resolve_scan_parallelism(n_compact);
        let (cscores, cslots): (Vec<f32>, Vec<i64>) = if t <= 1 {
            // Serial: one sub-index over all compact rows (the
            // pre-parallel path). `cslots` are compact-slot indices.
            let res = self.search_compact_chunk(
                &compact_codes,
                &compact_scales,
                0,
                n_compact,
                query,
                k,
                allow_compact.as_deref(),
            );
            (res.scores, res.indices)
        } else {
            self.search_compact_parallel(
                &compact_codes,
                &compact_scales,
                query,
                k,
                allow_compact.as_deref(),
                t,
            )
        };

        let mut ids = Vec::with_capacity(cslots.len());
        for &cslot in &cslots {
            let global = global_slots[cslot as usize] as usize;
            ids.push(self.slot_to_id[global]);
        }
        Some((cscores, ids))
    }

    /// Fine-scan a contiguous chunk `[row_start, row_end)` of the
    /// gathered compact codes for the local top-`k`, returning
    /// compact-slot indices (already offset by `row_start`, so they
    /// are global compact slots, not chunk-local). Builds a throwaway
    /// [`TurboQuantIndex`] over the chunk's rows with identity TQ+
    /// (matches the relfile codes) and the shared rotation / codebook
    /// handed over as borrowed prepared caches, so only the SIMD
    /// re-block is recomputed — bounded by the chunk row count. Pure
    /// compute over the passed-in slices (`Send`-safe); the parallel
    /// path calls it from rayon worker threads.
    #[allow(clippy::too_many_arguments)]
    fn search_compact_chunk(
        &self,
        compact_codes: &[u8],
        compact_scales: &[f32],
        row_start: usize,
        row_end: usize,
        query: &[f32],
        k: usize,
        allow_compact: Option<&[bool]>,
    ) -> SearchResults {
        let n_rows = row_end - row_start;
        let stride = self.codes_stride as usize;
        let codes = &compact_codes[row_start * stride..row_end * stride];
        let scales = &compact_scales[row_start..row_end];
        let dim_opt = if self.dim == 0 { None } else { Some(self.dim) };
        let prepared = turbovec::PreparedCachesBorrowed {
            blocked_codes: None,
            n_blocks: 0,
            centroids: Some(std::borrow::Cow::Borrowed(&self.codebook_centroids)),
            boundaries: Some(std::borrow::Cow::Borrowed(&self.codebook_boundaries)),
            rotation: Some(std::borrow::Cow::Borrowed(&self.rotation)),
        };
        let sub = TurboQuantIndex::from_parts_with_prepared_borrowed(
            dim_opt,
            self.bit_width,
            n_rows,
            std::borrow::Cow::Borrowed(codes),
            std::borrow::Cow::Borrowed(scales),
            prepared,
        );
        let mut res = match allow_compact {
            None => sub.search(query, k),
            Some(all) => sub.search_with_mask(query, k, Some(&all[row_start..row_end])),
        };
        // Lift chunk-local slots to global compact slots.
        if row_start != 0 {
            for idx in res.indices.iter_mut() {
                *idx += row_start as i64;
            }
        }
        res
    }

    /// Parallel fine-scan: split the compact rows into `t` roughly-
    /// equal contiguous chunks, top-`k` each in a bounded rayon pool,
    /// and merge the `t` local top-`k` lists into the global top-`k`.
    /// Returns `(scores, compact_slots)`, the same shape the serial
    /// path produces. Determinism of RESULTS (the top-`k` SET) is
    /// guaranteed by the union property of per-chunk top-`k`; tie
    /// order is not (nor need it be — the executor re-ranks exactly).
    fn search_compact_parallel(
        &self,
        compact_codes: &[u8],
        compact_scales: &[f32],
        query: &[f32],
        k: usize,
        allow_compact: Option<&[bool]>,
        t: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        use rayon::prelude::*;
        let n_compact = compact_scales.len();
        // Contiguous, roughly-equal row chunks. `chunk` rounds up so
        // the last chunk is the short one; every chunk is non-empty
        // because resolve_scan_parallelism keeps t <= n_compact/floor.
        let chunk = n_compact.div_ceil(t);
        let bounds: Vec<(usize, usize)> = (0..n_compact)
            .step_by(chunk)
            .map(|s| (s, (s + chunk).min(n_compact)))
            .collect();

        // Bounded pool so a scan does not grab rayon's global (all-core)
        // pool under concurrency; sized to the resolved chunk count.
        let work = || {
            bounds
                .par_iter()
                .map(|&(s, e)| {
                    self.search_compact_chunk(
                        compact_codes,
                        compact_scales,
                        s,
                        e,
                        query,
                        k,
                        allow_compact,
                    )
                })
                .collect()
        };
        let locals: Vec<SearchResults> = match crate::index::build_pool::scan_pool(t) {
            Some(pool) => pool.install(work),
            None => work(),
        };

        merge_topk(&locals, k)
    }
}

/// K-way merge of per-chunk local top-`k` results into the global
/// top-`k`. Each [`SearchResults`] is already sorted by score
/// descending (the kernel's output order). We concatenate the local
/// `(score, slot)` pairs and select the `k` highest by score,
/// breaking ties by ascending compact-slot so the result is a stable
/// function of the inputs (not thread scheduling). The union of the
/// local top-`k` lists provably contains the global top-`k`, so this
/// returns the same SET a serial scan would.
fn merge_topk(locals: &[SearchResults], k: usize) -> (Vec<f32>, Vec<i64>) {
    let total: usize = locals.iter().map(|r| r.indices.len()).sum();
    let mut pairs: Vec<(f32, i64)> = Vec::with_capacity(total);
    for r in locals {
        for (&s, &i) in r.scores.iter().zip(r.indices.iter()) {
            pairs.push((s, i));
        }
    }
    // Score descending; deterministic slot-ascending tie-break.
    pairs.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    pairs.truncate(k);
    let scores = pairs.iter().map(|p| p.0).collect();
    let slots = pairs.iter().map(|p| p.1).collect();
    (scores, slots)
}

/// A scan-facing handle over a cached index, regardless of which
/// [`Stored`] variant backs it. Lets the index-AM scan path reuse a
/// warm [`Stored::Mutable`] entry (e.g. one left behind by a
/// committed `aminsert`) without forcing a read-only rebuild, while
/// a fresh scan installs the cheaper [`Stored::ReadOnly`] variant.
///
/// Both arms expose the same `(scores, ids)` search and `len`
/// surface; the `Mutable` arm takes a read guard for the duration
/// of the call (uncontended in a single-threaded backend).
#[derive(Clone)]
pub(crate) enum ScanHandle {
    ReadOnly(Arc<ReadOnlyIndex>),
    Mutable(Arc<RwLock<IdMapIndex>>),
    /// Out-of-core cell-scoped IVF (Phase B-1/B-2). The big codes
    /// buffer is faulted per-probed-cell off the mmap; the resident
    /// set is bounded by `probes * cell_size`, not `O(n)`.
    Ooc(Arc<OocIvfIndex>),
    /// RAM-resident Vamana graph index (Phase G-2a). Navigated via
    /// greedy beam search instead of a flat/masked whole-index scan.
    Graph(Arc<GraphIndex>),
}

impl ScanHandle {
    pub fn len(&self) -> usize {
        match self {
            ScanHandle::ReadOnly(a) => a.len(),
            ScanHandle::Mutable(a) => a.read().len(),
            ScanHandle::Ooc(a) => a.len(),
            ScanHandle::Graph(a) => a.len(),
        }
    }

    /// True if the index has no live vectors.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        match self {
            ScanHandle::ReadOnly(a) => a.search(queries, k),
            ScanHandle::Mutable(a) => a.read().search(queries, k),
            // An OOC handle has no whole-index search; the scan path
            // never calls this arm (it always routes OOC through the
            // cell-scoped path). Returning empty is the safe inert
            // fallback should a future caller hit it.
            ScanHandle::Ooc(_) => (Vec::new(), Vec::new()),
            // The graph scan path always routes through `graph()` /
            // `GraphIndex::search`, not this whole-index arm; same
            // inert fallback as Ooc should a future caller hit it.
            ScanHandle::Graph(a) => a.search(queries, k),
        }
    }

    /// The out-of-core [`OocIvfIndex`], when this handle is the OOC
    /// variant. The IVF scan path uses it to coarse-probe + gather
    /// the probed cells off the mmap. `None` for the flat / mutable
    /// arms, which keep the whole-index search.
    pub(crate) fn ooc(&self) -> Option<Arc<OocIvfIndex>> {
        match self {
            ScanHandle::Ooc(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// The Vamana [`GraphIndex`], when this handle is the graph
    /// variant (Phase G-2a). The graph scan path uses this instead of
    /// `search`/`search_masked`. `None` for every other arm.
    pub(crate) fn graph(&self) -> Option<Arc<GraphIndex>> {
        match self {
            ScanHandle::Graph(a) => Some(a.clone()),
            _ => None,
        }
    }

    /// IVF cell-restricted search. Returns `Some((scores, ids))` only for
    /// the [`ScanHandle::ReadOnly`] arm, where slot order matches the
    /// on-disk cell directory the `mask` was derived from. Returns `None`
    /// for the [`ScanHandle::Mutable`] arm (a post-insert / dirty-xact
    /// mirror), whose slot order has diverged from the build-time cell
    /// layout — the caller must fall back to the flat [`Self::search`].
    pub fn search_masked(
        &self,
        queries: &[f32],
        k: usize,
        mask: &[bool],
    ) -> Option<(Vec<f32>, Vec<u64>)> {
        match self {
            ScanHandle::ReadOnly(a) => Some(a.search_masked(queries, k, mask)),
            ScanHandle::Mutable(_) => None,
            // OOC never uses the whole-index mask path; the scan
            // routes it through `OocIvfIndex::search_ooc` instead.
            ScanHandle::Ooc(_) => None,
            // The graph scan path routes through `graph()` /
            // `GraphIndex::search` instead of a slot mask.
            ScanHandle::Graph(_) => None,
        }
    }

    /// Build a by-slot allowlist mask (Phase C) from a set of external
    /// ids, for the [`ScanHandle::ReadOnly`] arm whose slot order
    /// matches the on-disk layout. Returns `None` for the `Mutable`
    /// arm (slot order diverged — the caller post-filters by the
    /// allowlist instead) and the `Ooc` arm (which masks the compact
    /// sub-index inside `search_ooc`).
    pub(crate) fn allow_slot_mask(&self, allowed: &HashSet<u64>) -> Option<Vec<bool>> {
        match self {
            ScanHandle::ReadOnly(a) => Some(a.allow_slot_mask(allowed)),
            ScanHandle::Mutable(_) | ScanHandle::Ooc(_) | ScanHandle::Graph(_) => None,
        }
    }
}

/// Composite cache key. `attnum = 0` is reserved for the index AM
/// path; positive values are heap attribute numbers from the
/// function-driven path.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct CacheKey {
    pub rel_oid: pg_sys::Oid,
    pub attnum: i16,
    pub bit_width: u8,
    pub dim: u32,
}

/// Mutable mirror of relfile meta-page state alongside an AM-path
/// cache entry. Maintained by `aminsert` and flushed by the
/// `PreCommit` xact callback. `None` for the knn path (read-only
/// snapshots).
#[derive(Clone)]
pub struct PersistState {
    pub bit_width: i32,
    pub dim: i32,
    pub n_vectors: i64,
    pub version: i32,
    pub live_ids: Vec<u64>,
}

/// What a cache entry actually holds. The index-AM scan path
/// installs the lightweight [`Stored::ReadOnly`] variant (no
/// `id_to_slot` `HashMap`); `aminsert` and `turbovec.knn()` install
/// the full [`Stored::Mutable`] [`IdMapIndex`].
///
/// For `attnum = 0` (the AM path) a single relfile may be cached as
/// either variant over its lifetime: a read-only scan installs
/// `ReadOnly`; the first `aminsert` in a transaction evicts it (via
/// [`am_lookup_for_mutation`] returning `None`) and reinstalls a
/// `Mutable` entry through [`am_install`]. The HashMap is therefore
/// built lazily, only when a mutation actually needs it.
enum Stored {
    /// Mutable, id-addressed index. Used by `aminsert` (write guard)
    /// and `turbovec.knn()`.
    Mutable(Arc<RwLock<IdMapIndex>>),
    /// Read-only positional index + slot table, no `id_to_slot`
    /// map. Used by the index-AM scan path.
    ReadOnly(Arc<ReadOnlyIndex>),
    /// Out-of-core cell-scoped IVF (Phase B-1/B-2): bounded resident
    /// metadata + a relfile mmap; the codes buffer is faulted per
    /// probed cell. Installed by the IVF scan path when
    /// `turbovec.out_of_core` is on.
    Ooc(Arc<OocIvfIndex>),
    /// RAM-resident Vamana graph index (Phase G-2a). Installed by the
    /// graph scan path for a `kind = KIND_GRAPH` index.
    Graph(Arc<GraphIndex>),
}

impl Stored {
    /// Cheap scan-facing view over whichever variant this is.
    fn scan_handle(&self) -> ScanHandle {
        match self {
            Stored::Mutable(a) => ScanHandle::Mutable(a.clone()),
            Stored::ReadOnly(a) => ScanHandle::ReadOnly(a.clone()),
            Stored::Ooc(a) => ScanHandle::Ooc(a.clone()),
            Stored::Graph(a) => ScanHandle::Graph(a.clone()),
        }
    }
}

struct Entry {
    /// The materialised index. See [`Stored`] for which variant a
    /// given caller installs and how the AM path upgrades a
    /// read-only entry to a mutable one on first mutation.
    index: Stored,
    /// Approximate bytes the entry occupies. Used for the LRU cap.
    bytes: usize,
    /// `pg_class.relfilenode` snapshot. Zero means we didn't track it
    /// (treated as "always stale" so the next lookup rebuilds).
    relfilenode: u32,
    /// Freshness signal. For the knn path this is `count(*)`; for
    /// the AM path this is the relfile meta page's `am_version`
    /// at load time, advanced to `persist.version` after a
    /// successful commit-time persist.
    n_rows: i64,
    /// Insertion order for LRU eviction. Higher = more recently used.
    seq: u64,
    /// Set by `aminsert` once the in-memory index has been mutated
    /// past the persisted snapshot. Cleared by the `PreCommit` hook
    /// after the relfile rewrite succeeds, or by `invalidate_dirty`
    /// after `XACT_EVENT_ABORT`.
    dirty: bool,
    /// AM-path mirror of the relfile meta-page fields. `None` for
    /// entries installed by the read-only knn path.
    persist: Option<PersistState>,
}

static CACHE: LazyLock<Mutex<HashMap<CacheKey, Entry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SEQ: LazyLock<Mutex<u64>> = LazyLock::new(|| Mutex::new(0));

fn next_seq() -> u64 {
    let mut s = SEQ.lock();
    *s += 1;
    *s
}

/// Look up the entry for `key`, validating it against the current
/// `(relfilenode, freshness)`. On hit, returns the cached
/// [`Stored::Mutable`] `Arc<RwLock<IdMapIndex>>`. Used by the
/// `turbovec.knn()` path (positive `attnum`), which only ever
/// installs `Mutable` entries; a `ReadOnly` entry under the same
/// key (impossible today given the disjoint `attnum` namespaces) is
/// treated as a miss. On miss the caller calls [`insert`].
pub fn lookup(
    key: CacheKey,
    expected_relfile: u32,
    expected_n_rows: i64,
) -> Option<Arc<RwLock<IdMapIndex>>> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile || entry.n_rows != expected_n_rows {
        // Don't evict if we have unflushed mutations — the on-disk
        // version is intentionally behind the in-memory state until
        // the xact commits. The mutating backend is the only one
        // that sees a stale-looking version while dirty.
        if entry.dirty {
            if let Stored::Mutable(a) = &entry.index {
                let a = a.clone();
                entry.seq = next_seq();
                return Some(a);
            }
        }
        g.remove(&key);
        return None;
    }
    match &entry.index {
        Stored::Mutable(a) => {
            let a = a.clone();
            entry.seq = next_seq();
            Some(a)
        }
        Stored::ReadOnly(_) => None,
        // OOC entries are installed only under the AM key (attnum=0);
        // the knn `lookup` (positive attnum) never matches one. Treat
        // it as a miss for the knn path (it can't be mutated in place).
        Stored::Ooc(_) => None,
        // Same reasoning as Ooc: a Graph entry is AM-key-only and has
        // no id-addressed mutable form.
        Stored::Graph(_) => None,
    }
}

/// Index-AM scan lookup. Returns a [`ScanHandle`] over whichever
/// [`Stored`] variant is cached for `key`, so a fresh read-only
/// scan can reuse a warm `Mutable` entry left by a committed
/// `aminsert` instead of rebuilding. On miss the caller builds a
/// `ReadOnlyIndex` and installs it via [`scan_install`].
///
/// Freshness semantics match [`lookup`]: a `(relfilenode,
/// am_version)` mismatch evicts and returns `None` (unless the
/// entry is dirty, in which case the mutating backend keeps its
/// own un-flushed view).
pub(crate) fn scan_lookup(
    key: CacheKey,
    expected_relfile: u32,
    expected_n_rows: i64,
) -> Option<ScanHandle> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile || entry.n_rows != expected_n_rows {
        if entry.dirty {
            entry.seq = next_seq();
            return Some(entry.index.scan_handle());
        }
        g.remove(&key);
        return None;
    }
    entry.seq = next_seq();
    Some(entry.index.scan_handle())
}

/// AM-mutation lookup: returns the cached entry whenever the
/// `relfilenode` matches, regardless of the version freshness slot.
/// `aminsert` uses this so a bulk insert doesn't pay a meta-page
/// version read per row — the in-backend cache is the
/// authoritative copy for the duration of the transaction. The
/// scan path uses [`scan_lookup`] so cross-session committed
/// inserts are visible to other backends.
///
/// Returns `None` when the entry is absent, is a read-only scan
/// entry (which the caller must rebuild as a mutable
/// [`IdMapIndex`], paying the deferred `HashMap` build), lacks a
/// persist mirror, or when the relation has been rewritten
/// (CLUSTER / VACUUM FULL / REINDEX / TRUNCATE) since the entry
/// was installed.
pub fn am_lookup_for_mutation(
    key: CacheKey,
    expected_relfile: u32,
) -> Option<Arc<RwLock<IdMapIndex>>> {
    let mut g = CACHE.lock();
    let entry = g.get_mut(&key)?;
    // A read-only scan entry can't be mutated in place (it has no
    // `id_to_slot` map and the inner index may borrow a read-only
    // mmap). Drop it so the caller rebuilds a full `IdMapIndex` via
    // `am_install` — this is where the deferred `HashMap` build
    // finally happens, on the first mutation that needs it.
    let Stored::Mutable(arc) = &entry.index else {
        g.remove(&key);
        return None;
    };
    let arc = arc.clone();
    if entry.relfilenode != expected_relfile {
        if entry.dirty {
            // Dirty + relfile mismatch is impossible in practice
            // (we don't reindex our own index mid-aminsert), but be
            // conservative and keep the dirty entry rather than
            // silently dropping unflushed mutations.
            entry.seq = next_seq();
            return Some(arc);
        }
        g.remove(&key);
        return None;
    }
    if entry.persist.is_none() {
        // A mutable entry without a persist mirror would be a knn
        // install under an AM key (impossible given disjoint attnum
        // namespaces); drop it so the caller reloads via
        // `am_install`.
        g.remove(&key);
        return None;
    }
    entry.seq = next_seq();
    Some(arc)
}

/// AM-scan visibility lookup: find the dirty AM-path cache entry
/// for `rel_oid` with `attnum = 0`, regardless of `bit_width` or
/// `dim`. Used by the scan path when the relfile meta page is
/// the `(dim = 0, n_vectors = 0)` sentinel written by
/// `ambuildempty` — the in-memory mirror has the truthful
/// `(bit_width, dim, n_vectors, version)` tuple. Returns the cache
/// key and a snapshot of the persist-state mirror alongside the
/// shared index, so the caller can install a freshness signal that
/// matches what the next `aminsert` would see.
pub fn am_find_dirty_by_rel(
    rel_oid: pg_sys::Oid,
) -> Option<(CacheKey, Arc<RwLock<IdMapIndex>>, PersistState)> {
    let g = CACHE.lock();
    for (k, e) in g.iter() {
        if k.rel_oid == rel_oid && k.attnum == 0 {
            if let (Stored::Mutable(a), Some(p)) = (&e.index, e.persist.as_ref()) {
                return Some((*k, a.clone(), p.clone()));
            }
        }
    }
    None
}

/// knn-path install: insert or replace the entry for `key` with no
/// persistence-state mirror attached. The cached index is treated
/// as read-only by the knn callers.
pub fn insert(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> Arc<RwLock<IdMapIndex>> {
    let arc = Arc::new(RwLock::new(index));
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: Stored::Mutable(arc.clone()),
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
            dirty: false,
            persist: None,
        },
    );
    enforce_cap(&mut g);
    arc
}

/// Index-AM scan install: cache a freshly-built [`ReadOnlyIndex`]
/// (no `id_to_slot` `HashMap`) under `key`. Returns a [`ScanHandle`]
/// the caller drains. This is the cold-scan fast path: a read-only
/// backend that only ever scans never pays the O(n) `HashMap` build.
/// All index data is read through the buffer manager.
pub(crate) fn scan_install(
    key: CacheKey,
    index: ReadOnlyIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> ScanHandle {
    let arc = Arc::new(index);
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: Stored::ReadOnly(arc.clone()),
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
            dirty: false,
            persist: None,
        },
    );
    enforce_cap(&mut g);
    ScanHandle::ReadOnly(arc)
}

/// Out-of-core IVF scan install (Phase B-1/B-2): cache an
/// [`OocIvfIndex`] (bounded resident metadata; the codes buffer is
/// gathered per probed cell through the buffer manager). Returns a
/// [`ScanHandle::Ooc`] the caller drains via the cell-scoped path.
/// `bytes` is the (bounded) resident footprint estimate for the LRU
/// cap — NOT `O(n)` codes, just the centroids/scales/ids/directory
/// tables.
pub(crate) fn scan_install_ooc(
    key: CacheKey,
    index: OocIvfIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> ScanHandle {
    let arc = Arc::new(index);
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: Stored::Ooc(arc.clone()),
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
            dirty: false,
            persist: None,
        },
    );
    enforce_cap(&mut g);
    ScanHandle::Ooc(arc)
}

/// Index-AM scan install for a Vamana graph index (Phase G-2a): cache
/// a freshly-built [`GraphIndex`] under `key`. Returns the installed
/// [`ScanHandle::Graph`].
pub(crate) fn scan_install_graph(
    key: CacheKey,
    index: GraphIndex,
    bytes: usize,
    relfilenode: u32,
    n_rows: i64,
) -> ScanHandle {
    let arc = Arc::new(index);
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: Stored::Graph(arc.clone()),
            bytes,
            relfilenode,
            n_rows,
            seq: next_seq(),
            dirty: false,
            persist: None,
        },
    );
    enforce_cap(&mut g);
    ScanHandle::Graph(arc)
}

/// AM-path install: insert or replace the entry for `key` and
/// attach the supplied `PersistState` mirror so subsequent
/// `aminsert` calls can mutate the in-memory index and defer the
/// relfile rewrite to commit time.
pub fn am_install(
    key: CacheKey,
    index: IdMapIndex,
    bytes: usize,
    relfilenode: u32,
    freshness: i64,
    persist: PersistState,
) -> Arc<RwLock<IdMapIndex>> {
    let arc = Arc::new(RwLock::new(index));
    let mut g = CACHE.lock();
    g.insert(
        key,
        Entry {
            index: Stored::Mutable(arc.clone()),
            bytes,
            relfilenode,
            n_rows: freshness,
            seq: next_seq(),
            dirty: false,
            persist: Some(persist),
        },
    );
    enforce_cap(&mut g);
    arc
}

/// Mutate the AM-path persist mirror under the cache mutex. Returns
/// the `CacheKey` if the entry exists and has a persist state,
/// otherwise `None` (caller must install a fresh entry).
///
/// The closure is invoked with `&mut PersistState` and is
/// responsible for advancing `n_vectors`, `version`, and
/// `live_ids`. The `dirty` flag is set after the closure returns.
pub fn am_mark_dirty<F: FnOnce(&mut PersistState)>(key: CacheKey, f: F) -> bool {
    let mut g = CACHE.lock();
    let Some(entry) = g.get_mut(&key) else {
        return false;
    };
    let Some(p) = entry.persist.as_mut() else {
        return false;
    };
    f(p);
    entry.dirty = true;
    true
}

/// Snapshot of a dirty AM-path entry that the `PreCommit` xact
/// callback can flush to the relfile main fork. We hand the caller
/// the `Arc<RwLock<IdMapIndex>>` so it can take a read guard for
/// the duration of the relfile rewrite without holding the cache
/// mutex.
pub struct DirtyEntry {
    pub key: CacheKey,
    pub index: Arc<RwLock<IdMapIndex>>,
    pub persist: PersistState,
}

/// Snapshot every currently-dirty AM-path entry. Does **not**
/// clear the dirty flag — call [`clear_dirty`] after each
/// relfile rewrite succeeds, so a panic mid-flush leaves the
/// remaining entries dirty for the matching `Abort` callback to
/// invalidate.
pub fn drain_dirty() -> Vec<DirtyEntry> {
    let g = CACHE.lock();
    let mut out = Vec::new();
    for (k, e) in g.iter() {
        if !e.dirty {
            continue;
        }
        let Some(p) = e.persist.as_ref() else {
            continue;
        };
        // Dirty entries are always `Mutable` — only `aminsert`
        // sets `dirty`, and it only ever installs `Mutable`
        // entries. A dirty `ReadOnly` entry is structurally
        // impossible; skip it defensively rather than panic.
        let Stored::Mutable(a) = &e.index else {
            continue;
        };
        out.push(DirtyEntry {
            key: *k,
            index: a.clone(),
            persist: p.clone(),
        });
    }
    out
}

/// Mark `key`'s entry clean and advance its freshness slot to the
/// current `persist.version`, so subsequent in-backend lookups hit
/// without forcing another reload. Called after the relfile
/// rewrite succeeds.
pub fn clear_dirty(key: CacheKey) {
    let mut g = CACHE.lock();
    if let Some(entry) = g.get_mut(&key) {
        entry.dirty = false;
        if let Some(p) = entry.persist.as_ref() {
            entry.n_rows = p.version as i64;
        }
    }
}

/// Drop every dirty AM-path entry. Called from the `Abort` xact
/// callback so a rolled-back transaction cannot leave in-memory
/// mutations visible to the next scan in this backend.
pub fn invalidate_dirty() {
    let mut g = CACHE.lock();
    g.retain(|_, e| !e.dirty);
}

/// Drop every entry referencing `rel_oid`. Called from index/table
/// DROP paths; harmless to call unconditionally.
#[allow(dead_code)]
pub fn invalidate(rel_oid: pg_sys::Oid) {
    let mut g = CACHE.lock();
    g.retain(|k, _| k.rel_oid != rel_oid);
}

/// Drop the entire cache. Used by tests.
#[allow(dead_code)]
pub fn invalidate_all() {
    CACHE.lock().clear();
}

/// Number of cached entries. Test/diagnostic only.
#[allow(dead_code)]
pub fn len() -> usize {
    CACHE.lock().len()
}

/// Test/diagnostic: report the [`Stored`] variant cached for an AM
/// (attnum = 0) entry on `rel_oid`, as a short tag (`"ooc"`,
/// `"readonly"`, `"mutable"`), or `None` if no AM entry is cached.
/// Used by the Phase B-1/B-2 mechanism test to prove an
/// `out_of_core = on` IVF scan installs a cell-scoped `Ooc` entry
/// (the codes are NOT loaded whole) while `off` installs the
/// whole-index `ReadOnly` entry.
#[allow(dead_code)]
pub fn am_entry_variant(rel_oid: pg_sys::Oid) -> Option<&'static str> {
    let g = CACHE.lock();
    // Prefer an Ooc entry if any exists for this rel (the cache is
    // process-global across pg_tests; report the variant the current
    // GUC would have installed rather than an arbitrary iteration
    // order). Fall back to the first AM entry's variant.
    let mut fallback: Option<&'static str> = None;
    for (k, e) in g.iter() {
        if k.rel_oid == rel_oid && k.attnum == 0 {
            let tag = match &e.index {
                Stored::Ooc(_) => "ooc",
                Stored::ReadOnly(_) => "readonly",
                Stored::Mutable(_) => "mutable",
                Stored::Graph(_) => "graph",
            };
            if tag == "ooc" {
                return Some("ooc");
            }
            fallback.get_or_insert(tag);
        }
    }
    fallback
}

fn enforce_cap(map: &mut HashMap<CacheKey, Entry>) {
    let cap_mb = guc::CACHE_SIZE_MB.get();
    if cap_mb <= 0 {
        // GUC = 0 disables caching entirely. Don't drop dirty
        // entries — flushing them is the PreCommit hook's job.
        map.retain(|_, e| e.dirty);
        return;
    }
    let cap = (cap_mb as usize).saturating_mul(1024 * 1024);
    let mut total: usize = map.values().map(|e| e.bytes).sum();
    while total > cap && map.len() > 1 {
        // Find LRU entry by lowest `seq`. Skip dirty entries — they
        // hold un-persisted mutations and can only be evicted via
        // the xact-end callbacks.
        let lru_key = map
            .iter()
            .filter(|(_, e)| !e.dirty)
            .min_by_key(|(_, e)| e.seq)
            .map(|(k, _)| *k);
        match lru_key {
            Some(k) => {
                if let Some(e) = map.remove(&k) {
                    total = total.saturating_sub(e.bytes);
                }
            }
            None => break,
        }
    }
}

/// Look up the relation's current `relfilenode` via `pg_class`.
/// Returns 0 on lookup failure (callers treat that as "unknown" — a
/// `0 != stored.relfilenode` comparison forces a rebuild).
pub fn current_relfilenode(rel_oid: pg_sys::Oid) -> u32 {
    let v: Option<i64> = Spi::get_one_with_args(
        "SELECT (relfilenode)::int8 FROM pg_class WHERE oid = $1",
        &[rel_oid.into()],
    )
    .ok()
    .flatten();
    v.unwrap_or(0) as u32
}

/// Pull the current relfilenode straight off the in-memory
/// `Relation` struct without an SPI roundtrip. The field name
/// changed between PG 15 and PG 16 (`rd_node` -> `rd_locator`,
/// `relNode` -> `relNumber`); both encode the same `Oid` /
/// `RelFileNumber` value as `u32`.
///
/// # Safety
///
/// Caller must pass a non-null `Relation` pointer that's pinned
/// in the relcache for the duration of the call (true for any
/// `Relation` Postgres hands an AM callback).
#[allow(dead_code)]
pub unsafe fn relfilenode_from_relation(rel: pg_sys::Relation) -> u32 {
    if rel.is_null() {
        return 0;
    }
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
    {
        // pg13/14/15: `rd_node.relNode` is an `Oid`.
        let oid: pg_sys::Oid = (*rel).rd_node.relNode;
        oid.to_u32()
    }
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    {
        // pg16+: `rd_locator.relNumber` is a `RelFileNumber`, which
        // is a typedef for `Oid`. Use `Oid::to_u32` for the
        // conversion — `as u32` doesn't work on the newtype.
        let oid: pg_sys::Oid = (*rel).rd_locator.relNumber;
        oid.to_u32()
    }
}

#[cfg(test)]
mod graph_scorer_tests {
    use super::ReadOnlyIndex;
    use crate::index::graph;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use turbovec::IdMapIndex;

    /// Deterministic L2-normalised synthetic corpus (rows unit-norm,
    /// matching turbovec's assumption and the graph build's space).
    fn corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut v = vec![0.0f32; n * dim];
        for row in v.chunks_mut(dim) {
            for x in row.iter_mut() {
                *x = rng.gen_range(-1.0f32..1.0);
            }
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v
    }

    /// Quantize `flat` via the SAME `IdMapIndex` path `ambuild` uses
    /// for the graph kind, then wrap the parts in a `ReadOnlyIndex`
    /// (the exact shape `install_graph_index` builds). Slot ids are
    /// 0..n (a graph build's synthetic ids).
    fn read_only_index(flat: &[f32], n: usize, dim: usize, bit_width: usize) -> ReadOnlyIndex {
        let mut idx = IdMapIndex::new(dim, bit_width).unwrap();
        let ids: Vec<u64> = (0..n as u64).collect();
        idx.add_with_ids(flat, &ids).unwrap();
        idx.prepare_eager();
        ReadOnlyIndex::from_prepared_parts(
            bit_width,
            dim,
            n,
            idx.packed_codes().to_vec(),
            idx.scales().to_vec(),
            idx.slot_to_id().to_vec(),
            idx.blocked_codes().to_vec(),
            idx.n_blocks(),
            idx.centroids().to_vec(),
            idx.boundaries().to_vec(),
            Some(idx.rotation().to_vec()),
        )
    }

    /// The load-bearing Phase G-2c equivalence property: the SIMD
    /// LUT-based per-hop `GraphScorer` drives `graph_search` to
    /// essentially the SAME top-k result SET as the G-2a scalar
    /// `score_slots` path — this is a speed change, not a recall
    /// change. Both feed the identical beam search; only the per-hop
    /// scoring kernel differs. The scorer reproduces turbovec's
    /// identity-TQ+ score formula UNQUANTIZED, whereas the
    /// masked-search kernel quantises its LUT to u8 [0,127], so the
    /// two rank identically EXCEPT on exact near-ties the u8 rounding
    /// can reorder — measured ~96–98% top-10 set overlap on this
    /// synthetic corpus, with the residual divergence being genuine
    /// near-tie reordering (both are valid ANN results, reranked
    /// exactly by `xs_recheckorderby` downstream anyway). We assert a
    /// high overlap floor rather than exact equality: exact-set
    /// equality would be asserting the fp32 and u8-quantized kernels
    /// never disagree on a tie, which is false by construction and
    /// not what "no recall change" means.
    #[test]
    fn graph_scan_simd_matches_scalar_result_set() {
        for &bit_width in &[2usize, 4] {
            for &dim in &[64usize, 128] {
                let n = 400usize;
                let flat = corpus(n, dim, 0xC0FFEE + bit_width as u64 + dim as u64);
                let roi = read_only_index(&flat, n, dim, bit_width);
                let (adj, entry) = graph::build_vamana(&flat, n, dim);

                let mut qseed = 0x1234_5678u64;
                let mut total = 0usize;
                let mut overlap = 0usize;
                for _ in 0..20 {
                    let mut q = vec![0.0f32; dim];
                    for v in q.iter_mut() {
                        qseed = qseed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        *v = ((qseed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                    }
                    for &k in &[1usize, 10, 20] {
                        // Scalar (G-2a) path: score_slots per hop.
                        let scalar = graph::graph_search(&adj, entry, k, &[], |ids| {
                            roi.score_slots(&q, ids)
                        });
                        // SIMD (G-2c) path: per-query LUT scorer.
                        let scorer = roi.graph_scorer(&q);
                        let simd =
                            graph::graph_search(&adj, entry, k, &[], |ids| scorer.score_batch(ids));

                        let scalar_set: std::collections::HashSet<u32> =
                            scalar.iter().map(|&(_, id)| id).collect();
                        let simd_set: std::collections::HashSet<u32> =
                            simd.iter().map(|&(_, id)| id).collect();
                        total += k.min(n);
                        overlap += scalar_set.intersection(&simd_set).count();
                        // Structural contract still exact: both return
                        // exactly min(k, n) distinct results.
                        assert_eq!(simd.len(), k.min(n));
                    }
                }
                let frac = overlap as f64 / total as f64;
                assert!(
                    frac >= 0.9,
                    "SIMD/scalar graph-scan result-set overlap {frac:.3} too low \
                     (bit_width={bit_width} dim={dim}) — the SIMD path changed recall, \
                     not just speed"
                );
            }
        }
    }

    /// The `GraphScorer` is deterministic: the same (query, id batch)
    /// always produces the same scores, call to call. Underpins the
    /// scan path's stability.
    #[test]
    fn graph_scorer_is_deterministic() {
        let n = 200usize;
        let dim = 64usize;
        let flat = corpus(n, dim, 99);
        let roi = read_only_index(&flat, n, dim, 4);
        let q = corpus(1, dim, 7);
        let ids: Vec<u32> = (0..n as u32).step_by(3).collect();
        let s1 = roi.graph_scorer(&q).score_batch(&ids);
        let s2 = roi.graph_scorer(&q).score_batch(&ids);
        assert_eq!(s1, s2, "GraphScorer scores are not deterministic");
    }

    /// The per-slot `GraphScorer` score reproduces turbovec's own
    /// masked-search score (identity-TQ+ formula) up to the kernel's
    /// u8-LUT quantization. Pins the score IDENTITY the equivalence
    /// rests on. The kernel quantises its LUT to a per-query u8 range,
    /// so the per-score error is bounded by the QUANTIZATION STEP
    /// (`≈ score_range / 127`), an ABSOLUTE bound — a per-score
    /// RELATIVE bound is meaningless where the true score is near zero
    /// (many slots are near-orthogonal to the query). We therefore
    /// assert the max absolute error is a small fraction of the score
    /// RANGE, and that the nearest-slot argmax agrees (the ranking
    /// crux the beam search actually depends on).
    #[test]
    fn graph_scorer_score_matches_kernel_within_quantization() {
        let n = 300usize;
        let dim = 128usize;
        let flat = corpus(n, dim, 55);
        let roi = read_only_index(&flat, n, dim, 4);
        let q = corpus(1, dim, 8);
        let ids: Vec<u32> = (0..n as u32).collect();
        let scorer_scores = roi.graph_scorer(&q).score_batch(&ids);
        let kernel_scores = roi.score_slots(&q, &ids);

        let smin = scorer_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let smax = scorer_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (smax - smin).max(1e-6);
        let mut max_abs = 0.0f32;
        for (&a, &b) in scorer_scores.iter().zip(kernel_scores.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        // u8 LUT => ~1/127 of the range per sub-table; summed over
        // dim/2 groups the worst case is looser, but empirically well
        // under 10% of the range. Assert a comfortable ceiling.
        assert!(
            max_abs < 0.1 * range,
            "GraphScorer diverges from kernel beyond quantization: max_abs={max_abs} range={range}"
        );
        // Argmax (nearest slot) must agree — the ranking crux.
        let argmax = |v: &[f32]| -> Option<usize> {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
        };
        assert_eq!(
            argmax(&scorer_scores),
            argmax(&kernel_scores),
            "nearest-slot argmax disagrees"
        );
    }

    /// G-2c local RELATIVE scan timing: the SIMD LUT `GraphScorer`
    /// per-hop path vs the G-2a scalar `score_slots` path, same
    /// graph, same queries. This box's absolute latency is
    /// untrustworthy (AGENTS.md), but the RELATIVE per-query ratio is
    /// meaningful. `#[ignore]` — run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_scan_simd_vs_scalar() {
        use std::time::Instant;
        let n = 20_000usize;
        let dim = 256usize;
        let bw = 4usize;
        let flat = corpus(n, dim, 7);
        let roi = read_only_index(&flat, n, dim, bw);
        let (adj, entry) = graph::build_vamana(&flat, n, dim);
        let queries: Vec<Vec<f32>> = (0..50).map(|s| corpus(1, dim, 1000 + s)).collect();
        let k = 10usize;

        // Warm + scalar path.
        let mut sink = 0u64;
        let t = Instant::now();
        for q in &queries {
            let hits = graph::graph_search(&adj, entry, k, &[], |ids| roi.score_slots(q, ids));
            sink = sink.wrapping_add(hits.len() as u64);
        }
        let scalar = t.elapsed();

        // SIMD LUT path.
        let t = Instant::now();
        for q in &queries {
            let scorer = roi.graph_scorer(q);
            let hits = graph::graph_search(&adj, entry, k, &[], |ids| scorer.score_batch(ids));
            sink = sink.wrapping_add(hits.len() as u64);
        }
        let simd = t.elapsed();

        eprintln!(
            "[G-2c scan] n={n} dim={dim} bw={bw} q={} k={k}  scalar(score_slots)={:?}  simd(GraphScorer)={:?}  speedup={:.2}x  (sink={sink})",
            queries.len(),
            scalar,
            simd,
            scalar.as_secs_f64() / simd.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod merge_tests {
    use super::merge_topk;
    use turbovec::SearchResults;

    fn sr(pairs: &[(f32, i64)]) -> SearchResults {
        // Kernel output is score-descending; mimic that so merge sees
        // the same shape it would in production.
        let mut p = pairs.to_vec();
        p.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        SearchResults {
            scores: p.iter().map(|x| x.0).collect(),
            indices: p.iter().map(|x| x.1).collect(),
            nq: 1,
            k: p.len(),
        }
    }

    /// The load-bearing property: merging per-chunk local top-k lists
    /// yields the SAME top-k SET a serial scan of all rows would. We
    /// build a global row set, compute the true top-k, split the rows
    /// into chunks, take each chunk's local top-k, merge, and assert
    /// the merged set equals the global top-k set.
    #[test]
    fn merge_matches_global_topk() {
        // 12 rows, distinct scores + a couple of ties.
        let rows: Vec<(f32, i64)> = vec![
            (0.90, 0),
            (0.10, 1),
            (0.55, 2),
            (0.55, 3),
            (0.80, 4),
            (0.20, 5),
            (0.70, 6),
            (0.30, 7),
            (0.95, 8),
            (0.40, 9),
            (0.60, 10),
            (0.50, 11),
        ];
        let k = 5;

        // Global top-k SET (serial ground truth).
        let mut g = rows.clone();
        g.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        let global: std::collections::HashSet<i64> = g.iter().take(k).map(|p| p.1).collect();

        // Split into 3 chunks, each takes its local top-k.
        let chunks = [&rows[0..4], &rows[4..8], &rows[8..12]];
        let locals: Vec<SearchResults> = chunks
            .iter()
            .map(|c| {
                let mut cc = c.to_vec();
                cc.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                cc.truncate(k);
                sr(&cc)
            })
            .collect();

        let (scores, slots) = merge_topk(&locals, k);
        assert_eq!(slots.len(), k, "merge returned {} != k", slots.len());
        // Scores descending.
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "merge not score-descending: {scores:?}");
        }
        let merged: std::collections::HashSet<i64> = slots.iter().copied().collect();
        assert_eq!(merged, global, "merged top-k SET != global top-k SET");
    }

    /// Deterministic tie-break: equal scores resolve by ascending
    /// slot, so the merge is a pure function of its inputs (not thread
    /// scheduling). Two ties at 0.55 -> slots 2 then 3.
    #[test]
    fn merge_tie_break_is_slot_ascending() {
        let locals = vec![sr(&[(0.55, 3), (0.55, 2), (0.90, 0)])];
        let (_s, slots) = merge_topk(&locals, 3);
        assert_eq!(slots, vec![0, 2, 3], "tie-break not slot-ascending");
    }
}
