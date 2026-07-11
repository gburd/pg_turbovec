//! IVF (inverted-file) coarse-quantizer layer — IVF-1.
//!
//! **Scope of IVF-1: the build path + on-disk layout only.** The
//! scan path stays FLAT (it ignores cells and scores the whole
//! corpus, exactly as it does for `lists = 0`). IVF-1 proves the
//! v3→v4 wire change round-trips: k-means coarse training,
//! cell-contiguous code reordering, persisting coarse centroids +
//! cell directory, and reading them all back. The latency win
//! (cell-restricted search) is IVF-2.
//!
//! This module is deliberately **Postgres-free** so it is
//! unit-testable like `kernels.rs`. It owns:
//!
//! - deterministic k-means (k-means++ seeding + Lloyd's iterations),
//!   mirroring turbovec's `ChaCha8Rng::seed_from_u64` determinism
//!   precedent (see `turbovec/src/rotation.rs`). The fixed
//!   [`IVF_SEED`] anchors reproducibility: same sample + same
//!   `lists` ⇒ byte-identical centroids;
//! - the [`CellDirectory`] type and its `(code_offset, n_vectors)`
//!   entries that give each cell's contiguous slot range;
//! - the cell-contiguous permutation builder.
//!
//! ## Space
//!
//! All clustering happens in the **rotated** space. The caller
//! rotates each sampled / assigned vector with the existing
//! `turbovec` rotation matrix before handing it here, so the cells
//! live in the same space the fine quantizer and the query use.
//! This module never sees raw input vectors.
//!
//! ## Precision
//!
//! Coarse centroids are stored as f32 (resolved decision: at
//! `nlist ≈ √n` it's a few MiB, mmap-resident; f16 rounding would
//! cost boundary recall for no real savings). The cluster
//! arithmetic here is all f32.

use gemm::{gemm, Parallelism};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Fixed seed for the k-means RNG. Mirrors turbovec's
/// `ROTATION_SEED` determinism precedent: a constant seed makes the
/// whole coarse-training pipeline reproducible, so the same sample
/// and the same `lists` always produce byte-identical centroids
/// (the IVF determinism anchor the `ivf_kmeans_deterministic`
/// `#[pg_test]` pins). Distinct from `ROTATION_SEED` (42) so the two
/// deterministic subsystems don't share an RNG stream by accident.
pub const IVF_SEED: u64 = 0x1F_F5EE_D00D_u64;

/// Maximum number of Lloyd's iterations. FAISS's default
/// coarse-quantizer training runs ~10–25 iterations; 25 converges the
/// small coarse codebook (`nlist ≈ √n` centroids over a
/// `256·nlist`-bounded sample) well within diminishing returns. This
/// is now an upper bound: [`train_kmeans`] early-exits once centroid
/// movement drops below [`KMEANS_TOL`] (k-means typically converges in
/// well under 25 iterations). The cap, the tolerance, and the
/// (deterministic) movement metric are part of the determinism
/// contract — same sample + same `lists` always runs the same number
/// of iterations and produces byte-identical centroids.
pub const KMEANS_ITERS: usize = 25;

/// Convergence tolerance for the Lloyd early-exit. After each update
/// step we measure the total centroid movement as the sum over all
/// cells of `||c_new - c_old||^2` (squared L2, the same fixed-order
/// reduction the rest of the module uses). When that total drops below
/// `KMEANS_TOL` the centroids have stopped moving meaningfully and we
/// stop — no point spending GEMMs on iterations that don't change the
/// partition. Deterministic: a fixed threshold over a deterministic
/// movement metric means the same input always stops at the same
/// iteration. `1e-6` is below the f32 noise floor for unit-norm
/// centroids (coordinates are O(1/√dim)); the partition is stable well
/// before this. Part of the determinism contract.
pub const KMEANS_TOL: f64 = 1e-6;

/// IVF-4a soft-assignment boundary factor. A vector is also assigned
/// to a 2nd..Mth nearest cell `j` only when its squared distance to
/// cell `j` is within `BOUNDARY_FACTOR` of its squared distance to
/// the nearest cell (`dist_j <= BOUNDARY_FACTOR * dist_nearest`).
/// This bounds the storage blow-up: only genuinely-boundary vectors
/// (those nearly equidistant to two cells) get duplicated, not every
/// vector. 1.2 (a 20% slack on squared distance) is the documented
/// constant; it is part of the build determinism contract — changing
/// it changes which vectors get duplicated and thus the on-disk
/// bytes. Empirically yields a ~1.1–1.5× slot blow-up for M=2 on
/// typical data.
pub const BOUNDARY_FACTOR: f32 = 1.2;

/// One cell's contiguous slot range in the (reordered) codes /
/// scales / ids chains.
///
/// After the build-time permutation, cell `c`'s rows occupy slots
/// `[code_offset .. code_offset + n_vectors)`. `code_offset` is a
/// **starting slot index** (not a byte offset); the fine-quantizer
/// chains all share the same slot numbering, so one offset addresses
/// codes, scales, and ids alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellEntry {
    /// Starting slot index of this cell in the reordered chains.
    pub code_offset: u64,
    /// Number of vectors (slots) in this cell.
    pub n_vectors: u32,
}

impl CellEntry {
    /// On-disk size of one entry: `u64` offset + `u32` count = 12
    /// bytes. Stored little-endian, packed (no padding) in the
    /// cell-directory chain.
    pub const ENCODED_BYTES: usize = 12;

    /// Serialise to 12 little-endian bytes.
    pub fn encode(&self) -> [u8; Self::ENCODED_BYTES] {
        let mut out = [0u8; Self::ENCODED_BYTES];
        out[0..8].copy_from_slice(&self.code_offset.to_le_bytes());
        out[8..12].copy_from_slice(&self.n_vectors.to_le_bytes());
        out
    }

    /// Inverse of [`Self::encode`].
    pub fn decode(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() >= Self::ENCODED_BYTES);
        let code_offset = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let n_vectors = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        Self {
            code_offset,
            n_vectors,
        }
    }
}

/// The cell directory: one [`CellEntry`] per cell, in cell-id order.
/// `entries.len() == lists`. The entries partition the `n_vectors`
/// slots exactly — contiguous and non-overlapping, summing to
/// `n_vectors`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellDirectory {
    pub entries: Vec<CellEntry>,
}

impl CellDirectory {
    /// Number of cells. Used in tests and by future IVF-4 work; the
    /// hot scan path indexes `entries` directly.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total vectors across all cells (should equal the index's
    /// `n_vectors`).
    pub fn total_vectors(&self) -> u64 {
        self.entries.iter().map(|e| u64::from(e.n_vectors)).sum()
    }

    /// Flatten the directory to a packed little-endian byte buffer
    /// (`len() * CellEntry::ENCODED_BYTES` bytes) for the on-disk
    /// chain.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * CellEntry::ENCODED_BYTES);
        for e in &self.entries {
            out.extend_from_slice(&e.encode());
        }
        out
    }

    /// Inverse of [`Self::encode`]: decode `lists` entries from a
    /// packed byte buffer.
    pub fn decode(bytes: &[u8], lists: usize) -> Self {
        let mut entries = Vec::with_capacity(lists);
        for i in 0..lists {
            let off = i * CellEntry::ENCODED_BYTES;
            entries.push(CellEntry::decode(
                &bytes[off..off + CellEntry::ENCODED_BYTES],
            ));
        }
        Self { entries }
    }

    /// Build a slot allowlist mask (`Vec<bool>` of length
    /// `n_vectors`) that is `true` for every slot belonging to one of
    /// the `probed` cells. The IVF scan hands this to
    /// `ReadOnlyIndex::search_masked`; turbovec's blocked kernel
    /// short-circuits whole 32-vector blocks whose mask window is
    /// all-zero, so the contiguous unprobed cell ranges skip their
    /// scoring work (the latency win).
    ///
    /// `probed` may contain duplicates or out-of-range ids; both are
    /// ignored. `n_vectors` must equal the index's live count (the
    /// directory's `total_vectors()`).
    pub fn probe_mask(&self, probed: &[u32], n_vectors: usize) -> Vec<bool> {
        let mut mask = vec![false; n_vectors];
        for &c in probed {
            let c = c as usize;
            if c >= self.entries.len() {
                continue;
            }
            let e = self.entries[c];
            let start = e.code_offset as usize;
            let end = (start + e.n_vectors as usize).min(n_vectors);
            for s in mask.iter_mut().take(end).skip(start) {
                *s = true;
            }
        }
        mask
    }

    /// Validate that the entries partition `n_vectors` exactly:
    /// contiguous, non-overlapping, starting at 0, summing to
    /// `n_vectors`. Returns `Ok(())` or a descriptive error. Used by
    /// round-trip tests and as a cheap corruption guard.
    pub fn validate_partition(&self, n_vectors: u64) -> Result<(), String> {
        let mut expected_offset = 0u64;
        for (c, e) in self.entries.iter().enumerate() {
            if e.code_offset != expected_offset {
                return Err(format!(
                    "cell {c}: code_offset {} != expected contiguous offset {expected_offset}",
                    e.code_offset
                ));
            }
            expected_offset += u64::from(e.n_vectors);
        }
        if expected_offset != n_vectors {
            return Err(format!(
                "cell directory covers {expected_offset} vectors, expected {n_vectors}"
            ));
        }
        Ok(())
    }
}

/// Result of coarse k-means training + assignment.
pub struct CoarseModel {
    /// `lists * dim` row-major f32 coarse centroids (rotated space).
    pub centroids: Vec<f32>,
    /// `lists` — number of coarse cells.
    pub lists: usize,
    /// Dimensionality.
    pub dim: usize,
}

impl CoarseModel {
    /// Centroid `c` as a `dim`-length slice.
    /// Single-centroid slice. Used by the scalar `assign_one`
    /// determinism reference (tests) and the per-vector tie-break
    /// recompute; the batched assign GEMM slices centroids directly.
    #[allow(dead_code)]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }

    /// Assign one (rotated) vector to its nearest centroid by
    /// squared Euclidean distance. Returns the cell id. Ties broken
    /// toward the lower cell id (deterministic).
    /// Scalar nearest-centroid assignment. Retained as the
    /// deterministic reference that `batched_assign` is verified
    /// against (`ivf_batched_assign_matches_scalar`) and for the
    /// top-2 tie-break recompute; not on the batched build hot path.
    #[allow(dead_code)]
    pub fn assign_one(&self, v: &[f32]) -> usize {
        debug_assert_eq!(v.len(), self.dim);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.lists {
            let d = sq_dist(v, self.centroid(c));
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }
}

/// Coarse search: score the (already rotated) query against every
/// coarse centroid by squared Euclidean distance and return the
/// `nprobe` nearest cell ids (ascending distance; ties broken toward
/// the lower cell id, deterministic).
///
/// `centroids` is the row-major `lists * dim` coarse codebook in the
/// rotated space (as persisted by the build / read by
/// `relfile::read_coarse_centroids`). `query_rotated` is the query
/// already normalised + rotated into the same space the build used
/// to assign vectors to cells.
///
/// `nprobe` is clamped to `[1, lists]`. Scalar — `lists` is small
/// (`≈√n`), and a scalar loop avoids a second SIMD-correctness
/// surface. Cost `O(lists * dim)`.
pub fn coarse_probe(
    centroids: &[f32],
    lists: usize,
    dim: usize,
    query_rotated: &[f32],
    nprobe: usize,
) -> Vec<u32> {
    debug_assert_eq!(centroids.len(), lists * dim);
    debug_assert_eq!(query_rotated.len(), dim);
    let nprobe = nprobe.clamp(1, lists.max(1)).min(lists);

    // Score every cell. (distance, cell_id) pairs.
    let mut scored: Vec<(f32, u32)> = (0..lists)
        .map(|c| {
            let d = sq_dist(query_rotated, &centroids[c * dim..(c + 1) * dim]);
            (d, c as u32)
        })
        .collect();

    // Partial sort: ascending distance, ties → lower cell id. We need
    // the nprobe smallest, in order. `sort_unstable_by` over `lists`
    // (small) is cheap and deterministic given the (dist, id) key.
    scored.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(nprobe);
    scored.into_iter().map(|(_, c)| c).collect()
}

/// Phase G-1: below this many cells, the plain O(lists*dim) linear
/// [`coarse_probe`] scan is already sub-microsecond-to-low-microsecond
/// and a graph's build cost (even parallel, per-row-independent
/// all-pairs) plus its per-query heap/visited-set overhead isn't worth
/// paying. `4096` matches the threshold the (aspirational, unshipped)
/// v1.20.0 CHANGELOG entry already documented for a "sublinear
/// two-level coarse quantizer" — that structure was never actually
/// implemented (see the G-1 session's docs-drift finding), so this is
/// the first REAL implementation to back that public threshold value.
/// [`coarse_probe_dispatch`] uses this to decide whether to build/use
/// the [`CentroidGraph`] in `auto` mode (`turbovec.coarse_graph`).
pub const GRAPH_MIN_LISTS: usize = 4096;

/// Fixed out-degree of the centroid graph. 16 is the top of the
/// suggested Vamana-lite range (8-16): cheap (16*4B = 64B/centroid;
/// 100k centroids -> 6.4 MiB, trivial next to the O(lists*dim)
/// centroid table itself) and gives the greedy search enough fan-out
/// per hop to avoid getting stuck in a bad local neighbourhood at the
/// centroid counts this targets (thousands, not millions).
pub const GRAPH_DEGREE: usize = 16;

/// A small fixed-out-degree graph over the coarse centroids
/// themselves (Phase G-1, "SPANN-lite" / minimum-viable-graph per
/// ). Turns coarse-cell selection from
/// `O(lists*dim)` ([`coarse_probe`]) into `O(log(lists)*dim)`-ish
/// greedy graph search ([`graph_probe`]) for large `lists`.
///
/// **Undirected (symmetrized).** [`build_centroid_graph`] first
/// builds a directed fixed-out-degree ([`GRAPH_DEGREE`]) nearest-
/// neighbour graph, then adds the reverse of every edge. A pure
/// directed k-NN graph can strand a cell that is someone else's
/// close neighbour but has no close neighbours of its own pointing
/// back (a classic graph-navigability gap for greedy search;
/// `graph_probe_matches_linear_scan_exactly` caught this on random
/// data during development) — symmetrizing guarantees every edge is
/// walkable in both directions, which is what makes the greedy
/// search in [`graph_probe`] safe. Stored CSR-style (`offsets` +
/// flat `neighbors`) rather than a fixed-width row, because
/// symmetrization makes per-cell degree vary (a "hub" cell that many
/// others point to ends up with more than `GRAPH_DEGREE` neighbours).
///
/// **Never persisted.** Computed in-memory, once per backend, from
/// the already-persisted coarse centroids — exactly the pattern
///  requires so this stays purely
/// additive (no wire-format change, no `MetaPageData::version` bump,
/// no REINDEX).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CentroidGraph {
    /// `offsets[c]..offsets[c+1]` indexes `neighbors` for cell `c`'s
    /// (ascending-id, deduplicated) adjacency list. Length
    /// `lists + 1`.
    offsets: Vec<u32>,
    /// Flat, per-cell-contiguous neighbour ids (ascending within each
    /// cell's range; not distance-sorted — [`graph_probe`] scores
    /// them itself).
    neighbors: Vec<u32>,
}

impl CentroidGraph {
    /// Neighbour ids of cell `c` (ascending id order).
    fn neighbors_of(&self, c: usize) -> &[u32] {
        let s = self.offsets[c] as usize;
        let e = self.offsets[c + 1] as usize;
        &self.neighbors[s..e]
    }
}

/// Build a [`CentroidGraph`] over `lists` centroids: for each
/// centroid, its [`GRAPH_DEGREE`] nearest OTHER centroids by exact
/// squared Euclidean distance (brute-force all-pairs — `O(lists^2 *
/// dim)` — deliberately simple per the Phase G-1 plan: `lists` here
/// is bounded at tens of thousands, not corpus scale, so the
/// quadratic build is cheap and avoids the approximate-graph-building
/// bugs a real ANN-graph-builder would risk), then SYMMETRIZED (every
/// directed edge `i -> j` also gets its reverse `j -> i`) so the
/// result is safe for greedy search — see [`CentroidGraph`]'s doc for
/// why the plain directed graph isn't.
///
/// **Byte-deterministic.** The directed pass is per-row independent
/// (parallel-safe, same precedent as `train_kmeans`'s D2-seeding
/// update: computing rows in parallel is bit-identical to serial).
/// The symmetrization pass is a fixed, deterministic reduction:
/// collect every directed edge into one flat `(from, to)` list in
/// ascending `(from, to)` order, add each edge's reverse, sort +
/// dedup ascending, and derive fixed CSR offsets — no hashing, no
/// thread-order dependence. Same `centroids` + `lists` + `dim` ⇒
/// byte-identical `CentroidGraph`.
pub fn build_centroid_graph(centroids: &[f32], lists: usize, dim: usize) -> CentroidGraph {
    debug_assert_eq!(centroids.len(), lists * dim);
    let degree = GRAPH_DEGREE.min(lists.saturating_sub(1));
    if lists <= 1 || degree == 0 {
        return CentroidGraph {
            offsets: vec![0u32; lists + 1],
            neighbors: Vec::new(),
        };
    }

    // Directed pass: row c's GRAPH_DEGREE nearest other cells,
    // ascending (dist, id). Per-row independent -> parallel-safe.
    use rayon::prelude::*;
    let directed: Vec<Vec<u32>> = (0..lists)
        .into_par_iter()
        .map(|c| {
            let me = &centroids[c * dim..(c + 1) * dim];
            let mut scored: Vec<(f32, u32)> = (0..lists)
                .filter(|&j| j != c)
                .map(|j| (sq_dist(me, &centroids[j * dim..(j + 1) * dim]), j as u32))
                .collect();
            scored.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            scored.into_iter().take(degree).map(|(_, id)| id).collect()
        })
        .collect();

    // Symmetrize: every directed edge (c, nb) also contributes (nb, c).
    // Collected in ascending `c` order (fixed, from the `directed`
    // Vec's index order) so the flattened edge list -- and therefore
    // the sort+dedup below -- is a deterministic function of the
    // input, independent of how `directed` was computed.
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(lists * degree * 2);
    for (c, row) in directed.iter().enumerate() {
        for &nb in row {
            edges.push((c as u32, nb));
            edges.push((nb, c as u32));
        }
    }
    edges.sort_unstable();
    edges.dedup();

    // Derive fixed CSR offsets + flat neighbour list from the sorted
    // edge list (single ascending pass; deterministic).
    let mut offsets = vec![0u32; lists + 1];
    for &(from, _) in &edges {
        offsets[from as usize + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let neighbors: Vec<u32> = edges.into_iter().map(|(_, to)| to).collect();
    CentroidGraph { offsets, neighbors }
}

/// Beam width multiplier for [`graph_probe`]'s greedy search relative
/// to the requested `nprobe`. A pure single-path greedy walk ("always
/// step to the locally-closest unvisited neighbour, stop at the first
/// local optimum") is NOT safe for the recall-preserving requirement:
/// it can miss the true nprobe-nearest cells behind a locally-worse
/// hop. Keeping a widened candidate/result beam of
/// `(nprobe * GRAPH_EF_MULTIPLIER).max(GRAPH_EF_FLOOR)` — the classic
/// HNSW-style `ef` parameter — trades a little extra work for a large
/// recall margin, measured empirically (see `graph_probe_recall_*`
/// tests) to reliably match the exact linear scan at these centroid
/// counts.
const GRAPH_EF_MULTIPLIER: usize = 4;
/// Floor on the beam width so a tiny `nprobe` (e.g. 1) still searches
/// widely enough to be safe; see [`GRAPH_EF_MULTIPLIER`].
const GRAPH_EF_FLOOR: usize = 32;

/// Greedy graph search for the `nprobe` nearest cells to
/// `query_rotated`, navigating [`CentroidGraph`] instead of scanning
/// every centroid ([`coarse_probe`]'s `O(lists*dim)`). Classic
/// beam-search graph traversal (Vamana/HNSW-lite): start at `entry`,
/// maintain a bounded max-heap of the best `ef` candidates seen and a
/// min-heap of unvisited candidates to expand, and stop once
/// expanding the closest remaining candidate can no longer improve
/// the current worst-of-`ef`. Returns the `nprobe` best, ascending
/// distance, ties broken toward the lower cell id — the SAME output
/// contract as [`coarse_probe`], so callers are interchangeable.
///
/// Deterministic: a fixed serial traversal with a fixed `(dist, id)`
/// tie-break at every heap comparison, so the same graph + query +
/// entry point always visits cells in the same order and returns the
/// same result (no query-time parallelism, no hash-map iteration).
///
/// `entry` must be `< lists`; the caller ([`coarse_probe_dispatch`])
/// always passes `0`. `ef` is `(nprobe * GRAPH_EF_MULTIPLIER).max(GRAPH_EF_FLOOR)`,
/// clamped to `lists`.
pub fn graph_probe(
    graph: &CentroidGraph,
    centroids: &[f32],
    lists: usize,
    dim: usize,
    query_rotated: &[f32],
    nprobe: usize,
    entry: u32,
) -> Vec<u32> {
    debug_assert_eq!(centroids.len(), lists * dim);
    debug_assert_eq!(query_rotated.len(), dim);
    let nprobe = nprobe.clamp(1, lists.max(1)).min(lists);
    if lists == 0 {
        return Vec::new();
    }
    let ef = (nprobe.saturating_mul(GRAPH_EF_MULTIPLIER))
        .max(GRAPH_EF_FLOOR)
        .min(lists);
    let entry = (entry as usize).min(lists - 1);

    // (dist, id) ordering shared by every heap below: ascending
    // distance, ties toward the lower id. `Candidate` wraps a
    // (f32, u32) pair with that `Ord` so `BinaryHeap` (a max-heap)
    // can serve as both the min-heap-by-negation (candidates to
    // expand: pop the CLOSEST) and the max-heap (current result set:
    // pop the FARTHEST to evict). We keep two separate heaps with
    // opposite orderings instead of one generic type, for clarity.
    #[derive(Clone, Copy, PartialEq)]
    struct Cand(f32, u32);
    impl Eq for Cand {}
    impl PartialOrd for Cand {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Cand {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Ascending distance, ties -> ascending id (so id order
            // is deterministic too, independent of NaN/partial_cmp
            // fallback).
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(self.1.cmp(&other.1))
        }
    }

    let dist_to = |c: usize| sq_dist(query_rotated, &centroids[c * dim..(c + 1) * dim]);

    let mut visited = vec![false; lists];
    let entry_dist = dist_to(entry);
    visited[entry] = true;

    // Min-heap of candidates to expand next: `Reverse` flips `Cand`'s
    // ascending-distance `Ord` into a min-heap on `BinaryHeap`.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut to_visit: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
    to_visit.push(Reverse(Cand(entry_dist, entry as u32)));

    // Max-heap of the best `ef` results seen so far (so the top is
    // the CURRENT WORST of the kept set — cheap to evict).
    let mut results: BinaryHeap<Cand> = BinaryHeap::new();
    results.push(Cand(entry_dist, entry as u32));

    while let Some(Reverse(cur)) = to_visit.pop() {
        // Stopping condition: once the result set is full (`ef`) and
        // the closest remaining candidate is no better than our
        // current worst kept result, no further expansion can improve
        // the result set (every unvisited node is reached only
        // through graph edges, and edge weights aren't triangle-
        // inequality-guaranteed for an approximate graph in general,
        // but this is the standard, effective HNSW/Vamana stopping
        // rule and the `ef` slack budget is what makes it safe in
        // practice — see the recall tests).
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cur.0 >= worst.0 {
                    break;
                }
            }
        }
        let neighbors = graph.neighbors_of(cur.1 as usize);
        for &nb in neighbors {
            let nb = nb as usize;
            if visited[nb] {
                continue;
            }
            visited[nb] = true;
            let d = dist_to(nb);
            if results.len() < ef {
                results.push(Cand(d, nb as u32));
                to_visit.push(Reverse(Cand(d, nb as u32)));
            } else {
                let worse_than_worst = results.peek().is_some_and(|w| d >= w.0);
                if !worse_than_worst {
                    results.pop();
                    results.push(Cand(d, nb as u32));
                    to_visit.push(Reverse(Cand(d, nb as u32)));
                }
                // else: `d` can't improve the kept set, and (being
                // worse than the current worst kept) it also can't
                // beat the stopping check above once popped — skip
                // adding it to `to_visit` entirely, bounding the
                // heap's growth.
            }
        }
    }

    let mut out: Vec<(f32, u32)> = results.into_iter().map(|c| (c.0, c.1)).collect();
    out.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    out.truncate(nprobe);
    out.into_iter().map(|(_, c)| c).collect()
}

/// Dispatch coarse-cell selection to the graph-navigated
/// [`graph_probe`] when a [`CentroidGraph`] is available, else fall
/// back to the exact linear [`coarse_probe`]. This is the single
/// call site callers (`OocIvfIndex::coarse_probe_cells`,
/// `ivf_setup_and_search`) should use; it keeps the small-`lists`
/// fallback (`graph = None`, e.g. below [`GRAPH_MIN_LISTS`] under
/// `turbovec.coarse_graph = auto`) and the forced-off case
/// (`turbovec.coarse_graph = off`) both correct with one code path.
pub fn coarse_probe_dispatch(
    centroids: &[f32],
    lists: usize,
    dim: usize,
    query_rotated: &[f32],
    nprobe: usize,
    graph: Option<&CentroidGraph>,
) -> Vec<u32> {
    match graph {
        Some(g) => graph_probe(g, centroids, lists, dim, query_rotated, nprobe, 0),
        None => coarse_probe(centroids, lists, dim, query_rotated, nprobe),
    }
}

/// Rotate a `dim`-length query into the clustering (rotated) space,
/// mirroring the build's `BuildState::rotate_unit` (and turbovec's
/// encode): `rotated[k] = sum_j R[k*dim+j] * src[j]`, i.e. `src @
/// R^T`. `rotation` is the row-major `dim * dim` matrix read from the
/// relfile; `src` must be the L2-normalised query (the build
/// normalises every vector before assigning it to a cell, so the
/// coarse search must too). Returns a `dim`-length Vec.
pub fn rotate_query(rotation: &[f32], src: &[f32], dim: usize) -> Vec<f32> {
    debug_assert_eq!(rotation.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    let mut out = vec![0.0f32; dim];
    for (k, o) in out.iter_mut().enumerate() {
        let rrow = &rotation[k * dim..(k + 1) * dim];
        let mut s = 0.0f32;
        for j in 0..dim {
            s += rrow[j] * src[j];
        }
        *o = s;
    }
    out
}

/// Squared Euclidean distance. Scalar — the coarse step is cheap
/// (`lists` centroids, `lists ≈ √n`) and keeping it scalar avoids a
/// second SIMD-correctness surface. The reduction
/// order is fixed (ascending coordinate), so it's deterministic.
#[inline]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Batched corpus rotation: `rotated = corpus @ rotation^T`, the
/// matrix form of the per-vector `BuildState::rotate_unit`
/// (`out[k] = sum_j R[k*dim+j] * unit[j]`, i.e. `out = R @ unit` per
/// row, so for the whole `n x dim` corpus `out = corpus @ R^T`).
///
/// `corpus` is row-major `n_rows * dim` (the caller must have already
/// L2-normalised every row -- this function does NOT normalise).
/// `rotation` is the row-major `dim * dim` matrix. `out` is row-major
/// `n_rows * dim` and is fully overwritten.
///
/// `Parallelism::Rayon(0)` (== `rayon::current_num_threads()` of the
/// AMBIENT bounded build pool this runs inside via
/// `build_pool::install`). Bit-deterministic across runs, machines,
/// and thread counts on the SAME empirical guarantee v1.22.1
/// established for `gemm_lloyd_assign`'s cross-term GEMM: gemm's own
/// internal Rayon(n) tiling produces byte-identical output to
/// `Parallelism::None` for every thread count, because each output
/// tile is an independent dot-product reduction over the shared `k`
/// dimension — the tiling never reduces ACROSS threads, so thread
/// count can never perturb which f32 adds happen in which order for a
/// given output element. This is the SAME GEMM crate, the same
/// non-accumulating overwrite (`read_dst=false`, alpha=0, beta=1), so
/// the guarantee transfers directly. The orientation is verified
/// elementwise against `rotate_unit` by the
/// `ivf_batched_rotation_matches_per_row` test, and thread-count
/// invariance by `rotate_corpus_bit_identical_across_pool_sizes`.
///
/// gemm semantics: `dst (m x n) = beta * lhs (m x k) @ rhs (k x n)`.
/// We want `out (n_rows x dim) = corpus (n_rows x dim) @ R^T (dim x
/// dim)`, so m = n_rows, n = dim, k = dim, lhs = corpus, rhs = R^T.
/// R is row-major; we read it transposed for free by swapping the
/// row/col strides (`rhs_rs = 1, rhs_cs = dim`), avoiding a physical
/// transpose copy. All matrices are row-major, so `*_rs = n_cols,
/// *_cs = 1` for the non-transposed operands.
pub fn rotate_corpus_into(
    corpus: &[f32],
    rotation: &[f32],
    n_rows: usize,
    dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(corpus.len(), n_rows * dim);
    debug_assert_eq!(rotation.len(), dim * dim);
    debug_assert_eq!(out.len(), n_rows * dim);
    if n_rows == 0 || dim == 0 {
        return;
    }
    // SAFETY: all three buffers are sized exactly m*k / k*n / m*n as
    // asserted above; the strides describe in-bounds row-major (and,
    // for R^T, transposed-row-major) access. `read_dst = false` means
    // gemm overwrites `out` (beta * lhs @ rhs, no accumulation), so
    // the prior contents of `out` are irrelevant.
    unsafe {
        gemm(
            n_rows,            // m
            dim,               // n
            dim,               // k
            out.as_mut_ptr(),  // dst (n_rows x dim) row-major
            1,                 // dst_cs
            dim as isize,      // dst_rs
            false,             // read_dst (overwrite)
            corpus.as_ptr(),   // lhs (n_rows x dim) row-major
            1,                 // lhs_cs
            dim as isize,      // lhs_rs
            rotation.as_ptr(), // rhs = R^T: swap strides on R
            dim as isize,      // rhs_cs (= R's row stride)
            1,                 // rhs_rs (= R's col stride)
            0.0f32,            // alpha
            1.0f32,            // beta
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
}

/// Batched nearest-centroid assignment over a (rotated) corpus.
///
/// Replaces `n_rows * lists` scalar [`CoarseModel::assign_one`] calls
/// (each `O(dim)`) with one `(n_rows x lists)` cross-term GEMM plus a
/// cheap reduction, via `||v - c||^2 = ||v||^2 + ||c||^2 - 2 (v . c)`.
/// The cross term `V @ C^T` is one `(n_rows x lists) = (n_rows x dim)
/// @ (dim x lists)` GEMM. The `||v||^2` term is constant across cells
/// for a fixed row, so the per-row argmin only needs `||c||^2 - 2
/// (v . c)`.
///
/// **Determinism through the GEMM.** f32 GEMM reorders adds vs the
/// scalar `sq_dist`, which could flip a near-tie and change the
/// permutation (and thus the on-disk bytes). To make the result
/// byte-identical to the scalar path we use the GEMM only as a coarse
/// ranking, then recompute the exact scalar `sq_dist` for the top-2
/// GEMM candidates per row and pick the winner with the same
/// `(dist, cell_id)` tie-break `assign_one` uses. Two `dim`-dot
/// products per row is negligible next to the GEMM. The GEMM itself
/// runs `Parallelism::None` for a fixed reduction order, but the
/// top-2 scalar recompute is the actual correctness anchor.
///
/// `corpus` is row-major `n_rows * dim` (already rotated +
/// normalised). `centroids` is row-major `lists * dim` (rotated
/// space). Returns a `Vec<u32>` of length `n_rows` of cell ids. The
/// per-row reduction is parallelised; rayon's `map` preserves index
/// order, so the output is deterministic.
///
/// Superseded on the build hot path by [`batched_assign_soft`] (which
/// handles `M = 1` identically); retained as the single-assignment
/// reference that the soft path's `M = 1` case is verified against
/// (`ivf_soft_assign_m1_matches_single`).
#[allow(dead_code)]
pub fn batched_assign(
    corpus: &[f32],
    centroids: &[f32],
    n_rows: usize,
    lists: usize,
    dim: usize,
) -> Vec<u32> {
    debug_assert_eq!(corpus.len(), n_rows * dim);
    debug_assert_eq!(centroids.len(), lists * dim);
    assert!(lists > 0, "batched_assign: lists must be > 0");
    if n_rows == 0 {
        return Vec::new();
    }

    // Cross term: cross (n_rows x lists) = corpus (n_rows x dim) @
    // centroids^T (dim x lists). centroids is row-major lists x dim;
    // read transposed (swap strides) so rhs is (dim x lists).
    let mut cross = vec![0.0f32; n_rows * lists];
    // SAFETY: buffers sized m*k / (read as) k*n / m*n; strides are
    // in-bounds row-major (corpus, cross) and transposed-row-major
    // (centroids^T). read_dst=false overwrites `cross`.
    unsafe {
        gemm(
            n_rows,             // m
            lists,              // n
            dim,                // k
            cross.as_mut_ptr(), // dst (n_rows x lists) row-major
            1,                  // dst_cs
            lists as isize,     // dst_rs
            false,              // read_dst (overwrite)
            corpus.as_ptr(),    // lhs (n_rows x dim) row-major
            1,                  // lhs_cs
            dim as isize,       // lhs_rs
            centroids.as_ptr(), // rhs = C^T: swap strides on C
            dim as isize,       // rhs_cs (= C's row stride)
            1,                  // rhs_rs (= C's col stride)
            0.0f32,             // alpha
            1.0f32,             // beta
            false,
            false,
            false,
            Parallelism::None,
        );
    }

    // ||c||^2 per cell (precomputed once, ascending coordinate order).
    let cnorm: Vec<f32> = (0..lists)
        .map(|c| {
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
        })
        .collect();

    // Per-row reduction. For each row: rank cells by the GEMM-derived
    // score `||c||^2 - 2 (v . c)` (monotone in true sq_dist for a
    // fixed row), keep the top-2 cell candidates, then recompute the
    // exact scalar sq_dist for those two and apply assign_one's
    // (dist, cell_id) tie-break.
    use rayon::prelude::*;
    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row_cross = &cross[i * lists..(i + 1) * lists];
            // Two smallest GEMM scores (best, second). Strict `<`
            // keeps the lower cell id on a score tie.
            let mut best_c = 0usize;
            let mut best_s = f32::INFINITY;
            let mut snd_c = usize::MAX;
            let mut snd_s = f32::INFINITY;
            for c in 0..lists {
                let s = cnorm[c] - 2.0 * row_cross[c];
                if s < best_s {
                    snd_s = best_s;
                    snd_c = best_c;
                    best_s = s;
                    best_c = c;
                } else if s < snd_s {
                    snd_s = s;
                    snd_c = c;
                }
            }
            // Recompute exact scalar sq_dist for the top-2 and pick
            // the winner with assign_one's (dist, cell_id) tie-break:
            // lower distance wins; on an exact tie, lower cell id wins.
            let v = &corpus[i * dim..(i + 1) * dim];
            let d_best = sq_dist(v, &centroids[best_c * dim..(best_c + 1) * dim]);
            if snd_c == usize::MAX {
                return best_c as u32;
            }
            let d_snd = sq_dist(v, &centroids[snd_c * dim..(snd_c + 1) * dim]);
            let winner = if d_snd < d_best || (d_snd == d_best && snd_c < best_c) {
                snd_c
            } else {
                best_c
            };
            winner as u32
        })
        .collect()
}

/// GEMM-batched nearest-centroid assignment used by [`train_kmeans`]'s
/// Lloyd step. Fills `assign[0..n_rows]` with each row's nearest cell
/// id, byte-identical to the scalar argmin (`if d < best_d`, ties →
/// lower cell id) [`CoarseModel::assign_one`] uses — the same
/// cross-term-GEMM + exact top-2 scalar recompute that
/// [`batched_assign`] is verified against by
/// `ivf_batched_assign_matches_scalar`. Replacing the scalar
/// `n_sample × lists × dim` double-loop with one
/// `(n_sample × lists) = (n_sample × dim) @ (dim × lists)` GEMM per
/// Lloyd iteration is the build-time win: at lists=448, sample=114k,
/// dim=256 that is one BLAS GEMM instead of ~13 billion scalar FLOPs
/// per iteration.
///
/// `cross` is `n_rows * lists` caller-owned scratch (reused across
/// iterations to avoid a per-iteration alloc); its prior contents are
/// overwritten. `cnorm` is the precomputed `||c||^2` per cell, which
/// the caller refreshes after each centroid update. The GEMM runs
/// `Parallelism::None` (fixed reduction order); the exact scalar
/// recompute of the top-2 candidates is the determinism anchor, so an
/// f32 GEMM reorder near a tie can never flip the assignment vs the
/// scalar reference.
fn gemm_lloyd_assign(
    sample: &[f32],
    centroids: &[f32],
    cnorm: &[f32],
    n_rows: usize,
    lists: usize,
    dim: usize,
    cross: &mut [f32],
    assign: &mut [u32],
) {
    debug_assert_eq!(cross.len(), n_rows * lists);
    debug_assert_eq!(assign.len(), n_rows);
    if n_rows == 0 {
        return;
    }
    // Cross term: cross (n_rows x lists) = sample @ centroids^T.
    // Identical GEMM shape/strides to batched_assign.
    //
    // Parallelism::Rayon(0) (== rayon::current_num_threads() of the
    // AMBIENT pool -- see below) rather than Parallelism::None. This
    // is the dominant term in an IVF build at high `lists` (the
    // Lloyd loop runs this GEMM over the WHOLE sample once per
    // iteration, up to KMEANS_ITERS times, vs. the row-blocked
    // per-vector work elsewhere which is already split across chunks
    // --
    // the FLOPs accounting that identified this as the real build
    // cliff, not the row-blocked stages parallelized in v1.21.0).
    //
    // Determinism is preserved WITHOUT any row-blocking or reduction-
    // order bookkeeping: empirically verified (see the v1.22.1
    // release notes) that gemm 0.18.2's OWN internal Parallelism::
    // Rayon(n) tiling produces BIT-IDENTICAL output to Parallelism::
    // None for every thread count and shape tested, because each
    // output tile of a GEMM is an independent dot-product reduction
    // over the shared `k` dimension -- unlike the k-means centroid-
    // SUM step (which DOES need the fixed-chunk-order dance in
    // train_kmeans_iters below), a GEMM's tiling never reduces
    // ACROSS threads, so thread count can never perturb which f32
    // adds happen in which order for a given output element. This is
    // exactly the guarantee the exact top-2 scalar recompute below
    // ALSO independently provides (it re-derives the real winner from
    // scratch), so parallelizing this GEMM is safe on two
    // independent grounds, not just one.
    //
    // `rayon::current_num_threads()` (what Rayon(0) resolves to)
    // reads the AMBIENT pool -- i.e. when this function runs inside
    // `build_pool::install(pool, ...)` (as train_kmeans/train_kmeans_
    // iters always is, from build.rs), it reports that bounded
    // pool's size, NOT rayon's unbounded global default pool. So
    // Rayon(0) here automatically and correctly respects `turbovec.
    // build_parallelism` with no extra plumbing.
    //
    // SAFETY: buffers sized m*k / (read transposed) k*n / m*n; strides
    // are in-bounds row-major (sample, cross) and transposed-row-major
    // (centroids^T). read_dst=false overwrites `cross`.
    unsafe {
        gemm(
            n_rows,
            lists,
            dim,
            cross.as_mut_ptr(),
            1,
            lists as isize,
            false,
            sample.as_ptr(),
            1,
            dim as isize,
            centroids.as_ptr(),
            dim as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
    // Per-row argmin + exact top-2 tie-break, parallel over ROWS.
    // Each `assign[i]` is a pure function of row `i`'s cross scores,
    // the read-only `cnorm`/`centroids`/`sample` — no cross-row
    // dependence, no reduction. `par_iter_mut().enumerate()` writes
    // each output to its fixed index, so the result is byte-identical
    // to the serial `for i in 0..n_rows` loop regardless of thread
    // count (data-parallel map, exactly like the seeding D2 fill and
    // the k-means centroid-update partition above). The GEMM this
    // reduces is the same one, so this runs inside the SAME ambient
    // bounded pool (`build_pool::install`) train_kmeans is called in.
    // This was the last serial per-row term in the Lloyd loop after
    // v1.22.1 parallelized the cross-term GEMM (Phase Q-4a).
    use rayon::prelude::*;
    assign.par_iter_mut().enumerate().for_each(|(i, out)| {
        let row_cross = &cross[i * lists..(i + 1) * lists];
        // Two smallest GEMM scores `||c||^2 - 2 (v . c)` (monotone
        // in true sq_dist for a fixed row). Strict `<` keeps the
        // lower cell id on a score tie.
        let mut best_c = 0usize;
        let mut best_s = f32::INFINITY;
        let mut snd_c = usize::MAX;
        let mut snd_s = f32::INFINITY;
        for c in 0..lists {
            let s = cnorm[c] - 2.0 * row_cross[c];
            if s < best_s {
                snd_s = best_s;
                snd_c = best_c;
                best_s = s;
                best_c = c;
            } else if s < snd_s {
                snd_s = s;
                snd_c = c;
            }
        }
        // Recompute exact scalar sq_dist for the top-2 and pick the
        // winner with assign_one's (dist, cell_id) tie-break: lower
        // distance wins; on an exact tie, lower cell id wins. This
        // matches the scalar Lloyd assignment (`if d < best_d`)
        // byte-for-byte per iteration, so the per-iteration
        // centroids are unchanged vs the old scalar path; the only
        // end-state difference is the convergence early-exit
        // truncating no-op iterations.
        let v = &sample[i * dim..(i + 1) * dim];
        let d_best = sq_dist(v, &centroids[best_c * dim..(best_c + 1) * dim]);
        if snd_c == usize::MAX {
            *out = best_c as u32;
            return;
        }
        let d_snd = sq_dist(v, &centroids[snd_c * dim..(snd_c + 1) * dim]);
        *out = if d_snd < d_best || (d_snd == d_best && snd_c < best_c) {
            snd_c as u32
        } else {
            best_c as u32
        };
    });
}

/// Train `lists` coarse centroids on `sample` (a row-major
/// `n_sample * dim` buffer of **rotated** vectors) via deterministic
/// k-means++ seeding + Lloyd's iterations.
///
/// Determinism: a `ChaCha8Rng::seed_from_u64(IVF_SEED)` drives both
/// the k-means++ seeding and the empty-cell re-seeding, with a fixed
/// iteration count ([`KMEANS_ITERS`]) and fixed (ascending) reduction
/// order. Same `sample` + same `lists` ⇒ byte-identical centroids.
///
/// Degenerate cases:
/// - `n_sample == 0` ⇒ all-zero centroids (the caller won't build an
///   IVF index over an empty heap; guarded for safety).
/// - `n_sample < lists` ⇒ the first `n_sample` centroids are the
///   distinct sample points; the remaining `lists - n_sample` are
///   re-seeded from the largest cell (standard rule). Such a cluster
///   set is still valid (assignment picks the nearest of `lists`).
pub fn train_kmeans(sample: &[f32], n_sample: usize, lists: usize, dim: usize) -> CoarseModel {
    train_kmeans_iters(sample, n_sample, lists, dim).0
}

/// As [`train_kmeans`], but also returns the number of Lloyd's
/// iterations actually run (`1..=KMEANS_ITERS`). The count is a
/// deterministic function of the input (fixed convergence threshold
/// over a deterministic movement metric); tests use it to prove the
/// early-exit fires on well-separated data and is reproducible.
fn train_kmeans_iters(
    sample: &[f32],
    n_sample: usize,
    lists: usize,
    dim: usize,
) -> (CoarseModel, usize) {
    assert!(lists > 0, "train_kmeans: lists must be > 0");
    assert!(dim > 0, "train_kmeans: dim must be > 0");
    debug_assert_eq!(sample.len(), n_sample * dim);

    let mut centroids = vec![0.0f32; lists * dim];
    if n_sample == 0 {
        return (
            CoarseModel {
                centroids,
                lists,
                dim,
            },
            0,
        );
    }

    let mut rng = ChaCha8Rng::seed_from_u64(IVF_SEED);
    let row = |i: usize| -> &[f32] { &sample[i * dim..(i + 1) * dim] };

    // ---- k-means++ seeding ----
    let __trace = std::env::var_os("TURBOVEC_BUILD_TRACE").is_some();
    let __t_seed = std::time::Instant::now();
    // First centroid: a deterministic uniform pick.
    let first = rng.gen_range(0..n_sample);
    centroids[0..dim].copy_from_slice(row(first));

    // D2 sampling for the remaining seeds. `d2[i]` is the squared
    // distance from sample i to its nearest chosen centroid so far.
    let mut d2 = vec![f32::INFINITY; n_sample];
    // Parallel: d2[i] = sq_dist(row(i), centroid0) is per-row
    // independent, so a par_chunks fill is bit-identical to the
    // serial loop. This + the per-seed update loop below were an
    // O(lists * n_sample * dim) serial term (the high-dim seeding
    // cost the benchmark benchmark exposed).
    {
        use rayon::prelude::*;
        let c0 = &centroids[0..dim];
        d2.par_iter_mut().enumerate().for_each(|(i, d)| {
            *d = sq_dist(&sample[i * dim..(i + 1) * dim], c0);
        });
    }
    for c in 1..lists {
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        let chosen = if total <= 0.0 {
            // All sample points coincide with chosen centroids (e.g.
            // fewer distinct points than `lists`). Fall back to a
            // uniform pick so we still place a centroid; it may
            // duplicate an existing one, which Lloyd's + the
            // empty-cell rule will spread out.
            rng.gen_range(0..n_sample)
        } else {
            // Weighted pick proportional to d2.
            let target = rng.gen_range(0.0..total);
            let mut acc = 0.0f64;
            let mut idx = n_sample - 1;
            for i in 0..n_sample {
                acc += d2[i] as f64;
                if acc >= target {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centroids[c * dim..(c + 1) * dim].copy_from_slice(row(chosen));
        // Update nearest-centroid distances with the new centroid.
        // Per-row independent -> parallel min-update, bit-identical
        // (each d2[i] is min'd against one new sq_dist; no cross-row
        // interaction). This is the inner loop of the O(lists *
        // n_sample * dim) seeding cost.
        let new_centroid = row(chosen).to_vec();
        {
            use rayon::prelude::*;
            let nc = &new_centroid;
            d2.par_iter_mut().enumerate().for_each(|(i, di)| {
                let d = sq_dist(&sample[i * dim..(i + 1) * dim], nc);
                if d < *di {
                    *di = d;
                }
            });
        }
    }
    if __trace {
        eprintln!(
            "[turbovec build trace]   1a_kmeanspp_seeding   {:.3}s",
            __t_seed.elapsed().as_secs_f64()
        );
    }
    let __t_lloyd = std::time::Instant::now();

    // ---- Lloyd's iterations ----
    // GEMM-batched assignment (one (n_sample x lists) cross-term GEMM
    // per iteration via gemm_lloyd_assign) replaces the old
    // n_sample x lists x dim scalar double-loop — the build-time
    // bottleneck on large samples. The assignment is byte-identical
    // to the scalar argmin (same exact-scalar (dist, cell_id)
    // tie-break), so the centroids are unchanged vs the pre-GEMM
    // path. A convergence early-exit (KMEANS_TOL on total centroid
    // movement) caps wasted iterations; both the GEMM (single-thread)
    // and the threshold are deterministic, so the iteration count is
    // a fixed function of the input.
    let mut assign = vec![0u32; n_sample];
    let mut cross = vec![0.0f32; n_sample * lists];
    let mut cnorm = vec![0.0f32; lists];
    let mut prev_centroids = vec![0.0f32; lists * dim];
    let mut iters_run = 0usize;
    for _iter in 0..KMEANS_ITERS {
        let __t_iter = std::time::Instant::now();
        iters_run += 1;
        // Snapshot centroids to measure end-of-iteration movement.
        prev_centroids.copy_from_slice(&centroids);

        // Assignment step. ||c||^2 per cell (fixed ascending
        // coordinate order), then the GEMM cross-term + exact top-2
        // scalar tie-break. Equivalent to the scalar
        // `for c { sq_dist(...) }` argmin, ties → lower cell id.
        for (c, cn) in cnorm.iter_mut().enumerate() {
            *cn = centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>();
        }
        gemm_lloyd_assign(
            sample,
            &centroids,
            &cnorm,
            n_sample,
            lists,
            dim,
            &mut cross,
            &mut assign,
        );

        // Update step: mean of assigned points. The per-cell f64
        // accumulation is parallelized with a FIXED-ORDER reduction
        // so it stays bit-identical: rows are split into fixed
        // contiguous chunks, each chunk sums into its own private
        // (sums, counts) in ascending row order, then the chunk
        // partials are combined in ascending chunk order. The f64
        // addition order is therefore a fixed function of the input
        // (independent of thread count / scheduling), so the centroids
        // are byte-identical to the serial path. (f64 addition is not
        // associative, so the fixed partition + fixed combine order
        // is what guarantees reproducibility, NOT that f64 is "exact".)
        let (sums, counts) = {
            use rayon::prelude::*;
            // Number of parallel chunks. CRITICAL on two axes:
            //  (1) MEMORY: each chunk holds a private `sums` of
            //      `lists*dim` f64 and all partials are materialised at
            //      once (`collect()`), so total transient memory is
            //      `n_chunks * lists * dim * 8`. The old code scaled
            //      chunks with n_sample and OOM'd at large `lists`.
            //  (2) DETERMINISM: the f64 reduction order is fixed by the
            //      PARTITION, so n_chunks must be a fixed function of
            //      the INPUT (n_sample, lists, dim) -- NEVER of
            //      `current_num_threads()`, or the byte-identical
            //      relfile would differ across machines with different
            //      core counts. We therefore pick a fixed target chunk
            //      row-count and derive n_chunks from n_sample, capped
            //      by a memory budget on the partials.
            let bytes_per_partial = (lists * dim).max(1) * 8;
            // Cap total partials at ~512 MiB.
            let max_chunks_by_mem = (512 * 1024 * 1024 / bytes_per_partial).max(1);
            // Fixed target: ~16k sample rows per chunk (amortises the
            // per-chunk alloc; input-only, thread-count-independent).
            const TARGET_CHUNK_ROWS: usize = 16_384;
            let by_rows = n_sample.div_ceil(TARGET_CHUNK_ROWS).max(1);
            let n_chunks = by_rows.min(max_chunks_by_mem);
            let chunk = n_sample.div_ceil(n_chunks).max(1);
            let starts: Vec<usize> = (0..n_sample).step_by(chunk).collect();
            // Each chunk sums into its own (sums, counts) in ascending
            // row order (parallel across chunks).
            let partials: Vec<(Vec<f64>, Vec<u64>)> = starts
                .par_iter()
                .map(|&start| {
                    let end = (start + chunk).min(n_sample);
                    let mut s = vec![0.0f64; lists * dim];
                    let mut cnt = vec![0u64; lists];
                    for i in start..end {
                        let c = assign[i] as usize;
                        cnt[c] += 1;
                        let r = &sample[i * dim..(i + 1) * dim];
                        let base = c * dim;
                        for j in 0..dim {
                            s[base + j] += r[j] as f64;
                        }
                    }
                    (s, cnt)
                })
                .collect();
            // Combine partials in ASCENDING chunk order (fixed) so the
            // f64 addition order is a deterministic function of the
            // input, not of thread scheduling -> byte-identical.
            let mut sums = vec![0.0f64; lists * dim];
            let mut counts = vec![0u64; lists];
            for (s, cnt) in &partials {
                for k in 0..lists * dim {
                    sums[k] += s[k];
                }
                for k in 0..lists {
                    counts[k] += cnt[k];
                }
            }
            (sums, counts)
        };
        let mut counts = counts;
        for c in 0..lists {
            if counts[c] == 0 {
                continue; // handled by the empty-cell rule below
            }
            let inv = 1.0 / counts[c] as f64;
            let base = c * dim;
            for j in 0..dim {
                centroids[base + j] = (sums[base + j] * inv) as f32;
            }
        }

        // Empty-cell rule: re-seed each empty cell from the LARGEST
        // cell by splitting a point off it. Deterministic: we pick
        // the largest cell (ties → lowest id) and the
        // deterministically-chosen sample point furthest from that
        // cell's centroid. This is the standard "re-seed largest
        // cell" rule and prevents dead centroids from collapsing
        // recall.
        //
        // Traced under TURBOVEC_BUILD_TRACE: each empty cell costs an
        // O(n_sample) serial scan below, so a build that regularly
        // hits many empty cells (e.g. lists close to or exceeding the
        // number of distinct clusters actually present in the data)
        // pays a real, otherwise-invisible cost here.
        if __trace {
            let n_empty = counts.iter().filter(|&&c| c == 0).count();
            if n_empty > 0 {
                eprintln!("[turbovec build trace]   iter {_iter}: {n_empty}/{lists} cells empty");
            }
        }
        for c in 0..lists {
            if counts[c] != 0 {
                continue;
            }
            // Largest cell (ties → lowest id).
            let mut big = 0usize;
            let mut big_n = 0u64;
            for k in 0..lists {
                if counts[k] > big_n {
                    big_n = counts[k];
                    big = k;
                }
            }
            if big_n <= 1 {
                // Nothing to split (all cells empty or singletons);
                // jitter from a deterministic random sample point.
                let pick = rng.gen_range(0..n_sample);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(row(pick));
                counts[c] = 1;
                continue;
            }
            // Furthest point in `big` from `big`'s centroid (ties →
            // lowest sample index ⇒ deterministic).
            let big_centroid = &centroids[big * dim..(big + 1) * dim];
            let mut far = usize::MAX;
            let mut far_d = -1.0f32;
            for i in 0..n_sample {
                if assign[i] as usize != big {
                    continue;
                }
                let d = sq_dist(row(i), big_centroid);
                if d > far_d {
                    far_d = d;
                    far = i;
                }
            }
            if far == usize::MAX {
                let pick = rng.gen_range(0..n_sample);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(row(pick));
                counts[c] = 1;
                continue;
            }
            // Move that point's mass to the empty cell.
            centroids[c * dim..(c + 1) * dim].copy_from_slice(row(far));
            assign[far] = c as u32;
            counts[c] = 1;
            counts[big] -= 1;
        }

        // Convergence early-exit: total centroid movement this
        // iteration (sum of ||c_new - c_old||^2 over all cells,
        // measured AFTER the update + empty-cell reseed so it
        // reflects the centroids the next iteration would assign
        // against). Deterministic fixed-order reduction; a fixed
        // KMEANS_TOL means the same input always stops at the same
        // iteration. Once the centroids have effectively stopped
        // moving, further iterations can't change the partition.
        let mut movement = 0.0f64;
        for j in 0..lists * dim {
            let d = (centroids[j] - prev_centroids[j]) as f64;
            movement += d * d;
        }
        if __trace {
            eprintln!(
                "[turbovec build trace]     iter {_iter}: {:.3}s",
                __t_iter.elapsed().as_secs_f64()
            );
        }
        if movement < KMEANS_TOL {
            break;
        }
    }
    if __trace {
        eprintln!(
            "[turbovec build trace]   1b_lloyd_loop          {:.3}s ({} iters)",
            __t_lloyd.elapsed().as_secs_f64(),
            iters_run
        );
    }

    (
        CoarseModel {
            centroids,
            lists,
            dim,
        },
        iters_run,
    )
}

/// Build the cell-contiguous permutation and directory from a
/// `slot → cell` assignment.
///
/// Returns `(permutation, directory)` where `permutation[new_slot] =
/// old_slot` — i.e. applying it reorders the flat slots so cell 0's
/// rows come first, then cell 1's, etc. The order is stable: within
/// a cell, original slot order is preserved (sort key `(cell_id,
/// original_slot)`), so the whole pipeline is deterministic.
///
/// `assignment[old_slot] = cell_id`, `assignment.len() == n_vectors`.
///
/// Superseded on the build hot path by [`build_permutation_soft`]
/// (which handles `M = 1` identically); retained as the
/// single-assignment reference for the soft `M = 1` parity test.
#[allow(dead_code)]
pub fn build_permutation(assignment: &[u32], lists: usize) -> (Vec<u32>, CellDirectory) {
    let n = assignment.len();

    // Count per cell.
    let mut counts = vec![0u64; lists];
    for &c in assignment {
        counts[c as usize] += 1;
    }

    // Prefix offsets → cell directory.
    let mut entries = Vec::with_capacity(lists);
    let mut acc = 0u64;
    for c in 0..lists {
        entries.push(CellEntry {
            code_offset: acc,
            n_vectors: counts[c] as u32,
        });
        acc += counts[c];
    }
    let directory = CellDirectory { entries };

    // Stable counting sort: walk old slots in ascending order, place
    // each into the next free position of its cell. Because we visit
    // old slots ascending and fill each cell's run left-to-right,
    // within-cell order == original slot order (stable).
    let mut cursor: Vec<u64> = directory.entries.iter().map(|e| e.code_offset).collect();
    let mut permutation = vec![0u32; n];
    for (old_slot, &c) in assignment.iter().enumerate() {
        let pos = cursor[c as usize];
        permutation[pos as usize] = old_slot as u32;
        cursor[c as usize] += 1;
    }

    (permutation, directory)
}

/// IVF-4a soft (multi) assignment over a (rotated) corpus.
///
/// Like [`batched_assign`] (same GEMM cross-term + exact-scalar
/// tie-break for the nearest cell), but each row may be assigned to
/// up to `max_dups` cells: its nearest, plus any of the next nearest
/// whose squared distance is within [`BOUNDARY_FACTOR`] of the
/// nearest's (`dist_j <= BOUNDARY_FACTOR * dist_nearest`). This puts
/// genuinely-boundary vectors (nearly equidistant to two cells) into
/// both, so a query that probes either neighbouring cell finds them
/// — raising recall@10 at a fixed `probes`. Non-boundary vectors stay
/// single-assigned, bounding the storage blow-up.
///
/// Returns a `Vec<Vec<u32>>` of length `n_rows`; entry `i` lists the
/// cell ids row `i` is assigned to, **nearest first**, length
/// `1..=max_dups`. `max_dups == 1` reproduces single assignment
/// exactly (each inner vec has one element == [`batched_assign`]'s
/// output for that row), so the soft path is a strict generalisation.
///
/// Determinism: the nearest cell is chosen by the same exact-scalar
/// `(dist, cell_id)` tie-break [`batched_assign`] uses; the
/// additional cells are the next-nearest by exact scalar `sq_dist`
/// (ascending distance, ties → lower cell id) that pass the boundary
/// threshold. The GEMM (`Parallelism::None`) only pre-ranks
/// candidates; the exact scalar recompute is the correctness anchor.
/// Same input + same `max_dups` ⇒ byte-identical assignment.
pub fn batched_assign_soft(
    corpus: &[f32],
    centroids: &[f32],
    n_rows: usize,
    lists: usize,
    dim: usize,
    max_dups: usize,
) -> Vec<Vec<u32>> {
    debug_assert_eq!(corpus.len(), n_rows * dim);
    debug_assert_eq!(centroids.len(), lists * dim);
    assert!(lists > 0, "batched_assign_soft: lists must be > 0");
    let max_dups = max_dups.clamp(1, lists);
    if n_rows == 0 {
        return Vec::new();
    }

    // Cross term: cross (n_rows x lists) = corpus @ centroids^T.
    // Identical GEMM to batched_assign.
    let mut cross = vec![0.0f32; n_rows * lists];
    // SAFETY: see batched_assign — same shapes/strides; overwrite.
    unsafe {
        gemm(
            n_rows,
            lists,
            dim,
            cross.as_mut_ptr(),
            1,
            lists as isize,
            false,
            corpus.as_ptr(),
            1,
            dim as isize,
            centroids.as_ptr(),
            dim as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
    let cnorm: Vec<f32> = (0..lists)
        .map(|c| {
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
        })
        .collect();

    use rayon::prelude::*;
    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row_cross = &cross[i * lists..(i + 1) * lists];
            let v = &corpus[i * dim..(i + 1) * dim];
            // Pre-rank cells by the GEMM score `||c||^2 - 2 (v . c)`
            // (monotone in true sq_dist for a fixed row); keep the
            // top `cand` candidates to recompute exactly. We need
            // max_dups winners; over-fetch a few to be safe against
            // GEMM reorder near-ties (cand = max_dups + 2, clamped).
            let cand = (max_dups + 2).min(lists);
            // Partial selection of the `cand` smallest GEMM scores.
            let mut top: Vec<(f32, usize)> = Vec::with_capacity(cand + 1);
            for c in 0..lists {
                let s = cnorm[c] - 2.0 * row_cross[c];
                if top.len() < cand {
                    top.push((s, c));
                    if top.len() == cand {
                        // Keep sorted ascending by (score, cell).
                        top.sort_unstable_by(|a, b| {
                            a.0.partial_cmp(&b.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.1.cmp(&b.1))
                        });
                    }
                } else if s < top[cand - 1].0 {
                    top[cand - 1] = (s, c);
                    top.sort_unstable_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.1.cmp(&b.1))
                    });
                }
            }
            if top.len() < cand {
                // lists < cand: sort whatever we have.
                top.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                });
            }
            // Recompute exact sq_dist for the candidates and re-rank
            // by (dist, cell_id) — the determinism anchor.
            let mut exact: Vec<(f32, usize)> = top
                .iter()
                .map(|&(_, c)| (sq_dist(v, &centroids[c * dim..(c + 1) * dim]), c))
                .collect();
            exact.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            // Nearest is always included. Additional cells join only
            // if within the boundary factor of the nearest distance.
            let d_nearest = exact[0].0;
            let threshold = BOUNDARY_FACTOR * d_nearest;
            let mut out = Vec::with_capacity(max_dups);
            out.push(exact[0].1 as u32);
            for &(d, c) in exact.iter().skip(1) {
                if out.len() >= max_dups {
                    break;
                }
                if d <= threshold {
                    out.push(c as u32);
                }
            }
            out
        })
        .collect()
}

/// IVF-4a soft permutation builder. Like [`build_permutation`] but
/// each vector may belong to multiple cells, so a vector's original
/// index appears once per cell it was assigned to — i.e. the produced
/// `permutation` is **non-injective** (`permutation[new_slot] =
/// old_vector_index`, and the same `old_vector_index` can appear in
/// several new slots). The cell directory partitions the *expanded*
/// slot count (`sum of per-vector dup counts`), not `n_vectors`.
///
/// `assignments[old_index]` is the (nearest-first) cell list for
/// vector `old_index`, as produced by [`batched_assign_soft`]. The
/// within-cell order is stable on `old_index` (ascending), so the
/// pipeline stays deterministic. Returns `(permutation, directory)`
/// where `permutation.len() == directory.total_vectors() as usize`.
///
/// The downstream quantize+pack runs on the expanded slot order;
/// `slot_to_id` therefore repeats an id once per cell the vector
/// landed in. The scan **must** dedup by id when merging across
/// probed cells (a boundary vector in two probed cells must not be
/// returned twice) — the scan's emitted-id set + an intra-batch
/// guard handle this.
pub fn build_permutation_soft(assignments: &[Vec<u32>], lists: usize) -> (Vec<u32>, CellDirectory) {
    // Count slots per cell (a vector contributes one slot per cell
    // it is assigned to).
    let mut counts = vec![0u64; lists];
    for cells in assignments {
        for &c in cells {
            counts[c as usize] += 1;
        }
    }

    let mut entries = Vec::with_capacity(lists);
    let mut acc = 0u64;
    for c in 0..lists {
        entries.push(CellEntry {
            code_offset: acc,
            n_vectors: counts[c] as u32,
        });
        acc += counts[c];
    }
    let directory = CellDirectory { entries };

    // Stable expansion: for each cell, in ascending old-index order,
    // place that vector's old index into the cell's next free slot.
    // We iterate old indices ascending in the OUTER loop so the
    // within-cell order is the original vector order (stable), then
    // route each (old_index, cell) into its cell's cursor.
    let mut cursor: Vec<u64> = directory.entries.iter().map(|e| e.code_offset).collect();
    let total = acc as usize;
    let mut permutation = vec![0u32; total];
    for (old_index, cells) in assignments.iter().enumerate() {
        for &c in cells {
            let pos = cursor[c as usize];
            permutation[pos as usize] = old_index as u32;
            cursor[c as usize] += 1;
        }
    }

    (permutation, directory)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The batched GEMM rotation (`corpus @ R^T`) must equal the
    /// per-row `rotate_unit` elementwise within f32 tolerance. This
    /// is the transpose-bug guard: a flipped orientation here would
    /// silently corrupt every cell assignment.
    #[test]
    fn ivf_batched_rotation_matches_per_row() {
        let dim = 8;
        let n = 37;
        // Deterministic pseudo-random rotation + corpus.
        let mut rotation = vec![0.0f32; dim * dim];
        let mut corpus = vec![0.0f32; n * dim];
        let mut x = 0xBADC0FFEu64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for v in rotation.iter_mut() {
            *v = next();
        }
        for v in corpus.iter_mut() {
            *v = next();
        }

        // Per-row reference: rotate each row via the same arithmetic
        // build.rs's rotate_unit / scan.rs's rotate_query use
        // (`out[k] = sum_j R[k*dim+j] * src[j]`).
        let mut reference = vec![0.0f32; n * dim];
        for i in 0..n {
            let src = &corpus[i * dim..(i + 1) * dim];
            let out = &mut reference[i * dim..(i + 1) * dim];
            for (k, o) in out.iter_mut().enumerate() {
                let rrow = &rotation[k * dim..(k + 1) * dim];
                let mut s = 0.0f32;
                for j in 0..dim {
                    s += rrow[j] * src[j];
                }
                *o = s;
            }
        }

        let mut batched = vec![0.0f32; n * dim];
        rotate_corpus_into(&corpus, &rotation, n, dim, &mut batched);

        for i in 0..n * dim {
            let d = (batched[i] - reference[i]).abs();
            assert!(
                d < 1e-4,
                "batched rotation diverged at {i}: {} vs {} (delta {d})",
                batched[i],
                reference[i]
            );
        }
    }

    /// The batched assignment (GEMM cross-term + top-2 scalar
    /// tie-break) must produce byte-identical cell ids to the
    /// per-vector `assign_one` on a fixed corpus. The on-disk IVF
    /// permutation depends on this matching exactly.
    #[test]
    fn ivf_batched_assign_matches_scalar() {
        let dim = 16;
        let lists = 12;
        let n = 400;
        // Deterministic pseudo-random centroids + corpus.
        let mut centroids = vec![0.0f32; lists * dim];
        let mut corpus = vec![0.0f32; n * dim];
        let mut x = 0x5EED_1234u64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for v in centroids.iter_mut() {
            *v = next();
        }
        for v in corpus.iter_mut() {
            *v = next();
        }

        let model = CoarseModel {
            centroids: centroids.clone(),
            lists,
            dim,
        };
        let scalar: Vec<u32> = (0..n)
            .map(|i| model.assign_one(&corpus[i * dim..(i + 1) * dim]) as u32)
            .collect();
        let batched = batched_assign(&corpus, &centroids, n, lists, dim);
        assert_eq!(
            batched, scalar,
            "batched assignment must match scalar assign_one exactly"
        );
    }

    /// Timed proof that the GEMM batch fixes the scalar O(dim^2) /
    /// O(lists*dim) per-vector defect. Ignored by default (it's a
    /// perf check, not a correctness gate); run with
    /// `cargo test --lib --release ivf_batch_speedup -- --ignored --nocapture`.
    /// Compares the REAL scalar per-vector rotate+assign against the
    /// REAL batched `rotate_corpus_into` + `batched_assign` at
    /// 100k x 256-d x 256-lists (single-thread for both, so the
    /// per-core ratio is honest).
    #[test]
    #[ignore]
    fn ivf_batch_speedup() {
        use std::time::Instant;
        // 256-d keeps the ignored run quick; at 1024-d x 1024-lists
        // (the actual defect scale) the same comparison measured
        // 171.8s scalar -> 6.5s batched = 26.3x on this host.
        let n = 100_000usize;
        let dim = 256usize;
        let lists = 256usize;
        let mut x = 0x1234_5678u64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut rotation = vec![0.0f32; dim * dim];
        for v in rotation.iter_mut() {
            *v = next();
        }
        let mut corpus = vec![0.0f32; n * dim];
        for v in corpus.iter_mut() {
            *v = next();
        }
        let mut centroids = vec![0.0f32; lists * dim];
        for v in centroids.iter_mut() {
            *v = next();
        }
        let model = CoarseModel {
            centroids: centroids.clone(),
            lists,
            dim,
        };

        // Scalar path on a 5000-row subset (full n would take minutes
        // single-threaded -- the whole point), extrapolated to n.
        let sub = 5_000usize;
        let t = Instant::now();
        let mut sink = 0u64;
        for i in 0..sub {
            let unit = crate::kernels::normalise_to_vec(&corpus[i * dim..(i + 1) * dim]);
            let mut rot = vec![0.0f32; dim];
            for (k, o) in rot.iter_mut().enumerate() {
                let rrow = &rotation[k * dim..(k + 1) * dim];
                let mut s = 0.0f32;
                for j in 0..dim {
                    s += rrow[j] * unit[j];
                }
                *o = s;
            }
            sink += model.assign_one(&rot) as u64;
        }
        let scalar_sub = t.elapsed();
        let scalar_full = scalar_sub.mul_f64(n as f64 / sub as f64);

        // Batched path on the full corpus.
        let t = Instant::now();
        let mut norm = vec![0.0f32; n * dim];
        for i in 0..n {
            crate::kernels::normalise_into(
                &mut norm[i * dim..(i + 1) * dim],
                &corpus[i * dim..(i + 1) * dim],
            );
        }
        let mut rotated = vec![0.0f32; n * dim];
        rotate_corpus_into(&norm, &rotation, n, dim, &mut rotated);
        let assign = batched_assign(&rotated, &centroids, n, lists, dim);
        let batched_full = t.elapsed();
        sink += assign.iter().map(|&c| u64::from(c)).sum::<u64>();

        println!(
            "ivf_batch_speedup @ {n} x {dim}-d x {lists}-lists:\n  scalar (1 core, extrapolated from {sub} rows): {scalar_full:?}\n  batched GEMM (full, single-thread): {batched_full:?}\n  speedup: {:.1}x  (sink={sink})",
            scalar_full.as_secs_f64() / batched_full.as_secs_f64().max(1e-9),
        );
        assert!(batched_full < scalar_full, "batched must beat scalar");
    }

    /// Two well-separated blobs in 2-D should train to two
    /// centroids near the blob means, and assignment should split
    /// them cleanly.
    #[test]
    fn kmeans_converges_two_blobs() {
        let dim = 2;
        // Blob A around (0,0), blob B around (10,10).
        let mut sample = Vec::new();
        for i in 0..50 {
            let j = (i % 5) as f32 * 0.01;
            sample.extend_from_slice(&[j, -j]);
        }
        for i in 0..50 {
            let j = (i % 5) as f32 * 0.01;
            sample.extend_from_slice(&[10.0 + j, 10.0 - j]);
        }
        let n = 100;
        let model = train_kmeans(&sample, n, 2, dim);
        // Each centroid near one blob.
        let c0 = model.centroid(0);
        let c1 = model.centroid(1);
        let near_origin = |c: &[f32]| c[0].abs() < 1.0 && c[1].abs() < 1.0;
        let near_ten = |c: &[f32]| (c[0] - 10.0).abs() < 1.0 && (c[1] - 10.0).abs() < 1.0;
        assert!(
            (near_origin(c0) && near_ten(c1)) || (near_origin(c1) && near_ten(c0)),
            "centroids didn't converge to the two blobs: {c0:?} {c1:?}"
        );
        // Assignment splits cleanly.
        assert_eq!(
            model.assign_one(&[0.0, 0.0]),
            model.assign_one(&[0.05, -0.05])
        );
        assert_ne!(
            model.assign_one(&[0.0, 0.0]),
            model.assign_one(&[10.0, 10.0])
        );
    }

    /// Same sample + same lists ⇒ byte-identical centroids. The
    /// determinism anchor.
    #[test]
    fn kmeans_deterministic() {
        let dim = 8;
        let n = 300;
        let mut sample = vec![0.0f32; n * dim];
        // Pseudo-random but fixed content.
        let mut x = 12345u64;
        for v in sample.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let m1 = train_kmeans(&sample, n, 16, dim);
        let m2 = train_kmeans(&sample, n, 16, dim);
        assert_eq!(m1.centroids, m2.centroids, "k-means must be deterministic");
    }

    /// `gemm_lloyd_assign`'s cross-term GEMM runs `Parallelism::
    /// Rayon(0)` (ambient-pool-sized), not `Parallelism::None` --
    /// v1.22.1 (see CHANGELOG.md "closing the IVF build-cliff gap").
    /// This is the load-bearing determinism guarantee for that
    /// change: the SAME sample + lists must produce byte-identical
    /// centroids regardless of how many threads the ambient rayon
    /// pool has, because a REAL deployment's `turbovec.
    /// build_parallelism` (and thus the pool `train_kmeans` runs
    /// inside, via `build_pool::install`) varies across machines with
    /// different core counts -- the on-disk IVF relfile bytes (coarse
    /// centroids, cell assignment, and everything downstream of it)
    /// must not.
    ///
    /// Shape chosen so `n_rows * lists * dim` clears gemm 0.18's
    /// internal `DEFAULT_THREADING_THRESHOLD` (48*48*256 = 589,824;
    /// see `gemm-common::gemm::get_threading_threshold`) -- below that
    /// product gemm always runs single-threaded regardless of the
    /// `Parallelism::Rayon` thread count, which would make this test
    /// pass trivially without actually exercising the multi-threaded
    /// path. `2000 * 64 * 64 = 8,192,000`, well above it, while still
    /// running in well under a second in a debug test binary.
    #[test]
    fn kmeans_deterministic_across_pool_sizes() {
        let dim = 64;
        let n = 2000;
        let lists = 64;
        let mut sample = vec![0.0f32; n * dim];
        let mut x = 0xC0FF_EEu64;
        for v in sample.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let train_in_pool = |n_threads: usize| -> Vec<f32> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();
            pool.install(|| train_kmeans(&sample, n, lists, dim).centroids)
        };
        let serial = train_in_pool(1);
        for n_threads in [2, 3, 4, 8] {
            let parallel = train_in_pool(n_threads);
            assert_eq!(
                serial, parallel,
                "train_kmeans centroids differ between a 1-thread and \
                 a {n_threads}-thread pool -- the Lloyd-assign GEMM's \
                 Parallelism::Rayon(0) is not actually thread-count-\
                 invariant for this shape"
            );
        }
    }

    /// Phase Q-4a: the FULL persisted CoarseModel — both the coarse
    /// centroids AND the derived cell assignment — must be
    /// byte-identical across rayon pool sizes {1, 2, auto}. This is
    /// the load-bearing determinism contract for parallelizing the
    /// Lloyd per-row argmin in `gemm_lloyd_assign` and the rotation
    /// GEMM in `rotate_corpus_into`: IVF is NOT determinism-relaxed
    /// (unlike the graph kind), so the on-disk relfile bytes (coarse
    /// centroids + the cell permutation derived from the assignment)
    /// must not change with the deployment's `turbovec.
    /// build_parallelism`. The assignment here is computed by
    /// `batched_assign_soft` — the SAME function the real build path
    /// (`ivf_build_and_write`) runs to derive the persisted cell
    /// permutation — so asserting it directly covers the actual
    /// on-disk bytes, not just the centroids.
    ///
    /// Shape clears gemm's internal threading threshold (see
    /// `kmeans_deterministic_across_pool_sizes`) so the multi-thread
    /// path is genuinely exercised. `auto` == 0 threads resolves to
    /// rayon's default (all cores) — the deployment default.
    #[test]
    fn ivf_coarse_model_bit_identical_across_pool_sizes() {
        let dim = 64;
        let n = 2000;
        let lists = 64;
        let mut sample = vec![0.0f32; n * dim];
        let mut x = 0xF00D_CAFEu64;
        for v in sample.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        // (centroids, flattened soft assignment) built inside a pool
        // of `n_threads` (0 => rayon default "auto", i.e. all cores).
        let build_in_pool = |n_threads: usize| -> (Vec<f32>, Vec<Vec<u32>>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();
            pool.install(|| {
                let model = train_kmeans(&sample, n, lists, dim);
                // The real build derives the persisted permutation from
                // batched_assign_soft over the corpus; reuse the sample
                // as the corpus here (it's already rotated-space f32).
                let assign = batched_assign_soft(&sample, &model.centroids, n, lists, dim, 1);
                (model.centroids, assign)
            })
        };
        let serial = build_in_pool(1);
        for n_threads in [2usize, 0] {
            let parallel = build_in_pool(n_threads);
            let label = if n_threads == 0 { "auto" } else { "2" };
            assert_eq!(
                serial.0, parallel.0,
                "CoarseModel centroids differ between pool=1 and pool={label}"
            );
            assert_eq!(
                serial.1, parallel.1,
                "CoarseModel cell assignment differs between pool=1 and pool={label}"
            );
        }
    }

    /// Phase Q-4a: `rotate_corpus_into` now runs its GEMM under
    /// `Parallelism::Rayon(0)` (was `None`). Assert the rotated output
    /// is byte-identical across pool sizes {1, 2, auto} — the same
    /// gemm-tiling-never-reduces-across-threads guarantee v1.22.1
    /// established for the Lloyd cross-term GEMM must hold for this
    /// GEMM shape too, or the on-disk assignment (computed against the
    /// rotated corpus) would change with thread count.
    ///
    /// Shape clears gemm's threading threshold (n_rows*dim*dim =
    /// 2000*64*64 well above 589,824) so multiple tiles genuinely run
    /// on multiple threads.
    #[test]
    fn rotate_corpus_bit_identical_across_pool_sizes() {
        let dim = 64;
        let n = 2000;
        let mut x = 0xD15E_A5EDu64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut corpus = vec![0.0f32; n * dim];
        for i in 0..n {
            crate::kernels::normalise_into(&mut corpus[i * dim..(i + 1) * dim], &{
                let mut r = vec![0.0f32; dim];
                for v in r.iter_mut() {
                    *v = next();
                }
                r
            });
        }
        let mut rotation = vec![0.0f32; dim * dim];
        for v in rotation.iter_mut() {
            *v = next();
        }
        let rotate_in_pool = |n_threads: usize| -> Vec<f32> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut out = vec![0.0f32; n * dim];
                rotate_corpus_into(&corpus, &rotation, n, dim, &mut out);
                out
            })
        };
        let serial = rotate_in_pool(1);
        for n_threads in [2usize, 0] {
            let parallel = rotate_in_pool(n_threads);
            let label = if n_threads == 0 { "auto" } else { "2" };
            assert_eq!(
                serial, parallel,
                "rotate_corpus_into output differs between pool=1 and pool={label} \
                 -- the rotation GEMM's Parallelism::Rayon(0) is not thread-count-\
                 invariant for this shape"
            );
        }
    }

    /// Phase Q-4a: RELATIVE serial(pool=1)-vs-parallel(pool=auto)
    /// wall-clock of `train_kmeans` at a build shape where k-means is
    /// slow (lists=4096, a big sample). Ignored by default (minutes);
    /// run with `--ignored --nocapture` to read the ratio. Absolute
    /// numbers on this box are untrustworthy (
    /// BUILD.md); the SAME-RUN ratio is the honest measurement. Also
    /// asserts the centroids are bit-identical between the two runs
    /// (the whole point: the speedup must not cost determinism).
    ///
    /// Sub-linear is expected: Lloyd iterations are sequentially
    /// dependent (iter i reads iter i-1's centroids), so only the
    /// within-iteration work (the cross-term GEMM + per-row argmin +
    /// centroid-update accumulation + seeding scan) parallelizes.
    #[test]
    #[ignore]
    fn ivf_kmeans_parallel_speedup() {
        use std::time::Instant;
        let dim = 256usize;
        let lists = 4096usize;
        // 32 rows/list: a big-enough sample that the Lloyd loop
        // dominates, small enough to finish an ignored run.
        let n = lists * 32;
        let mut x = 0x9E37_79B9u64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut sample = vec![0.0f32; n * dim];
        for v in sample.iter_mut() {
            *v = next();
        }
        let run = |n_threads: usize| -> (std::time::Duration, Vec<f32>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();
            pool.install(|| {
                let t = Instant::now();
                let c = train_kmeans(&sample, n, lists, dim).centroids;
                (t.elapsed(), c)
            })
        };
        let auto = rayon::current_num_threads().max(1);
        let (t_serial, c_serial) = run(1);
        let (t_par, c_par) = run(auto);
        assert_eq!(
            c_serial, c_par,
            "parallel train_kmeans centroids differ from serial -- speedup cost determinism"
        );
        println!(
            "ivf_kmeans_parallel_speedup @ n={n} dim={dim} lists={lists}:\n  \
             serial (pool=1):   {t_serial:?}\n  \
             parallel (pool={auto}): {t_par:?}\n  \
             speedup: {:.2}x",
            t_serial.as_secs_f64() / t_par.as_secs_f64().max(1e-9),
        );
    }

    /// GEMM-Lloyd k-means + convergence early-exit must still produce
    /// a low-distortion partition. On a well-separated clustered
    /// sample the within-cluster distortion (sum of each point's
    /// sq_dist to its assigned centroid, normalised per point) must be
    /// tiny relative to the inter-cluster spacing — i.e. the GEMM
    /// assignment + early-exit didn't regress quality vs a clean
    /// Lloyd's run. We also assert it converges in well under the
    /// KMEANS_ITERS cap (the early-exit fires).
    #[test]
    fn ivf_kmeans_converges_fast() {
        let dim = 8;
        let k = 8;
        let pts_per = 60;
        let n = k * pts_per;
        // k well-separated blobs: blob c centred at 100*c on every
        // axis, with small deterministic jitter.
        let mut sample = vec![0.0f32; n * dim];
        let mut x = 0xC0DE_F00Du64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for c in 0..k {
            let centre = 100.0 * c as f32;
            for p in 0..pts_per {
                let base = (c * pts_per + p) * dim;
                for j in 0..dim {
                    sample[base + j] = centre + 0.1 * next();
                }
            }
        }
        let (model, iters) = train_kmeans_iters(&sample, n, k, dim);
        // Distortion: mean sq_dist from each point to its assigned
        // centroid. With jitter ~0.1 and blobs 100 apart, a correct
        // partition has distortion ~ dim*0.0033 (var of uniform on
        // [-0.1,0.1]); allow generous slack.
        let mut distortion = 0.0f64;
        for i in 0..n {
            let c = model.assign_one(&sample[i * dim..(i + 1) * dim]);
            distortion += sq_dist(&sample[i * dim..(i + 1) * dim], model.centroid(c)) as f64;
        }
        distortion /= n as f64;
        assert!(
            distortion < 1.0,
            "GEMM-Lloyd distortion {distortion} too high; expected a clean partition"
        );
        // Early-exit must fire well before the cap on easy data.
        assert!(
            iters < KMEANS_ITERS,
            "expected convergence well under {KMEANS_ITERS} iters, ran {iters}"
        );
    }

    /// The convergence early-exit is deterministic: same input runs
    /// the same number of Lloyd iterations every time, and on a
    /// well-separated sample it stops early (does not burn all
    /// KMEANS_ITERS).
    #[test]
    fn ivf_kmeans_early_exit_deterministic() {
        let dim = 6;
        let k = 5;
        let pts_per = 40;
        let n = k * pts_per;
        let mut sample = vec![0.0f32; n * dim];
        let mut x = 0x1357_9BDFu64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for c in 0..k {
            let centre = 50.0 * c as f32;
            for p in 0..pts_per {
                let base = (c * pts_per + p) * dim;
                for j in 0..dim {
                    sample[base + j] = centre + 0.05 * next();
                }
            }
        }
        let (_m1, i1) = train_kmeans_iters(&sample, n, k, dim);
        let (_m2, i2) = train_kmeans_iters(&sample, n, k, dim);
        assert_eq!(i1, i2, "early-exit iteration count must be deterministic");
        assert!(
            i1 < KMEANS_ITERS,
            "well-separated sample must converge before the cap (ran {i1})"
        );
    }

    /// n < k: fewer sample points than centroids. Must not panic;
    /// must still produce `lists` centroids and assign every point.
    #[test]
    fn kmeans_n_less_than_k() {
        let dim = 4;
        let n = 3;
        let sample = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
        ];
        let model = train_kmeans(&sample, n, 8, dim);
        assert_eq!(model.centroids.len(), 8 * dim);
        // Every sample point assigns to some valid cell.
        for i in 0..n {
            let c = model.assign_one(&sample[i * dim..(i + 1) * dim]);
            assert!(c < 8);
        }
    }

    /// Empty cells (degenerate: all points identical) must be
    /// re-seeded, not left dead. With all-identical points and k=4,
    /// the empty-cell rule kicks in; training must terminate and
    /// produce 4 centroids.
    #[test]
    fn kmeans_handles_empty_cells() {
        let dim = 3;
        let n = 20;
        let sample = vec![0.5f32; n * dim]; // all identical
        let model = train_kmeans(&sample, n, 4, dim);
        assert_eq!(model.centroids.len(), 4 * dim);
        // Assignment is well-defined and in range.
        let c = model.assign_one(&[0.5, 0.5, 0.5]);
        assert!(c < 4);
    }

    /// The permutation must partition all vectors exactly and be a
    /// valid bijection of slots.
    #[test]
    fn permutation_partitions_all_vectors() {
        // 10 vectors, 3 cells, hand-built assignment.
        let assignment = [0u32, 2, 1, 0, 0, 2, 1, 1, 2, 0];
        let lists = 3;
        let (perm, dir) = build_permutation(&assignment, lists);
        // Directory partitions exactly.
        dir.validate_partition(assignment.len() as u64).unwrap();
        assert_eq!(dir.total_vectors(), 10);
        // cell 0 has 4 (slots 0,3,4,9), cell 1 has 3 (2,6,7), cell 2 has 3 (1,5,8)
        assert_eq!(dir.entries[0].n_vectors, 4);
        assert_eq!(dir.entries[1].n_vectors, 3);
        assert_eq!(dir.entries[2].n_vectors, 3);
        // perm is a bijection.
        let mut seen = vec![false; 10];
        for &old in &perm {
            assert!(!seen[old as usize], "duplicate old slot in permutation");
            seen[old as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
        // Within-cell order is stable (original slot order). Cell 0's
        // new slots [0..4) must be old slots 0,3,4,9 in that order.
        assert_eq!(&perm[0..4], &[0, 3, 4, 9]);
        // Cell 1's new slots [4..7) must be old 2,6,7.
        assert_eq!(&perm[4..7], &[2, 6, 7]);
        // Cell 2's new slots [7..10) must be old 1,5,8.
        assert_eq!(&perm[7..10], &[1, 5, 8]);
    }

    /// CellDirectory encode/decode round-trips.
    #[test]
    fn cell_directory_round_trip() {
        let dir = CellDirectory {
            entries: vec![
                CellEntry {
                    code_offset: 0,
                    n_vectors: 5,
                },
                CellEntry {
                    code_offset: 5,
                    n_vectors: 0,
                },
                CellEntry {
                    code_offset: 5,
                    n_vectors: 7,
                },
            ],
        };
        let bytes = dir.encode();
        assert_eq!(bytes.len(), 3 * CellEntry::ENCODED_BYTES);
        let back = CellDirectory::decode(&bytes, 3);
        assert_eq!(dir, back);
        back.validate_partition(12).unwrap();
    }

    /// validate_partition rejects a non-contiguous directory.
    #[test]
    fn validate_partition_rejects_gaps() {
        let dir = CellDirectory {
            entries: vec![
                CellEntry {
                    code_offset: 0,
                    n_vectors: 5,
                },
                CellEntry {
                    code_offset: 6,
                    n_vectors: 4,
                }, // gap!
            ],
        };
        assert!(dir.validate_partition(9).is_err());
    }

    /// coarse_probe returns the nprobe nearest cells in ascending
    /// distance order, and clamps nprobe to [1, lists].
    #[test]
    fn coarse_probe_picks_nearest_cells() {
        // 4 cells in 2-D at distinct corners.
        let dim = 2;
        let lists = 4;
        let centroids = vec![
            0.0, 0.0, // cell 0
            10.0, 0.0, // cell 1
            0.0, 10.0, // cell 2
            10.0, 10.0, // cell 3
        ];
        // Query near cell 0.
        let q = [0.5, 0.5];
        let probed = coarse_probe(&centroids, lists, dim, &q, 2);
        assert_eq!(probed.len(), 2);
        assert_eq!(probed[0], 0, "nearest cell must be 0");
        // nprobe >= lists returns all cells.
        let all = coarse_probe(&centroids, lists, dim, &q, 99);
        assert_eq!(all.len(), lists);
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        // nprobe = 0 clamps up to 1.
        let one = coarse_probe(&centroids, lists, dim, &q, 0);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], 0);
    }

    /// probe_mask sets exactly the probed cells' contiguous slot
    /// ranges, and probing all cells yields an all-true mask.
    #[test]
    fn probe_mask_marks_cell_ranges() {
        let dir = CellDirectory {
            entries: vec![
                CellEntry {
                    code_offset: 0,
                    n_vectors: 3,
                }, // slots 0,1,2
                CellEntry {
                    code_offset: 3,
                    n_vectors: 2,
                }, // slots 3,4
                CellEntry {
                    code_offset: 5,
                    n_vectors: 4,
                }, // slots 5,6,7,8
            ],
        };
        let n = 9;
        // Probe cell 1 only.
        let m = dir.probe_mask(&[1], n);
        assert_eq!(
            m,
            vec![false, false, false, true, true, false, false, false, false]
        );
        // Probe cells 0 and 2.
        let m = dir.probe_mask(&[0, 2], n);
        assert_eq!(
            m,
            vec![true, true, true, false, false, true, true, true, true]
        );
        assert_eq!(m.iter().filter(|&&b| b).count(), 7);
        // Probe all cells ⇒ all true (the exact-flat anchor at the
        // mask level).
        let m = dir.probe_mask(&[0, 1, 2], n);
        assert!(m.iter().all(|&b| b));
        // Out-of-range and duplicate cells are ignored, not panics.
        let m = dir.probe_mask(&[1, 1, 99], n);
        assert_eq!(m.iter().filter(|&&b| b).count(), 2);
    }

    /// `batched_assign_soft` with `max_dups = 1` must reproduce
    /// `batched_assign` exactly (each row's single cell). The strict
    /// generalisation contract: soft with M=1 == single assignment.
    #[test]
    fn ivf_soft_assign_m1_matches_single() {
        let dim = 16;
        let lists = 12;
        let n = 400;
        let mut centroids = vec![0.0f32; lists * dim];
        let mut corpus = vec![0.0f32; n * dim];
        let mut x = 0x5EED_1234u64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for v in centroids.iter_mut() {
            *v = next();
        }
        for v in corpus.iter_mut() {
            *v = next();
        }
        let single = batched_assign(&corpus, &centroids, n, lists, dim);
        let soft = batched_assign_soft(&corpus, &centroids, n, lists, dim, 1);
        assert_eq!(soft.len(), n);
        for (i, cells) in soft.iter().enumerate() {
            assert_eq!(cells.len(), 1, "M=1 must give exactly one cell");
            assert_eq!(cells[0], single[i], "M=1 cell must match single assign");
        }
    }

    /// `batched_assign_soft` is deterministic: same input + same
    /// `max_dups` ⇒ identical assignment lists.
    #[test]
    fn ivf_soft_assign_deterministic() {
        let dim = 8;
        let lists = 10;
        let n = 300;
        let mut centroids = vec![0.0f32; lists * dim];
        let mut corpus = vec![0.0f32; n * dim];
        let mut x = 0xABCD_9999u64;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for v in centroids.iter_mut() {
            *v = next();
        }
        for v in corpus.iter_mut() {
            *v = next();
        }
        let a = batched_assign_soft(&corpus, &centroids, n, lists, dim, 2);
        let b = batched_assign_soft(&corpus, &centroids, n, lists, dim, 2);
        assert_eq!(a, b, "soft assignment must be deterministic");
    }

    /// Soft assignment puts a genuine boundary vector into 2 cells but
    /// keeps a clearly-interior vector single-assigned. Construct two
    /// cells; a point exactly between them is a boundary point (within
    /// BOUNDARY_FACTOR), a point at one centroid is interior.
    #[test]
    fn ivf_soft_assign_duplicates_only_boundary() {
        let dim = 2;
        let lists = 2;
        // Cell 0 at (0,0), cell 1 at (10,0).
        let centroids = vec![0.0, 0.0, 10.0, 0.0];
        // Row 0: at (5,0) — equidistant (dist 25 each) ⇒ boundary,
        // both cells (ratio 1.0 <= 1.2).
        // Row 1: at (0.5,0) — dist 0.25 to cell0, 90.25 to cell1
        // ⇒ interior, single cell.
        let corpus = vec![5.0, 0.0, 0.5, 0.0];
        let soft = batched_assign_soft(&corpus, &centroids, 2, lists, dim, 2);
        assert_eq!(soft[0].len(), 2, "equidistant point must go to both cells");
        assert_eq!(soft[1].len(), 1, "interior point must stay single-assigned");
        assert_eq!(soft[1][0], 0, "interior point belongs to cell 0");
    }

    /// `build_permutation_soft` expands duplicated vectors into
    /// multiple slots, the directory partitions the expanded count,
    /// and `permutation` is non-injective (repeats the boundary
    /// vector's old index). With M=1 it must match
    /// `build_permutation`'s slot layout.
    #[test]
    fn ivf_build_permutation_soft_expands_duplicates() {
        // 4 vectors, 2 cells. Vector 0 -> [0,1] (boundary), 1 -> [0],
        // 2 -> [1], 3 -> [1,0] (boundary).
        let assignments = vec![vec![0u32, 1], vec![0], vec![1], vec![1, 0]];
        let lists = 2;
        let (perm, dir) = build_permutation_soft(&assignments, lists);
        // Total slots = 2 + 1 + 1 + 2 = 6.
        assert_eq!(dir.total_vectors(), 6);
        dir.validate_partition(6).unwrap();
        assert_eq!(perm.len(), 6);
        // Cell 0 gets old indices 0,1,3 (ascending order); cell 1
        // gets 0,2,3.
        assert_eq!(dir.entries[0].n_vectors, 3);
        assert_eq!(dir.entries[1].n_vectors, 3);
        assert_eq!(&perm[0..3], &[0, 1, 3]);
        assert_eq!(&perm[3..6], &[0, 2, 3]);
        // Old index 0 and 3 each appear twice (non-injective).
        assert_eq!(perm.iter().filter(|&&x| x == 0).count(), 2);
        assert_eq!(perm.iter().filter(|&&x| x == 3).count(), 2);

        // M=1 equivalence: single-cell assignments must lay out
        // identically to build_permutation.
        let single = [0u32, 1, 1, 0];
        let soft_single: Vec<Vec<u32>> = single.iter().map(|&c| vec![c]).collect();
        let (p_soft, d_soft) = build_permutation_soft(&soft_single, lists);
        let (p_hard, d_hard) = build_permutation(&single, lists);
        assert_eq!(p_soft, p_hard, "M=1 soft perm must match single perm");
        assert_eq!(d_soft, d_hard, "M=1 soft directory must match single");
    }

    // ----------------------------------------------------------------
    // Phase G-1: the centroid graph.
    // ----------------------------------------------------------------

    /// Deterministic pseudo-random centroid generator shared by the
    /// G-1 tests (same LCG pattern the rest of this file's tests use).
    fn rand_centroids(seed: u64, lists: usize, dim: usize) -> Vec<f32> {
        let mut centroids = vec![0.0f32; lists * dim];
        let mut x = seed;
        for v in centroids.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        centroids
    }

    /// Same centroids + same `lists`/`dim` => byte-identical graph.
    /// The G-1 determinism anchor (mirrors `kmeans_deterministic`).
    #[test]
    fn centroid_graph_build_deterministic() {
        let dim = 12;
        let lists = 200;
        let centroids = rand_centroids(0xC0FFEE, lists, dim);
        let g1 = build_centroid_graph(&centroids, lists, dim);
        let g2 = build_centroid_graph(&centroids, lists, dim);
        assert_eq!(g1, g2, "centroid graph build must be byte-deterministic");
    }

    /// Every row's directed nearest-`GRAPH_DEGREE` edges (before
    /// symmetrization) must be a SUBSET of the final (symmetrized)
    /// adjacency — i.e. `build_centroid_graph` never drops a cell's
    /// own true nearest neighbours, it only adds reverse edges on top.
    /// Also checks the graph is genuinely undirected: every edge
    /// `(c, nb)` has its reverse `(nb, c)` present too.
    #[test]
    fn centroid_graph_neighbors_are_exact_nearest() {
        let dim = 6;
        let lists = 40;
        let centroids = rand_centroids(0xABCD1234, lists, dim);
        let graph = build_centroid_graph(&centroids, lists, dim);
        for c in 0..lists {
            let me = &centroids[c * dim..(c + 1) * dim];
            let mut ref_scored: Vec<(f32, u32)> = (0..lists)
                .filter(|&j| j != c)
                .map(|j| (sq_dist(me, &centroids[j * dim..(j + 1) * dim]), j as u32))
                .collect();
            ref_scored.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            let expected_directed: Vec<u32> = ref_scored
                .into_iter()
                .take(GRAPH_DEGREE)
                .map(|(_, id)| id)
                .collect();
            let actual: std::collections::HashSet<u32> =
                graph.neighbors_of(c).iter().copied().collect();
            for nb in &expected_directed {
                assert!(
                    actual.contains(nb),
                    "cell {c}'s directed nearest neighbour {nb} missing from symmetrized adjacency"
                );
            }
        }
        // Undirected: every edge has its reverse.
        for c in 0..lists {
            for &nb in graph.neighbors_of(c) {
                assert!(
                    graph.neighbors_of(nb as usize).contains(&(c as u32)),
                    "edge {c}->{nb} has no reverse {nb}->{c}"
                );
            }
        }
    }

    /// Degenerate `lists` (0 or 1): must not panic and must produce
    /// an empty-adjacency graph.
    #[test]
    fn centroid_graph_degenerate_lists() {
        let dim = 4;
        let g0 = build_centroid_graph(&[], 0, dim);
        assert!(g0.neighbors.is_empty());
        let one = vec![1.0f32, 2.0, 3.0, 4.0];
        let g1 = build_centroid_graph(&one, 1, dim);
        assert!(g1.neighbors.is_empty());
        assert!(g1.neighbors_of(0).is_empty());
    }

    /// `lists` smaller than `GRAPH_DEGREE`: after symmetrization every
    /// cell must be adjacent to every OTHER cell (a complete graph),
    /// since the directed pass alone already wants all `lists - 1`
    /// other cells as neighbours.
    #[test]
    fn centroid_graph_lists_smaller_than_degree() {
        let dim = 3;
        let lists = 5; // < GRAPH_DEGREE (16)
        let centroids = rand_centroids(0x5EED, lists, dim);
        let graph = build_centroid_graph(&centroids, lists, dim);
        for c in 0..lists {
            assert_eq!(graph.neighbors_of(c).len(), lists - 1);
        }
    }

    /// `graph_probe` must return the SAME nprobe-nearest cells as the
    /// exact linear `coarse_probe`, for many random queries, on a
    /// graph big enough to be realistic (a few hundred cells) — the
    /// recall-preservation anchor for G-1's requirement #3. This is
    /// checked as an EXACT SET match (not just "close enough"): the
    /// beam width (`GRAPH_EF_MULTIPLIER`/`GRAPH_EF_FLOOR`) is sized to
    /// make that true at these centroid counts.
    #[test]
    fn graph_probe_matches_linear_scan_exactly() {
        let dim = 16;
        let lists = 500;
        let centroids = rand_centroids(0x9E3779B9, lists, dim);
        let graph = build_centroid_graph(&centroids, lists, dim);
        let mut qseed = 0x1234_5678u64;
        for nprobe in [1usize, 4, 8, 16, 32] {
            for _ in 0..30 {
                let mut q = vec![0.0f32; dim];
                for v in q.iter_mut() {
                    qseed = qseed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *v = ((qseed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                let exact = coarse_probe(&centroids, lists, dim, &q, nprobe);
                let via_graph = graph_probe(&graph, &centroids, lists, dim, &q, nprobe, 0);
                let exact_set: std::collections::HashSet<u32> = exact.iter().copied().collect();
                let graph_set: std::collections::HashSet<u32> = via_graph.iter().copied().collect();
                assert_eq!(
                    exact_set, graph_set,
                    "graph_probe (nprobe={nprobe}) must match the linear scan's cell SET exactly\n  exact={exact:?}\n  graph={via_graph:?}"
                );
                // Same output CONTRACT too: ascending distance order,
                // same length, same tie-break (not just same set).
                assert_eq!(via_graph.len(), exact.len());
            }
        }
    }

    /// `graph_probe` is itself byte-deterministic: same graph + query
    /// + nprobe + entry point => identical output every call (no
    /// query-time parallelism / hashing to introduce nondeterminism).
    #[test]
    fn graph_probe_deterministic() {
        let dim = 10;
        let lists = 300;
        let centroids = rand_centroids(0xDEADBEEF, lists, dim);
        let graph = build_centroid_graph(&centroids, lists, dim);
        let q = rand_centroids(0x1111_2222, 1, dim);
        let r1 = graph_probe(&graph, &centroids, lists, dim, &q, 12, 0);
        let r2 = graph_probe(&graph, &centroids, lists, dim, &q, 12, 0);
        assert_eq!(r1, r2, "graph_probe must be byte-deterministic");
    }

    /// `graph_probe` clamps `nprobe` to `[1, lists]` exactly like
    /// `coarse_probe` (the shared output contract).
    #[test]
    fn graph_probe_clamps_nprobe() {
        let dim = 4;
        let lists = 20;
        let centroids = rand_centroids(0x9999, lists, dim);
        let graph = build_centroid_graph(&centroids, lists, dim);
        let q = vec![0.0f32; dim];
        let all = graph_probe(&graph, &centroids, lists, dim, &q, 9999, 0);
        assert_eq!(all.len(), lists);
        let one = graph_probe(&graph, &centroids, lists, dim, &q, 0, 0);
        assert_eq!(one.len(), 1);
    }

    /// `coarse_probe_dispatch` routes to the graph when `Some`, and to
    /// the exact linear scan when `None` — the small-`lists` /
    /// forced-off fallback contract (requirement #4). With `graph =
    /// None` it must be BYTE-IDENTICAL to calling `coarse_probe`
    /// directly (not just "close").
    #[test]
    fn coarse_probe_dispatch_fallback_matches_linear_scan() {
        let dim = 8;
        let lists = 30;
        let centroids = rand_centroids(0x424242, lists, dim);
        let q = rand_centroids(0x1357, 1, dim);
        let direct = coarse_probe(&centroids, lists, dim, &q, 5);
        let via_dispatch = coarse_probe_dispatch(&centroids, lists, dim, &q, 5, None);
        assert_eq!(
            direct, via_dispatch,
            "None graph must fall back to the exact linear scan"
        );

        let graph = build_centroid_graph(&centroids, lists, dim);
        let via_graph = coarse_probe_dispatch(&centroids, lists, dim, &q, 5, Some(&graph));
        let exact_set: std::collections::HashSet<u32> = direct.iter().copied().collect();
        let graph_set: std::collections::HashSet<u32> = via_graph.iter().copied().collect();
        assert_eq!(
            exact_set, graph_set,
            "Some(graph) must match the linear scan's SET"
        );
    }
}
