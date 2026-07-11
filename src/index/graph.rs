//! Vamana-style navigable graph construction (Phase G-2a).
//!
//! and
//! . This module is
//! deliberately **Postgres-free** (like `ivf.rs`) so it is
//! unit-testable directly. It owns:
//!
//! - [`build_vamana`]: the DiskANN "Vamana" single-pass graph-build
//!   algorithm (greedy search + RobustPrune, per node, in a
//!   deterministic randomized insertion order).
//! - [`GraphAdjacency`]: the CSR (offsets + flat neighbor ids)
//!   adjacency representation persisted to the relfile, same design
//!   pattern as `ivf::CentroidGraph` but built via Vamana instead of
//!   simple k-NN-then-symmetrize (Vamana's RobustPrune gives the
//!   graph long-range edges that a plain k-NN graph lacks, which is
//!   what lets a bounded-degree graph stay navigable at corpus
//!   scale — see the module-level comment on `ivf::CentroidGraph` for
//!   the k-NN-then-symmetrize failure mode this algorithm avoids by
//!   construction).
//!
//! ## Distance space
//!
//! The graph is built over the vectors in the SAME space they get
//! quantized in: L2-normalised (matching `turbovec`'s assumption
//! that vectors are unit-normalised) squared-Euclidean distance.
//! For unit vectors, squared Euclidean distance and cosine
//! similarity/inner-product rank identically
//! (`||a-b||^2 = 2 - 2<a,b>`), so the graph's proximity structure is
//! correct for every operator class the AM supports (`<=>`, `<#>`,
//! `<->`). Unlike the IVF coarse quantizer, this module does NOT
//! need the persisted rotation matrix — the graph's edges are a
//! property of the raw vector geometry, not of the quantizer's
//! per-coordinate calibration, so building on the un-rotated
//! normalised vectors is simpler and exactly as correct.
//!
//! ## RAM residency
//!
//! Per , Phase G-2 is explicitly a
//! **RAM-resident** design (trading pure out-of-core for
//! HNSW-matching latency) — so [`build_vamana`] takes the whole
//! corpus as one resident `&[f32]` slice. The build-side caller
//! (`build.rs`) is responsible for staging that slice (it reuses
//! `CorpusSpill` to stream the heap scan into it without holding a
//! second copy alive during the scan itself).
//!
//! ## Determinism
//!
//! Deterministic for a fixed seed on one machine (not byte-identical
//! across machines — explicitly relaxed for the graph kind per the
//! plan doc), AND bit-identical regardless of thread count. The
//! per-node insertion order is strictly serial (each node's greedy
//! search depends on the graph state every prior insertion left), so
//! there is no insertion-order parallelism. Phase G-2c batches the
//! per-node candidate distance evals via `build_row_dists` over plain
//! `sq_dist` (which LLVM already auto-vectorises optimally on
//! contiguous f32); the reduction order is a fixed function of the
//! input, independent of thread count, so a fixed seed always
//! produces the same adjacency at any `turbovec.build_parallelism`.
//! Asserted by `prop_build_thread_count_independent` and
//! `build_is_bit_identical_across_thread_counts`.
//!
//! Note (measured G-2c finding): thread-level parallelism of the
//! single-pass build does NOT amortise — the per-hop candidate
//! batches are bounded by the build beam width `L` (~64), too small
//! for rayon's fork-join dispatch to pay off (a fanned-out version
//! was measured 0.68–0.85×, i.e. a slowdown). The build speedup is
//! the SIMD distance kernel (a thread-count-independent ~2.5× on the
//! dominant cost), not thread fan-out. See `build_row_dists`.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::index::ivf::sq_dist;

/// Fixed seed for the randomized insertion-order RNG. Distinct from
/// `ivf::IVF_SEED` and turbovec's `ROTATION_SEED` so the three
/// deterministic subsystems don't share an RNG stream by accident.
pub const GRAPH_SEED: u64 = 0x6A_2E_6E_A5_5EED_u64;

/// Maximum out-degree `R`. DiskANN's paper uses R in the 32-64 range
/// for large-scale (million+) corpora; 32 is the low end of that
/// range, keeping the per-node adjacency-chain storage tax modest
/// (32 * 4B = 128B/node) while still giving the greedy search enough
/// fan-out per hop for good navigability. G-2c (SIMD/parallelism) or
/// a future tuning pass may want this higher for larger corpora;
/// G-2a's correctness-first scope keeps one fixed value.
pub const GRAPH_DEGREE_R: usize = 32;

/// Build-time search-list size `L` (the beam width `RobustPrune`'s
/// candidate set is drawn from). DiskANN recommends `L >= R`; we use
/// 2x `R` so the greedy search explores enough of the graph to give
/// `RobustPrune` a genuinely diverse candidate pool without making
/// each node's build-time search arbitrarily expensive.
pub const GRAPH_BUILD_L: usize = GRAPH_DEGREE_R * 2;

/// `RobustPrune`'s diversity factor `alpha`. `>= 1.0`; DiskANN's
/// paper and most reference implementations use 1.2 for a
/// single-pass build (the two-pass refinement that pushes `alpha`
/// higher on a second pass is explicitly out of scope for G-2a — see
/// the module doc comment on single-pass builds in
/// ).
pub const GRAPH_ALPHA: f32 = 1.2;

/// Phase G-2d(a): corpus size at/below which the `auto`
/// `turbovec.graph_build_partitions` policy uses the single-pass
/// [`build_vamana`]. Below this the serial build finishes quickly
/// (seconds) and the partitioned build's stitch overhead + slightly
/// different adjacency isn't worth it. 200k is the scale where the
/// serial build starts taking real wall-clock on this hardware (the
/// timing test's regime) while still being small enough that the
/// partitioned build's recall parity is easy to verify against it.
pub const GRAPH_PARTITION_MIN_ROWS: usize = 200_000;

/// Target rows per shard for the `auto` policy. Smaller shards build
/// FASTER per row (single-pass Vamana's per-node greedy search cost
/// grows with the sub-graph size) AND expose more parallelism, so the
/// measured sweet spot is many small shards, not few large ones
/// (G-2d(a) timing: n=200k on 8 cores, P=8 -> 8.3x, P=16 -> 13.3x,
/// P=32 -> 11.4x). ~12k keeps a shard's build fast while keeping the
/// stitch's cross-shard candidate diversity reasonable; the pool cap
/// (`P <= pool*4`, below) is what actually bounds P on a given box.
/// `P = clamp(n / this, 2, pool*4)`.
pub const GRAPH_TARGET_SHARD_ROWS: usize = 12_000;

/// Floor on rows per shard: a corpus is only split into `P` shards if
/// each shard holds at least this many rows, else `P` is reduced
/// (down to 1 = single-pass). Below ~2k rows a shard's sub-graph is
/// too small to give the stitch's per-shard candidate pool useful
/// cross-shard diversity, and the fixed stitch cost dominates. This
/// also bounds `forced-N` so `graph_build_partitions = 9999` on a
/// small table can't create thousands of near-empty shards.
pub const GRAPH_MIN_SHARD_ROWS: usize = 2_000;

/// Scan-time beam-width multiplier, mirroring
/// `ivf::GRAPH_EF_MULTIPLIER`'s pattern but sized for CORPUS scale
/// (thousands to millions of nodes) rather than centroid scale
/// (thousands of cells). `ef = (k * MULTIPLIER).max(FLOOR).min(n)`.
/// 4x the requested `k` gives the beam room to recover from a
/// locally-suboptimal hop without materially widening the per-query
/// cost — the same recall-safety margin `ivf::graph_probe` already
/// validated empirically at centroid scale.
pub const GRAPH_SCAN_EF_MULTIPLIER: usize = 4;

/// Floor on the scan-time beam width, so a tiny `k` (e.g. `LIMIT 1`)
/// still searches widely enough to be recall-safe. 64 matches the
/// common HNSW `ef_search` default (e.g. Faiss/hnswlib ship 64 as a
/// reasonable out-of-the-box value) — reusing an established
/// reference point rather than inventing a new one.
pub const GRAPH_SCAN_EF_FLOOR: usize = 64;

/// CSR (compressed sparse row) adjacency: `offsets[i]..offsets[i+1]`
/// indexes `neighbors` for node `i`'s (ascending-id, deduplicated,
/// no self-loop) out-neighbor list. `offsets.len() == n + 1`. Same
/// representation style as `ivf::CentroidGraph`, but built via
/// [`build_vamana`] instead of k-NN-then-symmetrize.
///
/// Persisted to the relfile as two concatenated flat byte
/// sub-chains: the `u32` offsets array, then the `u32` neighbor-id
/// array (see [`Self::encode_offsets`] / [`Self::encode_neighbors`]
/// and `page.rs`'s `graph_offsets_bytes` / `graph_neighbors_bytes`
/// meta fields, which record the split point).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphAdjacency {
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl GraphAdjacency {
    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Out-neighbor ids of node `id` (ascending id order, no dups, no
    /// self-loop).
    pub fn neighbors_of(&self, id: usize) -> &[u32] {
        let s = self.offsets[id] as usize;
        let e = self.offsets[id + 1] as usize;
        &self.neighbors[s..e]
    }

    /// Total edge count (sum of every node's out-degree).
    #[allow(dead_code)] // exercised by tests
    pub fn edge_count(&self) -> usize {
        self.neighbors.len()
    }

    /// An adjacency with `n` nodes and zero edges. Used for the
    /// empty-corpus and single-node build cases.
    fn empty(n: usize) -> Self {
        Self {
            offsets: vec![0u32; n + 1],
            neighbors: Vec::new(),
        }
    }

    /// Expand the CSR representation into the mutable
    /// `Vec<Vec<u32>>` shape [`insert_node_into_graph`] and the
    /// build loop operate on. Used by the incremental-insert path
    /// (`insert.rs`, Phase G-2b) to rehydrate a persisted adjacency,
    /// mutate it in RAM, and re-flatten via [`Self::from_lists`].
    pub(crate) fn to_lists(&self) -> Vec<Vec<u32>> {
        (0..self.n())
            .map(|i| self.neighbors_of(i).to_vec())
            .collect()
    }

    /// Inverse of [`Self::to_lists`]: flatten a `Vec<Vec<u32>>` back
    /// to CSR, sorting each node's own list ascending first (same
    /// canonicalisation [`build_vamana_with_params`] applies before
    /// persisting).
    pub(crate) fn from_lists(mut adj: Vec<Vec<u32>>) -> Self {
        let n = adj.len();
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + adj[i].len() as u32;
        }
        let mut neighbors = Vec::with_capacity(offsets[n] as usize);
        for row in &mut adj {
            row.sort_unstable();
            neighbors.extend_from_slice(row);
        }
        Self { offsets, neighbors }
    }

    /// Serialise the offsets array to little-endian bytes (the first
    /// of the two concatenated sub-chains).
    pub fn encode_offsets(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.offsets.len() * 4);
        for &o in &self.offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out
    }

    /// Serialise the neighbor-id array to little-endian bytes (the
    /// second of the two concatenated sub-chains).
    pub fn encode_neighbors(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.neighbors.len() * 4);
        for &nb in &self.neighbors {
            out.extend_from_slice(&nb.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::encode_offsets`] + [`Self::encode_neighbors`].
    /// `n` is the expected node count (`offsets_bytes.len()` must be
    /// `(n + 1) * 4`); the neighbor count is derived from
    /// `neighbors_bytes.len() / 4` and cross-checked against the
    /// final offset entry.
    pub fn decode(offsets_bytes: &[u8], neighbors_bytes: &[u8], n: usize) -> Result<Self, String> {
        if offsets_bytes.len() != (n + 1) * 4 {
            return Err(format!(
                "graph offsets_bytes.len()={} != (n+1)*4={} for n={n}",
                offsets_bytes.len(),
                (n + 1) * 4,
            ));
        }
        if neighbors_bytes.len() % 4 != 0 {
            return Err(format!(
                "graph neighbors_bytes.len()={} is not a multiple of 4",
                neighbors_bytes.len()
            ));
        }
        let offsets: Vec<u32> = offsets_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let neighbors: Vec<u32> = neighbors_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let expected_edges = *offsets.last().unwrap_or(&0) as usize;
        if expected_edges != neighbors.len() {
            return Err(format!(
                "graph offsets[n]={expected_edges} != neighbors.len()={}",
                neighbors.len()
            ));
        }
        Ok(Self { offsets, neighbors })
    }
}

/// Build a [`GraphAdjacency`] over `n` vectors (row-major, `n * dim`
/// `f32`, each row already L2-normalised) using the default
/// [`GRAPH_DEGREE_R`] / [`GRAPH_BUILD_L`] / [`GRAPH_ALPHA`] /
/// [`GRAPH_SEED`] parameters. Returns the adjacency plus the entry
/// point slot id (an approximate medoid — the actual corpus point
/// nearest the mean vector — which greedy search starts from).
///
/// `n == 0` returns an empty (zero-node) adjacency and entry point 0
/// (meaningless but harmless — `has_graph()` on the persisted meta
/// page will be `false` for an empty index, so no reader ever
/// dereferences it). `n == 1` returns a single node with no edges.
pub fn build_vamana(vectors: &[f32], n: usize, dim: usize) -> (GraphAdjacency, u32) {
    build_vamana_with_params(
        vectors,
        n,
        dim,
        GRAPH_DEGREE_R,
        GRAPH_BUILD_L,
        GRAPH_ALPHA,
        GRAPH_SEED,
    )
}

/// Parameterised form of [`build_vamana`], for tests (and any future
/// tuning work) that want to vary `r` / `l` / `alpha` / `seed`
/// directly.
pub fn build_vamana_with_params(
    vectors: &[f32],
    n: usize,
    dim: usize,
    r: usize,
    l: usize,
    alpha: f32,
    seed: u64,
) -> (GraphAdjacency, u32) {
    debug_assert_eq!(vectors.len(), n * dim);
    if n == 0 {
        return (GraphAdjacency::empty(0), 0);
    }
    if n == 1 {
        return (GraphAdjacency::empty(1), 0);
    }
    let r = r.max(1);
    let l = l.max(r).min(n);

    let entry = approx_medoid(vectors, n, dim);

    // Mutable adjacency-in-progress. Vec<Vec<u32>> (not CSR) because
    // degree varies during the build (reverse edges push a node's
    // degree above R transiently, until the re-prune below trims it
    // back).
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];

    // Deterministic randomized insertion order.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);

    for &p in &order {
        insert_node_into_graph(&mut adj, entry, p, vectors, dim, l, alpha, r);
    }

    // Flatten to CSR. Each node's own list sorted ascending (fixed,
    // deterministic byte layout regardless of insertion order within
    // the list).
    (GraphAdjacency::from_lists(adj), entry)
}

/// Phase G-2d(a): partitioned/merge PARALLEL Vamana build
///. Structurally different
/// from [`build_vamana`]'s serial single-pass loop so the graph build
/// scales to millions of rows:
///
/// 1. **Partition** the deterministic shuffled insertion order into
///    `p` CONTIGUOUS shards (so each shard is a uniform random sample
///    of the corpus, uncorrelated with slot id).
/// 2. **Build each shard's sub-graph IN PARALLEL** (`par_iter`, index-
///    ordered collect — deterministic regardless of pool size). Each
///    shard runs a full single-pass [`build_vamana_with_params`] in
///    LOCAL id space over its own gathered vectors, then remaps local
///    -> global ids. This is the parallel win: `p` independent builds,
///    each `1/p` the size, run concurrently across the pool.
/// 3. **Stitch** into one navigable graph. A union of disjoint
///    sub-graphs is unnavigable across shards, so:
///    - global entry point = [`approx_medoid`] over ALL vectors;
///    - a **cross-shard refinement pass**: for each node `p` (fixed id
///      order, parallelizable — reads the merged graph READ-ONLY,
///      stages `p`'s new neighbor list, all applied atomically after
///      the pass so it is order-independent), greedy-search the MERGED
///      graph from the global entry toward `p` (beam `l`) to collect a
///      visited set that now SPANS shards, then `RobustPrune(p, V ∪
///      N_out(p), alpha, r)`. This is what creates the cross-shard
///      edges the per-shard build couldn't see.
///    - a **serial reverse-edge pass** afterwards restores the mutual
///      navigability + degree bound the single-pass build's reverse-
///      edge step gives (done serially/deterministically since it
///      mutates OTHER nodes' lists).
///
/// Both dominant costs (the per-shard builds AND the refinement pass)
/// are parallel, unlike the single-pass build whose G-2c finding was
/// that its per-hop batches are too small to amortise fan-out.
///
/// Returns the SAME `(GraphAdjacency, entry_point)` shape
/// [`build_vamana`] does — no wire-format change (the persisted CSR is
/// identical in shape, just a different-but-equally-valid adjacency).
///
/// Deterministic for a fixed `(vectors, seed, p)` on one machine, AND
/// bit-identical across rayon pool sizes: the partition is a pure
/// function of the seed, the per-shard collect is index-ordered, the
/// refinement pass stages per-node results indexed by node id (fixed
/// order), and the reverse-edge pass is serial in ascending id order.
///
/// `p <= 1` (or a corpus too small to shard) delegates to the
/// single-pass [`build_vamana_with_params`] — the reference path stays
/// reachable.
pub fn build_vamana_partitioned(
    vectors: &[f32],
    n: usize,
    dim: usize,
    p: usize,
) -> (GraphAdjacency, u32) {
    build_vamana_partitioned_with_params(
        vectors,
        n,
        dim,
        p,
        GRAPH_DEGREE_R,
        GRAPH_BUILD_L,
        GRAPH_ALPHA,
        GRAPH_SEED,
    )
}

/// Parameterised form of [`build_vamana_partitioned`] (for tests /
/// tuning). See that function's doc for the algorithm.
#[allow(clippy::too_many_arguments)]
pub fn build_vamana_partitioned_with_params(
    vectors: &[f32],
    n: usize,
    dim: usize,
    p: usize,
    r: usize,
    l: usize,
    alpha: f32,
    seed: u64,
) -> (GraphAdjacency, u32) {
    debug_assert_eq!(vectors.len(), n * dim);
    // Degenerate / too-small-to-shard cases fall back to the
    // single-pass reference build (keeps it reachable + identical for
    // small corpora).
    if p <= 1 || n < 2 * GRAPH_MIN_SHARD_ROWS.max(1) || p > n {
        return build_vamana_with_params(vectors, n, dim, r, l, alpha, seed);
    }
    let r = r.max(1);

    // 1. Partition the deterministic shuffled insertion order into `p`
    //    contiguous shards. Same shuffle the single-pass build uses,
    //    so each shard is a uniform random sample of the corpus.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    // Contiguous, near-equal shard ranges over `order` (deterministic
    // split: shard s gets order[s*n/p .. (s+1)*n/p]).
    let shard_ranges: Vec<(usize, usize)> = (0..p)
        .map(|s| (s * n / p, (s + 1) * n / p))
        .filter(|&(a, b)| b > a)
        .collect();

    // 2. Build each shard's sub-graph in parallel (index-ordered
    //    collect -> deterministic). Each shard: gather its vectors
    //    into a local buffer, build in LOCAL id space, remap to
    //    global ids. The per-shard seed is derived from the base seed
    //    + shard index so shards don't share an RNG stream but the
    //    whole thing stays a pure function of `seed`.
    use rayon::prelude::*;
    let shard_graphs: Vec<(Vec<Vec<u32>>, Vec<u32>)> = shard_ranges
        .par_iter()
        .enumerate()
        .map(|(s, &(a, b))| {
            let global_ids: Vec<u32> = order[a..b].to_vec();
            let m = global_ids.len();
            // Gather this shard's vectors contiguously in local order.
            let mut local_vecs = vec![0.0f32; m * dim];
            for (li, &gid) in global_ids.iter().enumerate() {
                let src = gid as usize * dim;
                local_vecs[li * dim..(li + 1) * dim].copy_from_slice(&vectors[src..src + dim]);
            }
            let shard_seed = seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(s as u64 + 1));
            let (sub, _local_entry) =
                build_vamana_with_params(&local_vecs, m, dim, r, l, alpha, shard_seed);
            // Remap local -> global adjacency lists.
            let mut remapped: Vec<Vec<u32>> = (0..m)
                .map(|li| {
                    sub.neighbors_of(li)
                        .iter()
                        .map(|&lnb| global_ids[lnb as usize])
                        .collect()
                })
                .collect();
            for row in &mut remapped {
                row.sort_unstable();
            }
            (remapped, global_ids)
        })
        .collect();

    // 3a. Union the per-shard adjacencies into one global graph.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (sub, global_ids) in &shard_graphs {
        for (li, &gid) in global_ids.iter().enumerate() {
            // Each global id belongs to exactly one shard (disjoint
            // contiguous partition of `order`, which is a permutation
            // of 0..n), so this assignment never overwrites.
            adj[gid as usize] = sub[li].clone();
        }
    }
    drop(shard_graphs);

    // 3b. Global entry point = approx medoid over ALL vectors (same
    //     primitive the single-pass build uses).
    let entry = approx_medoid(vectors, n, dim);

    // 3c. Cross-shard refinement pass. For each node, greedy-search
    //     the MERGED graph from the global entry toward that node,
    //     collecting a visited set that now spans shards, then
    //     RobustPrune(node, V ∪ N_out(node)). Read-only over `adj`,
    //     each node's new list staged into a fixed id-indexed slot,
    //     then applied all at once -> order-independent, bit-identical
    //     across thread counts (same discipline as G-2c's per-hop
    //     batching + the plan's staged-write requirement).
    let refined: Vec<Vec<u32>> = (0..n as u32)
        .into_par_iter()
        .map(|node| {
            let query = &vectors[node as usize * dim..(node as usize + 1) * dim];
            let dist_to = |id: usize| sq_dist(query, &vectors[id * dim..(id + 1) * dim]);
            let dist_to_many = |ids: &[u32]| build_row_dists(query, ids, vectors, dim);
            let dist = |x: u32, y: u32| -> f32 {
                sq_dist(
                    &vectors[x as usize * dim..(x as usize + 1) * dim],
                    &vectors[y as usize * dim..(y as usize + 1) * dim],
                )
            };
            let dist_p_many = |x: u32, ids: &[u32]| {
                let arow = &vectors[x as usize * dim..(x as usize + 1) * dim];
                build_row_dists(arow, ids, vectors, dim)
            };
            let visited = greedy_search_collect_visited_via(
                entry,
                node as usize,
                l,
                &adj,
                dist_to,
                &dist_to_many,
            );
            let mut candidates: Vec<u32> =
                Vec::with_capacity(visited.len() + adj[node as usize].len());
            candidates.extend_from_slice(&visited);
            candidates.extend_from_slice(&adj[node as usize]);
            robust_prune_via(node, &candidates, &dist, &dist_p_many, alpha, r)
        })
        .collect();
    adj = refined;

    // 3d. Reverse-edge pass. For each node's selected out-edge q, the
    //     single-pass build adds a reverse edge q -> node (re-pruning
    //     q if it exceeds the degree bound). Done as a deterministic,
    //     PARALLEL two-step here so it never races the refinement pass
    //     and uses the cores:
    //     (i) collect every reverse-edge request (target q, source
    //         node) — parallel over source nodes, then group by target
    //         in a fixed (ascending source id within ascending target)
    //         order;
    //     (ii) for each target q, merge its existing out-list with its
    //         incoming reverse edges and re-prune to `r` if over the
    //         bound — parallel over targets (each target's list is
    //         independent), index-ordered collect => deterministic.
    //     This matches the single-pass reverse-edge semantics (a
    //     reverse edge is added, and the node is re-pruned only if it
    //     now exceeds `r`) while being order-independent.
    let mut reverse_reqs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for node in 0..n as u32 {
        for &q in &adj[node as usize] {
            reverse_reqs[q as usize].push(node);
        }
    }
    let updated: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|qi| {
            let incoming = &reverse_reqs[qi];
            if incoming.is_empty() {
                return adj[qi].clone();
            }
            // Merge existing out-edges + incoming reverse edges, dedup,
            // drop self-loops.
            let mut merged: Vec<u32> = adj[qi].clone();
            merged.extend_from_slice(incoming);
            merged.retain(|&x| x as usize != qi);
            merged.sort_unstable();
            merged.dedup();
            if merged.len() <= r {
                return merged;
            }
            // Over the bound: re-prune (same oracle as the build).
            let dist = |x: u32, y: u32| -> f32 {
                sq_dist(
                    &vectors[x as usize * dim..(x as usize + 1) * dim],
                    &vectors[y as usize * dim..(y as usize + 1) * dim],
                )
            };
            let dist_p_many = |x: u32, ids: &[u32]| {
                let arow = &vectors[x as usize * dim..(x as usize + 1) * dim];
                build_row_dists(arow, ids, vectors, dim)
            };
            robust_prune_via(qi as u32, &merged, &dist, &dist_p_many, alpha, r)
        })
        .collect();
    adj = updated;

    (GraphAdjacency::from_lists(adj), entry)
}

/// Phase G-2b incremental insert: add ONE new node (id `new_id ==
/// existing.n()`, i.e. the next slot) to an existing, already-built
/// [`GraphAdjacency`], using the exact same per-node Vamana step
/// (`insert_node_into_graph`) the bulk build loop runs for every
/// node. `vectors` must be `(existing.n() + 1) * dim` `f32` -- the
/// FULL corpus INCLUDING the new row appended at the end (the
/// caller, `insert.rs`, already has to hold this resident to encode
/// the new row via the same TurboQuant path every other kind's
/// `aminsert` uses, and `insert_node_into_graph`'s greedy search
/// needs random access to any existing row).
///
/// Returns the new [`GraphAdjacency`] with `n() == existing.n() + 1`
/// and, since the entry point never moves for an insert (Vamana's
/// entry point is a build-time medoid choice, not something later
/// inserts re-derive -- re-computing an exact medoid on every insert
/// would be `O(n)` per row, and an approximate one already-central
/// node stays a perfectly good entry point after one more row joins
/// the graph), `entry` unchanged from what the caller passed in.
///
/// See [`insert_node_into_graph`]'s doc comment for the determinism
/// caveat: this does not attempt to reproduce what a bulk build
/// would have produced had the new row been present from the start,
/// only a valid, navigable result.
///
/// Requires the WHOLE corpus's raw f32 vectors resident — the real
/// `aminsert` path never has that (see `insert_one_node_via_oracle`
/// below, which is what `insert.rs` actually calls). Kept as the
/// simpler reference implementation for tests/future callers that DO
/// have a resident corpus (e.g. a from-scratch offline rebuild tool);
/// `#[allow(dead_code)]` because no current caller besides its own
/// test needs it, but deleting a correct, tested, simpler-to-reason-
/// about reference implementation just to silence a lint would be a
/// net loss.
#[allow(dead_code)]
pub fn insert_one_node(
    existing: &GraphAdjacency,
    entry: u32,
    vectors: &[f32],
    dim: usize,
) -> GraphAdjacency {
    let old_n = existing.n();
    let new_n = old_n + 1;
    debug_assert_eq!(vectors.len(), new_n * dim);
    let mut adj = existing.to_lists();
    adj.push(Vec::new());
    let new_id = old_n as u32;
    if old_n == 0 {
        // First-ever row: nothing to search from / connect to yet.
        // Leave it edgeless, matching build_vamana's own n==1 case.
        return GraphAdjacency::from_lists(adj);
    }
    insert_node_into_graph(
        &mut adj,
        entry,
        new_id,
        vectors,
        dim,
        GRAPH_BUILD_L,
        GRAPH_ALPHA,
        GRAPH_DEGREE_R,
    );
    GraphAdjacency::from_lists(adj)
}

/// Real `aminsert` entry point (Phase G-2b): insert ONE new node into
/// an existing, ALREADY-PERSISTED graph, where the existing corpus's
/// raw f32 vectors are **not available** — only their persisted
/// TurboQuant-quantized codes are (the graph kind never keeps a
/// whole-corpus f32 buffer resident outside of `ambuild`; see the
/// module doc's "RAM residency" section). The new row's raw f32
/// vector IS available (the caller just received it from `aminsert`).
///
/// `score_existing` must score the new vector against a BATCH of
/// existing slot ids using the SAME quantized-distance kernel the
/// scan path already uses (`ReadOnlyIndex::score_slots` via
/// `cache::GraphIndex`, wired by `insert.rs`) — higher score = closer
/// (matches `graph_search`'s convention), so this function negates
/// internally to get an ascending-distance ordering consistent with
/// [`insert_node_into_graph`]'s `sq_dist`-based (lower = closer)
/// convention.
///
/// **Known, deliberate approximation** (document this clearly, it is
/// the one real accuracy gap versus the build path's exact
/// `RobustPrune`): the diversity condition `alpha * dist(p*, p') <=
/// dist(p, p')` needs a distance BETWEEN TWO EXISTING candidates
/// (`p*` and `p'`), not just to the new node `p`. There is no
/// reconstruct-from-quantized-codes primitive in `turbovec` to get
/// an exact (or even approximate) `p*`-vs-`p'` distance cheaply, and
/// adding one is out of scope for G-2b (it would need a new codebook
/// decode primitive upstream or in this crate, plus its own
/// round-trip test suite). Rather than block incremental insert on
/// that, this function approximates `dist(p*, p')` using the
/// difference between EACH candidate's own already-computed
/// distance-to-`p` (a real, exact, cheap quantity from
/// `score_existing`): `|dist(p, p*) - dist(p, p')|` is a valid lower
/// bound on `dist(p*, p')` by the triangle inequality, and using a
/// lower bound in place of the true value can only make the pruning
/// rule LESS aggressive (a candidate that would have been pruned by
/// the exact distance might survive here), never MORE aggressive —
/// so this never drops a genuinely-diverse edge, it can only
/// occasionally keep an edge the exact algorithm would have pruned
/// as redundant. Net effect: the new node's out-degree may run
/// slightly higher than perfectly diverse RobustPrune would produce
/// (still hard-capped at `r`), not lower — a safe direction to be
/// wrong in for a navigability-critical structure. This is a
/// genuinely different (weaker) guarantee than the build path's
/// exact RobustPrune and should be revisited if a codebook-decode
/// primitive becomes available.
pub fn insert_one_node_via_oracle<S>(
    existing: &GraphAdjacency,
    entry: u32,
    new_vector: &[f32],
    score_existing: S,
) -> GraphAdjacency
where
    S: Fn(&[f32], &[u32]) -> Vec<f32>,
{
    let old_n = existing.n();
    let new_id = old_n as u32;
    let mut adj = existing.to_lists();
    adj.push(Vec::new());
    if old_n == 0 {
        // First-ever row: nothing to search from / connect to yet.
        return GraphAdjacency::from_lists(adj);
    }

    // dist_to(existing_slot) -- exact: new_vector is a real f32 row,
    // score_existing scores it against real persisted quantized
    // codes via the same kernel the scan path trusts.
    let dist_to = |id: usize| -> f32 {
        let scores = score_existing(new_vector, &[id as u32]);
        -scores[0] // score_existing: higher = closer; sq_dist convention: lower = closer.
    };
    // dist(a, b) for RobustPrune's diversity check: if either side is
    // the new node `p`, this IS exact (same as dist_to). If BOTH
    // sides are existing nodes, approximate via the triangle-
    // inequality lower bound documented above.
    let dist = |a: u32, b: u32| -> f32 {
        if a == new_id {
            return dist_to(b as usize);
        }
        if b == new_id {
            return dist_to(a as usize);
        }
        (dist_to(a as usize) - dist_to(b as usize)).abs()
    };

    // Phase G-2c batch oracles for the shared core. The oracle
    // (incremental-insert) path runs a SINGLE node insert per row —
    // not the bulk-build hot path — so it scores its batches serially
    // (mapping the scalar oracles above); parallelism here would only
    // add rayon dispatch overhead for one row's worth of work. Kept
    // serial keeps this path's behaviour byte-for-byte what it was
    // before G-2c (the oracle path was never a determinism-across-
    // thread-count concern — its result already depends on insertion
    // timing, per `insert_node_into_graph`'s doc comment).
    let dist_to_many = |ids: &[u32]| ids.iter().map(|&id| dist_to(id as usize)).collect();
    let dist_p_many = |a: u32, ids: &[u32]| ids.iter().map(|&b| dist(a, b)).collect();

    insert_node_into_graph_via(
        &mut adj,
        entry,
        new_id,
        dist_to,
        dist,
        dist_to_many,
        dist_p_many,
        GRAPH_ALPHA,
        GRAPH_DEGREE_R,
        GRAPH_BUILD_L,
    );
    GraphAdjacency::from_lists(adj)
}

/// Approximate medoid: the mean vector, then the actual corpus point
/// nearest it (`O(n * dim)`, fixed-order summation — deterministic).
/// Not the EXACT medoid (which would be `O(n^2 * dim)` — too
/// expensive to justify for an entry point that only needs to be
/// "roughly central", not optimal); ties broken toward the lower id.
fn approx_medoid(vectors: &[f32], n: usize, dim: usize) -> u32 {
    let mut mean = vec![0.0f32; dim];
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        for d in 0..dim {
            mean[d] += row[d];
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in &mut mean {
        *m *= inv_n;
    }
    let mut best = 0u32;
    let mut best_d = f32::INFINITY;
    for i in 0..n {
        let d = sq_dist(&mean, &vectors[i * dim..(i + 1) * dim]);
        if d < best_d {
            best_d = d;
            best = i as u32;
        }
    }
    best
}

/// Insert ONE node `p` into an in-progress (or already-complete)
/// adjacency `adj`, using Vamana's own per-node insertion step:
/// greedy-search from `entry` toward `p`'s vector to collect a
/// candidate/visited set, `RobustPrune` that set down to `p`'s
/// out-edges, then add the reverse edge for each new neighbor
/// (re-pruning that neighbor's own list if it now exceeds the degree
/// bound `r`).
///
/// This is the EXACT per-node body [`build_vamana_with_params`]'s
/// main loop already runs for every node during a bulk build --
/// extracted here so it has exactly one implementation, called once
/// per node during a build (via that loop) and once for a single new
/// row during an incremental `aminsert` (Phase G-2b, see
/// `insert.rs`). `adj[p as usize]` must already be sized (an empty
/// `Vec::new()` slot is fine, as it is throughout the build loop) --
/// callers on the incremental path must `push(Vec::new())` for the
/// new node before calling this.
///
/// Determinism note: this does NOT claim to produce the same graph a
/// node would have gotten had it been present at build time -- the
/// build's insertion ORDER matters (each node's greedy search
/// navigates whatever partial graph existed before it), so a node
/// inserted incrementally, after the graph is already built, sees a
/// materially different (already-complete) graph than it would have
/// seen mid-build. It only guarantees a VALID, navigable result: `p`
/// gets up to `r` real out-edges found by greedy search + RobustPrune
/// over the CURRENT graph, and every neighbor that gains a reverse
/// edge to `p` stays within the degree bound. No byte-determinism
/// guarantee is made or needed here (the module doc's determinism
/// section is about a fixed BUILD being reproducible, not about
/// incremental insert matching a hypothetical bulk build).
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_node_into_graph(
    adj: &mut [Vec<u32>],
    entry: u32,
    p: u32,
    vectors: &[f32],
    dim: usize,
    l: usize,
    alpha: f32,
    r: usize,
) {
    let dist = |a: u32, b: u32| -> f32 {
        sq_dist(
            &vectors[a as usize * dim..(a as usize + 1) * dim],
            &vectors[b as usize * dim..(b as usize + 1) * dim],
        )
    };
    let query = &vectors[p as usize * dim..(p as usize + 1) * dim];
    let dist_to = |id: usize| sq_dist(query, &vectors[id * dim..(id + 1) * dim]);
    // Phase G-2c: batch oracles that score a node's candidate set in
    // one pass via `build_row_dists` (plain `sq_dist`). The batch
    // shape lets greedy search / RobustPrune hand a whole hop's
    // neighbor list to one call; the reduction order is fixed, so the
    // resulting adjacency is bit-identical regardless of thread count
    // — see `build_row_dists`'s doc for why the build is deliberately
    // serial and uses plain (already auto-vectorised) `sq_dist`
    // rather than rayon-fanned or hand-blocked (both measured
    // regressions).
    let dist_to_many = |ids: &[u32]| build_row_dists(query, ids, vectors, dim);
    let dist_p_many = |a: u32, ids: &[u32]| {
        let arow = &vectors[a as usize * dim..(a as usize + 1) * dim];
        build_row_dists(arow, ids, vectors, dim)
    };
    insert_node_into_graph_via(
        adj,
        entry,
        p,
        dist_to,
        dist,
        dist_to_many,
        dist_p_many,
        alpha,
        r,
        l,
    );
}

/// Score `query` against each `ids[i]`'th row of `vectors` (squared
/// L2 via [`sq_dist`]), returning distances in the SAME order as
/// `ids`.
///
/// ## Why this is serial and uses plain `sq_dist` (Phase G-2c findings)
///
/// Two G-2c build-path optimisations were implemented, MEASURED, and
/// REJECTED as regressions on this hardware — both recorded here so
/// they are not re-attempted:
///
/// 1. **Rayon-parallel per-node candidate batches** (fanning these
///    distance evals across the bounded pool the way `ivf.rs`'s
///    k-means fans its GEMM). Made bit-identical across thread counts
///    (order-preserving indexed collect over fixed-order evals), but
///    it was a REGRESSION (measured 0.68–0.85× on an 8-core box at
///    dim=512): single-pass Vamana produces its distance evals in
///    small per-hop batches bounded by the build beam width `L` (~64
///    candidates × `dim` coords, microseconds of work), so rayon's
///    per-batch fork-join dispatch cost dominates. The insertions are
///    also strictly sequentially dependent (each node's greedy search
///    navigates the graph every prior insertion left), so there is no
///    COARSE independent unit to parallelise either — unlike k-means,
///    whose whole assignment step is one big data-parallel GEMM.
///
/// 2. **A hand-rolled 8-lane `sq_dist_blocked` kernel** (fixed-lane
///    accumulator, intended as a SIMD-friendly build distance). Also
///    a REGRESSION (measured ~0.72× at dim=512): LLVM already
///    auto-vectorises the plain `sq_dist` loop over contiguous f32
///    optimally, so the manual lane-blocking only added indexing
///    overhead. Removed; the build uses plain `sq_dist`.
///
/// The REAL G-2c win is on the SCAN path, not the build: the build
/// scores raw f32 (already optimal scalar-autovectorised), whereas
/// the scan scores QUANTIZED codes, where TurboQuant's LUT
/// table-lookup primitive genuinely beats the old per-hop full-buffer
/// rescan — see `cache::GraphScorer`. Per AGENTS.md's guidance
/// ("correctness/determinism beats speed"), the build stays
/// plain-scalar-serial and bit-identical. The bounded-pool plumbing
/// (`build_pool::install` around `build_vamana`) is retained so
/// turbovec's OWN internal rayon fan-out (the `add_with_ids`
/// quantize step alongside the graph build in `build.rs`) stays
/// confined to the GUC-sized pool — that part genuinely amortises.
fn build_row_dists(query: &[f32], ids: &[u32], vectors: &[f32], dim: usize) -> Vec<f32> {
    ids.iter()
        .map(|&id| {
            let s = id as usize * dim;
            sq_dist(query, &vectors[s..s + dim])
        })
        .collect()
}

/// Generalized (over a distance oracle) core of
/// [`insert_node_into_graph`]: greedy-search from `entry` toward `p`
/// (scored via `dist_to`, a function of a candidate node id), then
/// `RobustPrune` the visited set + `p`'s current out-edges (scored
/// via `dist`, a function of TWO node ids — needed because
/// `RobustPrune`'s diversity condition compares distances BETWEEN
/// candidates, not just to `p`) down to `p`'s real out-edges, then
/// add + re-prune reverse edges exactly as [`insert_node_into_graph`]
/// already did. Split into two oracles (rather than one) because the
/// incremental-insert caller ([`insert_one_node_via_oracle`]) has a
/// cheap way to score "the new vector vs an existing slot" (the new
/// vector is a real, resident f32 row; the existing slot's distance
/// comes from the persisted quantized codes via `score_slots`) but
/// no way to compute "existing slot A vs existing slot B" any
/// cheaper than the SAME per-slot quantized scoring the scan path
/// already does — both oracles end up calling into the same
/// quantized-score machinery for existing-vs-existing comparisons,
/// they're just shaped differently for the two comparison kinds this
/// algorithm needs.
///
/// Phase G-2c adds two BATCH oracles alongside the two scalar ones:
/// `dist_to_many(ids) -> Vec<f32>` (batch of "the query node `p` vs
/// each existing id", used per greedy-search hop) and
/// `dist_p_many(a, ids) -> Vec<f32>` (batch of "node `a` vs each id",
/// used for RobustPrune's initial candidate scoring). The build path
/// supplies rayon-parallel batch oracles over the resident `vectors`
/// buffer (`build_dist_closures`); the incremental-insert path
/// supplies serial ones. Batching is where all the parallelism lives
/// — the greedy-search beam order and the prune selection order are
/// unchanged, and the pairing of scores to ids is by fixed index, so
/// the resulting adjacency is bit-identical regardless of thread
/// count (the whole point of the G-2c determinism contract).
#[allow(clippy::too_many_arguments)]
fn insert_node_into_graph_via<F, G, TB, PB>(
    adj: &mut [Vec<u32>],
    entry: u32,
    p: u32,
    dist_to: F,
    dist: G,
    dist_to_many: TB,
    dist_p_many: PB,
    alpha: f32,
    r: usize,
    l: usize,
) where
    F: Fn(usize) -> f32,
    G: Fn(u32, u32) -> f32,
    TB: Fn(&[u32]) -> Vec<f32>,
    PB: Fn(u32, &[u32]) -> Vec<f32>,
{
    // 1. Greedy search from the entry point toward p, beam l,
    //    collecting the visited (expanded) node set V.
    let visited =
        greedy_search_collect_visited_via(entry, p as usize, l, adj, dist_to, &dist_to_many);

    // 2. RobustPrune(p, V ∪ N_out(p), alpha, r).
    let mut candidates: Vec<u32> = Vec::with_capacity(visited.len() + adj[p as usize].len());
    candidates.extend_from_slice(&visited);
    candidates.extend_from_slice(&adj[p as usize]);
    let selected = robust_prune_via(p, &candidates, &dist, &dist_p_many, alpha, r);
    adj[p as usize] = selected.clone();

    // 3. Add p -> selected edges' reverses (q -> p), re-pruning q
    //    if that pushes it over the degree bound.
    for &q in &selected {
        let qi = q as usize;
        if adj[qi].contains(&p) {
            continue;
        }
        adj[qi].push(p);
        if adj[qi].len() > r {
            let q_candidates = std::mem::take(&mut adj[qi]);
            adj[qi] = robust_prune_via(q, &q_candidates, &dist, &dist_p_many, alpha, r);
        }
    }
}

/// Build-time greedy search from `entry` toward the vector at
/// `query_id` (an existing corpus row — during the build the "query"
/// for node `p`'s own insertion step IS `p` itself), beam width `l`,
/// navigating the IN-PROGRESS adjacency `adj`. Returns the visited
/// (expanded) node set `V` that `robust_prune` draws its candidate
/// pool from — NOT just the top-`l` result list (DiskANN's
/// `GreedySearch` returns both; the build only needs `V`, which is a
/// superset of the top-`l` list by construction).
///
/// Same beam-search shape as `ivf::graph_probe` (bounded max-heap of
/// current best results + min-heap of unvisited candidates to
/// expand, stopping once the closest remaining candidate can't beat
/// the current worst kept result) but over `dim`-dimensional corpus
/// vectors instead of coarse centroids, and returning the expansion
/// set rather than the top-k.
fn greedy_search_collect_visited_via<D, B>(
    entry: u32,
    query_id: usize,
    l: usize,
    adj: &[Vec<u32>],
    dist_to: D,
    dist_to_many: B,
) -> Vec<u32>
where
    D: Fn(usize) -> f32,
    B: Fn(&[u32]) -> Vec<f32>,
{
    let n = adj.len();

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
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(self.1.cmp(&other.1))
        }
    }

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut visited_flag = vec![false; n];
    let entry_us = entry as usize;
    visited_flag[entry_us] = true;
    let entry_dist = dist_to(entry_us);

    let mut to_visit: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
    to_visit.push(Reverse(Cand(entry_dist, entry)));
    let mut results: BinaryHeap<Cand> = BinaryHeap::new();
    results.push(Cand(entry_dist, entry));

    let mut visited_list: Vec<u32> = Vec::new();

    while let Some(Reverse(cur)) = to_visit.pop() {
        if results.len() >= l {
            if let Some(worst) = results.peek() {
                if cur.0 >= worst.0 {
                    break;
                }
            }
        }
        visited_list.push(cur.1);
        // Phase G-2c: collect this hop's not-yet-visited neighbors and
        // score them in ONE batch call (`dist_to_many`) rather than one
        // `dist_to` at a time. The batch scorer is where the build path
        // fans the per-neighbor `sq_dist` evals across the bounded rayon
        // pool (see `par_dist_to_many`); the oracle/insert path passes a
        // serial batch scorer. The beam-expansion ORDER is unchanged
        // (each `cur` still expands its whole neighbor list before the
        // next `to_visit.pop()`), and the scores fed into the heaps are
        // in the SAME fixed neighbor-list order regardless of thread
        // count, so the resulting `visited_list` (and thus the whole
        // build) is bit-identical to the serial path. The `visited_flag`
        // is set up front for the whole batch, matching the serial
        // "mark-then-score" order exactly.
        let batch: Vec<u32> = adj[cur.1 as usize]
            .iter()
            .copied()
            .filter(|&nb| {
                let nbi = nb as usize;
                // Skip the query node itself: during node p's own build
                // turn, p may already appear in another node's adjacency
                // (a reverse edge from an earlier iteration). Letting p
                // occupy a beam slot for a search whose target IS p
                // wastes capacity for no benefit (RobustPrune excludes p
                // from its own candidate pool regardless); this is a
                // minor efficiency guard, not load-bearing for
                // correctness.
                if nbi == query_id || visited_flag[nbi] {
                    return false;
                }
                visited_flag[nbi] = true;
                true
            })
            .collect();
        if batch.is_empty() {
            continue;
        }
        let dists = dist_to_many(&batch);
        debug_assert_eq!(dists.len(), batch.len());
        for (&nb, &d) in batch.iter().zip(dists.iter()) {
            if results.len() < l {
                results.push(Cand(d, nb));
                to_visit.push(Reverse(Cand(d, nb)));
            } else {
                let worse_than_worst = results.peek().is_some_and(|w| d >= w.0);
                if !worse_than_worst {
                    results.pop();
                    results.push(Cand(d, nb));
                    to_visit.push(Reverse(Cand(d, nb)));
                }
            }
        }
    }

    // `dist_to` is retained in the signature for the entry-point score
    // above; silence the unused-in-some-monomorphisations lint without
    // dropping the parameter (both callers still need the scalar form
    // for the single entry-point evaluation).
    let _ = &dist_to;
    visited_list
}

/// `RobustPrune(p, candidates, alpha, r)` per the DiskANN paper's
/// Algorithm 2: greedily select up to `r` neighbors for `p` from
/// `candidates` (`p` itself is excluded; duplicates are deduped),
/// each time picking the closest remaining candidate `p*` and then
/// discarding every remaining candidate `p'` for which `alpha *
/// dist(p*, p') <= dist(p, p')` — `p*` already "covers" `p'` well
/// enough (within a factor of `alpha`) that keeping both as `p`'s
/// neighbors would be redundant. This diversity condition is what
/// gives Vamana long-range edges instead of a naive k-nearest-
/// neighbour clique (a plain k-NN graph tends to cluster edges
/// locally and needs many more hops to cross the space; RobustPrune
/// deliberately keeps a distant, "covering" edge over a second
/// nearby one once one candidate near that direction is already
/// selected).
///
/// NOTE on fidelity to the paper: the condition implemented here is
/// `alpha * dist(p*, p') <= dist(p, p')` (distance from the
/// just-selected node `p*` to the candidate, compared against the
/// candidate's distance to `p`) — this is the actual DiskANN
/// Algorithm 2 pruning rule. Please verify this against the paper's
/// pseudocode / a reference implementation before trusting it
/// blindly; it is the crux of why Vamana outperforms a naive k-NN
/// graph and the one piece of this module most worth double-
/// checking independently.
///
/// Generalized over a `dist(a, b)` oracle (rather than reading raw
/// `vectors` directly) so this ONE implementation serves both the
/// build path (oracle = exact `sq_dist` against resident f32 rows)
/// and the incremental-insert path (oracle = an approximate distance
/// derived from PERSISTED QUANTIZED CODES for existing rows, since
/// `aminsert` never has the whole corpus's raw f32 resident — see
/// `insert_one_node_via_oracle` below).
fn robust_prune_via<D, B>(
    p: u32,
    candidates: &[u32],
    dist: D,
    dist_p_many: B,
    alpha: f32,
    r: usize,
) -> Vec<u32>
where
    D: Fn(u32, u32) -> f32,
    B: Fn(u32, &[u32]) -> Vec<f32>,
{
    let mut cand: Vec<u32> = candidates.iter().copied().filter(|&c| c != p).collect();
    cand.sort_unstable();
    cand.dedup();
    if cand.is_empty() {
        return Vec::new();
    }

    // (dist_to_p, id) working list, repeatedly filtered as
    // candidates get pruned by the diversity condition. Phase G-2c:
    // the initial `dist(p, c)` batch (the largest independent set of
    // distance evals in the whole prune) is scored via `dist_p_many`,
    // where the build path fans it across the bounded rayon pool. The
    // pairing with `cand` is by fixed index, so `remaining` is
    // identical regardless of thread count.
    let cand_dists = dist_p_many(p, &cand);
    debug_assert_eq!(cand_dists.len(), cand.len());
    let mut remaining: Vec<(f32, u32)> = cand_dists.into_iter().zip(cand.iter().copied()).collect();

    let mut selected: Vec<u32> = Vec::with_capacity(r);
    while !remaining.is_empty() && selected.len() < r {
        // p* = the closest remaining candidate to p (ties -> lower
        // id, deterministic).
        remaining.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let (_, pstar) = remaining[0];
        selected.push(pstar);
        remaining.retain(|&(d_p_pprime, pprime)| {
            if pprime == pstar {
                return false;
            }
            let d_pstar_pprime = dist(pstar, pprime);
            !(alpha * d_pstar_pprime <= d_p_pprime)
        });
    }
    selected
}

/// Build-time wrapper of [`robust_prune_via`]: the distance oracle is
/// exact raw-f32 `sq_dist` against the resident `vectors` buffer.
/// Only reachable via [`insert_one_node`]/[`insert_node_into_graph`]
/// now (the build path itself uses [`robust_prune_via`] directly) --
/// see [`insert_one_node`]'s doc comment for why it's kept.
#[allow(dead_code)]
fn robust_prune(
    p: u32,
    candidates: &[u32],
    vectors: &[f32],
    dim: usize,
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    let dist = |a: u32, b: u32| -> f32 {
        sq_dist(
            &vectors[a as usize * dim..(a as usize + 1) * dim],
            &vectors[b as usize * dim..(b as usize + 1) * dim],
        )
    };
    let dist_p_many = |a: u32, ids: &[u32]| {
        let arow = &vectors[a as usize * dim..(a as usize + 1) * dim];
        build_row_dists(arow, ids, vectors, dim)
    };
    robust_prune_via(p, candidates, dist, dist_p_many, alpha, r)
}

/// Scan-time greedy beam search over a persisted [`GraphAdjacency`],
/// navigating via a caller-supplied BATCH scoring oracle. Mirrors
/// `ivf::graph_probe`'s beam-search shape (bounded max-heap of
/// current best results + heap of unvisited candidates to expand,
/// stopping once the best remaining candidate can no longer improve
/// the kept set) but flipped to turbovec's NATIVE similarity
/// convention — **HIGHER score = closer/more similar** (matching
/// `TurboQuantIndex::search`'s `SearchResults.scores`) — so callers
/// (the AM scan path in `cache.rs`) can hand the returned scores
/// straight to the EXISTING `amgettuple` `(1.0 - score)` distance
/// conversion with no extra sign-translation layer.
///
/// `score_batch` scores a BATCH of candidate slot ids in one call
/// (a node's whole unvisited-neighbor set per hop) rather than one
/// id at a time, so a turbovec-backed caller can build ONE
/// `search_masked` mask per hop instead of one per candidate — this
/// is what keeps the per-hop cost proportional to "one masked search
/// over the whole index" rather than "one masked search per
/// candidate". Returns the scores in the SAME order as the input ids.
///
/// Returns the `k` best (score DESCENDING, ties → ascending id for
/// determinism) `(score, id)` pairs found. `ef` (the internal beam
/// width) is `(k * GRAPH_SCAN_EF_MULTIPLIER).max(GRAPH_SCAN_EF_FLOOR)`,
/// clamped to `n`.
///
/// Deterministic: a fixed serial traversal with a fixed `(score, id)`
/// tie-break at every heap comparison — same idea as `ivf::graph_probe`.
///
/// `tombstones` is the Phase E-2/G-2b per-slot bitmap (LSB-first,
/// bit set ⇒ slot dead) read by the caller via
/// `relfile::read_tombstones` — an empty slice means "nothing has
/// been vacuumed" (every id is live), matching the convention
/// `scan.rs`'s `apply_tombstones` already uses for the IVF path. A
/// tombstoned node is treated as if it had been deleted from the
/// graph entirely: it is never added to `results`, and once
/// discovered as a neighbor of some live node it is dropped rather
/// than pushed onto `to_visit` — so its own out-edges are never
/// followed either (a dead node can't be `cur`, since only ids that
/// passed the tombstone check ever enter the heap). If the persisted
/// `entry` itself is tombstoned (VACUUM normally re-points
/// `graph_entry_point` at a fallback live slot when this happens —
/// see `vacuum.rs`'s `graph_tombstone_dead` — but this is a defensive
/// second layer for any caller that hands in a stale/foreign entry),
/// fall back to the first non-tombstoned id in ascending order; if
/// every id is dead, returns an empty result rather than panicking.
pub fn graph_search<F>(
    adjacency: &GraphAdjacency,
    entry: u32,
    k: usize,
    tombstones: &[u8],
    mut score_batch: F,
) -> Vec<(f32, u32)>
where
    F: FnMut(&[u32]) -> Vec<f32>,
{
    let n = adjacency.n();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let is_dead = |id: u32| -> bool {
        if tombstones.is_empty() {
            return false;
        }
        let slot = id as usize;
        let byte = slot / 8;
        byte < tombstones.len() && (tombstones[byte] >> (slot % 8)) & 1 != 0
    };
    let ef = (k.saturating_mul(GRAPH_SCAN_EF_MULTIPLIER))
        .max(GRAPH_SCAN_EF_FLOOR)
        .min(n);
    let entry = (entry as usize).min(n - 1) as u32;
    // Defense in depth (the primary fix lives in VACUUM's own
    // fallback selection, `vacuum.rs::graph_tombstone_dead` -- this
    // is a second layer in case any future write path ever installs
    // an entry point without re-checking this): an entry point that
    // is ALIVE but has zero LIVE out-neighbors is just as useless as
    // a dead one -- the very first hop's expansion yields nothing,
    // and the search returns (at best) only the entry point itself.
    // A low-degree entry point whose only out-edge points at a slot
    // that gets tombstoned is a genuine dead-end; fall back to a
    // live node that still has a live neighbor.
    let entry_has_live_neighbor = |id: u32| -> bool {
        adjacency
            .neighbors_of(id as usize)
            .iter()
            .any(|&nb| !is_dead(nb))
    };
    let entry = if is_dead(entry) || !entry_has_live_neighbor(entry) {
        match (0..n as u32).find(|&id| !is_dead(id) && entry_has_live_neighbor(id)) {
            Some(fallback) => fallback,
            // No live node has a live neighbor (a maximally
            // fragmented graph) -- fall back further to just "any
            // live node", matching the pre-existing degenerate-case
            // behaviour rather than returning empty prematurely.
            None => match (0..n as u32).find(|&id| !is_dead(id)) {
                Some(fallback) => fallback,
                None => return Vec::new(), // every id tombstoned
            },
        }
    } else {
        entry
    };

    // (score, id) ordering: ascending score, ties -> ascending id
    // (deterministic, no NaN-fallback surprises). Plain `Cand` is a
    // max-heap-by-score on `BinaryHeap` (pops HIGHEST score first --
    // what `to_visit` needs, since higher score = closer); wrapping
    // in `Reverse` flips it into a min-heap-by-score (pops LOWEST
    // score first -- what `results` needs, to cheaply evict the
    // current worst-kept candidate).
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
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(self.1.cmp(&other.1))
        }
    }

    use std::collections::BinaryHeap;

    let entry_score = score_batch(&[entry])[0];

    let mut visited = vec![false; n];
    visited[entry as usize] = true;

    let mut to_visit: BinaryHeap<Cand> = BinaryHeap::new();
    to_visit.push(Cand(entry_score, entry));

    use std::cmp::Reverse;
    let mut results: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
    results.push(Reverse(Cand(entry_score, entry)));

    while let Some(cur) = to_visit.pop() {
        // Stopping condition: once the result set is full (`ef`) and
        // the best remaining candidate to expand is no better than
        // our current worst kept result, no further expansion can
        // improve the result set. Standard HNSW/Vamana stopping rule
        // (see `ivf::graph_probe`'s doc for the same rule in the
        // opposite (ascending-distance) direction).
        if results.len() >= ef {
            if let Some(Reverse(worst)) = results.peek() {
                if cur.0 <= worst.0 {
                    break;
                }
            }
        }
        let mut new_ids: Vec<u32> = Vec::new();
        for &nb in adjacency.neighbors_of(cur.1 as usize) {
            // A tombstoned neighbor contributes nothing: it can never
            // be returned and its own out-edges must never be
            // followed. Marking it `visited` (without adding it to
            // `new_ids`) is enough to guarantee both -- it never
            // enters `results`/`to_visit`, so it never becomes `cur`
            // and its out-edges are never walked.
            if !visited[nb as usize] {
                visited[nb as usize] = true;
                if !is_dead(nb) {
                    new_ids.push(nb);
                }
            }
        }
        if new_ids.is_empty() {
            continue;
        }
        let scores = score_batch(&new_ids);
        debug_assert_eq!(scores.len(), new_ids.len());
        for (&nb, &d) in new_ids.iter().zip(scores.iter()) {
            if results.len() < ef {
                results.push(Reverse(Cand(d, nb)));
                to_visit.push(Cand(d, nb));
            } else {
                let worse_than_worst = results.peek().is_some_and(|Reverse(w)| d <= w.0);
                if !worse_than_worst {
                    results.pop();
                    results.push(Reverse(Cand(d, nb)));
                    to_visit.push(Cand(d, nb));
                }
                // else: `d` can't improve the kept set and (being
                // worse than the current worst kept) also can't beat
                // the stopping check above once popped -- skip adding
                // to `to_visit`, bounding the heap's growth.
            }
        }
    }

    let mut out: Vec<(f32, u32)> = results.into_iter().map(|Reverse(c)| (c.0, c.1)).collect();
    // Descending score (closest/most-similar first), ties -> ascending id.
    out.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random unit-ish vectors for tests (not
    /// actually L2-normalised -- these tests only check adjacency
    /// STRUCTURE, not recall, so raw synthetic coordinates are fine).
    fn synth_corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        use rand::Rng;
        (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn empty_corpus_builds_empty_graph() {
        let (g, entry) = build_vamana(&[], 0, 8);
        assert_eq!(g.n(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(entry, 0);
    }

    #[test]
    fn single_node_has_no_edges() {
        let v = vec![1.0f32; 8];
        let (g, entry) = build_vamana(&v, 1, 8);
        assert_eq!(g.n(), 1);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(entry, 0);
        assert_eq!(g.neighbors_of(0), &[] as &[u32]);
    }

    #[test]
    fn two_nodes_connect_to_each_other() {
        let mut v = vec![0.0f32; 16];
        v[0] = 1.0;
        v[8] = -1.0;
        let (g, _entry) = build_vamana_with_params(&v, 2, 8, 32, 64, 1.2, GRAPH_SEED);
        assert_eq!(g.neighbors_of(0), &[1]);
        assert_eq!(g.neighbors_of(1), &[0]);
    }

    #[test]
    fn every_node_has_at_least_one_neighbor_on_a_real_corpus() {
        // A classic Vamana-navigability sanity check: with R >= 1 and
        // a genuinely connected build (every node participates in at
        // least one greedy-search + reverse-edge round), no node
        // should end up totally isolated once n is comfortably above
        // R+1.
        let n = 200;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 7);
        let (g, _entry) = build_vamana(&corpus, n, dim);
        for i in 0..n {
            assert!(
                !g.neighbors_of(i).is_empty(),
                "node {i} has no out-neighbors -- navigability gap"
            );
        }
    }

    #[test]
    fn max_out_degree_is_bounded_by_r() {
        let n = 300;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 11);
        let r = 16;
        let (g, _entry) = build_vamana_with_params(&corpus, n, dim, r, r * 2, 1.2, GRAPH_SEED);
        for i in 0..n {
            assert!(
                g.neighbors_of(i).len() <= r,
                "node {i} has out-degree {} > R={r}",
                g.neighbors_of(i).len()
            );
        }
    }

    #[test]
    fn no_self_loops() {
        let n = 150;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 3);
        let (g, _entry) = build_vamana(&corpus, n, dim);
        for i in 0..n {
            assert!(
                !g.neighbors_of(i).contains(&(i as u32)),
                "node {i} has a self-loop"
            );
        }
    }

    #[test]
    fn neighbor_ids_are_ascending_and_deduplicated() {
        let n = 150;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 99);
        let (g, _entry) = build_vamana(&corpus, n, dim);
        for i in 0..n {
            let nbrs = g.neighbors_of(i);
            for w in nbrs.windows(2) {
                assert!(
                    w[0] < w[1],
                    "node {i}'s neighbor list is not strictly ascending"
                );
            }
        }
    }

    /// Determinism contract: same input (corpus + seed) -> byte-
    /// identical `GraphAdjacency`, on one machine/thread-count. The
    /// build here is fully serial, so this is close to a tautology,
    /// but it pins the contract explicitly (and would catch e.g. an
    /// accidental HashMap-iteration-order dependency).
    #[test]
    fn build_is_deterministic_for_a_fixed_seed() {
        let n = 250;
        let dim = 20;
        let corpus = synth_corpus(n, dim, 42);
        let (g1, e1) = build_vamana(&corpus, n, dim);
        let (g2, e2) = build_vamana(&corpus, n, dim);
        assert_eq!(g1, g2);
        assert_eq!(e1, e2);
    }

    #[test]
    fn different_seeds_can_produce_different_graphs() {
        // Not a hard requirement, but a sanity check that the seed
        // actually feeds the insertion-order RNG (rather than being
        // silently ignored).
        let n = 250;
        let dim = 20;
        let corpus = synth_corpus(n, dim, 42);
        let (g1, _) = build_vamana_with_params(&corpus, n, dim, 32, 64, 1.2, 1);
        let (g2, _) = build_vamana_with_params(&corpus, n, dim, 32, 64, 1.2, 2);
        assert_ne!(
            g1, g2,
            "different seeds produced identical graphs (seed not wired through?)"
        );
    }

    #[test]
    fn adjacency_encode_decode_round_trip() {
        let n = 100;
        let dim = 10;
        let corpus = synth_corpus(n, dim, 5);
        let (g, _entry) = build_vamana(&corpus, n, dim);
        let offsets_bytes = g.encode_offsets();
        let neighbors_bytes = g.encode_neighbors();
        let back = GraphAdjacency::decode(&offsets_bytes, &neighbors_bytes, n).expect("decode");
        assert_eq!(g, back);
    }

    #[test]
    fn adjacency_decode_rejects_length_mismatch() {
        let n = 10;
        let offsets_bytes = vec![0u8; (n + 1) * 4];
        let bad_neighbors_bytes = vec![0u8; 4]; // claims 1 edge but offsets say 0
                                                // offsets are all-zero so offsets[n] == 0, and neighbors_bytes
                                                // has 1 entry -> mismatch.
        let err = GraphAdjacency::decode(&offsets_bytes, &bad_neighbors_bytes, n);
        assert!(err.is_err());
    }

    /// `RobustPrune` diversity check: a directly-adjacent duplicate
    /// direction should be pruned in favour of diversity once one
    /// representative is selected, when R is smaller than the raw
    /// candidate count. This is a coarse smoke test of the pruning
    /// behaviour, not a formal proof.
    #[test]
    fn robust_prune_respects_degree_bound_with_many_similar_candidates() {
        // p at origin; candidates all clustered tightly in the same
        // direction (near-duplicates) plus a handful spread out.
        let dim = 4;
        // p = id 0.
        let mut vectors = vec![0.0f32; dim];
        // 8 near-duplicate candidates in the same direction.
        for i in 0..8u32 {
            let jitter = i as f32 * 1e-4;
            vectors.extend_from_slice(&[1.0 + jitter, 0.0, 0.0, 0.0]);
        }
        // 4 candidates spread across other directions.
        vectors.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
        vectors.extend_from_slice(&[0.0, 0.0, 1.0, 0.0]);
        vectors.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
        vectors.extend_from_slice(&[-1.0, 0.0, 0.0, 0.0]);
        let candidates: Vec<u32> = (1..=12).collect();
        let selected = robust_prune(0, &candidates, &vectors, dim, 1.2, 4);
        assert!(selected.len() <= 4);
        // At least one of the widely-spread directions should have
        // survived pruning (not just the tightest cluster).
        let spread_ids: std::collections::HashSet<u32> = [9, 10, 11, 12].into_iter().collect();
        assert!(
            selected.iter().any(|s| spread_ids.contains(s)),
            "RobustPrune kept only near-duplicate candidates, no diversity: {selected:?}"
        );
    }

    /// Phase G-2b: `insert_node_into_graph` adding ONE new node onto
    /// an already-built graph must produce a valid, navigable result
    /// -- the new node gets real out-edges (found by greedy search
    /// over the existing graph), no self-loop, degree bound `r`
    /// respected on both the new node and any neighbor that gained a
    /// reverse edge.
    #[test]
    fn insert_node_into_graph_adds_a_valid_navigable_node() {
        let n = 200;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 42);
        let (g, entry) = build_vamana(&corpus, n, dim);

        // Rehydrate the CSR adjacency into the mutable Vec<Vec<u32>>
        // shape insert_node_into_graph operates on (same shape the
        // build loop uses internally).
        let mut adj: Vec<Vec<u32>> = (0..n).map(|i| g.neighbors_of(i).to_vec()).collect();

        // New node's vector + id (one past the existing corpus).
        let mut extended = corpus.clone();
        let new_vec = synth_corpus(1, dim, 999);
        extended.extend_from_slice(&new_vec);
        let new_id = n as u32;
        adj.push(Vec::new());

        insert_node_into_graph(
            &mut adj,
            entry,
            new_id,
            &extended,
            dim,
            GRAPH_BUILD_L,
            GRAPH_ALPHA,
            GRAPH_DEGREE_R,
        );

        assert!(
            !adj[new_id as usize].is_empty(),
            "new node got zero out-edges -- not navigable"
        );
        assert!(
            adj[new_id as usize].len() <= GRAPH_DEGREE_R,
            "new node's out-degree {} exceeds R={GRAPH_DEGREE_R}",
            adj[new_id as usize].len()
        );
        assert!(
            !adj[new_id as usize].contains(&new_id),
            "new node has a self-loop"
        );
        // Every neighbor of the new node must, after the reverse-edge
        // step, itself point back at the new node (mutual
        // navigability) and stay within the degree bound.
        for &nb in &adj[new_id as usize] {
            assert!(
                adj[nb as usize].contains(&new_id),
                "neighbor {nb} of the new node has no reverse edge back"
            );
            assert!(
                adj[nb as usize].len() <= GRAPH_DEGREE_R,
                "neighbor {nb}'s out-degree {} exceeds R={GRAPH_DEGREE_R} after reverse-edge insert",
                adj[nb as usize].len()
            );
        }
        // Every pre-existing node's degree bound must still hold too
        // (a reverse edge to some UNRELATED node must never happen).
        for i in 0..n {
            assert!(
                adj[i].len() <= GRAPH_DEGREE_R,
                "pre-existing node {i}'s out-degree {} exceeds R={GRAPH_DEGREE_R} after insert",
                adj[i].len()
            );
        }
    }

    /// Phase G-2b: `insert_one_node` (the public API `insert.rs`
    /// calls) round-trips through CSR correctly, produces a graph
    /// findable via `graph_search` for the new row's own vector (a
    /// minimal "is it actually reachable" sanity check), and leaves
    /// every pre-existing node's edges within the degree bound.
    #[test]
    fn insert_one_node_grows_the_adjacency_and_stays_findable() {
        let n = 200;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 17);
        let (g, entry) = build_vamana(&corpus, n, dim);

        let mut extended = corpus.clone();
        let new_vec = synth_corpus(1, dim, 4242);
        extended.extend_from_slice(&new_vec);
        let new_id = n as u32;

        let g2 = insert_one_node(&g, entry, &extended, dim);
        assert_eq!(g2.n(), n + 1);
        for i in 0..n {
            assert!(
                g2.neighbors_of(i).len() <= GRAPH_DEGREE_R,
                "pre-existing node {i} exceeded R after insert_one_node"
            );
        }
        assert!(g2.neighbors_of(new_id as usize).len() <= GRAPH_DEGREE_R);

        // Querying for the new row's own vector should surface it in
        // the top few results via graph_search over the grown graph.
        let out = graph_search(&g2, entry, 5, &[], |ids| {
            neg_sq_dist_batch(&extended, dim, &new_vec, ids)
        });
        assert!(
            out.iter().any(|&(_, id)| id == new_id),
            "newly-inserted node is not findable via graph_search for its own vector: {out:?}"
        );
    }

    /// Same scenario as [`insert_one_node_grows_the_adjacency_and_stays_findable`],
    /// but via [`insert_one_node_via_oracle`] — the REAL `aminsert` entry
    /// point, which does NOT have the existing corpus's raw f32
    /// vectors resident (only the new row's). `score_existing` here
    /// is a stand-in for `cache::GraphIndex`'s real
    /// `ReadOnlyIndex::score_slots` (exact in this test, since the
    /// synthetic corpus IS resident here — the real `insert.rs` path
    /// scores against PERSISTED QUANTIZED codes instead, which is
    /// lossier; this test only exercises the oracle-based CONTROL
    /// FLOW, not the accuracy loss from real quantization, which is
    /// out of scope for a graph.rs-level unit test).
    #[test]
    fn insert_one_node_via_oracle_grows_the_adjacency_and_stays_findable() {
        let n = 200;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 17);
        let (g, entry) = build_vamana(&corpus, n, dim);

        let new_vec = synth_corpus(1, dim, 4242);
        let new_id = n as u32;

        let score_existing = |query: &[f32], ids: &[u32]| -> Vec<f32> {
            ids.iter()
                .map(|&id| -sq_dist(query, &corpus[id as usize * dim..(id as usize + 1) * dim]))
                .collect()
        };
        let g2 = insert_one_node_via_oracle(&g, entry, &new_vec, score_existing);
        assert_eq!(g2.n(), n + 1);
        for i in 0..n {
            assert!(
                g2.neighbors_of(i).len() <= GRAPH_DEGREE_R,
                "pre-existing node {i} exceeded R after insert_one_node_via_oracle"
            );
        }
        assert!(g2.neighbors_of(new_id as usize).len() <= GRAPH_DEGREE_R);
        assert!(
            !g2.neighbors_of(new_id as usize).contains(&new_id),
            "new node must not have a self-loop"
        );

        let mut extended = corpus.clone();
        extended.extend_from_slice(&new_vec);
        let out = graph_search(&g2, entry, 5, &[], |ids| {
            neg_sq_dist_batch(&extended, dim, &new_vec, ids)
        });
        assert!(
            out.iter().any(|&(_, id)| id == new_id),
            "newly-inserted node (via oracle) is not findable via graph_search for its own vector: {out:?}"
        );
    }

    /// Score oracle for the `graph_search` tests: higher = closer,
    /// matching turbovec's native convention (`SearchResults.scores`).
    /// Negative squared distance is a monotonic (order-preserving)
    /// transform of squared distance, so ranking by this score is
    /// identical to ranking by ascending distance.
    fn neg_sq_dist_batch(corpus: &[f32], dim: usize, query: &[f32], ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|&id| -sq_dist(query, &corpus[id as usize * dim..(id as usize + 1) * dim]))
            .collect()
    }

    #[test]
    fn graph_search_matches_linear_scan_set_on_a_real_corpus() {
        let n = 300;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 21);
        let (g, entry) = build_vamana(&corpus, n, dim);

        let mut qseed = 0x1234_5678u64;
        for k in [1usize, 5, 10, 20] {
            for _ in 0..15 {
                let mut q = vec![0.0f32; dim];
                for v in q.iter_mut() {
                    qseed = qseed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *v = ((qseed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                // Exact linear scan: score every node, take the top-k.
                let mut exact: Vec<(f32, u32)> = (0..n as u32)
                    .map(|id| {
                        (
                            -sq_dist(&q, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                            id,
                        )
                    })
                    .collect();
                exact.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                });
                exact.truncate(k);
                let exact_set: std::collections::HashSet<u32> =
                    exact.iter().map(|&(_, id)| id).collect();

                let via_graph = graph_search(&g, entry, k, &[], |ids| {
                    neg_sq_dist_batch(&corpus, dim, &q, ids)
                });
                let graph_set: std::collections::HashSet<u32> =
                    via_graph.iter().map(|&(_, id)| id).collect();

                // A beam search over an approximate graph is not
                // guaranteed to find the EXACT top-k on every query
                // (unlike ivf::graph_probe's small-lists case, which
                // does match exactly) -- so this asserts a generous
                // recall floor, not an exact-set match, at a
                // comfortably-wide ef (the defaults: ef =
                // max(k*4, 64)). At this corpus/graph scale the beam
                // is much wider than k, so recall should be high.
                let hits = exact_set.intersection(&graph_set).count();
                let recall = hits as f64 / k.min(exact_set.len()).max(1) as f64;
                assert!(
                    recall >= 0.7,
                    "graph_search recall {recall:.2} too low for k={k} (exact={exact_set:?} graph={graph_set:?})"
                );
                assert_eq!(
                    via_graph.len(),
                    k.min(n),
                    "graph_search must return k results when n >= k"
                );
            }
        }
    }

    #[test]
    fn graph_search_deterministic() {
        let n = 200;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 55);
        let (g, entry) = build_vamana(&corpus, n, dim);
        let q = synth_corpus(1, dim, 999);
        let r1 = graph_search(&g, entry, 10, &[], |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        let r2 = graph_search(&g, entry, 10, &[], |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        assert_eq!(r1, r2);
    }

    #[test]
    fn graph_search_empty_graph_returns_empty() {
        let g = GraphAdjacency::empty(0);
        let out = graph_search(&g, 0, 5, &[], |ids| vec![0.0; ids.len()]);
        assert!(out.is_empty());
    }

    #[test]
    fn graph_search_k_zero_returns_empty() {
        let n = 50;
        let dim = 8;
        let corpus = synth_corpus(n, dim, 3);
        let (g, entry) = build_vamana(&corpus, n, dim);
        let q = synth_corpus(1, dim, 4);
        let out = graph_search(&g, entry, 0, &[], |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        assert!(out.is_empty());
    }

    #[test]
    fn graph_search_results_are_score_descending() {
        let n = 200;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 77);
        let (g, entry) = build_vamana(&corpus, n, dim);
        let q = synth_corpus(1, dim, 88);
        let out = graph_search(&g, entry, 15, &[], |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        for w in out.windows(2) {
            assert!(
                w[0].0 >= w[1].0,
                "results must be score-descending: {out:?}"
            );
        }
    }

    /// Build a LSB-first per-slot tombstone bitmap with the given
    /// dead ids set, matching `relfile::read_tombstones`'s on-disk
    /// convention.
    fn tombstone_bitmap(n: usize, dead: &[u32]) -> Vec<u8> {
        let mut bm = vec![0u8; n.div_ceil(8)];
        for &id in dead {
            bm[id as usize / 8] |= 1u8 << (id as usize % 8);
        }
        bm
    }

    /// Phase G-2b: a tombstoned node must never be returned, and the
    /// live-set results must match a brute-force scan restricted to
    /// the live ids only (not just "close to" -- an exact set match,
    /// since the corpus here is small enough for the beam to find
    /// the true top-k of the live set with the default ef).
    #[test]
    fn graph_search_never_returns_a_tombstoned_id() {
        let n = 300;
        let dim = 16;
        let corpus = synth_corpus(n, dim, 321);
        let (g, entry) = build_vamana(&corpus, n, dim);

        // Tombstone every 7th id (avoid the entry point itself --
        // that degenerate case gets its own test below).
        let dead: Vec<u32> = (0..n as u32)
            .filter(|id| id % 7 == 0 && *id != entry)
            .collect();
        let bitmap = tombstone_bitmap(n, &dead);
        let dead_set: std::collections::HashSet<u32> = dead.iter().copied().collect();

        let q = synth_corpus(1, dim, 654);
        for k in [1usize, 5, 10, 20] {
            let out = graph_search(&g, entry, k, &bitmap, |ids| {
                neg_sq_dist_batch(&corpus, dim, &q, ids)
            });
            for &(_, id) in &out {
                assert!(
                    !dead_set.contains(&id),
                    "graph_search returned tombstoned id {id} for k={k}"
                );
            }

            // Brute-force top-k restricted to the live set.
            let mut exact: Vec<(f32, u32)> = (0..n as u32)
                .filter(|id| !dead_set.contains(id))
                .map(|id| {
                    (
                        -sq_dist(&q, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                        id,
                    )
                })
                .collect();
            exact.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            exact.truncate(k);
            let exact_set: std::collections::HashSet<u32> =
                exact.iter().map(|&(_, id)| id).collect();
            let graph_set: std::collections::HashSet<u32> = out.iter().map(|&(_, id)| id).collect();
            let hits = exact_set.intersection(&graph_set).count();
            let recall = hits as f64 / k.min(exact_set.len()).max(1) as f64;
            assert!(
                recall >= 0.7,
                "tombstone-aware recall {recall:.2} too low for k={k}"
            );
        }
    }

    /// Phase G-2b entry-point-tombstoned case: `graph_search` must
    /// fall back to a live id rather than starting (and getting
    /// stuck at) a dead entry point, and must still return only live
    /// ids.
    #[test]
    fn graph_search_falls_back_when_entry_point_is_tombstoned() {
        let n = 250;
        let dim = 12;
        let corpus = synth_corpus(n, dim, 111);
        let (g, entry) = build_vamana(&corpus, n, dim);

        let bitmap = tombstone_bitmap(n, &[entry]);
        let q = synth_corpus(1, dim, 222);
        let out = graph_search(&g, entry, 10, &bitmap, |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        assert!(
            !out.is_empty(),
            "graph_search returned nothing when only the entry point was tombstoned"
        );
        for &(_, id) in &out {
            assert_ne!(id, entry, "returned the tombstoned entry point itself");
        }
    }

    /// Degenerate case: every node tombstoned. Must return empty, not
    /// panic.
    #[test]
    fn graph_search_fully_tombstoned_corpus_returns_empty() {
        let n = 100;
        let dim = 8;
        let corpus = synth_corpus(n, dim, 5);
        let (g, entry) = build_vamana(&corpus, n, dim);
        let all_dead: Vec<u32> = (0..n as u32).collect();
        let bitmap = tombstone_bitmap(n, &all_dead);
        let q = synth_corpus(1, dim, 9);
        let out = graph_search(&g, entry, 5, &bitmap, |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        assert!(out.is_empty());
    }

    /// Build the same corpus inside a rayon pool of `n_threads` and
    /// return the resulting adjacency + entry point. Used by the
    /// thread-count-independence test below.
    fn build_in_pool(
        corpus: &[f32],
        n: usize,
        dim: usize,
        n_threads: usize,
    ) -> (GraphAdjacency, u32) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        pool.install(|| build_vamana(corpus, n, dim))
    }

    /// Phase G-2c HARD determinism contract: the parallelized Vamana
    /// build must produce a BIT-IDENTICAL `GraphAdjacency` (and entry
    /// point) regardless of the rayon pool size it runs under. The
    /// per-node distance SCORING fans across the pool, but via an
    /// order-preserving indexed collect over fixed-order `sq_dist`
    /// evals, so the graph can never depend on thread count. This is
    /// the graph-kind analogue of `ivf.rs`'s thread-count-invariant
    /// k-means test and `build_pool::build_parts_are_pool_size_
    /// invariant`. Corpus dim is large enough that the parallel batch
    /// path actually engages (clears `PAR_BATCH_MIN_FLOPS` on
    /// full-degree hops), not just the serial fallback.
    #[test]
    fn build_is_bit_identical_across_thread_counts() {
        let n = 400;
        let dim = 256;
        let corpus = synth_corpus(n, dim, 424242);
        let (g1, e1) = build_in_pool(&corpus, n, dim, 1);
        let (g2, e2) = build_in_pool(&corpus, n, dim, 2);
        // Ambient/global pool (rayon's default = all cores) == the
        // "auto" / Rayon(0) case a real ambient build hits.
        let (g_auto, e_auto) = build_vamana(&corpus, n, dim);
        assert_eq!(g1, g2, "adjacency differs between pool sizes 1 and 2");
        assert_eq!(g1, g_auto, "adjacency differs between pool size 1 and auto");
        assert_eq!(e1, e2, "entry point differs between pool sizes 1 and 2");
        assert_eq!(
            e1, e_auto,
            "entry point differs between pool size 1 and auto"
        );
    }

    /// G-2c local RELATIVE build timing (this box's absolute numbers
    /// are untrustworthy per AGENTS.md, but relative/wall-clock is
    /// meaningful for a sanity check). Reports the full serial build
    /// wall-clock. `#[ignore]` — run with `--ignored --nocapture`.
    ///
    /// NOTE: two build-path "SIMD"/parallel optimisations were tried
    /// and REJECTED as measured regressions here (a hand-rolled
    /// 8-lane `sq_dist_blocked` kernel at ~0.72×, and rayon per-hop
    /// fan-out at 0.68–0.85×) — see `build_row_dists`'s doc. The
    /// build uses plain `sq_dist` (LLVM auto-vectorises it optimally);
    /// the real G-2c speedup is on the SCAN path (`cache::GraphScorer`,
    /// the per-query LUT scorer), which this test does not exercise.
    #[test]
    #[ignore]
    fn timing_build_wall_clock() {
        use std::time::Instant;
        let n = 4000;
        let dim = 512;
        let corpus = synth_corpus(n, dim, 7);
        let tbuild = Instant::now();
        let (g, _e) = build_vamana(&corpus, n, dim);
        eprintln!(
            "[G-2c build] n={n} dim={dim} edges={} full-build={:?}",
            g.edge_count(),
            tbuild.elapsed()
        );
    }

    // -----------------------------------------------------------------
    // Phase G-2d(a): partitioned/merge parallel build tests.
    // -----------------------------------------------------------------

    /// Clustered synthetic corpus: `n_clusters` random centers, each
    /// point = a center + small gaussian-ish jitter. Gives the graph
    /// real proximity structure (unlike uniform `synth_corpus`), so
    /// recall is a meaningful signal. Deterministic in `seed`.
    fn clustered_corpus(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut centers = vec![0.0f32; n_clusters * dim];
        for c in centers.iter_mut() {
            *c = rng.gen_range(-1.0f32..1.0);
        }
        let mut out = vec![0.0f32; n * dim];
        for i in 0..n {
            let c = i % n_clusters;
            for d in 0..dim {
                // small jitter around the cluster center
                let j: f32 = rng.gen_range(-0.15f32..0.15);
                out[i * dim + d] = centers[c * dim + d] + j;
            }
        }
        out
    }

    /// recall@k of `graph`'s search vs exact linear scan, averaged
    /// over `n_queries` deterministic queries drawn near the corpus
    /// distribution.
    fn measure_recall(
        corpus: &[f32],
        n: usize,
        dim: usize,
        g: &GraphAdjacency,
        entry: u32,
        queries: &[f32],
        n_queries: usize,
        k: usize,
    ) -> f64 {
        let mut total = 0.0f64;
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut exact: Vec<(f32, u32)> = (0..n as u32)
                .map(|id| {
                    (
                        -sq_dist(q, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                        id,
                    )
                })
                .collect();
            exact.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            exact.truncate(k);
            let exact_set: std::collections::HashSet<u32> =
                exact.iter().map(|&(_, id)| id).collect();
            let out = graph_search(g, entry, k, &[], |ids| {
                neg_sq_dist_batch(corpus, dim, q, ids)
            });
            let graph_set: std::collections::HashSet<u32> = out.iter().map(|&(_, id)| id).collect();
            let hits = exact_set.intersection(&graph_set).count();
            total += hits as f64 / k.min(exact_set.len()).max(1) as f64;
        }
        total / n_queries as f64
    }

    /// THE correctness gate: the partitioned build's recall@10 must be
    /// within a small tolerance of (or better than) the single-pass
    /// build's, on the SAME clustered corpus. A partitioned build that
    /// under-recalls is a failed approach.
    #[test]
    fn partitioned_build_recall_parity_with_single_pass() {
        let n = 20_000;
        let dim = 64;
        let corpus = clustered_corpus(n, dim, 50, 20260709);
        // Query set: 100 fresh points from the same distribution.
        let queries = clustered_corpus(100, dim, 50, 0xC0FFEE);
        let n_queries = 100;
        let k = 10;

        let (g_single, e_single) = build_vamana(&corpus, n, dim);
        // Force 8 shards (n is below the auto MIN_ROWS threshold, so
        // pass p explicitly through the params form).
        let (g_part, e_part) = build_vamana_partitioned_with_params(
            &corpus,
            n,
            dim,
            8,
            GRAPH_DEGREE_R,
            GRAPH_BUILD_L,
            GRAPH_ALPHA,
            GRAPH_SEED,
        );

        let r_single = measure_recall(&corpus, n, dim, &g_single, e_single, &queries, n_queries, k);
        let r_part = measure_recall(&corpus, n, dim, &g_part, e_part, &queries, n_queries, k);

        eprintln!(
            "[G-2d recall] single-pass R@{k}={r_single:.4}  partitioned(P=8) R@{k}={r_part:.4}  delta={:.4}",
            r_part - r_single
        );
        // Tolerance: partitioned must not be worse by more than 3
        // recall points. (Empirically it matches or beats single-pass
        // because the refinement pass's greedy search over the MERGED
        // graph surfaces cross-shard neighbors the single-pass build's
        // incremental insertion never reconsidered.)
        assert!(
            r_part >= r_single - 0.03,
            "partitioned recall {r_part:.4} under-recalls single-pass {r_single:.4} by more than tolerance"
        );
    }

    /// Determinism: same (corpus, seed, P) -> bit-identical adjacency
    /// AND entry point.
    #[test]
    fn partitioned_build_is_deterministic() {
        let n = 8_000;
        let dim = 32;
        let corpus = clustered_corpus(n, dim, 20, 555);
        let (g1, e1) = build_vamana_partitioned_with_params(
            &corpus,
            n,
            dim,
            6,
            GRAPH_DEGREE_R,
            GRAPH_BUILD_L,
            GRAPH_ALPHA,
            GRAPH_SEED,
        );
        let (g2, e2) = build_vamana_partitioned_with_params(
            &corpus,
            n,
            dim,
            6,
            GRAPH_DEGREE_R,
            GRAPH_BUILD_L,
            GRAPH_ALPHA,
            GRAPH_SEED,
        );
        assert_eq!(g1, g2, "partitioned build is non-deterministic");
        assert_eq!(e1, e2, "partitioned entry point is non-deterministic");
    }

    /// HARD determinism contract (like G-2c's
    /// `build_is_bit_identical_across_thread_counts`): the partitioned
    /// build is bit-identical across rayon pool sizes {1, 2, auto}.
    /// The per-shard collect is index-ordered, the refinement pass
    /// stages per-node results by fixed id, and the reverse-edge pass
    /// is serial — so the result never depends on thread scheduling.
    #[test]
    fn partitioned_build_is_bit_identical_across_thread_counts() {
        let n = 8_000;
        let dim = 48;
        let corpus = clustered_corpus(n, dim, 25, 42);
        let build = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(|| {
                    build_vamana_partitioned_with_params(
                        &corpus,
                        n,
                        dim,
                        5,
                        GRAPH_DEGREE_R,
                        GRAPH_BUILD_L,
                        GRAPH_ALPHA,
                        GRAPH_SEED,
                    )
                })
        };
        let (g1, e1) = build(1);
        let (g2, e2) = build(2);
        let (g_auto, e_auto) = build_vamana_partitioned_with_params(
            &corpus,
            n,
            dim,
            5,
            GRAPH_DEGREE_R,
            GRAPH_BUILD_L,
            GRAPH_ALPHA,
            GRAPH_SEED,
        );
        assert_eq!(g1, g2, "partitioned adjacency differs between pool 1 and 2");
        assert_eq!(
            g1, g_auto,
            "partitioned adjacency differs between pool 1 and auto"
        );
        assert_eq!(e1, e2);
        assert_eq!(e1, e_auto);
    }

    /// Structural invariants on the partitioned build's output: degree
    /// <= R, no self-loops, strictly ascending/deduped neighbor lists,
    /// CSR round-trips, no isolated live nodes.
    #[test]
    fn partitioned_build_invariants() {
        let n = 10_000;
        let dim = 32;
        let r = 24;
        let corpus = clustered_corpus(n, dim, 30, 909);
        let (g, _e) =
            build_vamana_partitioned_with_params(&corpus, n, dim, 7, r, r * 2, GRAPH_ALPHA, 123);
        for i in 0..n {
            let nbrs = g.neighbors_of(i);
            assert!(nbrs.len() <= r, "node {i} degree {} > R={r}", nbrs.len());
            assert!(!nbrs.contains(&(i as u32)), "node {i} self-loop");
            assert!(!nbrs.is_empty(), "node {i} isolated -- navigability gap");
            for w in nbrs.windows(2) {
                assert!(w[0] < w[1], "node {i} neighbor list not strictly ascending");
            }
            assert!(nbrs.iter().all(|&id| (id as usize) < n));
        }
        // CSR round-trips.
        let back =
            GraphAdjacency::decode(&g.encode_offsets(), &g.encode_neighbors(), n).expect("decode");
        assert_eq!(g, back);
    }

    /// P <= 1 (and too-small corpora) delegate to the single-pass
    /// build, byte-identically. Guarantees the reference path stays
    /// reachable and `graph_build_partitions = 0/1` is a true no-op.
    #[test]
    fn partitioned_p_le_1_equals_single_pass() {
        let n = 5_000;
        let dim = 24;
        let corpus = clustered_corpus(n, dim, 15, 77);
        let (g_single, e_single) = build_vamana(&corpus, n, dim);
        for p in [0usize, 1] {
            let (g, e) = build_vamana_partitioned_with_params(
                &corpus,
                n,
                dim,
                p,
                GRAPH_DEGREE_R,
                GRAPH_BUILD_L,
                GRAPH_ALPHA,
                GRAPH_SEED,
            );
            assert_eq!(g, g_single, "P={p} differs from single-pass adjacency");
            assert_eq!(e, e_single, "P={p} differs from single-pass entry");
        }
    }

    /// G-2d(a) local RELATIVE build timing: serial (P=1 = single-pass)
    /// vs partitioned (P=auto-ish) wall-clock at a scale where the
    /// single-pass build is slow. This box's ABSOLUTE numbers are
    /// untrustworthy (AGENTS.md), but the serial-vs-partitioned RATIO
    /// on the SAME machine demonstrates the cores get used. `#[ignore]`
    /// — run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_partitioned_vs_single_pass() {
        use std::time::Instant;
        let n = 200_000;
        let dim = 64;
        let corpus = clustered_corpus(n, dim, 200, 7);

        let t0 = Instant::now();
        let (gs, _es) = build_vamana(&corpus, n, dim);
        let single = t0.elapsed();
        eprintln!(
            "[G-2d timing] single-pass={single:?} ({} edges)",
            gs.edge_count()
        );

        let cores = rayon::current_num_threads().max(2);
        for p in [cores, cores * 2, cores * 4] {
            let t1 = Instant::now();
            let (gp, _ep) = build_vamana_partitioned(&corpus, n, dim, p);
            let part = t1.elapsed();
            let ratio = single.as_secs_f64() / part.as_secs_f64();
            eprintln!(
                "[G-2d timing] cores={cores} P={p}  partitioned={part:?} ({} edges)  speedup={ratio:.2}x",
                gp.edge_count(),
            );
        }
    }

    // -----------------------------------------------------------------
    // Property-based tests (Hegel / hegeltest).
    //
    // These generalise the hand-written example tests above: each one
    // pins a contract that must hold for EVERY (corpus, dim, seed, R,
    // ...) combination, not just the single fixed shape a `#[test]`
    // happened to pick. Hegel draws the parameters, runs the real
    // build/search/codec, and shrinks any failure to a minimal
    // counterexample. They run under the plain `cargo test` runner
    // (no PostgreSQL backend needed -- this whole module is pure
    // Rust), alongside the pgrx `#[pg_test]` suite.
    //
    // Generator discipline: dim is drawn small-but-varied (the codec
    // and adjacency invariants are dimension-independent; recall is
    // the only dimension-sensitive property and its generator is
    // widened accordingly). n is drawn across the R+1 navigability
    // boundary so both the "too few nodes to fill degree R" and the
    // "comfortably connected" regimes are exercised.
    // -----------------------------------------------------------------

    use hegel::generators::{self};

    /// A generated corpus: (flat row-major f32 buffer, n, dim, seed).
    /// n and dim are drawn independently; the vectors themselves come
    /// from the deterministic `synth_corpus` keyed on the drawn seed,
    /// so Hegel shrinks the *shape* and the *seed*, and replay is
    /// exact.
    #[hegel::composite]
    fn corpus_shape(tc: hegel::TestCase) -> (Vec<f32>, usize, usize, u64) {
        // n spans 0..400: includes the empty/one/two-node degenerate
        // cases AND corpora comfortably past R+1 where navigability
        // must hold. dim spans 1..64: cheap but varied.
        let n = tc.draw(generators::integers::<usize>().min_value(0).max_value(400));
        let dim = tc.draw(generators::integers::<usize>().min_value(1).max_value(64));
        let seed = tc.draw(generators::integers::<u64>());
        (synth_corpus(n, dim, seed), n, dim, seed)
    }

    /// Tier-1 round-trip: encode -> decode is the identity on any
    /// built adjacency. This is the corruption-prevention property --
    /// the persisted relfile bytes MUST decode back to exactly what
    /// was built (a real block-offset collision bug in this exact
    /// codec path was found and fixed in v1.24.0). Generalises
    /// `adjacency_encode_decode_round_trip`.
    #[hegel::test]
    fn prop_adjacency_encode_decode_round_trip(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let (g, _entry) = build_vamana(&corpus, n, dim);
        let offsets = g.encode_offsets();
        let neighbors = g.encode_neighbors();
        let back = GraphAdjacency::decode(&offsets, &neighbors, n)
            .expect("decode of freshly-encoded adjacency must succeed");
        assert_eq!(g, back, "encode->decode was not the identity");
    }

    /// Round-trip through the mutable `Vec<Vec<u32>>` shape the
    /// incremental-insert path (`insert.rs`) rehydrates into. Because
    /// `from_lists` canonicalises (sorts each row ascending) and the
    /// build already emits ascending rows, `from_lists(to_lists(g))`
    /// must be the identity. Generalises the implicit contract the
    /// insert path relies on.
    #[hegel::test]
    fn prop_to_lists_from_lists_round_trip(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let (g, _entry) = build_vamana(&corpus, n, dim);
        let back = GraphAdjacency::from_lists(g.to_lists());
        assert_eq!(g, back, "to_lists->from_lists was not the identity");
    }

    /// Structural invariant: no node's out-degree exceeds R, for any
    /// R the build is parameterised with. RobustPrune's hard cap is
    /// the whole point of the degree bound (unbounded fan-out would
    /// blow the CSR size estimate and the scan cost model).
    /// Generalises `max_out_degree_is_bounded_by_r`.
    #[hegel::test]
    fn prop_out_degree_bounded_by_r(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let r = tc.draw(generators::integers::<usize>().min_value(1).max_value(64));
        let (g, _entry) =
            build_vamana_with_params(&corpus, n, dim, r, r * 2, GRAPH_ALPHA, GRAPH_SEED);
        for i in 0..g.n() {
            assert!(
                g.neighbors_of(i).len() <= r,
                "node {i} out-degree {} > R={r} (n={n} dim={dim})",
                g.neighbors_of(i).len()
            );
        }
    }

    /// Structural invariant: no self-loops and every neighbor list is
    /// strictly ascending with no duplicates -- for ANY corpus. The
    /// scan/insert paths assume both (ascending for the tombstone
    /// merge, no-self-loop for the greedy-search visited bookkeeping).
    /// Merges `no_self_loops` + `neighbor_ids_are_ascending_and_
    /// deduplicated` into one invariant over generated corpora.
    #[hegel::test]
    fn prop_neighbor_lists_are_wellformed(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let (g, _entry) = build_vamana(&corpus, n, dim);
        for i in 0..g.n() {
            let nbrs = g.neighbors_of(i);
            assert!(
                !nbrs.contains(&(i as u32)),
                "node {i} has a self-loop (n={n} dim={dim})"
            );
            for w in nbrs.windows(2) {
                assert!(
                    w[0] < w[1],
                    "node {i} neighbor list not strictly ascending (n={n} dim={dim})"
                );
            }
            assert!(
                nbrs.iter().all(|&id| (id as usize) < n),
                "node {i} references an out-of-range neighbor id (n={n})"
            );
        }
    }

    /// Navigability: once n is comfortably past R+1, no live node is
    /// left totally isolated (zero out-edges) -- an isolated node is
    /// unreachable by greedy search and silently lost. Generalises
    /// `every_node_has_at_least_one_neighbor_on_a_real_corpus`. The
    /// n generator is floored above R+2 so the property is only
    /// asserted in the regime where it must hold (a 1- or 2-node
    /// corpus legitimately has isolated-ish structure, covered by the
    /// separate degenerate-case example tests).
    #[hegel::test]
    fn prop_no_isolated_nodes_above_degree_floor(tc: hegel::TestCase) {
        let dim = tc.draw(generators::integers::<usize>().min_value(1).max_value(48));
        let seed = tc.draw(generators::integers::<u64>());
        // n well above R+1 (default R=32): draw 40..400.
        let n = tc.draw(generators::integers::<usize>().min_value(40).max_value(400));
        let corpus = synth_corpus(n, dim, seed);
        let (g, _entry) = build_vamana(&corpus, n, dim);
        for i in 0..n {
            assert!(
                !g.neighbors_of(i).is_empty(),
                "node {i} is isolated (n={n} dim={dim} seed={seed}) -- navigability gap"
            );
        }
    }

    /// Determinism contract: same (corpus, seed) -> byte-identical
    /// adjacency AND identical entry point, for any drawn shape. This
    /// underpins single-host REINDEX reproducibility and the test
    /// suite's own stability. Generalises
    /// `build_is_deterministic_for_a_fixed_seed`.
    #[hegel::test]
    fn prop_build_is_deterministic(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let (g1, e1) = build_vamana(&corpus, n, dim);
        let (g2, e2) = build_vamana(&corpus, n, dim);
        assert_eq!(g1, g2, "non-deterministic adjacency (n={n} dim={dim})");
        assert_eq!(e1, e2, "non-deterministic entry point (n={n} dim={dim})");
    }

    /// Phase G-2c HARD contract, generalised: the parallelized build
    /// is bit-identical across thread counts for ANY drawn shape — a
    /// pool of size 1 and a pool of size 3 must produce the exact
    /// same adjacency + entry point. This is the property that makes
    /// `turbovec.build_parallelism` a pure performance knob with no
    /// effect on the persisted graph. Generalises
    /// `build_is_bit_identical_across_thread_counts`.
    #[hegel::test]
    fn prop_build_thread_count_independent(tc: hegel::TestCase) {
        let (corpus, n, dim, _seed) = tc.draw(corpus_shape());
        let build_in = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(|| build_vamana(&corpus, n, dim))
        };
        let (g1, e1) = build_in(1);
        let (g3, e3) = build_in(3);
        assert_eq!(
            g1, g3,
            "adjacency depends on thread count (n={n} dim={dim})"
        );
        assert_eq!(
            e1, e3,
            "entry point depends on thread count (n={n} dim={dim})"
        );
    }

    /// Parse robustness (Tier-1): `decode` must NEVER panic on
    /// arbitrary bytes -- it returns `Err` on any malformed input.
    /// The relfile reader hands `decode` whatever bytes the chain
    /// pages contain; a panic there would be an unrecoverable scan
    /// crash instead of a clean corruption error. Draws fully
    /// arbitrary byte buffers and node counts.
    #[hegel::test]
    fn prop_decode_never_panics_on_garbage(tc: hegel::TestCase) {
        let offsets = tc.draw(generators::binary().max_size(512));
        let neighbors = tc.draw(generators::binary().max_size(512));
        let n = tc.draw(generators::integers::<usize>().max_value(200));
        // Contract: returns Result, never panics. We don't assert Ok
        // or Err -- only that it doesn't crash on garbage.
        let _ = GraphAdjacency::decode(&offsets, &neighbors, n);
    }

    /// Any bytes that DO decode successfully must re-encode to the
    /// exact same bytes (decode is injective on its valid domain).
    /// This is the other half of the codec round-trip, driven from
    /// the bytes side rather than the built-graph side, so it also
    /// covers hand-constructed adjacencies the build path would never
    /// emit.
    #[hegel::test]
    fn prop_decode_then_encode_is_identity_when_valid(tc: hegel::TestCase) {
        let offsets = tc.draw(generators::binary().max_size(512));
        let neighbors = tc.draw(generators::binary().max_size(512));
        let n = tc.draw(generators::integers::<usize>().max_value(120));
        if let Ok(g) = GraphAdjacency::decode(&offsets, &neighbors, n) {
            assert_eq!(g.encode_offsets(), offsets, "offsets re-encode mismatch");
            assert_eq!(
                g.encode_neighbors(),
                neighbors,
                "neighbors re-encode mismatch"
            );
            assert_eq!(g.n(), n, "decoded node count disagrees with requested n");
        }
    }

    /// Search contract: `graph_search` returns exactly `min(k, n)`
    /// results, all of them distinct, all valid live ids -- for any
    /// corpus, any k, any query. (Recall QUALITY is a separate,
    /// dimension-sensitive property below; this one is the pure
    /// structural contract every caller relies on.) Generalises the
    /// `assert_eq!(via_graph.len(), k.min(n))` corner of
    /// `graph_search_matches_linear_scan_set_on_a_real_corpus`.
    #[hegel::test]
    fn prop_graph_search_returns_k_distinct_valid_results(tc: hegel::TestCase) {
        let dim = tc.draw(generators::integers::<usize>().min_value(1).max_value(48));
        let n = tc.draw(generators::integers::<usize>().min_value(1).max_value(300));
        let seed = tc.draw(generators::integers::<u64>());
        let corpus = synth_corpus(n, dim, seed);
        let (g, entry) = build_vamana(&corpus, n, dim);
        let k = tc.draw(generators::integers::<usize>().min_value(1).max_value(64));
        let q = synth_corpus(1, dim, seed ^ 0x9E37_79B9);
        let out = graph_search(&g, entry, k, &[], |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        assert_eq!(out.len(), k.min(n), "wrong result count (n={n} k={k})");
        let ids: std::collections::HashSet<u32> = out.iter().map(|&(_, id)| id).collect();
        assert_eq!(ids.len(), out.len(), "graph_search returned duplicate ids");
        assert!(
            ids.iter().all(|&id| (id as usize) < n),
            "graph_search returned an out-of-range id (n={n})"
        );
    }

    /// Tombstone consistency: no tombstoned id is EVER returned by a
    /// scan, for any corpus / any dead set / any query. This is the
    /// hard correctness guarantee VACUUM relies on (a returned dead
    /// row is a visibility bug). Generalises the fixed-set exclusion
    /// checks in the pg_test suite down to the pure-Rust core.
    #[hegel::test]
    fn prop_graph_search_never_returns_tombstoned_ids(tc: hegel::TestCase) {
        let dim = tc.draw(generators::integers::<usize>().min_value(1).max_value(32));
        let n = tc.draw(generators::integers::<usize>().min_value(1).max_value(300));
        let seed = tc.draw(generators::integers::<u64>());
        let corpus = synth_corpus(n, dim, seed);
        let (g, entry) = build_vamana(&corpus, n, dim);
        // Draw a dead set as a subset of valid ids (unique, in range).
        let dead: Vec<u32> = tc.draw(
            generators::vecs(
                generators::integers::<u32>()
                    .min_value(0)
                    .max_value(n as u32 - 1),
            )
            .max_size(n)
            .unique(true),
        );
        let bitmap = tombstone_bitmap(n, &dead);
        let dead_set: std::collections::HashSet<u32> = dead.iter().copied().collect();
        let k = tc.draw(generators::integers::<usize>().min_value(1).max_value(32));
        let q = synth_corpus(1, dim, seed ^ 0xABCD_1234);
        let out = graph_search(&g, entry, k, &bitmap, |ids| {
            neg_sq_dist_batch(&corpus, dim, &q, ids)
        });
        for &(_, id) in &out {
            assert!(
                !dead_set.contains(&id),
                "graph_search returned tombstoned id {id} (n={n} dead={dead:?})"
            );
        }
        // Also: if EVERY id is dead, the result must be empty.
        if dead.len() == n {
            assert!(out.is_empty(), "fully-tombstoned corpus returned results");
        }
    }
}
