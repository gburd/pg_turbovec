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

/// Number of Lloyd's iterations. FAISS's default coarse-quantizer
/// training runs ~10–25 iterations; 25 converges the small coarse
/// codebook (`nlist ≈ √n` centroids over a `256·nlist`-bounded
/// sample) well within diminishing returns. Iteration count is part
/// of the determinism contract — changing it changes the centroids.
pub const KMEANS_ITERS: usize = 25;

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
            entries.push(CellEntry::decode(&bytes[off..off + CellEntry::ENCODED_BYTES]));
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
/// surface (per an internal design note §8). Cost `O(lists * dim)`.
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
/// second SIMD-correctness surface (per an internal design note §8). The reduction
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
/// One single-threaded `gemm` call (`Parallelism::None`) so the
/// reduction order is fixed and the result is bit-deterministic
/// across runs and machines -- the on-disk IVF bytes must not change.
/// The orientation is verified elementwise against `rotate_unit` by
/// the `ivf_batched_rotation_matches_per_row` test.
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
            Parallelism::None,
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
    assert!(lists > 0, "train_kmeans: lists must be > 0");
    assert!(dim > 0, "train_kmeans: dim must be > 0");
    debug_assert_eq!(sample.len(), n_sample * dim);

    let mut centroids = vec![0.0f32; lists * dim];
    if n_sample == 0 {
        return CoarseModel {
            centroids,
            lists,
            dim,
        };
    }

    let mut rng = ChaCha8Rng::seed_from_u64(IVF_SEED);
    let row = |i: usize| -> &[f32] { &sample[i * dim..(i + 1) * dim] };

    // ---- k-means++ seeding ----
    // First centroid: a deterministic uniform pick.
    let first = rng.gen_range(0..n_sample);
    centroids[0..dim].copy_from_slice(row(first));

    // D2 sampling for the remaining seeds. `d2[i]` is the squared
    // distance from sample i to its nearest chosen centroid so far.
    let mut d2 = vec![f32::INFINITY; n_sample];
    for i in 0..n_sample {
        d2[i] = sq_dist(row(i), &centroids[0..dim]);
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
        let new_c = chosen;
        let new_centroid = row(new_c).to_vec();
        for i in 0..n_sample {
            let d = sq_dist(row(i), &new_centroid);
            if d < d2[i] {
                d2[i] = d;
            }
        }
    }

    // ---- Lloyd's iterations ----
    let mut assign = vec![0u32; n_sample];
    for _iter in 0..KMEANS_ITERS {
        // Assignment step (ascending order ⇒ deterministic ties).
        for i in 0..n_sample {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..lists {
                let d = sq_dist(row(i), &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }

        // Update step: mean of assigned points.
        let mut sums = vec![0.0f64; lists * dim];
        let mut counts = vec![0u64; lists];
        for i in 0..n_sample {
            let c = assign[i] as usize;
            counts[c] += 1;
            let r = row(i);
            let base = c * dim;
            for j in 0..dim {
                sums[base + j] += r[j] as f64;
            }
        }
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
    }

    CoarseModel {
        centroids,
        lists,
        dim,
    }
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
        assert_eq!(model.assign_one(&[0.0, 0.0]), model.assign_one(&[0.05, -0.05]));
        assert_ne!(model.assign_one(&[0.0, 0.0]), model.assign_one(&[10.0, 10.0]));
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
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let m1 = train_kmeans(&sample, n, 16, dim);
        let m2 = train_kmeans(&sample, n, 16, dim);
        assert_eq!(m1.centroids, m2.centroids, "k-means must be deterministic");
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
                CellEntry { code_offset: 0, n_vectors: 5 },
                CellEntry { code_offset: 5, n_vectors: 0 },
                CellEntry { code_offset: 5, n_vectors: 7 },
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
                CellEntry { code_offset: 0, n_vectors: 5 },
                CellEntry { code_offset: 6, n_vectors: 4 }, // gap!
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
                CellEntry { code_offset: 0, n_vectors: 3 },  // slots 0,1,2
                CellEntry { code_offset: 3, n_vectors: 2 },  // slots 3,4
                CellEntry { code_offset: 5, n_vectors: 4 },  // slots 5,6,7,8
            ],
        };
        let n = 9;
        // Probe cell 1 only.
        let m = dir.probe_mask(&[1], n);
        assert_eq!(m, vec![false, false, false, true, true, false, false, false, false]);
        // Probe cells 0 and 2.
        let m = dir.probe_mask(&[0, 2], n);
        assert_eq!(m, vec![true, true, true, false, false, true, true, true, true]);
        assert_eq!(m.iter().filter(|&&b| b).count(), 7);
        // Probe all cells ⇒ all true (the exact-flat anchor at the
        // mask level).
        let m = dir.probe_mask(&[0, 1, 2], n);
        assert!(m.iter().all(|&b| b));
        // Out-of-range and duplicate cells are ignored, not panics.
        let m = dir.probe_mask(&[1, 1, 99], n);
        assert_eq!(m.iter().filter(|&&b| b).count(), 2);
    }
}
