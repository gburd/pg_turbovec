//! `ambuild` and `ambuildempty` — initial index materialisation.
//!
//! v0.7 uses the table AM's `index_build_range_scan` callback rather
//! than SPI for the heap scan. This is the same path the built-in
//! btree / GIN / hash AMs use, and crucially it doesn't try to take
//! locks on the index that's currently being (re)built — SPI did,
//! and REINDEX broke as a result.

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::guc;
use crate::index::ivf;
use crate::index::{options, relfile};
use crate::kernels;
use crate::vec::Vector;

/// State threaded through `index_build_range_scan` into our callback.
///
/// Phase W (v1.6.0): we no longer accumulate the entire heap-scan output
/// into a single `Vec<f32>` before handing it to `IdMapIndex`. At
/// 10 M × 1536-d that buffer alone was 61 GiB, the dominant offender in
/// the 121 GiB peak RSS Phase V observed. Instead we keep two bounded
/// staging buffers (`pending_flat`, `pending_ids`) sized off
/// `maintenance_work_mem`, and flush them into `IdMapIndex::add_with_ids`
/// every `chunk_rows`. The IdMapIndex's row-major `packed_codes` still
/// grows linearly (~7.7 GiB at 10 M × 1536-d × 4-bit) but the f32
/// staging buffer is capped at min(0.75 × maintenance_work_mem, 1 GiB).
///
/// `bit_width` and `cfg_bit_width` are unused inside the staging path
/// (the bit-width lives on the constructed `IdMapIndex` itself) but we
/// keep them on the state for symmetry with the eventual
/// `relfile::write_full_with_prepared` call site.
struct BuildState {
    /// The IdMapIndex under construction. We add into it incrementally
    /// from the heap-scan callback rather than at end-of-scan.
    /// `Option` because we don't know the dim until the first non-NULL
    /// row when reloptions didn't pin it.
    idx: Option<IdMapIndex>,
    /// Configured bit-width (1, 2, 4, or 8). Captured here so the
    /// callback can lazily construct `idx` once the first row pins
    /// `dim`.
    bit_width: usize,
    /// Expected dim. Set on the first non-NULL row, validated against
    /// every subsequent row. Once set, also pins `idx`.
    dim: Option<usize>,
    /// Optionally L2-normalise on insert.
    normalise: bool,
    /// Pending f32 staging buffer; flushed into `idx.add_with_ids`
    /// every `chunk_rows`. Bounded by `chunk_rows * dim * 4` bytes.
    pending_flat: Vec<f32>,
    /// Pending u64 ids, parallel to `pending_flat`.
    pending_ids: Vec<u64>,
    /// How many rows trigger a flush. Computed from
    /// `maintenance_work_mem` once `dim` is known; until then the
    /// callback buffers everything (which is fine — flushing at every
    /// row before `dim` is known is impossible anyway).
    chunk_rows: usize,
    /// Number of heap tuples we were called for (alive + dead).
    heap_seen: u64,
    /// Bounded rayon pool for the quantize phase. `add_with_ids` →
    /// `encode` fans out across this pool's threads instead of rayon's
    /// global all-cores pool. Sized from `turbovec.build_parallelism`
    /// (parity gap #2). `None` collapses to inline single-threaded
    /// encode. Held as a raw pointer rather than a borrow because
    /// `BuildState` is threaded through a `*mut c_void` FFI callback;
    /// the pool outlives every callback invocation (it lives on
    /// `ambuild`'s stack for the whole scan), so deref is sound.
    pool: *const rayon::ThreadPool,

    // ---- IVF-1 build state (only used when `lists > 0`) ----
    /// IVF coarse-cell count from `WITH (lists = N)`. `0` = flat
    /// (today's behaviour); the IVF fields below stay unused.
    lists: usize,
    /// When `lists > 0` we cannot flush incrementally: the
    /// cell-contiguous permutation needs the whole flat corpus before
    /// quantizing. We accumulate the (optionally normalised) flat
    /// vectors and their ids here, then permute + build the
    /// `IdMapIndex` once at end-of-scan. Bounded by the corpus size
    /// (IVF is for medium corpora where this fits; the Phase W
    /// streaming cap applies to the flat `lists == 0` path).
    ivf_flat: Vec<f32>,
    ivf_ids: Vec<u64>,
    /// Reservoir sample of ROTATED vectors for k-means training,
    /// capped at `256 * lists` rows (FAISS's rule of thumb). Filled
    /// during the scan so we don't double-scan. Row-major
    /// `sample_count * dim`.
    ivf_sample: Vec<f32>,
    /// Rows currently in `ivf_sample`.
    ivf_sample_count: usize,
    /// Total rows seen so far (for reservoir replacement probability).
    ivf_seen: u64,
    /// Deterministic RNG for reservoir sampling. Seeded from
    /// `ivf::IVF_SEED` so the sample (and thus the trained centroids)
    /// are reproducible across identical builds.
    ivf_rng: rand_chacha::ChaCha8Rng,
    /// Lazily-built rotation matrix (row-major `dim * dim`), used to
    /// rotate sampled vectors into the clustering space. Built once
    /// `dim` is known. Cells must live in the rotated space (the same
    /// space the fine quantizer + query use), so we rotate the
    /// L2-normalised vector exactly as turbovec's encode does.
    ivf_rotation: Option<Vec<f32>>,
}

impl BuildState {
    /// Compute the chunk-row threshold from PG's `maintenance_work_mem`
    /// GUC. The GUC is in **kilobytes** (PG convention — the variable
    /// is named `*_mem` but the unit is KB). We allocate 75% of it to
    /// the staging buffer (leaving headroom for the IdMapIndex's own
    /// growth and the surrounding allocator), capped at 1 GiB so a
    /// large `SET maintenance_work_mem = '8GB'` doesn't blow past the
    /// memory we were trying to bound.
    fn compute_chunk_rows(dim: usize) -> usize {
        // 1 GiB hard ceiling. Hoisted out of the function body to
        // avoid clippy::items_after_statements; the const is
        // associated with the staging-buffer policy, not the
        // function.
        const MAX_STAGING_BYTES: usize = 1024 * 1024 * 1024;
        // SAFETY: pg_sys::maintenance_work_mem is a global C int that
        // is set during postmaster startup and never goes negative.
        // Reading it from a backend is safe.
        let mwm_kb = unsafe { pg_sys::maintenance_work_mem }.max(0) as usize;
        // Saturating math because mwm_kb * 1024 could overflow on a
        // hypothetical 32-bit build; on 64-bit it can't, but be safe.
        let bytes = mwm_kb
            .saturating_mul(1024)
            .saturating_mul(3)
            / 4;
        let chunk_bytes = bytes.min(MAX_STAGING_BYTES);
        let row_bytes = dim.saturating_mul(std::mem::size_of::<f32>()).max(1);
        (chunk_bytes / row_bytes).max(1)
    }

    /// IVF-1 sample cap: `256 * lists` rows (FAISS's rule of thumb).
    /// Bounds the k-means training set so the sample stays small
    /// regardless of corpus size (Phase W streaming spirit).
    fn ivf_sample_cap(&self) -> usize {
        self.lists.saturating_mul(256)
    }

    /// Rotate an L2-normalised vector into the clustering (rotated)
    /// space, mirroring turbovec's encode: `rotated[k] = sum_j R[k*dim+j]
    /// * unit[j]` (i.e. `unit @ R^T`). `unit` must already be
    /// L2-normalised and `dim`-length. Returns a `dim`-length Vec.
    fn rotate_unit(rotation: &[f32], unit: &[f32], dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dim];
        for (k, o) in out.iter_mut().enumerate() {
            let rrow = &rotation[k * dim..(k + 1) * dim];
            let mut s = 0.0f32;
            for j in 0..dim {
                s += rrow[j] * unit[j];
            }
            *o = s;
        }
        out
    }

    /// Reservoir-sample one ROTATED row for k-means training. Called
    /// per accepted vector when `lists > 0`. `raw` is the raw (un-
    /// normalised) input slice; we normalise + rotate before storing
    /// so the sample lives in the clustering space.
    fn ivf_reservoir_push(&mut self, raw: &[f32], dim: usize) {
        use rand::Rng;
        let rotation = self
            .ivf_rotation
            .as_ref()
            .expect("ivf_reservoir_push before rotation built");
        // Normalise then rotate, matching turbovec's encode pipeline.
        let unit = kernels::normalise_to_vec(raw);
        let rotated = Self::rotate_unit(rotation, &unit, dim);

        let cap = self.ivf_sample_cap();
        self.ivf_seen += 1;
        if self.ivf_sample_count < cap {
            self.ivf_sample.extend_from_slice(&rotated);
            self.ivf_sample_count += 1;
        } else {
            // Classic reservoir replacement: replace a random slot
            // with probability cap / seen.
            let j = self.ivf_rng.gen_range(0..self.ivf_seen);
            if (j as usize) < cap {
                let base = (j as usize) * dim;
                self.ivf_sample[base..base + dim].copy_from_slice(&rotated);
            }
        }
    }

    /// Drain `pending_flat` / `pending_ids` into the IdMapIndex and
    /// release the staging memory back to the allocator. The
    /// `shrink_to_fit` calls are load-bearing: without them, the
    /// staging buffers would stay sized at the peak and Phase W's
    /// memory cap would only apply to the *first* chunk.
    fn flush(&mut self) {
        if self.pending_ids.is_empty() {
            return;
        }
        let idx = self
            .idx
            .as_mut()
            .expect("turbovec ambuild: flush called before idx was constructed");
        // Run the per-vector quantize (turbovec's rayon-parallel
        // `encode`) inside the GUC-sized build pool when present.
        // SAFETY: `pool` either is null or points at a `ThreadPool`
        // that lives on `ambuild`'s stack for the entire scan, which
        // strictly outlives this callback-driven flush.
        let pool: Option<&rayon::ThreadPool> =
            unsafe { self.pool.as_ref() };
        let pending_flat = &self.pending_flat;
        let pending_ids = &self.pending_ids;
        let res = super::build_pool::install(pool, || {
            idx.add_with_ids(pending_flat, pending_ids)
        });
        if let Err(e) = res {
            error!("turbovec ambuild: add_with_ids failed: {:?}", e);
        }
        self.pending_flat.clear();
        self.pending_flat.shrink_to_fit();
        self.pending_ids.clear();
        self.pending_ids.shrink_to_fit();
    }
}

/// `ambuild`: scan the heap, build the IdMapIndex, persist it.
///
/// # Safety
///
/// Caller is PostgreSQL's index machinery. The two `Relation`
/// pointers are valid for the duration of the call; the
/// `IndexInfo` pointer too. We must return a palloc'd
/// `IndexBuildResult` populated with row counts.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn ambuild(
    heap_relation: pg_sys::Relation,
    index_relation: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    if result.is_null() {
        error!("turbovec: failed to allocate IndexBuildResult");
    }

    let (cfg_bit_width, cfg_dim, cfg_lists) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    let normalise = guc::NORMALIZE_ON_INSERT.get();
    let lists = cfg_lists.max(0) as usize;

    let initial_dim = if cfg_dim > 0 {
        if (cfg_dim as usize) % 8 != 0 {
            error!(
                "turbovec ambuild: dim must be a multiple of 8 (got {})",
                cfg_dim
            );
        }
        Some(cfg_dim as usize)
    } else {
        None
    };

    let mut state = BuildState {
        idx: None,
        bit_width: cfg_bit_width as usize,
        dim: initial_dim,
        normalise,
        pending_flat: Vec::new(),
        pending_ids: Vec::new(),
        // Lazily computed from `dim` on the first chunk; if dim was
        // pinned via reloptions we can compute it up front.
        chunk_rows: usize::MAX,
        heap_seen: 0,
        pool: std::ptr::null(),
        lists,
        ivf_flat: Vec::new(),
        ivf_ids: Vec::new(),
        ivf_sample: Vec::new(),
        ivf_sample_count: 0,
        ivf_seen: 0,
        ivf_rng: <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(ivf::IVF_SEED),
        ivf_rotation: None,
    };
    // Parity gap #2: a bounded rayon pool sized from
    // `turbovec.build_parallelism` (auto = max_parallel_maintenance_workers
    // + 1). The encode and repack phases `install` onto it. `None` =
    // single thread; we run inline. The pool lives on this stack frame
    // for the whole scan, so the raw pointer stashed on `state` stays
    // valid for every `build_callback` / `flush` invocation.
    let build_pool = super::build_pool::make_pool();
    state.pool = build_pool
        .as_ref()
        .map_or(std::ptr::null(), |p| p as *const rayon::ThreadPool);
    if let Some(d) = state.dim {
        // Pre-construct the IdMapIndex when reloptions pinned the dim.
        // Otherwise we wait for the first non-NULL row and construct
        // there. Either way the per-row callback path is identical.
        state.idx = Some(
            IdMapIndex::new(d, cfg_bit_width as usize)
                .expect("turbovec ambuild: invalid (dim, bit_width) for IdMapIndex::new"),
        );
        state.chunk_rows = BuildState::compute_chunk_rows(d);
        // IVF: build the rotation matrix up front so sampled vectors
        // can be rotated into the clustering space during the scan.
        if state.lists > 0 {
            state.ivf_rotation = Some(turbovec::rotation::make_rotation_matrix(d));
        }
    }

    // Pull the table AM's index_build_range_scan and invoke it with
    // our callback. This is functionally `table_index_build_scan` in
    // C — the static inline helper that pgrx-pg-sys doesn't expose
    // as a Rust function.
    let table_am = (*heap_relation).rd_tableam;
    if table_am.is_null() {
        error!("turbovec ambuild: heap relation has no table AM");
    }
    let scan_fn = (*table_am)
        .index_build_range_scan
        .unwrap_or_else(|| error!("turbovec ambuild: table AM lacks index_build_range_scan"));

    let n_seen = scan_fn(
        heap_relation,
        index_relation,
        index_info,
        /* allow_sync   */ true,
        /* anyvisible   */ false,
        /* progress     */ true,
        /* start_blockno */ 0,
        /* numblocks    */ pg_sys::InvalidBlockNumber,
        Some(build_callback),
        (&mut state) as *mut BuildState as *mut std::ffi::c_void,
        std::ptr::null_mut(),
    );
    state.heap_seen = n_seen as u64;
    // Drain any rows the heap scan left in the staging buffers.
    // (Phase W: heap scan flushes whenever `pending_ids.len() >=
    // chunk_rows`; the trailing partial chunk is flushed here.)
    state.flush();

    // If the heap was empty and dim was not pinned, write an
    // empty meta page so subsequent aminserts have stable state
    // to extend.
    let Some(dim) = state.dim else {
        relfile::write_full(
            index_relation,
            cfg_bit_width as u8,
            /*dim=*/ 0,
            /*n_vectors=*/ 0,
            &[],
            &[],
            &[],
            /*am_version=*/ 1,
        );
        let _ = indexrelid;
        (*result).heap_tuples = state.heap_seen as f64;
        (*result).index_tuples = 0.0;
        return result;
    };

    // IVF-1 build path: train coarse centroids, assign, permute the
    // flat corpus cell-contiguous, then run the EXISTING
    // quantize+pack on the permuted order. The scan path stays flat
    // in IVF-1; we persist cells purely to prove the v3->v4 wire
    // change round-trips (cell-restricted search is IVF-2).
    if state.lists > 0 {
        let n_built = ivf_build_and_write(
            index_relation,
            &mut state,
            dim,
            cfg_bit_width as u8,
            build_pool.as_ref(),
        );
        let _ = indexrelid;
        (*result).heap_tuples = state.heap_seen as f64;
        (*result).index_tuples = n_built as f64;
        return result;
    }

    // The IdMapIndex was either pre-constructed (dim pinned by
    // reloptions) or built lazily by the callback when the first
    // non-NULL row arrived. If `state.dim` is set we must have an
    // `idx`.
    let idx = state
        .idx
        .take()
        .expect("turbovec ambuild: state.dim is set but state.idx is None");
    let n_vectors = idx.slot_to_id().len() as i64;
    // Phase P: pre-bake the SIMD-blocked layout and the
    // Lloyd-Max codebook now, while we own the freshly-built
    // index. Persisting them alongside the row-major codes
    // means every backend opening this index for the first
    // time skips the per-backend ~12–15 s `pack::repack` and
    // ~5–8 s codebook compute. The prepare step is not free
    // here, but ambuild is the right place to pay it: it
    // already runs once per index and at the only point
    // where we definitively know n_vectors won't change
    // until the next mutation.
    //
    // v1.7.1 revert note: Phase W-2 (v1.7.0) split this into
    // write_packed_phase + take_packed_codes + prepare_eager +
    // write_blocked_phase_and_meta in an attempt to avoid
    // packed_codes and blocked being co-resident. Validation
    // on `meh` at 10 M × 1536-d showed the split made the
    // build 53% slower with no actual RSS reduction (the
    // "freed" heap pages just migrate to pinned shared
    // buffers, which `ps -o rss` still counts). Reverted to
    // the v1.6.0 single-call path. See
    // an internal design note and the validation JSON at
    // `benches/results/phase_w_2_validate_meh_10m_2026_05_27.json`.
    if n_vectors > 0 {
        // Phase R-2 + parity gap #2: prepare_eager builds the
        // SIMD-blocked layout (`pack::repack`) and the Lloyd-Max
        // codebook. We run it under the same GUC-sized pool so that
        // (a) the codebook's per-coord work and any future
        // parallelised repack inherit the bounded pool rather than
        // rayon's global all-cores pool, and (b) the pool's threads
        // are reused for this phase instead of falling back to the
        // global pool mid-build. The blocked layout is a deterministic
        // function of the (already-deterministic) row-major codes, so
        // this stays byte-identical to a serial build regardless of
        // pool size.
        super::build_pool::install(build_pool.as_ref(), || idx.prepare_eager());
        // also drive the rotation `OnceLock` so we
        // can persist the matrix alongside the codes. The
        // rotation is a deterministic function of `(dim,
        // ROTATION_SEED)` whose lazy QR was the warm-scan
        // hotspot Phase R diagnosed (~64% self time at
        // dim = 1536). Computing it once here — we already pay
        // QR on the first search of every backend in the
        // pre-Phase-R-2 path — lets every backend reading the
        // index skip it forever.
        let rotation = idx.rotation();
        let prepared = relfile::PreparedParts {
            blocked_codes: idx.blocked_codes(),
            n_blocks: idx.n_blocks() as u32,
            centroids: idx.centroids(),
            boundaries: idx.boundaries(),
            rotation,
        };
        relfile::write_full_with_prepared(
            index_relation,
            cfg_bit_width as u8,
            dim as u32,
            n_vectors as u64,
            idx.packed_codes(),
            idx.scales(),
            idx.slot_to_id(),
            1,
            prepared,
        );
    } else {
        // Empty index: no prepared layout to persist.
        relfile::write_full(
            index_relation,
            cfg_bit_width as u8,
            dim as u32,
            0,
            idx.packed_codes(),
            idx.scales(),
            idx.slot_to_id(),
            1,
        );
    }
    let _ = indexrelid;

    (*result).heap_tuples = state.heap_seen as f64;
    (*result).index_tuples = n_vectors as f64;
    result
}

/// IVF-1 build finisher (`lists > 0`). Trains coarse centroids on the
/// reservoir sample, single-assigns every accumulated vector to its
/// nearest centroid (in the rotated space), computes a stable
/// cell-contiguous permutation, applies it to the flat corpus + ids,
/// runs the EXISTING quantize+pack on the permuted order, then
/// persists codes/scales/ids (cell-contiguous) plus the coarse
/// centroids (f32) and the cell directory via the v4 relfile path.
///
/// Returns the number of vectors written. Consumes `state.ivf_flat` /
/// `state.ivf_ids` / `state.ivf_sample`.
///
/// # Safety
///
/// `index_relation` is a valid, exclusively-held index relation
/// (ambuild holds it for the whole build).
unsafe fn ivf_build_and_write(
    index_relation: pg_sys::Relation,
    state: &mut BuildState,
    dim: usize,
    bit_width: u8,
    build_pool: Option<&rayon::ThreadPool>,
) -> usize {
    let n_vectors = state.ivf_ids.len();

    // Empty corpus: write an empty (flat-shaped) meta page. An IVF
    // index over zero rows has no cells; readers treat it as empty.
    if n_vectors == 0 {
        relfile::write_full(
            index_relation,
            bit_width,
            dim as u32,
            0,
            &[],
            &[],
            &[],
            1,
        );
        return 0;
    }

    // `lists` can exceed the corpus size; cap it so we never train
    // more centroids than we have points (k-means handles n < k, but
    // an over-large nlist wastes cells). Keep at least 1.
    let lists = state.lists.min(n_vectors).max(1);

    let rotation = state
        .ivf_rotation
        .take()
        .expect("ivf_build_and_write: rotation matrix not built");

    // 1. Train coarse centroids on the (rotated) reservoir sample.
    //    Deterministic: seeded k-means++ + Lloyd's (see ivf.rs).
    let sample = std::mem::take(&mut state.ivf_sample);
    let sample_count = state.ivf_sample_count;
    let model = super::build_pool::install(build_pool, || {
        ivf::train_kmeans(&sample, sample_count, lists, dim)
    });
    drop(sample);

    // 2. Assign every (rotated) vector to its nearest centroid. One
    //    streamed sweep over the accumulated flat corpus; we rotate
    //    each L2-normalised vector on the fly (matching the sample
    //    space) so we don't keep a second rotated copy of the corpus.
    //    Parallelised over the build pool with a stable map (rayon
    //    preserves index order in `map`), so the assignment is
    //    deterministic.
    let flat = std::mem::take(&mut state.ivf_flat);
    let assignment: Vec<u32> = super::build_pool::install(build_pool, || {
        use rayon::prelude::*;
        (0..n_vectors)
            .into_par_iter()
            .map(|i| {
                let raw = &flat[i * dim..(i + 1) * dim];
                let unit = kernels::normalise_to_vec(raw);
                let rotated = BuildState::rotate_unit(&rotation, &unit, dim);
                model.assign_one(&rotated) as u32
            })
            .collect()
    });

    // 3. Stable cell-contiguous permutation + cell directory.
    let (permutation, directory) = ivf::build_permutation(&assignment, lists);
    debug_assert!(directory.validate_partition(n_vectors as u64).is_ok());

    // 4. Apply the permutation to the flat corpus + ids. `perm[new] =
    //    old`, so new_slot i takes old_slot perm[i].
    let ids = std::mem::take(&mut state.ivf_ids);
    let mut perm_flat = vec![0.0f32; n_vectors * dim];
    let mut perm_ids = vec![0u64; n_vectors];
    for new_slot in 0..n_vectors {
        let old_slot = permutation[new_slot] as usize;
        perm_flat[new_slot * dim..(new_slot + 1) * dim]
            .copy_from_slice(&flat[old_slot * dim..(old_slot + 1) * dim]);
        perm_ids[new_slot] = ids[old_slot];
    }
    drop(flat);
    drop(ids);

    // 5. Run the EXISTING quantize+pack on the permuted order. The
    //    fine quantizer doesn't care about order; slot_to_id reflects
    //    the permutation, and the codes/scales/ids land
    //    cell-contiguous on disk.
    let mut idx = IdMapIndex::new(dim, bit_width as usize)
        .expect("turbovec ambuild (ivf): invalid (dim, bit_width)");
    super::build_pool::install(build_pool, || {
        idx.add_with_ids(&perm_flat, &perm_ids)
            .expect("turbovec ambuild (ivf): add_with_ids failed")
    });
    drop(perm_flat);
    drop(perm_ids);

    let built = idx.slot_to_id().len();
    debug_assert_eq!(built, n_vectors);

    super::build_pool::install(build_pool, || idx.prepare_eager());
    let idx_rotation = idx.rotation();
    let prepared = relfile::PreparedParts {
        blocked_codes: idx.blocked_codes(),
        n_blocks: idx.n_blocks() as u32,
        centroids: idx.centroids(),
        boundaries: idx.boundaries(),
        rotation: idx_rotation,
    };
    // Coarse centroids are already in the rotated space (trained on
    // rotated samples); persist as-is. The cell directory packs the
    // per-cell (code_offset, n_vectors) ranges.
    let cell_dir_bytes = directory.encode();
    let ivf_parts = relfile::IvfParts {
        lists: lists as u32,
        coarse_centroids: &model.centroids,
        cell_dir_bytes: &cell_dir_bytes,
    };
    relfile::write_full_with_prepared_ivf(
        index_relation,
        bit_width,
        dim as u32,
        n_vectors as u64,
        idx.packed_codes(),
        idx.scales(),
        idx.slot_to_id(),
        1,
        prepared,
        ivf_parts,
    );

    built
}

/// Per-tuple callback invoked by `index_build_range_scan`. We treat
/// dead tuples (`tuple_is_alive == false`) like NULL: they are skipped
/// rather than indexed, matching pgvector's policy.
unsafe extern "C-unwind" fn build_callback(
    index_relation: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    tuple_is_alive: bool,
    state_ptr: *mut std::ffi::c_void,
) {
    if !tuple_is_alive {
        return;
    }
    let _ = index_relation;

    let state = &mut *(state_ptr as *mut BuildState);
    if *isnull {
        return;
    }
    let datum = *values;
    let value: Option<Vector> = pgrx::FromDatum::from_datum(datum, false);
    let Some(value) = value else {
        return;
    };

    let row_dim = value.dim();
    if row_dim == 0 {
        return;
    }
    if row_dim % 8 != 0 {
        error!(
            "turbovec ambuild: dim must be a multiple of 8 (got {})",
            row_dim
        );
    }
    match state.dim {
        Some(d) if d != row_dim => {
            error!(
                "turbovec ambuild: dim mismatch — first row had dim {}, this row has {}",
                d, row_dim
            );
        }
        None => {
            state.dim = Some(row_dim);
            // First non-NULL row pinning the dim. Lazily construct
            // the IdMapIndex and compute the chunk threshold now.
            state.idx = Some(
                IdMapIndex::new(row_dim, state.bit_width)
                    .expect("turbovec ambuild: invalid (dim, bit_width) for IdMapIndex::new"),
            );
            state.chunk_rows = BuildState::compute_chunk_rows(row_dim);
            // IVF: build the rotation matrix now that dim is pinned,
            // so reservoir samples land in the clustering space.
            if state.lists > 0 && state.ivf_rotation.is_none() {
                state.ivf_rotation =
                    Some(turbovec::rotation::make_rotation_matrix(row_dim));
            }
        }
        _ => {}
    }

    let id = pgrx::itemptr::item_pointer_to_u64(*tid);

    // IVF-1 build path (`lists > 0`): we can't flush incrementally
    // — the cell-contiguous permutation needs the whole flat corpus
    // before quantizing. Accumulate the (optionally normalised) flat
    // vector + id, and reservoir-sample the ROTATED vector for
    // k-means. The permute + quantize + IVF persist happens at
    // end-of-scan in `ambuild`.
    if state.lists > 0 {
        if state.normalise {
            let mut buf = vec![0.0_f32; row_dim];
            kernels::normalise_into(&mut buf, value.as_slice());
            state.ivf_flat.extend_from_slice(&buf);
        } else {
            state.ivf_flat.extend_from_slice(value.as_slice());
        }
        state.ivf_ids.push(id);
        // Reservoir sample always rotates the L2-normalised vector
        // (cells live in the rotated unit-sphere space, matching
        // turbovec's encode), regardless of the `normalise` GUC.
        state.ivf_reservoir_push(value.as_slice(), row_dim);
        return;
    }

    if state.normalise {
        let mut buf = vec![0.0_f32; row_dim];
        kernels::normalise_into(&mut buf, value.as_slice());
        state.pending_flat.extend_from_slice(&buf);
    } else {
        state.pending_flat.extend_from_slice(value.as_slice());
    }
    state.pending_ids.push(id);

    // Phase W: flush whenever the staging buffer reaches the
    // configured chunk size. Caps peak staging memory at
    // O(min(maintenance_work_mem, 1 GiB)) instead of
    // O(n_vectors * dim * 4 bytes).
    if state.pending_ids.len() >= state.chunk_rows {
        state.flush();
    }
}

/// `ambuildempty`: called only for **unlogged** indexes; PG uses
/// it to initialise the INIT fork (`INIT_FORKNUM`) so the index
/// can be reset after a crash. Logged indexes (the common case)
/// never invoke this callback — PG calls `ambuild` instead.
///
/// Phase L hardening (item 2): we allocate a single empty meta
/// page in `INIT_FORKNUM` and WAL-log it via `GenericXLog`.
/// After a crash PG copies the init fork over the main fork,
/// restoring the index to a known-empty state. This matches
/// pgvector's `HnswBuildEmpty` pattern.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn ambuildempty(index_relation: pg_sys::Relation) {
    let (bw, dim, _lists) = options::read(index_relation);

    // Plan an empty layout. dim may be 0 if the user didn't
    // pin it via reloptions — the meta page records 0/0/0,
    // which our reader handles as "empty index".
    let dim_u32 = if dim > 0 { dim as u32 } else { 0 };
    if dim_u32 % 8 == 0 {
        let meta = crate::index::page::MetaPageData::plan(
            bw as u8, dim_u32, /*n_vectors=*/ 0, /*am_version=*/ 1,
        );
        relfile::write_meta_in_fork(
            index_relation,
            pg_sys::ForkNumber::INIT_FORKNUM,
            &meta,
        );
    }
    // If dim was supplied as a non-multiple-of-8 we silently
    // skip the init-fork write; the next ambuild will error
    // with a more informative message.
}
