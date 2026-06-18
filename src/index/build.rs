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

/// Phase B-4: a disk-backed corpus spill for the out-of-core IVF
/// build. Each record is a fixed-stride `(u64 id, dim x f32 vector)`
/// row, appended in heap-scan order. The IVF build streams the
/// corpus through this file instead of holding the whole f32 corpus
/// (and a second cell-permuted copy) resident in RAM.
///
/// Backed by PG's `BufFile` temp-file machinery: the file lands in
/// `PGDATA/base/pgsql_tmp` (or a `temp_tablespaces` member), counts
/// against `temp_file_limit`, and is registered with the current
/// resource owner so it is unlinked at transaction end / on abort.
/// `Drop` also calls `BufFileClose`, which closes + unlinks the
/// segment files, so a build that errors partway (or panics) never
/// leaks a spill file. `BufFileCreateTemp(false)` makes it
/// transaction-local (not inter-xact), which is correct: the build
/// is a single transaction.
/// `BufFileWrite` shim. PG13-15 declare the pointer as `*mut
/// c_void`; PG16+ as `*const c_void`. Both return `void` (they
/// ereport on I/O failure). This casts to the right pointer type per
/// version so the build compiles across the pg13..pg18 matrix.
///
/// # Safety
/// `file` is a valid `BufFile`; `[ptr, ptr+size)` is readable.
#[inline]
unsafe fn buffile_write(file: *mut pg_sys::BufFile, ptr: *const std::ffi::c_void, size: usize) {
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
    pg_sys::BufFileWrite(file, ptr as *mut std::ffi::c_void, size);
    #[cfg(not(any(feature = "pg13", feature = "pg14", feature = "pg15")))]
    pg_sys::BufFileWrite(file, ptr, size);
}

/// `BufFileReadExact` shim. PG16+ ship `BufFileReadExact` (ereports
/// on a short read); PG13-15 do not, so we use `BufFileRead` (which
/// returns the bytes actually read, present on every version) and
/// ERROR ourselves on a short read. Uniform "read exactly `size`
/// bytes or error" semantics across the matrix.
///
/// # Safety
/// `file` is a valid `BufFile`; `[ptr, ptr+size)` is writable.
#[inline]
unsafe fn buffile_read_exact(file: *mut pg_sys::BufFile, ptr: *mut std::ffi::c_void, size: usize) {
    let got = pg_sys::BufFileRead(file, ptr, size);
    if got != size {
        error!(
            "turbovec ambuild (ivf): short read from corpus spill (wanted {size} bytes, got {got}) -- truncated temp file?"
        );
    }
}

struct CorpusSpill {
    /// The PG temp BufFile. Non-null between `new` and `Drop`.
    file: *mut pg_sys::BufFile,
    /// Vector dimensionality; the record stride is `8 + dim*4`.
    dim: usize,
    /// Bytes per record: `size_of::<u64>() + dim * size_of::<f32>()`.
    stride: usize,
    /// Number of records appended so far.
    rows: usize,
}

impl CorpusSpill {
    /// Create a fresh transaction-local spill file.
    ///
    /// # Safety
    /// Must run inside a transaction with a valid CurrentResourceOwner
    /// (true throughout `ambuild`).
    unsafe fn new(dim: usize) -> Self {
        let file = pg_sys::BufFileCreateTemp(false);
        if file.is_null() {
            error!("turbovec ambuild (ivf): BufFileCreateTemp returned null");
        }
        let stride = std::mem::size_of::<u64>() + dim * std::mem::size_of::<f32>();
        CorpusSpill {
            file,
            dim,
            stride,
            rows: 0,
        }
    }

    /// Append one `(id, vector)` record. `vector.len()` must equal
    /// `dim`. Writes the 8-byte little-endian id then the dim f32s
    /// (native-endian, matching how we read them back on the same
    /// host within the same build).
    fn push(&mut self, id: u64, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        // SAFETY: `file` is a valid BufFile for our lifetime; the two
        // writes total exactly `stride` bytes. BufFileWrite ereports
        // (PG longjmp) on I/O failure, which pgrx converts to a Rust
        // panic that unwinds through Drop -> BufFileClose.
        unsafe {
            let id_le = id.to_le_bytes();
            buffile_write(
                self.file,
                id_le.as_ptr() as *const std::ffi::c_void,
                id_le.len(),
            );
            buffile_write(
                self.file,
                vector.as_ptr() as *const std::ffi::c_void,
                self.dim * std::mem::size_of::<f32>(),
            );
        }
        self.rows += 1;
    }

    /// Sequentially read `rows` records starting at record index
    /// `start`, filling `ids[0..rows]` and the row-major
    /// `out[0..rows*dim]` vector block. Seeks to the start record
    /// then reads contiguously (the cheap path for the assign sweep).
    fn read_block(&self, start: usize, rows: usize, ids: &mut [u64], out: &mut [f32]) {
        debug_assert!(start + rows <= self.rows);
        debug_assert!(ids.len() >= rows);
        debug_assert!(out.len() >= rows * self.dim);
        unsafe {
            self.seek(start);
            for r in 0..rows {
                self.read_one_at_cursor(
                    &mut ids[r],
                    &mut out[r * self.dim..(r + 1) * self.dim],
                );
            }
        }
    }

    /// Read a single record at record index `idx` (random access).
    /// Used by the cell-order write sweep, where the permutation
    /// scatters reads across the file.
    fn read_one(&self, idx: usize, id: &mut u64, vector: &mut [f32]) {
        debug_assert!(idx < self.rows);
        debug_assert_eq!(vector.len(), self.dim);
        unsafe {
            self.seek(idx);
            self.read_one_at_cursor(id, vector);
        }
    }

    /// Seek to the byte offset of record `idx`. `BufFileSeek` with
    /// `whence = SEEK_SET` and `fileno = 0` takes an ABSOLUTE byte
    /// offset from the start of the logical stream: its internal
    /// `while (newOffset > MAX_PHYSICAL_FILESIZE)` loop walks the
    /// 1 GiB segments to find the right (segment, in-segment-offset)
    /// pair, so a single absolute offset addresses a multi-segment
    /// (> 1 GiB) spill correctly. Verified against PG's buffile.c.
    ///
    /// # Safety
    /// `file` must be valid; `idx <= rows`.
    unsafe fn seek(&self, idx: usize) {
        // SEEK_SET from <stdio.h>; not exposed by pgrx-pg-sys, and
        // its value (0) is fixed by POSIX. BufFileSeek takes the same
        // whence constants as fseek.
        const SEEK_SET: i32 = 0;
        let offset = (idx * self.stride) as pg_sys::off_t;
        // BufFileSeek(file, fileno=0, offset, SEEK_SET): position the
        // logical stream `offset` bytes from the start. BufFile maps
        // the (fileno=0, offset) pair onto its 1 GiB segments
        // internally, so a single absolute offset from segment 0 is
        // the correct addressing for our < a-few-GiB-per-row stream.
        let rc = pg_sys::BufFileSeek(self.file, 0, offset, SEEK_SET);
        if rc != 0 {
            error!("turbovec ambuild (ivf): BufFileSeek to record {idx} failed (rc={rc})");
        }
    }

    /// Read one record at the current cursor into `id` + `vector`.
    ///
    /// # Safety
    /// Cursor must be positioned at a record boundary with a full
    /// record remaining.
    unsafe fn read_one_at_cursor(&self, id: &mut u64, vector: &mut [f32]) {
        let mut idbuf = [0u8; std::mem::size_of::<u64>()];
        buffile_read_exact(
            self.file,
            idbuf.as_mut_ptr() as *mut std::ffi::c_void,
            idbuf.len(),
        );
        *id = u64::from_le_bytes(idbuf);
        buffile_read_exact(
            self.file,
            vector.as_mut_ptr() as *mut std::ffi::c_void,
            self.dim * std::mem::size_of::<f32>(),
        );
    }
}

impl Drop for CorpusSpill {
    fn drop(&mut self) {
        if !self.file.is_null() {
            // BufFileClose flushes, closes, and unlinks the temp
            // segment file(s). Runs on the normal path AND on an
            // unwinding panic (a failed build), so the spill never
            // leaks. SAFETY: `file` was created by BufFileCreateTemp
            // and not previously closed.
            unsafe { pg_sys::BufFileClose(self.file) };
            self.file = std::ptr::null_mut();
        }
    }
}

use crate::guc;
use crate::index::ivf;
use crate::index::{options, relfile};
use crate::kernels;
use crate::vec::Vector;

/// Phase F-2: is this index a ColBERT / multivector token index?
///
/// Detected from the indexed attribute's type: a ColBERT index is
/// built over a `turbovec.vector[]` column (opclass `vec_colbert_ops`),
/// so attribute 1's type is an ARRAY type (`get_element_type` returns
/// a valid element OID). A single-vector index is over a scalar
/// `vector` column (`get_element_type` returns `InvalidOid`). The only
/// way to reach the turbovec AM with an array column is via the
/// `vec_colbert_ops` opclass, so "attribute type is an array" is an
/// exact discriminator.
///
/// # Safety
/// `index_relation` is a valid index relation with a populated
/// tuple descriptor (true throughout `ambuild` / `ambuildempty`).
unsafe fn is_colbert_index(index_relation: pg_sys::Relation) -> bool {
    let tupdesc = (*index_relation).rd_att;
    if tupdesc.is_null() {
        return false;
    }
    // PgTupleDesc handles the pg13..pg18 attr-layout differences
    // (pg18 switched to CompactAttribute), so read attribute 0
    // through it rather than touching `.attrs` directly.
    let desc = pgrx::PgTupleDesc::from_pg_unchecked(tupdesc);
    let Some(att) = desc.get(0) else {
        return false;
    };
    // get_element_type(arrayoid) -> element oid, or InvalidOid for a
    // non-array type. A non-zero element type ⇒ the indexed column is
    // an array ⇒ ColBERT.
    pg_sys::get_element_type(att.atttypid) != pg_sys::InvalidOid
}

/// Default IVF `nlist` for a ColBERT token index built without an
/// explicit `WITH (lists = N)`. A token index has many more slots
/// than docs, so it is always IVF; this modest default lets a fresh
/// `CREATE INDEX ... (tokens vec_colbert_ops)` build real cells on a
/// small corpus without the user having to tune nlist. Production
/// corpora should set `WITH (lists ~ sqrt(n_tokens))`. The build caps
/// `lists` at the slot count, so an over-large default on a tiny
/// corpus is harmless.
const DEFAULT_COLBERT_LISTS: usize = 64;

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
    /// Phase F-2: this is a ColBERT / multivector token index (the
    /// indexed column is `turbovec.vector[]`, opclass
    /// `vec_colbert_ops`). When set, `build_callback` UNNESTS each
    /// heap tuple's `vector[]` into N token slots (each tagged with
    /// the doc's TID, repeated), and the finished meta is stamped
    /// `mark_colbert()` (wire v5). A token index is always IVF
    /// (n_tokens is large; cell-contiguous layout makes OOC stage-1
    /// work), so the colbert path forces `lists > 0`. The
    /// single-vector path leaves this `false` and is UNCHANGED.
    colbert: bool,
    /// IVF-4a soft-assignment multiplicity `M` from
    /// `WITH (assign_dups = M)`. `1` = single assignment (each vector
    /// in exactly one cell); `M > 1` stores boundary vectors in their
    /// top-M nearest cells. Only consulted when `lists > 0`.
    assign_dups: usize,
    /// When `lists > 0` we no longer accumulate the full f32 corpus
    /// in RAM. Phase B-4: the (optionally normalised) vectors + ids
    /// are spilled to a PG temp file (`CorpusSpill`) during the heap
    /// scan, then streamed back in `maintenance_work_mem`-bounded
    /// chunks for assignment and cell-order quantization. This
    /// eliminates the two full-corpus f32 copies (`ivf_flat` +
    /// `perm_flat`) that OOM-killed 1M+ IVF builds. `None` until the
    /// first non-NULL row pins `dim` (the record stride needs `dim`).
    ivf_spill: Option<CorpusSpill>,
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

    let (cfg_bit_width, cfg_dim, cfg_lists, cfg_assign_dups) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    let normalise = guc::NORMALIZE_ON_INSERT.get();
    let lists = cfg_lists.max(0) as usize;
    let assign_dups = cfg_assign_dups.clamp(1, options::MAX_ASSIGN_DUPS) as usize;

    // Phase F-2: detect the ColBERT / multivector token index kind
    // from the indexed column type (an array ⇒ `vec_colbert_ops`). A
    // token index is intrinsically IVF (n_tokens is large; the
    // cell-contiguous layout is what makes the out-of-core stage-1
    // token search work), so when the user didn't pin `lists` we
    // default it to a sane nlist. `assign_dups` is honoured as-is.
    let colbert = is_colbert_index(index_relation);
    let lists = if colbert && lists == 0 {
        // Default nlist for a colbert build that didn't specify
        // `WITH (lists = N)`. A token index has many more slots than
        // docs; a modest fixed default keeps small synthetic corpora
        // (the tests) building real cells while staying well under
        // MAX_LISTS. Production users tune via WITH (lists = ...).
        DEFAULT_COLBERT_LISTS
    } else {
        lists
    };

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
        ivf_spill: None,
        ivf_sample: Vec::new(),
        ivf_sample_count: 0,
        ivf_seen: 0,
        ivf_rng: <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(ivf::IVF_SEED),
        ivf_rotation: None,
        lists,
        colbert,
        assign_dups,
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
            // Phase B-4: open the disk spill now that the record
            // stride (8 + d*4) is known. The heap scan streams every
            // accepted vector into it instead of `ivf_flat`.
            state.ivf_spill = Some(CorpusSpill::new(d));
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
    // `docs/PHASE_W_PROGRESS.md` and the validation JSON at
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

/// Row-block size for the batched IVF assignment GEMM. We bound the
/// transient rotated row-block by 75% of `maintenance_work_mem`
/// (the same policy `compute_chunk_rows` uses for the flat staging
/// buffer), capped at 64k rows so a large `maintenance_work_mem`
/// doesn't make one GEMM tile absurdly tall. At dim = 1024 a 64k-row
/// block is ~256 MiB of f32; the full 1M-row rotated corpus would be
/// 4 GiB, so chunking keeps peak bounded. Floor of 1 row.
fn ivf_assign_block_rows(dim: usize) -> usize {
    const MAX_BLOCK_ROWS: usize = 64 * 1024;
    // Reuse the staging-buffer sizing: 75% of maintenance_work_mem,
    // capped at 1 GiB, divided by the per-row f32 byte cost.
    let by_mem = BuildState::compute_chunk_rows(dim);
    by_mem.clamp(1, MAX_BLOCK_ROWS)
}

/// IVF-1 build finisher (`lists > 0`). Phase B-4: an OUT-OF-CORE
/// streaming build. Trains coarse centroids on the bounded reservoir
/// sample, assigns every vector to its cell(s) in a streamed sweep
/// over the disk spill (keeping only the per-row cell-id assignment
/// array, not a corpus copy), computes the stable cell-contiguous
/// permutation, then feeds the quantizer in cell order by re-reading
/// the spill at the permuted offsets in `maintenance_work_mem`-bounded
/// chunks. Persists codes/scales/ids (cell-contiguous) plus the
/// coarse centroids (f32) and the cell directory via the v4 relfile
/// path.
///
/// The full f32 corpus is NEVER resident: it lives on `state.ivf_spill`
/// and only a bounded row-block / read-chunk is in RAM at a time. The
/// assignment, permutation (stable sort) and quantize order are
/// identical to the in-RAM path, so the on-disk bytes are unchanged.
///
/// Returns the number of vectors written. Consumes `state.ivf_spill`
/// and `state.ivf_sample`.
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
    // The spill is the authoritative corpus now; its row count is
    // the number of distinct heap vectors seen.
    let spill = state
        .ivf_spill
        .take()
        .expect("ivf_build_and_write: spill not opened");
    let n_vectors = spill.rows;

    // Empty corpus: write an empty (flat-shaped) meta page. An IVF
    // index over zero rows has no cells; readers treat it as empty.
    // (The spill drops here, unlinking the temp file.)
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

    // 2. Assign every vector to its nearest centroid in a STREAMED
    //    sweep over the disk spill. We read row-blocks (bounded by
    //    maintenance_work_mem), GEMM-rotate + GEMM-assign each block,
    //    and keep ONLY the per-row cell-id assignment array
    //    (`Vec<Vec<u32>>`, ~tiny) -- never a corpus copy. The spill
    //    stores the already-(optionally-)normalised vectors the heap
    //    scan wrote; we re-normalise each block exactly as the
    //    in-RAM path did (idempotent when normalise_on_insert is on,
    //    and matching the old `flat` re-normalise when it's off) so
    //    the assignment space is byte-identical.
    let block_rows = ivf_assign_block_rows(dim);
    // IVF-4a: soft assignment when assign_dups > 1. Cap at lists.
    let assign_dups = state.assign_dups.clamp(1, lists);
    // Per-vector cell lists (nearest-first, length 1..=assign_dups).
    // For single assignment (assign_dups == 1) each inner vec has
    // exactly one cell, so the downstream soft permutation reduces to
    // the single permutation bit-for-bit (verified by
    // ivf_build_permutation_soft_expands_duplicates's M=1 case).
    let assignments: Vec<Vec<u32>> = {
        let mut assignments: Vec<Vec<u32>> = Vec::with_capacity(n_vectors);
        // Reused per-block scratch: spill-read block + normalised
        // block + rotated block + id sink (ids discarded in pass 2).
        let mut raw_block = vec![0.0f32; block_rows * dim];
        let mut norm_block = vec![0.0f32; block_rows * dim];
        let mut rot_block = vec![0.0f32; block_rows * dim];
        let mut id_sink = vec![0u64; block_rows];
        let mut start = 0usize;
        while start < n_vectors {
            let rows = (n_vectors - start).min(block_rows);
            // Sequential read of this block straight from the spill.
            // Done on THIS thread (BufFile is not thread-safe and
            // `*mut BufFile` is not Send), outside the pool.
            spill.read_block(
                start,
                rows,
                &mut id_sink[..rows],
                &mut raw_block[..rows * dim],
            );
            // The parallel kernels (normalise -> rotate GEMM ->
            // batched assign) run on the bounded build pool; their
            // inputs are plain `&[f32]` (Send), so the spill never
            // crosses into the pool.
            let raw = &raw_block[..rows * dim];
            let norm = &mut norm_block[..rows * dim];
            let rot = &mut rot_block[..rows * dim];
            let model_centroids = &model.centroids;
            let rotation_ref = &rotation;
            let block_assign = super::build_pool::install(build_pool, || {
                // Normalise each row (matching the old per-vector
                // re-normalise of the `flat` corpus so the sample
                // space and the assignment space agree).
                for r in 0..rows {
                    let src = &raw[r * dim..(r + 1) * dim];
                    kernels::normalise_into(&mut norm[r * dim..(r + 1) * dim], src);
                }
                ivf::rotate_corpus_into(norm, rotation_ref, rows, dim, rot);
                ivf::batched_assign_soft(
                    rot,
                    model_centroids,
                    rows,
                    lists,
                    dim,
                    assign_dups,
                )
            });
            assignments.extend(block_assign);
            start += rows;
        }
        assignments
    };

    // 3. Stable cell-contiguous permutation + cell directory. With
    //    soft assignment a vector's old index appears once per cell
    //    it landed in, so `permutation` is non-injective and the
    //    directory partitions the EXPANDED slot count (>= n_vectors).
    let (permutation, directory) = ivf::build_permutation_soft(&assignments, lists);
    let n_slots = permutation.len();
    debug_assert!(directory.validate_partition(n_slots as u64).is_ok());
    debug_assert!(n_slots >= n_vectors, "soft expansion never shrinks");
    drop(assignments);

    // 4 + 5. Build `perm_ids` (the real external ids in cell order,
    //    with soft-assign duplicates) from the spill's id column, and
    //    feed the quantizer in cell order by re-reading the spill at
    //    the permuted offsets in bounded chunks. We NEVER materialise
    //    `perm_flat` whole: only one `chunk_rows`-row block of
    //    cell-ordered vectors is resident at a time. `add_with_ids`
    //    is incremental and order-preserving, so feeding the slots in
    //    contiguous chunks (synthetic ids start_slot..start_slot+rows)
    //    produces byte-identical packed_codes/scales to one big add.
    //
    //    CRITICAL: `IdMapIndex::add_with_ids` enforces UNIQUE ids, but
    //    soft assignment puts a boundary vector's real id in multiple
    //    slots. So we feed SYNTHETIC unique slot-ids (0..n_slots) and
    //    persist the REAL external ids (`perm_ids`, with duplicates)
    //    into the relfile ids chain separately (below). The scan maps
    //    slots through the persisted real ids and dedups returned
    //    TIDs across probed cells. For assign_dups == 1 perm_ids has
    //    no duplicates, behaviour-identical to before.
    let mut perm_ids = vec![0u64; n_slots];
    let mut idx = IdMapIndex::new(dim, bit_width as usize)
        .expect("turbovec ambuild (ivf): invalid (dim, bit_width)");
    // Bounded cell-order read chunk for the streamed feed: same
    // sizing as the assign block (mwm-bounded). One chunk of f32
    // (chunk_rows * dim * 4) + the growing QUANTIZED packed_codes
    // are the only large RAM terms; the full f32 corpus stays on the
    // spill.
    //
    // DETERMINISM / chunk-invariance: turbovec locks the TQ+
    // per-coord calibration on the FIRST `add_with_ids` batch (it
    // fits empirical quantiles over that batch, then reuses them for
    // all later adds). So the first batch's *size* is part of the
    // on-disk bytes. To keep the relfile byte-identical regardless of
    // `maintenance_work_mem`, we prime calibration with a FIXED-size
    // cell-ordered prefix (`IVF_CALIB_ROWS`, independent of mwm) as
    // the first add, then stream the remainder in mwm-bounded chunks.
    // The prefix is deterministic (cell-order + stable sort), so the
    // calibration is a fixed function of (table, lists, assign_dups)
    // alone -- not of the chunk size. The calibration buffer is
    // bounded (IVF_CALIB_ROWS * dim * 4).
    //
    // `IVF_CALIB_ROWS` is comfortably above turbovec's
    // `TQPLUS_MIN_SAMPLES` (1000), so the prefix yields a stable
    // calibration; capped so the priming buffer stays small even at
    // large dim.
    const IVF_CALIB_ROWS: usize = 16 * 1024;
    let chunk_rows = block_rows;
    let calib_rows = IVF_CALIB_ROWS.min(n_slots);
    // First buffer must hold the larger of the calibration prefix and
    // a normal stream chunk.
    let first_cap = calib_rows.max(chunk_rows);
    {
        let mut chunk_flat = vec![0.0f32; first_cap * dim];
        let mut chunk_ids = vec![0u64; first_cap];
        let mut new_slot = 0usize;
        while new_slot < n_slots {
            // First batch is the fixed calibration prefix; subsequent
            // batches are mwm-bounded stream chunks.
            let want = if new_slot == 0 { calib_rows } else { chunk_rows };
            let rows = (n_slots - new_slot).min(want);
            // Random-access reads of this chunk's cell-ordered
            // vectors + real ids from the spill, on THIS thread.
            for r in 0..rows {
                let old_idx = permutation[new_slot + r] as usize;
                let mut id = 0u64;
                spill.read_one(
                    old_idx,
                    &mut id,
                    &mut chunk_flat[r * dim..(r + 1) * dim],
                );
                perm_ids[new_slot + r] = id;
                // Synthetic contiguous slot id for the IdMapIndex.
                chunk_ids[r] = (new_slot + r) as u64;
            }
            // The encode fan-out runs on the bounded build pool; its
            // inputs are plain slices (Send).
            let flat_chunk = &chunk_flat[..rows * dim];
            let id_chunk = &chunk_ids[..rows];
            super::build_pool::install(build_pool, || {
                idx.add_with_ids(flat_chunk, id_chunk)
                    .expect("turbovec ambuild (ivf): add_with_ids failed");
            });
            new_slot += rows;
        }
    }
    // The spill is no longer needed; drop it now (unlinks the temp
    // file) before the prepare_eager + write, freeing disk early.
    drop(spill);
    drop(permutation);

    let built = idx.slot_to_id().len();
    debug_assert_eq!(built, n_slots);

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
        colbert: state.colbert,
    };
    relfile::write_full_with_prepared_ivf(
        index_relation,
        bit_width,
        dim as u32,
        // The persisted codes/scales/ids/slot_to_id are all n_slots
        // long (>= distinct n_vectors under soft assignment); the
        // meta's n_vectors field is the on-disk row count the scan
        // validates against, so it must be n_slots.
        n_slots as u64,
        idx.packed_codes(),
        idx.scales(),
        // Real external ids (with duplicates for soft-assigned
        // boundary vectors), NOT idx.slot_to_id() (which is the
        // synthetic 0..n_slots). This is the authoritative
        // slot -> external-id table the scan maps through.
        &perm_ids,
        1,
        prepared,
        ivf_parts,
    );

    built
}

/// Phase F-2: ColBERT token-index unnest, called from
/// `build_callback` when `state.colbert`. Decodes the heap tuple's
/// `turbovec.vector[]` (the doc's per-token vectors), pins the dim
/// from the first token, lazily opens the rotation matrix + disk
/// spill (identical to the single-vector IVF first-row path), then
/// spills one record per token tagged with the doc's TID (`tid`,
/// repeated across all the doc's tokens) and reservoir-samples each
/// token for k-means. Tokens are processed in ARRAY ORDER, so the
/// on-disk token order is deterministic.
///
/// # Safety
/// `state` is the live `BuildState`; `datum` is the indexed
/// `vector[]` value for this tuple (non-NULL, checked by the caller).
unsafe fn colbert_build_callback(
    state: &mut BuildState,
    tid: pg_sys::ItemPointerData,
    datum: pg_sys::Datum,
) {
    let tokens: Option<Vec<Vector>> = pgrx::FromDatum::from_datum(datum, false);
    let Some(tokens) = tokens else {
        return;
    };
    if tokens.is_empty() {
        return;
    }
    let id = pgrx::itemptr::item_pointer_to_u64(tid);
    for tok in &tokens {
        let row_dim = tok.dim();
        if row_dim == 0 {
            continue;
        }
        if row_dim % 8 != 0 {
            error!(
                "turbovec ambuild (colbert): token dim must be a multiple of 8 (got {})",
                row_dim
            );
        }
        match state.dim {
            Some(d) if d != row_dim => {
                error!(
                    "turbovec ambuild (colbert): dim mismatch — first token had dim {}, this token has {}",
                    d, row_dim
                );
            }
            None => {
                state.dim = Some(row_dim);
                state.idx = Some(
                    IdMapIndex::new(row_dim, state.bit_width)
                        .expect("turbovec ambuild (colbert): invalid (dim, bit_width)"),
                );
                state.chunk_rows = BuildState::compute_chunk_rows(row_dim);
                // A colbert index is always IVF (lists > 0), so open
                // the rotation matrix + disk spill on the first token,
                // exactly as the single-vector IVF path does.
                if state.ivf_rotation.is_none() {
                    state.ivf_rotation =
                        Some(turbovec::rotation::make_rotation_matrix(row_dim));
                    state.ivf_spill = Some(CorpusSpill::new(row_dim));
                }
            }
            _ => {}
        }
        let spill = state
            .ivf_spill
            .as_mut()
            .expect("turbovec ambuild (colbert): spill not opened before first token");
        if state.normalise {
            let mut buf = vec![0.0_f32; row_dim];
            kernels::normalise_into(&mut buf, tok.as_slice());
            spill.push(id, &buf);
        } else {
            spill.push(id, tok.as_slice());
        }
        // Reservoir sample the rotated, L2-normalised token (cells
        // live in the rotated unit-sphere space), regardless of the
        // normalise GUC — mirrors the single-vector IVF reservoir.
        state.ivf_reservoir_push(tok.as_slice(), row_dim);
    }
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

    // Phase F-2: ColBERT token index. The indexed value is a
    // `turbovec.vector[]` (the doc's token arrays); unnest it into N
    // token slots, each tagged with THIS doc's TID (repeated). The
    // IVF soft-assign machinery already handles many-slots-one-id, so
    // we feed the spill one record per token with the doc TID as the
    // external id, and reservoir-sample each token for k-means. Token
    // order = array order (determinism). All other build state
    // (spill, sample, rotation) is shared with the single-vector IVF
    // path and was opened the same way.
    if state.colbert {
        colbert_build_callback(state, *tid, datum);
        return;
    }

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
                // Phase B-4: open the disk spill (stride needs dim).
                state.ivf_spill = Some(CorpusSpill::new(row_dim));
            }
        }
        _ => {}
    }

    let id = pgrx::itemptr::item_pointer_to_u64(*tid);

    // IVF-1 build path (`lists > 0`): Phase B-4 spills each
    // (optionally normalised) vector + id to the disk-backed
    // `CorpusSpill` instead of accumulating the full f32 corpus in
    // RAM. We also reservoir-sample the ROTATED vector for k-means.
    // The train + streamed assign + cell-order quantize + IVF persist
    // happens at end-of-scan in `ambuild` -> `ivf_build_and_write`,
    // re-reading the spill in bounded chunks.
    if state.lists > 0 {
        let spill = state
            .ivf_spill
            .as_mut()
            .expect("turbovec ambuild (ivf): spill not opened before first row");
        if state.normalise {
            let mut buf = vec![0.0_f32; row_dim];
            kernels::normalise_into(&mut buf, value.as_slice());
            spill.push(id, &buf);
        } else {
            spill.push(id, value.as_slice());
        }
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
    let (bw, dim, _lists, _assign_dups) = options::read(index_relation);

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
