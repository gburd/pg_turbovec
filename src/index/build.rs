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
        if let Err(e) = idx.add_with_ids(&self.pending_flat, &self.pending_ids) {
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

    let (cfg_bit_width, cfg_dim) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    let normalise = guc::NORMALIZE_ON_INSERT.get();

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
    };
    if let Some(d) = state.dim {
        // Pre-construct the IdMapIndex when reloptions pinned the dim.
        // Otherwise we wait for the first non-NULL row and construct
        // there. Either way the per-row callback path is identical.
        state.idx = Some(
            IdMapIndex::new(d, cfg_bit_width as usize)
                .expect("turbovec ambuild: invalid (dim, bit_width) for IdMapIndex::new"),
        );
        state.chunk_rows = BuildState::compute_chunk_rows(d);
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
        idx.prepare_eager();
        // Phase R-2: also drive the rotation `OnceLock` so we
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
        }
        _ => {}
    }

    let id = pgrx::itemptr::item_pointer_to_u64(*tid);
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
    let (bw, dim) = options::read(index_relation);

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
