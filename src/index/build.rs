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
struct BuildState {
    /// Expected dim. Set on the first non-NULL row, validated against
    /// every subsequent row.
    dim: Option<usize>,
    /// Optionally L2-normalise on insert.
    normalise: bool,
    /// Concatenated f32 buffer (`flat[i*dim..(i+1)*dim]` is row i).
    flat: Vec<f32>,
    /// `IdMapIndex` u64 ids — one per row, in the same order as `flat`.
    ids: Vec<u64>,
    /// Number of heap tuples we were called for (alive + dead).
    heap_seen: u64,
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
        dim: initial_dim,
        normalise,
        flat: Vec::new(),
        ids: Vec::new(),
        heap_seen: 0,
    };

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

    let mut idx = IdMapIndex::new(dim, cfg_bit_width as usize);
    if !state.ids.is_empty() {
        if let Err(e) = idx.add_with_ids(&state.flat, &state.ids) {
            error!("turbovec ambuild: add_with_ids failed: {:?}", e);
        }
    }

    let n_vectors = state.ids.len() as i64;
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
        None => state.dim = Some(row_dim),
        _ => {}
    }

    let id = pgrx::itemptr::item_pointer_to_u64(*tid);
    if state.normalise {
        let mut buf = vec![0.0_f32; row_dim];
        kernels::normalise_into(&mut buf, value.as_slice());
        state.flat.extend_from_slice(&buf);
    } else {
        state.flat.extend_from_slice(value.as_slice());
    }
    state.ids.push(id);
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
