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
#[cfg(feature = "relfile_storage")]
use crate::index::relfile;
use crate::index::{options, persist};
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

    // If the heap was empty and dim was not pinned, persist an empty
    // marker so subsequent aminserts have a row to update.
    let Some(dim) = state.dim else {
        persist::save_empty(indexrelid, cfg_bit_width, 0);
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
    #[cfg(feature = "relfile_storage")]
    {
        // Phase L: write pages directly into the index relation's
        // main fork. The SPI side-table remains untouched
        // (compile-time choice; users opt in via the cargo
        // feature). am_version starts at 1 and bumps on every
        // mutation so the cache freshness check still works.
        relfile::write_full(
            index_relation,
            cfg_bit_width as u8,
            dim as u32,
            n_vectors as u64,
            idx.packed_codes(),
            idx.scales(),
            idx.slot_to_id(),
            1,
        );
        // We still write a side-table marker row so existing
        // tests that grep `turbovec.am_storage` for `n_vectors`
        // keep passing. Marked payload-empty so the SPI loader
        // would fail-loud if anything ever tried to use it.
        persist::save_empty_with_count(indexrelid, cfg_bit_width, dim as i32, n_vectors);
    }
    #[cfg(not(feature = "relfile_storage"))]
    persist::save(
        indexrelid,
        cfg_bit_width,
        dim as i32,
        n_vectors,
        &idx,
        1,
        &state.ids,
    );

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
/// Phase L stub: for the relfile path we *do not* write to the
/// init fork yet (that requires `RBM_NORMAL` reads against
/// `INIT_FORKNUM` and a small init-fork meta page). Unlogged
/// `turbovec` indexes therefore degrade to "rebuilds-from-scratch
/// after every crash", which is correct but expensive. Logged
/// indexes (PG's default) are unaffected.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn ambuildempty(index_relation: pg_sys::Relation) {
    let (bw, _dim) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    persist::save_empty(indexrelid, bw, 0);
}
