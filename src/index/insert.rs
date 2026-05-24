//! `aminsert` — incremental insert into an existing turbovec index.
//!
//! v0.4 strategy: load the persisted IdMapIndex from `am_storage`,
//! call `add_with_ids` with the new row's CTID-as-u64, write back.
//! This is O(payload-size) per insert because we re-serialise the
//! whole index. Phase 5 will introduce dirty-flagging and a commit
//! hook so we batch writes.

use pgrx::pg_sys;
use pgrx::prelude::*;
#[cfg(not(feature = "relfile_storage"))]
use turbovec::IdMapIndex;

use crate::guc;
#[cfg(feature = "relfile_storage")]
use crate::index::relfile;
use crate::index::{options, persist};
use crate::kernels;
use crate::vec::Vector;

/// `aminsert` callback. Returns `true` if the index now contains the
/// row; `false` if we deliberately skipped it. We never skip in v0.4
/// (any reason to skip should produce an `ERROR` instead).
///
/// The callback signature changed in PG 14 to add the
/// `indexUnchanged` flag (used by HOT chain elision); pg13 has the
/// 7-arg form. We expose two thin wrappers and pick which one to
/// install in `register_am`.
#[cfg(not(feature = "pg13"))]
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_impl(index_relation, values, isnull, heap_tid)
}

/// PG 13 `aminsert` shape — no `indexUnchanged` parameter.
#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_impl(index_relation, values, isnull, heap_tid)
}

unsafe fn aminsert_impl(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
) -> bool {
    let indexrelid = (*index_relation).rd_id;

    // Single-column indexes only — values[0] / isnull[0].
    if *isnull {
        // NULL embeddings simply don't get indexed (matches pgvector).
        return false;
    }
    let datum: pg_sys::Datum = *values;
    let value: Option<Vector> = pgrx::FromDatum::from_datum(datum, false);
    let Some(value) = value else {
        return false;
    };

    let dim = value.dim();
    if dim % 8 != 0 {
        error!(
            "turbovec aminsert: dim must be a multiple of 8 (got {})",
            dim
        );
    }

    // Encode CTID into u64 using the canonical pgrx layout.
    let id = pgrx::itemptr::item_pointer_to_u64(*heap_tid);

    let normalise = guc::NORMALIZE_ON_INSERT.get();

    let (bit_width, _) = options::read(index_relation);

    #[cfg(feature = "relfile_storage")]
    {
        aminsert_relfile(
            index_relation,
            indexrelid,
            bit_width,
            dim,
            normalise,
            value,
            id,
        )
    }

    #[cfg(not(feature = "relfile_storage"))]
    {
        aminsert_sidetable(indexrelid, bit_width, dim, normalise, value, id)
    }
}

#[cfg(feature = "relfile_storage")]
unsafe fn aminsert_relfile(
    index_relation: pg_sys::Relation,
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: usize,
    normalise: bool,
    value: Vector,
    id: u64,
) -> bool {
    use turbovec::IdMapIndex;

    // Phase L stub: full-rewrite per insert. Same asymptotic cost
    // as today's SPI path — we read every page back into RAM,
    // mutate the IdMapIndex, then rewrite every page. Phase K's
    // deferred-commit batching will turn this into a per-tx
    // append; for now correctness > throughput.
    let mut idx_index = match relfile::read_meta(index_relation) {
        Some(meta) if meta.n_vectors > 0 => {
            let (codes, scales, ids) = relfile::read_full(index_relation, &meta);
            IdMapIndex::from_id_map_parts(
                meta.bit_width as usize,
                meta.dim as usize,
                meta.n_vectors as usize,
                codes,
                scales,
                ids,
            )
            .unwrap_or_else(|e| error!("turbovec aminsert: corrupt relfile pages: {}", e))
        }
        _ => IdMapIndex::new(dim, bit_width as usize),
    };

    if idx_index.dim() != 0 && idx_index.dim() != dim {
        error!(
            "turbovec aminsert: dim mismatch — index expects {}, row has {}",
            idx_index.dim(),
            dim
        );
    }

    let buf = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    if let Err(e) = idx_index.add_with_ids(&buf, &[id]) {
        let msg = format!("{:?}", e);
        if msg.contains("IdAlreadyPresent") {
            idx_index.remove(id);
            if let Err(e2) = idx_index.add_with_ids(&buf, &[id]) {
                error!("turbovec aminsert: re-add after remove failed: {:?}", e2);
            }
        } else {
            error!("turbovec aminsert: add_with_ids failed: {:?}", e);
        }
    }

    let n_vectors = idx_index.len() as u64;
    // Bump am_version on every commit so the per-backend cache
    // invalidates. Read the previous meta to compute next version;
    // when the index was empty before, restart at 1.
    let next_version =
        relfile::read_meta(index_relation).map_or(1, |m| m.am_version.saturating_add(1));

    relfile::write_full(
        index_relation,
        bit_width as u8,
        dim as u32,
        n_vectors,
        idx_index.packed_codes(),
        idx_index.scales(),
        idx_index.slot_to_id(),
        next_version,
    );

    // Mirror the n_vectors counter into the SPI side-table so
    // existing tests / queries that read it keep working.
    persist::save_empty_with_count(indexrelid, bit_width, dim as i32, n_vectors as i64);

    true
}

#[cfg(not(feature = "relfile_storage"))]
unsafe fn aminsert_sidetable(
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: usize,
    normalise: bool,
    value: Vector,
    id: u64,
) -> bool {
    let mut state = persist::load(indexrelid).unwrap_or_else(|| {
        // No row yet — happens on the first insert after
        // `ambuildempty`. Build a fresh, empty IdMapIndex now.
        persist::StoredIndex {
            bit_width,
            dim: dim as i32,
            n_vectors: 0,
            index: IdMapIndex::new(dim, bit_width as usize),
            version: 1,
            live_ids: Vec::new(),
        }
    });

    if state.dim as usize != dim {
        error!(
            "turbovec aminsert: dim mismatch — index expects {}, row has {}",
            state.dim, dim
        );
    }

    let buf = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    if let Err(e) = state.index.add_with_ids(&buf, &[id]) {
        let msg = format!("{:?}", e);
        if msg.contains("IdAlreadyPresent") {
            state.index.remove(id);
            // live_ids already contains this id (CIC-validate or
            // HOT-update path) — don't push it again. n_vectors
            // unchanged.
            if let Err(e2) = state.index.add_with_ids(&buf, &[id]) {
                error!("turbovec aminsert: re-add after remove failed: {:?}", e2);
            }
            state.version += 1;
            persist::save(
                indexrelid,
                state.bit_width,
                state.dim,
                state.n_vectors,
                &state.index,
                state.version,
                &state.live_ids,
            );
            return true;
        }
        error!("turbovec aminsert: add_with_ids failed: {:?}", e);
    }
    state.live_ids.push(id);
    state.n_vectors += 1;
    state.version += 1;

    persist::save(
        indexrelid,
        state.bit_width,
        state.dim,
        state.n_vectors,
        &state.index,
        state.version,
        &state.live_ids,
    );

    true
}
