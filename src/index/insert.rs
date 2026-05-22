//! `aminsert` — incremental insert into an existing turbovec index.
//!
//! v0.4 strategy: load the persisted IdMapIndex from `am_storage`,
//! call `add_with_ids` with the new row's CTID-as-u64, write back.
//! This is O(payload-size) per insert because we re-serialise the
//! whole index. Phase 5 will introduce dirty-flagging and a commit
//! hook so we batch writes.

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::guc;
use crate::index::{options, persist};
use crate::kernels;
use crate::tvector::Tvector;

/// `aminsert` callback. Returns `true` if the index now contains the
/// row; `false` if we deliberately skipped it. We never skip in v0.4
/// (any reason to skip should produce an `ERROR` instead).
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
    let indexrelid = (*index_relation).rd_id;

    // Single-column indexes only — values[0] / isnull[0].
    if *isnull {
        // NULL embeddings simply don't get indexed (matches pgvector).
        return false;
    }
    let datum: pg_sys::Datum = *values;
    let value: Option<Tvector> = pgrx::FromDatum::from_datum(datum, false);
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
