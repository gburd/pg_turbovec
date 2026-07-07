//! `aminsert` — incremental insert into an existing turbovec index.
//!
//! Mutates the cached `IdMapIndex` under a `parking_lot::RwLock`
//! write guard, marks the entry dirty, and defers the relfile-page
//! write to the `PreCommit` xact callback (see `src/xact.rs`). One
//! relfile rewrite per transaction, regardless of how many rows
//! were inserted.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::guc;
use crate::index::{options, relfile};
use crate::kernels;
use crate::vec::Vector;
use turbovec::IdMapIndex;

/// `aminsert` callback. Returns `true` if the index now contains the
/// row; `false` if we deliberately skipped it. We never skip
/// without an explicit reason (NULL embeddings, decode failures);
/// any unexpected condition produces an `ERROR` instead.
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

    // Phase G-2a: a Vamana graph index is a build-time-only artifact
    // for this sub-phase (incremental insert/VACUUM integration is
    // G-2b, explicitly out of scope here — see
    // an internal design note). Rather than silently
    // corrupting the adjacency chain (a generic insert would rewrite
    // the relfile via the flat write_full_with_prepared path, which
    // knows nothing about `kind = KIND_GRAPH` and would drop the
    // graph fields on the next commit), refuse clearly and point at
    // REINDEX.
    if let Some(meta) = relfile::read_meta(index_relation) {
        if meta.is_graph() {
            error!(
                "turbovec aminsert: INSERT into a graph index (built WITH (graph = true)) is not yet supported (see an internal design note G-2b). REINDEX INDEX <name>; after bulk-loading new rows, or use a flat/IVF index for tables with ongoing INSERTs."
            );
        }
    }

    let normalise = guc::NORMALIZE_ON_INSERT.get();
    let (bit_width, _, _, _, _graph) = options::read(index_relation);

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

unsafe fn aminsert_relfile(
    index_relation: pg_sys::Relation,
    indexrelid: pg_sys::Oid,
    bit_width: i32,
    dim: usize,
    normalise: bool,
    value: Vector,
    id: u64,
) -> bool {
    use crate::cache::{self, CacheKey, PersistState};

    let buf = if normalise {
        kernels::normalise_to_vec(value.as_slice())
    } else {
        value.as_slice().to_vec()
    };

    let key = CacheKey {
        rel_oid: indexrelid,
        attnum: 0,
        bit_width: bit_width as u8,
        dim: dim as u32,
    };

    let relfile_node = cache::relfilenode_from_relation(index_relation);
    let arc = match cache::am_lookup_for_mutation(key, relfile_node) {
        Some(a) => a,
        None => {
            // First mutation in this tx: load from relfile pages.
            let (idx_index, n_vectors_existing, version_existing) =
                match relfile::read_meta(index_relation) {
                    Some(meta) if meta.n_vectors > 0 => {
                        let (codes, scales, ids) = relfile::read_full(index_relation, &meta);
                        let idx = IdMapIndex::from_id_map_parts(
                            meta.bit_width as usize,
                            meta.dim as usize,
                            meta.n_vectors as usize,
                            codes,
                            scales,
                            ids,
                        )
                        .unwrap_or_else(|e| {
                            error!("turbovec aminsert: corrupt relfile pages: {}", e)
                        });
                        (idx, meta.n_vectors as i64, meta.am_version as i32)
                    }
                    _ => (
                        IdMapIndex::new(dim, bit_width as usize).expect(
                            "turbovec aminsert: invalid (dim, bit_width) for IdMapIndex::new",
                        ),
                        0,
                        0,
                    ),
                };
            let bytes_per_vec = (dim * bit_width as usize) / 8 + 4 + 64;
            let total_bytes = bytes_per_vec * n_vectors_existing.max(1) as usize;
            let live_ids = idx_index.slot_to_id().to_vec();
            let persist_state = PersistState {
                bit_width,
                dim: dim as i32,
                n_vectors: n_vectors_existing,
                version: version_existing,
                live_ids,
            };
            cache::am_install(
                key,
                idx_index,
                total_bytes,
                relfile_node,
                version_existing as i64,
                persist_state,
            )
        }
    };

    let mut id_already_present = false;
    {
        let mut guard = arc.write();
        if guard.dim() != 0 && guard.dim() != dim {
            error!(
                "turbovec aminsert: dim mismatch — index expects {}, row has {}",
                guard.dim(),
                dim
            );
        }
        match guard.add_with_ids(&buf, &[id]) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{:?}", e);
                if msg.contains("IdAlreadyPresent") {
                    guard.remove(id);
                    if let Err(e2) = guard.add_with_ids(&buf, &[id]) {
                        error!("turbovec aminsert: re-add after remove failed: {:?}", e2);
                    }
                    id_already_present = true;
                } else {
                    error!("turbovec aminsert: add_with_ids failed: {:?}", e);
                }
            }
        }
    }

    let updated = cache::am_mark_dirty(key, |p| {
        if !id_already_present {
            p.live_ids.push(id);
            p.n_vectors += 1;
        }
        p.version += 1;
    });
    if !updated {
        error!("turbovec aminsert: cache entry vanished between install and mark_dirty");
    }

    crate::xact::ensure_xact_callbacks_registered();

    true
}
