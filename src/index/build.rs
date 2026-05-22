//! `ambuild` and `ambuildempty` — initial index materialisation.
//!
//! v0.4 uses SPI to enumerate the heap rather than the table AM
//! `index_build_range_scan` callback. The trade-off is documented
//! in `docs/INDEXAM.md`: SPI runs under the calling transaction's
//! snapshot rather than the dedicated build snapshot, which is fine
//! for experimental but must be replaced before promotion.

use pgrx::pg_sys;
use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::guc;
use crate::index::{options, persist};
use crate::kernels;

/// `ambuild`: scan the heap, build the IdMapIndex, persist it.
///
/// # Safety
///
/// Caller is PostgreSQL's index machinery. The two `Relation`
/// pointers are valid for the duration of the call; the
/// `IndexInfo` pointer too. We must return a palloc'd
/// `IndexBuildResult` populated with row counts.
pub(crate) unsafe extern "C-unwind" fn ambuild(
    heap_relation: pg_sys::Relation,
    index_relation: pg_sys::Relation,
    _index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    if result.is_null() {
        error!("turbovec: failed to allocate IndexBuildResult");
    }

    let (cfg_bit_width, cfg_dim) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    let heaprelid = (*heap_relation).rd_id;

    // Resolve the (single) indexed attribute name on the heap rel.
    // We only support one indexed column.
    let attr = resolve_indexed_attr(index_relation, heap_relation);

    let qualified = qualified_name(heaprelid);
    let attr_q = pgrx::spi::quote_identifier(&attr);

    let normalise = guc::NORMALIZE_ON_INSERT.get();

    // First pass: figure out dim if the user didn't pin it.
    let dim = if cfg_dim == 0 {
        autodetect_dim(&qualified, &attr_q)
    } else {
        cfg_dim as usize
    };
    if dim == 0 {
        // Empty table — emit an empty marker and return early.
        persist::save_empty(indexrelid, cfg_bit_width, 0);
        (*result).heap_tuples = 0.0;
        (*result).index_tuples = 0.0;
        return result;
    }
    if dim % 8 != 0 {
        error!(
            "turbovec ambuild: dim must be a multiple of 8 (got {}); cast or pad your column",
            dim
        );
    }

    // Second pass: collect (ctid_u64, vector).
    let sql = format!(
        "SELECT (ctid::text)::text, ({attr_q})::turbovec.tvector::real[] \
         FROM   {qualified} \
         WHERE  ({attr_q}) IS NOT NULL"
    );

    let mut idx = IdMapIndex::new(dim, cfg_bit_width as usize);
    let mut flat: Vec<f32> = Vec::new();
    let mut ids: Vec<u64> = Vec::new();
    let mut heap_seen: u64 = 0;

    Spi::connect(|client| {
        let tup_iter = match client.select(&sql, None, &[]) {
            Ok(t) => t,
            Err(e) => error!("turbovec ambuild: SPI select failed: {}", e),
        };
        for row in tup_iter {
            heap_seen += 1;
            let ctid_text: Option<String> = row.get(1).ok().flatten();
            let arr: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            let (Some(ctid_text), Some(arr)) = (ctid_text, arr) else {
                continue;
            };
            if arr.len() != dim {
                continue;
            }
            let values: Vec<f32> = arr.into_iter().map(|v| v.unwrap_or(f32::NAN)).collect();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let id = parse_ctid_to_u64(&ctid_text);
            if normalise {
                let mut buf = vec![0.0_f32; dim];
                kernels::normalise_into(&mut buf, &values);
                flat.extend_from_slice(&buf);
            } else {
                flat.extend_from_slice(&values);
            }
            ids.push(id);
        }
    });

    if !ids.is_empty() {
        if let Err(e) = idx.add_with_ids(&flat, &ids) {
            error!("turbovec ambuild: add_with_ids failed: {:?}", e);
        }
    }

    let n_vectors = ids.len() as i64;
    persist::save(
        indexrelid,
        cfg_bit_width,
        dim as i32,
        n_vectors,
        &idx,
        1,
    );

    (*result).heap_tuples = heap_seen as f64;
    (*result).index_tuples = n_vectors as f64;
    result
}

/// `ambuildempty`: called when the index is created over an empty
/// relation or via `CREATE INDEX ... NOT VALID`. We just persist an
/// empty marker so subsequent `aminsert` calls have a row to update.
pub(crate) unsafe extern "C-unwind" fn ambuildempty(index_relation: pg_sys::Relation) {
    let (bw, _dim) = options::read(index_relation);
    let indexrelid = (*index_relation).rd_id;
    persist::save_empty(indexrelid, bw, 0);
}

/// Look up the qualified name `"schema"."table"` for a relation oid.
fn qualified_name(relid: pg_sys::Oid) -> String {
    let txt: Option<String> = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[relid.into()],
    )
    .ok()
    .flatten();
    txt.unwrap_or_else(|| error!("turbovec: relation {:?} not found in pg_class", relid))
}

/// Resolve the (single) indexed attribute name for our index. We only
/// support one tvector column per index.
unsafe fn resolve_indexed_attr(
    index_relation: pg_sys::Relation,
    heap_relation: pg_sys::Relation,
) -> String {
    let indrel = (*index_relation).rd_index;
    if indrel.is_null() {
        error!("turbovec: rd_index is null");
    }
    let nkey = (*indrel).indnatts as usize;
    if nkey != 1 {
        error!(
            "turbovec: only single-column indexes are supported (got {} columns)",
            nkey
        );
    }
    // `indkey.values` is a `__IncompleteArrayField<i16>` flexible
    // array; read the first element via `.as_slice(n)`.
    let attno: i16 = *(*indrel).indkey.values.as_slice(nkey).first()
        .unwrap_or_else(|| error!("turbovec: indkey is empty"));
    if attno < 1 {
        error!("turbovec: indexed attribute number {} is invalid", attno);
    }
    let tupdesc = (*heap_relation).rd_att;
    if tupdesc.is_null() {
        error!("turbovec: heap rd_att is null");
    }
    // pg_sys exposes the attrs as a flexible array on TupleDescData.
    let n_attrs = (*tupdesc).natts as usize;
    let attrs: &[pg_sys::FormData_pg_attribute] =
        (*tupdesc).attrs.as_slice(n_attrs);
    let attr = attrs
        .get((attno - 1) as usize)
        .unwrap_or_else(|| error!("turbovec: indexed attribute {} not in tupdesc", attno));
    let raw_name = attr.attname.data.as_ptr() as *const std::os::raw::c_char;
    std::ffi::CStr::from_ptr(raw_name)
        .to_string_lossy()
        .into_owned()
}

/// Autodetect dimension by reading one row from the table.
fn autodetect_dim(qualified: &str, attr_q: &str) -> usize {
    let sql = format!(
        "SELECT array_length(({attr_q})::turbovec.tvector::real[], 1) \
         FROM {qualified} WHERE ({attr_q}) IS NOT NULL LIMIT 1"
    );
    let dim: Option<i32> = Spi::get_one(&sql).ok().flatten();
    dim.unwrap_or(0) as usize
}

/// Encode `(block, offset)` text "(B,O)" into a u64 using pgrx's
/// canonical 32/16 layout (top 32 bits = block, low 16 = offset).
fn parse_ctid_to_u64(s: &str) -> u64 {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    let mut parts = inner.split(',');
    let blk: u32 = parts
        .next()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(0);
    let off: u16 = parts
        .next()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(0);
    (u64::from(blk) << 32) | u64::from(off)
}
