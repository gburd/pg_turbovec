//! Phase 2 — function-driven ANN search backed by `turbovec::IdMapIndex`.
//!
//! v0.2.0 ships ANN as a SQL-callable function rather than a Postgres
//! index access method. The IndexAM path is staged for a later release
//! (see `docs/ARCHITECTURE.md` §6); the difference is purely
//! ergonomic — the underlying `turbovec` calls and the storage
//! semantics are identical, so promoting to an AM is mostly a
//! matter of wrapping the same code in `IndexAmRoutine` callbacks.
//!
//! Public surface:
//!
//! ```sql
//! turbovec.knn(
//!     rel       regclass,
//!     id_col    text,            -- bigint primary key column
//!     vec_col   text,            -- vector column
//!     query     vector,
//!     k         integer,
//!     bit_width integer DEFAULT 4
//! ) RETURNS TABLE(id bigint, score double precision)
//! ```
//!
//! The function is `STABLE PARALLEL SAFE`: results are deterministic
//! within a snapshot (modulo TurboQuant's internal `OnceLock`
//! initialisation, which is itself deterministic).
//!
//! v0.2 builds the `IdMapIndex` from scratch on every call. A
//! backend-local cache that respects relcache invalidation is
//! deferred to v0.3 — getting the invalidation right without a
//! running cluster to test against carries too much risk.

use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::vec::Vector;

/// `turbovec.knn(rel, id_col, vec_col, query, k, bit_width, allowed)`
/// — see module documentation. The optional `allowed` parameter
/// restricts results to the given `bigint[]` of ids; passing NULL
/// (or omitting the argument) does an unfiltered search.
#[pg_extern(stable, parallel_safe)]
fn knn(
    rel: pg_sys::Oid,
    id_col: &str,
    vec_col: &str,
    query: Vector,
    k: i32,
    bit_width: default!(i32, 4),
    allowed: default!(Option<Vec<i64>>, "NULL"),
) -> TableIterator<'static, (name!(id, i64), name!(score, f64))> {
    if k <= 0 {
        error!("turbovec.knn: k must be positive (got {})", k);
    }
    if !(2..=4).contains(&bit_width) {
        error!(
            "turbovec.knn: bit_width must be 2, 3, or 4 (got {})",
            bit_width
        );
    }
    if query.dim() % 8 != 0 {
        error!(
            "turbovec.knn: query dim must be a multiple of 8 (turbovec constraint), got {}",
            query.dim()
        );
    }

    let normalise = guc::NORMALIZE_ON_INSERT.get();

    // Cache lookup: key by (rel, vec_col_attnum, bit_width, dim).
    // We do not key on id_col because the cache stores u64
    // ids — a different id_col would change the id semantics, but
    // we don't currently track which column produced the ids. Phase
    // 9 will hash the id_col into the key for safety.
    let attnum = attnum_for(rel, vec_col);
    let key = CacheKey {
        rel_oid: rel,
        attnum,
        bit_width: bit_width as u8,
        dim: query.dim() as u32,
    };
    let relfile = cache::current_relfilenode(rel);
    let n_rows = relation_row_count(rel);

    let q_buf: Vec<f32> = if normalise {
        let mut out = vec![0.0_f32; query.dim()];
        crate::kernels::normalise_into(&mut out, query.as_slice());
        out
    } else {
        query.as_slice().to_vec()
    };

    if let Some(idx_arc) = cache::lookup(key, relfile, n_rows) {
        let n_live = idx_arc.read().len();
        // Over-fetch ANN candidates so the exact rerank has the true
        // nearest in hand even when the lossy quantized ranking places
        // it just outside k. This is what makes the top-k stable across
        // turbovec's layout-dependent scoring kernels (vector-major
        // permute-dot vs the classic LUT differ enough at low
        // dim / 4-bit to flip a near-tie -- see rerank_take).
        let take = rerank_take(k as usize).min(n_live);
        if take == 0 {
            return TableIterator::new(Vec::<(i64, f64)>::new());
        }
        let cand = run_search(&idx_arc.read(), &q_buf, take, allowed.as_deref());
        let result = exact_rerank(
            rel,
            id_col,
            vec_col,
            query.as_slice(),
            normalise,
            cand,
            k as usize,
        );
        return TableIterator::new(result);
    }

    // Cache miss: walk the heap via SPI, build the IdMapIndex, cache it.
    let rows = collect_via_spi(rel, id_col, vec_col, query.dim());
    if rows.is_empty() {
        return TableIterator::new(Vec::<(i64, f64)>::new());
    }

    let mut idx = IdMapIndex::new(query.dim(), bit_width as usize)
        .expect("turbovec_knn: invalid (dim, bit_width) for IdMapIndex::new");
    let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * query.dim());
    let mut ids: Vec<u64> = Vec::with_capacity(rows.len());
    for (id, vector) in &rows {
        if normalise {
            push_normalised(&mut flat, vector);
        } else {
            flat.extend_from_slice(vector);
        }
        ids.push(*id as u64);
    }
    idx.add_with_ids(&flat, &ids)
        .unwrap_or_else(|e| error!("turbovec.knn: add_with_ids failed: {:?}", e));

    // Approximate bytes: packed_codes (dim*bit_width/8 per vector)
    // plus 4-byte scale, plus a 64-byte id-map overhead heuristic.
    let bytes_per_vec = (query.dim() * bit_width as usize) / 8 + 4 + 64;
    let total_bytes = bytes_per_vec * rows.len();
    let idx_arc = cache::insert(key, idx, total_bytes, relfile, n_rows);

    let take = rerank_take(k as usize).min(rows.len());
    let cand = run_search(&idx_arc.read(), &q_buf, take, allowed.as_deref());
    // Miss path already holds the raw f32 rows -- rerank from them
    // without a second SPI fetch.
    let result = exact_rerank_from_rows(&rows, query.as_slice(), normalise, cand, k as usize);

    TableIterator::new(result)
}

/// Dispatch on whether the caller supplied an allowlist. Translates
/// the bigint[] to the u64 buffer the kernel expects, dropping NULL
/// entries.
fn run_search(
    idx: &IdMapIndex,
    query: &[f32],
    k: usize,
    allowed: Option<&[i64]>,
) -> Vec<(i64, f64)> {
    if k == 0 || idx.is_empty() {
        return Vec::new();
    }
    match allowed {
        None => {
            let (scores, hit_ids) = idx.search(query, k);
            hit_ids
                .iter()
                .zip(scores.iter())
                .map(|(id, s)| (*id as i64, f64::from(*s)))
                .collect()
        }
        Some(allow) => {
            let mut buf: Vec<u64> = allow.iter().map(|v| *v as u64).collect();
            buf.sort_unstable();
            buf.dedup();
            if buf.is_empty() {
                return Vec::new();
            }
            let take = k.min(buf.len());
            // wire v8 / turbovec 1.0.0: search_with_allowlist now returns
            // Result (AllowlistEmpty / UnknownId). buf is non-empty
            // (guarded above) and its ids come from the caller's allow
            // set; an unknown id is a caller error, surfaced clearly.
            let (scores, hit_ids) = idx
                .search_with_allowlist(query, take, Some(&buf))
                .unwrap_or_else(|e| error!("turbovec.knn: allowlist search failed: {:?}", e));
            hit_ids
                .iter()
                .zip(scores.iter())
                .map(|(id, s)| (*id as i64, f64::from(*s)))
                .collect()
        }
    }
}

/// Candidate over-fetch floor for the exact rerank. turbovec's
/// quantized ANN score is layout-dependent (the vector-major
/// permute-dot kernel and the classic LUT kernel are selected by CPU
/// feature — avx512vnni/vbmi — and differ enough at low dim / 4-bit to
/// place the true nearest just outside a small `k`, or to flip a
/// near-tie: e.g. for query [1,0,..] the exact match and a [0.9,0.1,..]
/// row land ~0.006 apart in true cosine, within the classic kernel's
/// 4-bit error). Fetching a wider candidate set and reranking the
/// survivors by EXACT cosine makes the returned top-k identical on
/// every host regardless of which kernel scored the codes. 64 is ample
/// for the toy corpora these knn() correctness tests use and negligible
/// at scale (the exact rerank is O(cand * dim)); an explicit larger `k`
/// always wins.
const RERANK_CANDIDATE_FLOOR: usize = 64;

/// The number of ANN candidates to pull before the exact rerank: at
/// least `k`, at least the floor.
fn rerank_take(k: usize) -> usize {
    k.max(RERANK_CANDIDATE_FLOOR)
}

/// Exact cosine (higher = nearer, matching turbovec's score
/// convention and `ORDER BY score DESC`). When `normalise` is set the
/// stored vectors were unit-normalised at insert, so normalise `v` the
/// same way; the query is normalised by the caller. Accumulate in f64
/// for a stable, platform-independent ordering. A zero-norm vector
/// scores 0 (degenerate but finite).
fn exact_cosine(query: &[f32], v: &[f32], normalise: bool) -> f64 {
    if v.len() != query.len() {
        return f64::NEG_INFINITY;
    }
    let mut dot = 0.0_f64;
    let mut vn = 0.0_f64;
    for (q, x) in query.iter().zip(v.iter()) {
        let xf = f64::from(*x);
        dot += f64::from(*q) * xf;
        vn += xf * xf;
    }
    if normalise {
        if vn == 0.0 {
            return 0.0;
        }
        dot / vn.sqrt()
    } else {
        dot
    }
}

/// Re-rank ANN `candidates` by exact cosine against `query`, fetching
/// each candidate's true vector via SPI, and return the top-`k`
/// `(id, exact_score)` in descending score order. Used on the
/// cache-hit path, where the raw vectors are not already in hand.
///
/// The exact score REPLACES the quantized score in the result, so
/// `ORDER BY score DESC` downstream sees the true ranking. Ties break
/// by ascending id for determinism.
fn exact_rerank(
    rel: pg_sys::Oid,
    id_col: &str,
    vec_col: &str,
    query_raw: &[f32],
    normalise: bool,
    candidates: Vec<(i64, f64)>,
    k: usize,
) -> Vec<(i64, f64)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }
    // Normalise the query the same way the corpus was, so the exact
    // cosine is comparable to the stored unit vectors.
    let q_norm: Vec<f32> = if normalise {
        let mut out = vec![0.0_f32; query_raw.len()];
        crate::kernels::normalise_into(&mut out, query_raw);
        out
    } else {
        query_raw.to_vec()
    };

    let ids: Vec<i64> = candidates.iter().map(|(id, _)| *id).collect();
    let vecs = fetch_vectors_by_id(rel, id_col, vec_col, &ids, query_raw.len());

    let mut scored: Vec<(i64, f64)> = candidates
        .into_iter()
        .filter_map(|(id, _quant)| {
            vecs.get(&id)
                .map(|v| (id, exact_cosine(&q_norm, v, normalise)))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored
}

/// Cache-miss variant of [`exact_rerank`]: the raw `(id, vector)` rows
/// are already collected, so rerank from them directly without a
/// second SPI round-trip.
fn exact_rerank_from_rows(
    rows: &[(i64, Vec<f32>)],
    query_raw: &[f32],
    normalise: bool,
    candidates: Vec<(i64, f64)>,
    k: usize,
) -> Vec<(i64, f64)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }
    let q_norm: Vec<f32> = if normalise {
        let mut out = vec![0.0_f32; query_raw.len()];
        crate::kernels::normalise_into(&mut out, query_raw);
        out
    } else {
        query_raw.to_vec()
    };
    let by_id: std::collections::HashMap<i64, &[f32]> =
        rows.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let mut scored: Vec<(i64, f64)> = candidates
        .into_iter()
        .filter_map(|(id, _quant)| {
            by_id
                .get(&id)
                .map(|v| (id, exact_cosine(&q_norm, v, normalise)))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored
}

/// Fetch the true `real[]` vectors for a set of ids in one SPI query,
/// keyed by id. Rows whose dim mismatches or are non-finite are
/// skipped (the caller then drops that candidate from the rerank).
fn fetch_vectors_by_id(
    rel: pg_sys::Oid,
    id_col: &str,
    vec_col: &str,
    ids: &[i64],
    expected_dim: usize,
) -> std::collections::HashMap<i64, Vec<f32>> {
    let mut out: std::collections::HashMap<i64, Vec<f32>> = std::collections::HashMap::new();
    if ids.is_empty() {
        return out;
    }
    let qualified: String = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c \
         JOIN   pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[rel.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_else(|| error!("turbovec.knn: relation oid {:?} not found", rel));

    let id_q = pgrx::spi::quote_identifier(id_col);
    let vec_q = pgrx::spi::quote_identifier(vec_col);
    // Pass the candidate ids as a bigint[] and filter with = ANY.
    let sql = format!(
        "SELECT ({id_q})::bigint, \
                ({vec_q})::turbovec.vector::real[] \
         FROM   {qualified} \
         WHERE  ({id_q}) = ANY($1) AND ({vec_q}) IS NOT NULL"
    );
    Spi::connect(|client| {
        let tup_iter = match client.select(&sql, None, &[ids.to_vec().into()]) {
            Ok(t) => t,
            Err(e) => error!("turbovec.knn: rerank SPI select failed: {}", e),
        };
        for row in tup_iter {
            let id: Option<i64> = row.get(1).ok().flatten();
            let arr: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            let (Some(id), Some(arr)) = (id, arr) else {
                continue;
            };
            if arr.len() != expected_dim {
                continue;
            }
            let values: Vec<f32> = arr.into_iter().map(|v| v.unwrap_or(f32::NAN)).collect();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            out.insert(id, values);
        }
    });
    out
}

/// Resolve the heap attribute number for a column name. Returns 1
/// (a valid attnum) if the lookup fails — the cache will simply
/// see a different effective key and miss.
fn attnum_for(rel: pg_sys::Oid, col: &str) -> i16 {
    let v: Option<i32> = Spi::get_one_with_args(
        "SELECT attnum::int4 FROM pg_attribute \
         WHERE attrelid = $1 AND attname = $2 AND NOT attisdropped",
        &[rel.into(), col.into()],
    )
    .ok()
    .flatten();
    v.unwrap_or(1) as i16
}

/// Cheap row count for cache invalidation. Returns -1 on failure so
/// the cache miss path runs.
fn relation_row_count(rel: pg_sys::Oid) -> i64 {
    let qualified: Option<String> = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[rel.into()],
    )
    .ok()
    .flatten();
    let Some(qualified) = qualified else {
        return -1;
    };
    let sql = format!("SELECT count(*)::int8 FROM {qualified}");
    Spi::get_one::<i64>(&sql).ok().flatten().unwrap_or(-1)
}

/// Append a unit-normalised copy of `src` to `dst`. If `src` is the
/// zero vector it is appended unchanged (the kernel will produce a
/// sensible-but-degenerate result; we avoid emitting NaNs).
fn push_normalised(dst: &mut Vec<f32>, src: &[f32]) {
    let mut acc: f64 = 0.0;
    for x in src {
        acc += f64::from(*x) * f64::from(*x);
    }
    if acc == 0.0 {
        dst.extend_from_slice(src);
        return;
    }
    let inv = (1.0_f64 / acc.sqrt()) as f32;
    for x in src {
        dst.push(*x * inv);
    }
}

/// Pull `(id, vector)` rows out of the relation via SPI. Skips rows
/// with NULL id, NULL vector, or vectors whose dim does not match
/// `expected_dim`.
fn collect_via_spi(
    rel: pg_sys::Oid,
    id_col: &str,
    vec_col: &str,
    expected_dim: usize,
) -> Vec<(i64, Vec<f32>)> {
    // Resolve qualified table name from the oid using SPI's own
    // identifier-quoting helper for the result.
    let qualified: String = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c \
         JOIN   pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[rel.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_else(|| error!("turbovec.knn: relation oid {:?} not found", rel));

    // Quote the user-supplied column identifiers defensively.
    let id_q = pgrx::spi::quote_identifier(id_col);
    let vec_q = pgrx::spi::quote_identifier(vec_col);

    let sql = format!(
        "SELECT ({id_q})::bigint, \
                ({vec_q})::turbovec.vector::real[] \
         FROM   {qualified} \
         WHERE  ({id_q}) IS NOT NULL AND ({vec_q}) IS NOT NULL"
    );

    let mut out: Vec<(i64, Vec<f32>)> = Vec::new();
    Spi::connect(|client| {
        let tup_iter = match client.select(&sql, None, &[]) {
            Ok(t) => t,
            Err(e) => error!("turbovec.knn: SPI select failed: {}", e),
        };
        for row in tup_iter {
            let id: Option<i64> = row.get(1).ok().flatten();
            let arr: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            let (Some(id), Some(arr)) = (id, arr) else {
                continue;
            };
            if arr.len() != expected_dim {
                continue;
            }
            let values: Vec<f32> = arr.into_iter().map(|v| v.unwrap_or(f32::NAN)).collect();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            out.push((id, values));
        }
    });
    out
}
