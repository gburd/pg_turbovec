//! Phase F-1 — index-native late interaction (ColBERT MaxSim), first
//! cut. See `docs/PHASE_F_COLBERT_PLAN.md`.
//!
//! ColBERT scores a (query, doc) pair as
//! `MaxSim(Q, D) = sum_{q in Q} max_{d in D} sim(q, d)` over per-token
//! vectors. pg_turbovec already ships the **stage-2** half: Phase D's
//! `turbovec.max_sim(query vector[], doc vector[])` (`src/hybrid.rs`)
//! computes MaxSim exactly. What Phase D cannot do is **stage-1
//! recall**: its candidate set is capped by the *pooled* (mean)
//! document vector, so a doc retrieved only if its mean token vector
//! is near the query. The queries ColBERT was built for — rare
//! entities, specific terms, long docs where one passage matters —
//! need retrieval by *best single token*, which this function adds.
//!
//! `turbovec.colbert_search` is a two-stage SET-returning function
//! (the `turbovec.knn` model — no index AM, no `ORDER BY` operator,
//! no `amrescan` rewrite):
//!
//!   1. **Stage 1 (index-accelerated candidate generation).** Build /
//!      load a backend-cached flat token index — ONE slot per token
//!      across all docs, the slot's id being the token's **doc id**
//!      (many slots share a doc id; `IdMapIndex` is fed synthetic
//!      unique slot-ids and the real doc-ids are kept separately, the
//!      same trick the IVF build uses for soft-assign duplicates).
//!      Run ONE batched search of all |Q| query tokens against the
//!      token index, union the hit doc-ids into a candidate set.
//!   2. **Stage 2 (exact re-rank).** For each candidate doc, compute
//!      exact MaxSim against the doc's full token array read from the
//!      heap (reusing Phase D's kernel). Return the top-`k` docs.
//!
//! No relfile, no wire-format change (the token index lives only in
//! the backend cache, like `turbovec.knn`). The on-disk
//! `MetaPageData::version` is untouched. The persistent token-index
//! AM (Phase F-2) is gated on F-1's measured recall delta — see the
//! plan doc.
//!
//! ```sql
//! turbovec.colbert_search(
//!     rel         regclass,
//!     id_col      text,        -- bigint doc key
//!     token_col   text,        -- a turbovec.vector[] column (per-doc tokens)
//!     query       turbovec.vector[],   -- the query's token vectors
//!     k           integer,     -- final top-k docs
//!     per_token_k integer DEFAULT 64,   -- stage-1 hits per query token
//!     candidate_n integer DEFAULT 256,  -- max candidate docs into stage 2
//!     bit_width   integer DEFAULT 4
//! ) RETURNS TABLE(id bigint, score double precision)
//! ```
//!
//! `STABLE PARALLEL SAFE`: deterministic within a snapshot.

use std::collections::HashMap;

use pgrx::prelude::*;
use turbovec::IdMapIndex;

use crate::cache::{self, CacheKey};
use crate::guc;
use crate::kernels;
use crate::vec::Vector;

#[allow(clippy::too_many_arguments)]
#[pg_extern(stable, parallel_safe)]
fn colbert_search(
    rel: pg_sys::Oid,
    id_col: &str,
    token_col: &str,
    query: Vec<Vector>,
    k: i32,
    per_token_k: default!(i32, 64),
    candidate_n: default!(i32, 256),
    bit_width: default!(i32, 4),
) -> TableIterator<'static, (name!(id, i64), name!(score, f64))> {
    if k <= 0 {
        error!("turbovec.colbert_search: k must be positive (got {})", k);
    }
    if per_token_k <= 0 {
        error!(
            "turbovec.colbert_search: per_token_k must be positive (got {})",
            per_token_k
        );
    }
    if candidate_n <= 0 {
        error!(
            "turbovec.colbert_search: candidate_n must be positive (got {})",
            candidate_n
        );
    }
    if !(2..=4).contains(&bit_width) {
        error!(
            "turbovec.colbert_search: bit_width must be 2, 3, or 4 (got {})",
            bit_width
        );
    }
    // Empty query: nothing to match -> no candidates (ColBERT
    // convention, mirrors Phase D max_sim).
    if query.is_empty() {
        return TableIterator::new(Vec::<(i64, f64)>::new());
    }

    // All query tokens must share one dimension (and it must be a
    // multiple of 8 for the turbovec kernel).
    let dim = query[0].dim();
    for (i, q) in query.iter().enumerate() {
        if q.dim() != dim {
            error!(
                "turbovec.colbert_search: query token {} has dim {} but token 0 has dim {} \
                 (all query tokens must share one dimension)",
                i,
                q.dim(),
                dim
            );
        }
    }
    if dim == 0 || dim % 8 != 0 {
        error!(
            "turbovec.colbert_search: query token dim must be a positive multiple of 8 \
             (turbovec constraint), got {}",
            dim
        );
    }

    let normalise = guc::NORMALIZE_ON_INSERT.get();

    // ---- Stage 1: candidate doc-ids from the token index ----
    let candidates = stage1_candidates(
        rel,
        id_col,
        token_col,
        &query,
        dim,
        bit_width,
        per_token_k as usize,
        candidate_n as usize,
        normalise,
    );
    if candidates.is_empty() {
        return TableIterator::new(Vec::<(i64, f64)>::new());
    }

    // ---- Stage 2: exact MaxSim re-rank from the heap token arrays ----
    let doc_tokens = fetch_doc_tokens(rel, id_col, token_col, dim, &candidates);

    // Normalise the query tokens once if configured (so stage-2 dot ==
    // cosine, matching the stage-1 index which was built normalised).
    let q_norm: Vec<Vec<f32>> = query
        .iter()
        .map(|q| {
            if normalise {
                kernels::normalise_to_vec(q.as_slice())
            } else {
                q.as_slice().to_vec()
            }
        })
        .collect();

    let mut scored: Vec<(i64, f64)> = doc_tokens
        .into_iter()
        .map(|(doc_id, tokens)| {
            let score = max_sim_dot(&q_norm, &tokens, dim);
            (doc_id, score)
        })
        .collect();

    // Top-k by score descending; deterministic tie-break by doc id so
    // the ranking is reproducible.
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(k as usize);

    TableIterator::new(scored)
}

/// Stage 1: build/load the backend-cached flat token index and return
/// the candidate doc-ids (union of the per-query-token nearest docs,
/// capped at `candidate_n`, ordered by best stage-1 token score so the
/// cap keeps the most promising docs).
#[allow(clippy::too_many_arguments)]
fn stage1_candidates(
    rel: pg_sys::Oid,
    id_col: &str,
    token_col: &str,
    query: &[Vector],
    dim: usize,
    bit_width: i32,
    per_token_k: usize,
    candidate_n: usize,
    normalise: bool,
) -> Vec<i64> {
    // Token-index cache key: NEGATE the token column's attnum so the
    // entry can never collide with a single-vector `turbovec.knn`
    // entry (positive attnum) or the index-AM entry (attnum = 0). The
    // token index is a fundamentally different shape (n_tokens slots,
    // not n_docs), so it must not share a key with a per-doc index on
    // the same column.
    let attnum = attnum_for(rel, token_col);
    let key = CacheKey {
        rel_oid: rel,
        attnum: -attnum.max(1),
        bit_width: bit_width as u8,
        dim: dim as u32,
    };
    let relfile = cache::current_relfilenode(rel);
    let n_rows = relation_row_count(rel);

    // Flatten the query tokens into the row-major buffer the kernel
    // wants: nq * dim, normalised if configured.
    let nq = query.len();
    let mut q_flat: Vec<f32> = Vec::with_capacity(nq * dim);
    for q in query {
        if normalise {
            q_flat.extend_from_slice(&kernels::normalise_to_vec(q.as_slice()));
        } else {
            q_flat.extend_from_slice(q.as_slice());
        }
    }

    // Run the batched stage-1 search against the (cached or freshly
    // built) token index, returning slot->doc-id hits and their
    // scores. The slot->doc map lives process-local in SLOT_DOC,
    // keyed identically to the index cache, so a warm hit reuses it.
    let warm = cache::lookup(key, relfile, n_rows);
    let warm_map = SLOT_DOC.with(|m| m.borrow().get(&key).cloned());
    let hits: Vec<(i64, f32)> = match (warm, warm_map) {
        // Warm cache hit AND we still have the slot->doc map: reuse both.
        (Some(idx_arc), Some(slot_doc)) => {
            let guard = idx_arc.read();
            search_tokens(&guard, &slot_doc, &q_flat, nq, per_token_k)
        }
        // Either the index or its slot->doc map is missing (cold, or
        // the thread-local was cleared / a different backend warmed
        // the index). Rebuild from the heap so the two stay paired.
        _ => {
            let (idx, slot_doc, n_tokens) =
                build_token_index(rel, id_col, token_col, dim, bit_width, normalise);
            if n_tokens == 0 {
                return Vec::new();
            }
            // Approximate resident bytes for the cache budget: per-token
            // packed codes + scale + the slot->doc map (8 B/slot) +
            // id-map overhead heuristic.
            let bytes_per_tok = (dim * bit_width as usize) / 8 + 4 + 8 + 64;
            let total_bytes = bytes_per_tok * n_tokens;
            let idx_arc = cache::insert(key, idx, total_bytes, relfile, n_rows);
            SLOT_DOC.with(|m| {
                m.borrow_mut().insert(key, slot_doc.clone());
            });
            let guard = idx_arc.read();
            search_tokens(&guard, &slot_doc, &q_flat, nq, per_token_k)
        }
    };

    // Union hits by doc-id, keeping the BEST (max) stage-1 token score
    // per doc. Then take the top `candidate_n` docs by that score so a
    // large per_token_k * nq hit set is capped to the most promising
    // candidates before the (more expensive) exact stage-2 rerank.
    let mut best: HashMap<i64, f32> = HashMap::new();
    for (doc_id, score) in hits {
        best.entry(doc_id)
            .and_modify(|s| {
                if score > *s {
                    *s = score;
                }
            })
            .or_insert(score);
    }
    let mut docs: Vec<(i64, f32)> = best.into_iter().collect();
    // Deterministic: best score desc, then doc id asc.
    docs.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    docs.truncate(candidate_n);
    docs.into_iter().map(|(id, _)| id).collect()
}

thread_local! {
    /// Process-local slot->doc-id map for each cached token index. The
    /// `IdMapIndex` stores synthetic slot-ids (0..n_tokens); the real
    /// per-token doc-ids are kept here, parallel to the cache entry,
    /// so a warm cache hit doesn't have to rebuild the index just to
    /// recover the mapping. Keyed identically to the index cache.
    static SLOT_DOC: std::cell::RefCell<HashMap<CacheKey, Vec<i64>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Run a batched search of `nq` query tokens (row-major `q_flat`,
/// length `nq*dim`) against the token index, returning `(doc_id,
/// score)` for every hit, mapping each result slot through
/// `slot_doc`. The kernel returns `nq * k` results row-major per
/// query; a slot id out of range of `slot_doc` (shouldn't happen) is
/// skipped.
fn search_tokens(
    idx: &IdMapIndex,
    slot_doc: &[i64],
    q_flat: &[f32],
    nq: usize,
    per_token_k: usize,
) -> Vec<(i64, f32)> {
    if idx.is_empty() || nq == 0 || slot_doc.is_empty() {
        return Vec::new();
    }
    let take = per_token_k.min(idx.len());
    if take == 0 {
        return Vec::new();
    }
    // The index was built with synthetic slot-ids 0..n_tokens, so the
    // ids the kernel returns ARE slot indices into `slot_doc`.
    let (scores, slot_ids) = idx.search(q_flat, take);
    let mut out: Vec<(i64, f32)> = Vec::with_capacity(scores.len());
    for (slot, score) in slot_ids.iter().zip(scores.iter()) {
        let s = *slot as usize;
        if s < slot_doc.len() {
            out.push((slot_doc[s], *score));
        }
    }
    out
}

/// Build the flat token index from the heap: unnest each doc's
/// `token_col` array into per-token slots. Returns the index (keyed by
/// synthetic slot-ids 0..n_tokens), the parallel slot->doc-id map, and
/// the token count.
fn build_token_index(
    rel: pg_sys::Oid,
    id_col: &str,
    token_col: &str,
    dim: usize,
    bit_width: i32,
    normalise: bool,
) -> (IdMapIndex, Vec<i64>, usize) {
    let rows = collect_token_rows(rel, id_col, token_col, dim);
    if rows.is_empty() {
        let empty = IdMapIndex::new(dim, bit_width as usize)
            .expect("turbovec.colbert_search: invalid (dim, bit_width)");
        return (empty, Vec::new(), 0);
    }

    let mut flat: Vec<f32> = Vec::new();
    let mut slot_doc: Vec<i64> = Vec::new();
    for (doc_id, tokens) in &rows {
        // `tokens` is a flat n_tok*dim buffer for this doc.
        let n_tok = tokens.len() / dim;
        for t in 0..n_tok {
            let tok = &tokens[t * dim..(t + 1) * dim];
            if normalise {
                flat.extend_from_slice(&kernels::normalise_to_vec(tok));
            } else {
                flat.extend_from_slice(tok);
            }
            slot_doc.push(*doc_id);
        }
    }
    let n_tokens = slot_doc.len();
    if n_tokens == 0 {
        let empty = IdMapIndex::new(dim, bit_width as usize)
            .expect("turbovec.colbert_search: invalid (dim, bit_width)");
        return (empty, Vec::new(), 0);
    }

    // Synthetic unique slot-ids 0..n_tokens (the IVF/soft-assign trick:
    // IdMapIndex requires unique ids, the real doc-ids live in
    // slot_doc with duplicates).
    let synthetic: Vec<u64> = (0..n_tokens as u64).collect();
    let mut idx = IdMapIndex::new(dim, bit_width as usize)
        .expect("turbovec.colbert_search: invalid (dim, bit_width)");
    idx.add_with_ids(&flat, &synthetic)
        .unwrap_or_else(|e| error!("turbovec.colbert_search: add_with_ids failed: {:?}", e));

    (idx, slot_doc, n_tokens)
}

/// Exact stage-2 MaxSim using dot similarity (the Phase D kernel,
/// inlined to avoid re-wrapping `Vector`). `q_norm` is the list of
/// (already-normalised-if-configured) query token slices; `doc` is the
/// doc's flat n_tok*dim token buffer.
fn max_sim_dot(q_norm: &[Vec<f32>], doc: &[f32], dim: usize) -> f64 {
    if q_norm.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let n_tok = doc.len() / dim;
    let mut total = 0.0_f64;
    for q in q_norm {
        let mut best = f64::NEG_INFINITY;
        for t in 0..n_tok {
            let d = &doc[t * dim..(t + 1) * dim];
            let s = kernels::dot(q, d);
            if s > best {
                best = s;
            }
        }
        // An empty doc (n_tok == 0) leaves best = -inf; guard to 0.
        if best.is_finite() {
            total += best;
        }
    }
    total
}

/// Fetch the full token arrays for the given candidate doc-ids from
/// the heap. Returns `(doc_id, flat n_tok*dim buffer)` per doc. Docs
/// whose tokens have a wrong inner dim or non-finite values are
/// skipped (defensive; the build path applied the same filter).
fn fetch_doc_tokens(
    rel: pg_sys::Oid,
    id_col: &str,
    token_col: &str,
    dim: usize,
    candidates: &[i64],
) -> Vec<(i64, Vec<f32>)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let qualified = qualified_name(rel);
    let id_q = pgrx::spi::quote_identifier(id_col);
    let tok_q = pgrx::spi::quote_identifier(token_col);

    // Build an IN-list of the candidate ids. They're our own i64s
    // (no injection risk) but format defensively.
    let in_list = candidates
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    // token_col is a turbovec.vector[]; unnest to one row per token
    // (doc_id, token::real[]), ordered by doc_id so we can group the
    // tokens of each doc contiguously in Rust. We add a stable
    // WITH ORDINALITY tie so token order within a doc is the array
    // order (determinism).
    let sql = format!(
        "SELECT ({id_q})::bigint AS doc_id, \
                t::turbovec.vector::real[] AS tok \
         FROM   {qualified}, unnest({tok_q}) WITH ORDINALITY AS u(t, ord) \
         WHERE  ({id_q}) IN ({in_list}) \
         ORDER  BY doc_id, u.ord"
    );

    let mut out: Vec<(i64, Vec<f32>)> = Vec::new();
    let mut cur_id: Option<i64> = None;
    let mut cur_flat: Vec<f32> = Vec::new();
    Spi::connect(|client| {
        let tup_iter = match client.select(&sql, None, &[]) {
            Ok(t) => t,
            Err(e) => error!("turbovec.colbert_search: SPI select failed: {}", e),
        };
        for row in tup_iter {
            let id: Option<i64> = row.get(1).ok().flatten();
            let tok: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            let (Some(id), Some(tok)) = (id, tok) else {
                continue;
            };
            if tok.len() != dim {
                continue;
            }
            // Doc boundary: flush the previous doc.
            if cur_id != Some(id) {
                if let Some(prev) = cur_id.take() {
                    if !cur_flat.is_empty() {
                        out.push((prev, std::mem::take(&mut cur_flat)));
                    }
                }
                cur_id = Some(id);
                cur_flat = Vec::new();
            }
            let mut ok = true;
            for v in &tok {
                let v = v.unwrap_or(f32::NAN);
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                cur_flat.push(v);
            }
            if !ok {
                // Drop this whole doc on a bad token (defensive).
                cur_flat.clear();
            }
        }
        if let Some(prev) = cur_id.take() {
            if !cur_flat.is_empty() {
                out.push((prev, std::mem::take(&mut cur_flat)));
            }
        }
    });
    out
}

/// Pull `(doc_id, flat token buffer)` rows for building the token
/// index: unnest every doc's token array. Same shape as
/// `fetch_doc_tokens` but over the whole table.
fn collect_token_rows(
    rel: pg_sys::Oid,
    id_col: &str,
    token_col: &str,
    dim: usize,
) -> Vec<(i64, Vec<f32>)> {
    let qualified = qualified_name(rel);
    let id_q = pgrx::spi::quote_identifier(id_col);
    let tok_q = pgrx::spi::quote_identifier(token_col);

    let sql = format!(
        "SELECT ({id_q})::bigint AS doc_id, \
                t::turbovec.vector::real[] AS tok \
         FROM   {qualified}, unnest({tok_q}) WITH ORDINALITY AS u(t, ord) \
         WHERE  ({id_q}) IS NOT NULL AND ({tok_q}) IS NOT NULL \
         ORDER  BY doc_id, u.ord"
    );

    let mut out: Vec<(i64, Vec<f32>)> = Vec::new();
    let mut cur_id: Option<i64> = None;
    let mut cur_flat: Vec<f32> = Vec::new();
    let mut cur_bad = false;
    Spi::connect(|client| {
        let tup_iter = match client.select(&sql, None, &[]) {
            Ok(t) => t,
            Err(e) => error!("turbovec.colbert_search: SPI select failed: {}", e),
        };
        for row in tup_iter {
            let id: Option<i64> = row.get(1).ok().flatten();
            let tok: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            let (Some(id), Some(tok)) = (id, tok) else {
                continue;
            };
            if cur_id != Some(id) {
                if let Some(prev) = cur_id.take() {
                    if !cur_bad && !cur_flat.is_empty() {
                        out.push((prev, std::mem::take(&mut cur_flat)));
                    }
                }
                cur_id = Some(id);
                cur_flat = Vec::new();
                cur_bad = false;
            }
            if tok.len() != dim {
                cur_bad = true;
                continue;
            }
            for v in &tok {
                let v = v.unwrap_or(f32::NAN);
                if !v.is_finite() {
                    cur_bad = true;
                    break;
                }
                cur_flat.push(v);
            }
        }
        if let Some(prev) = cur_id.take() {
            if !cur_bad && !cur_flat.is_empty() {
                out.push((prev, std::mem::take(&mut cur_flat)));
            }
        }
    });
    out
}

/// Resolve the `schema.table` qualified name for an oid, or ERROR.
fn qualified_name(rel: pg_sys::Oid) -> String {
    Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM   pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE  c.oid = $1",
        &[rel.into()],
    )
    .ok()
    .flatten()
    .unwrap_or_else(|| error!("turbovec.colbert_search: relation oid {:?} not found", rel))
}

/// Heap attnum for a column, or 1 if not found (a valid attnum; the
/// negated key just won't match anything useful, which is safe).
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

/// Heap row (doc) count for cache invalidation, -1 on failure.
fn relation_row_count(rel: pg_sys::Oid) -> i64 {
    let qualified = qualified_name(rel);
    let sql = format!("SELECT count(*)::int8 FROM {qualified}");
    Spi::get_one::<i64>(&sql).ok().flatten().unwrap_or(-1)
}
