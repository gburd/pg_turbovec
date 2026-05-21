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
//!     vec_col   text,            -- tvector column
//!     query     tvector,
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

use crate::guc;
use crate::tvector::Tvector;

/// `turbovec.knn(rel, id_col, vec_col, query, k, bit_width)` — see
/// module documentation.
#[pg_extern(stable, parallel_safe)]
fn knn(
    rel: pg_sys::Oid,
    id_col: &str,
    vec_col: &str,
    query: Tvector,
    k: i32,
    bit_width: default!(i32, 4),
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

    let rows = collect_via_spi(rel, id_col, vec_col, query.dim());
    if rows.is_empty() {
        return TableIterator::new(Vec::<(i64, f64)>::new());
    }

    // Optionally L2-normalise to match TurboQuant's unit-norm
    // assumption. Controlled by `turbovec.normalize_on_insert`.
    let normalise = guc::NORMALIZE_ON_INSERT.get();

    let mut idx = IdMapIndex::new(query.dim(), bit_width as usize);
    let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * query.dim());
    let mut ids: Vec<u64> = Vec::with_capacity(rows.len());
    for (id, vec) in &rows {
        if normalise {
            push_normalised(&mut flat, vec);
        } else {
            flat.extend_from_slice(vec);
        }
        ids.push(*id as u64);
    }
    idx.add_with_ids(&flat, &ids)
        .unwrap_or_else(|e| error!("turbovec.knn: add_with_ids failed: {:?}", e));

    let q_buf: Vec<f32> = if normalise {
        let mut out = Vec::with_capacity(query.dim());
        push_normalised(&mut out, query.as_slice());
        out
    } else {
        query.as_slice().to_vec()
    };

    let take = (k as usize).min(rows.len());
    let (scores, hit_ids) = idx.search(&q_buf, take);

    let result: Vec<(i64, f64)> = hit_ids
        .iter()
        .zip(scores.iter())
        .map(|(id, s)| (*id as i64, f64::from(*s)))
        .collect();

    TableIterator::new(result)
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

/// Pull `(id, vec)` rows out of the relation via SPI. Skips rows
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
                ({vec_q})::turbovec.tvector::real[] \
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
            let values: Vec<f32> = arr
                .into_iter()
                .map(|v| v.unwrap_or(f32::NAN))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            out.push((id, values));
        }
    });
    out
}
