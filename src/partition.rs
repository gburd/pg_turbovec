//! Phase S-1 — partition-level coarse quantizer for partition pruning.
//!
//! At 1T scale the design (see `docs/PARTITIONED_SCALE.md` and the
//! SCALE_1T_PLAN investigation) is a *partitioned* index: a declaratively
//! hash-partitioned parent table with one turbovec index per partition, and
//! PostgreSQL's native `Merge Append` doing the scatter → gather → global
//! top-k merge for free. That is correct for any N, but the per-query
//! fan-out is `O(N)` in the partition count: at N ≈ 100 000 partitions
//! (1T / 10M) opening an `Index Scan` node per partition dominates and
//! queries take seconds.
//!
//! This module is the query-side fix: a two-level IVF, lifted one level up
//! to the *partition* tier. Each partition gets a small **summary** — its
//! centroid (the mean of its vectors) in the ORIGINAL (un-rotated) vector
//! space — stored in the plain catalog table `turbovec.partition_summary`.
//! At query time [`nearest_partitions`] scores the query against every
//! summary and returns the `k_partitions` nearest partitions, so the user
//! fans out to only those `Kp` partitions instead of all `N`. This is
//! exactly how IVF picks the nearest `probes` cells, one tier up.
//!
//! ## Why original space, not the persisted coarse centroids
//!
//! Each partition trains its OWN rotation matrix, so a partition's persisted
//! coarse centroids live in a *partition-specific* rotated space and are not
//! comparable across partitions. The partition mean in original space IS
//! comparable — and it is scored with the same `<=>` / `<->` / `<#>`
//! distance operator the user queries with. So the summary is derived by
//! averaging the partition's rows (via SPI over the heap), independent of
//! any index's internal rotation.
//!
//! ## No wire / catalog-format change
//!
//! `turbovec.partition_summary` is an ordinary derived table (like a
//! materialized view of partition means), NOT part of any index's on-disk
//! wire format. `MetaPageData::version` is untouched (still 7). Existing
//! single and partitioned indexes are unchanged; this whole mechanism is
//! ADDITIVE and OPTIONAL — a query that does not call it behaves exactly as
//! before.
//!
//! ## Recall / pruning tradeoff (honest)
//!
//! Pruning to `Kp < N` partitions can lose a true global-top-k neighbour if
//! that neighbour lives in a partition whose *mean* is not among the `Kp`
//! nearest to the query — the same recall/probe tradeoff IVF has. With hash
//! partitioning (the default), each partition is a uniform random sample of
//! the corpus, so every partition's mean is ≈ the global mean and the true
//! neighbours are spread roughly evenly across partitions — pruning by mean
//! then helps latency but does NOT reliably preserve recall unless `Kp` is
//! large. Summary-based pruning pays off when partitions are *content-
//! clustered* (range/list-partitioned by a semantic key, or hash-partitioned
//! then re-clustered) so a partition's mean is representative of its
//! contents. The `#[pg_test]` proves the *mechanism* (pruned == full fan-out
//! when `Kp` covers the partitions holding the true top-k); the doc spells
//! out the clustering caveat. See `docs/PARTITIONED_SCALE.md`.

use pgrx::prelude::*;

use crate::kernels;
use crate::vec::Vector;

/// Distance metric used to rank partition summaries against a query. Mirrors
/// the three turbovec distance operators so the pruning metric matches the
/// operator the user's `ORDER BY` uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    /// `<=>` — cosine distance (`1 - cosθ`).
    Cosine,
    /// `<->` — Euclidean (L2) distance.
    L2,
    /// `<#>` — negative inner product.
    NegInnerProduct,
}

impl Metric {
    /// Parse the operator-name / short-name form a user passes. Case- and
    /// whitespace-insensitive. Accepts both the operator glyph and a word.
    pub fn parse(s: &str) -> Option<Metric> {
        match s.trim().to_ascii_lowercase().as_str() {
            "<=>" | "cosine" | "cos" => Some(Metric::Cosine),
            "<->" | "l2" | "euclidean" => Some(Metric::L2),
            "<#>" | "ip" | "inner_product" | "neg_ip" => Some(Metric::NegInnerProduct),
            _ => None,
        }
    }

    /// Score `query` against one summary `centroid`. Lower is nearer for all
    /// three (negative inner product makes IP a "lower-is-nearer" key, same
    /// as the `<#>` operator).
    #[inline]
    fn distance(self, query: &[f32], centroid: &[f32]) -> f64 {
        match self {
            Metric::Cosine => kernels::cosine_distance(query, centroid),
            Metric::L2 => kernels::l2_sq(query, centroid).sqrt(),
            Metric::NegInnerProduct => -kernels::dot(query, centroid),
        }
    }
}

/// Pure ranking core (load-independent, unit-tested). Given a query and a
/// list of `(key, centroid)` partition summaries, return the keys of the
/// `k` nearest summaries in ascending-distance order.
///
/// `key` is generic so this is testable with plain integers and reusable
/// with partition OIDs. Ties break toward the lower key (via a stable sort
/// on distance followed by the input order), so the result is deterministic.
/// NaN distances (e.g. cosine against a zero centroid) sort last.
///
/// Cost `O(N·dim + N·log N)` in the partition count `N` — a flat scan over
/// the summaries. At N ≈ 100k summaries × ~1k dim this is a few million
/// f32 ops, single-digit milliseconds, and turns the `O(N)` executor
/// fan-out into an `O(Kp)` one. A sublinear centroid-graph over the
/// summaries (reusing `ivf::CentroidGraph`) is the documented follow-up if
/// even this flat scan becomes the bottleneck.
pub fn rank_nearest<K: Copy>(
    query: &[f32],
    summaries: &[(K, Vec<f32>)],
    k: usize,
    metric: Metric,
) -> Vec<K> {
    if k == 0 || summaries.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(f64, usize)> = summaries
        .iter()
        .enumerate()
        .map(|(i, (_, c))| (metric.distance(query, c), i))
        .collect();
    // Ascending distance; NaN sorts last; ties break toward earlier input
    // (lower index) for determinism.
    scored.sort_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
        (true, true) => a.1.cmp(&b.1),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        }
    });
    scored
        .into_iter()
        .take(k)
        .map(|(_, i)| summaries[i].0)
        .collect()
}

/// Resolve a parent partitioned table's child partitions to
/// `(child_oid, child_regclass_text)` pairs. Errors if `parent` is not a
/// partitioned table (no children in `pg_inherits`).
fn child_partitions(parent: pg_sys::Oid) -> Vec<(pg_sys::Oid, String)> {
    let mut out: Vec<(pg_sys::Oid, String)> = Vec::new();
    Spi::connect(|client| {
        let rows = client
            .select(
                "SELECT c.oid::oid, format('%I.%I', n.nspname, c.relname) \
                 FROM pg_inherits i \
                 JOIN pg_class c ON c.oid = i.inhrelid \
                 JOIN pg_namespace n ON n.oid = c.relnamespace \
                 WHERE i.inhparent = $1 \
                 ORDER BY c.relname",
                None,
                &[parent.into()],
            )
            .unwrap_or_else(|e| error!("turbovec.partition: SPI failed: {e}"));
        for row in rows {
            let oid: Option<pg_sys::Oid> = row.get(1).ok().flatten();
            let name: Option<String> = row.get(2).ok().flatten();
            if let (Some(oid), Some(name)) = (oid, name) {
                out.push((oid, name));
            }
        }
    });
    out
}

/// Compute the mean vector (partition summary) of `vec_col` over one child
/// partition, in ORIGINAL space, via SPI. Returns `None` for an empty
/// partition (no rows) — such a partition has no summary and is never
/// selected. Skips NULL / wrong-dim / non-finite rows.
fn partition_mean(child_qualified: &str, vec_col: &str, expected_dim: usize) -> Option<Vec<f32>> {
    let vec_q = pgrx::spi::quote_identifier(vec_col);
    let sql = format!(
        "SELECT ({vec_q})::turbovec.vector::real[] \
         FROM {child_qualified} \
         WHERE ({vec_q}) IS NOT NULL"
    );
    let mut acc: Vec<f64> = Vec::new();
    let mut n: u64 = 0;
    Spi::connect(|client| {
        let rows = client
            .select(&sql, None, &[])
            .unwrap_or_else(|e| error!("turbovec.partition: SPI select failed: {e}"));
        for row in rows {
            let arr: Option<Vec<Option<f32>>> = row.get(1).ok().flatten();
            let Some(arr) = arr else { continue };
            if arr.len() != expected_dim {
                continue;
            }
            if acc.is_empty() {
                acc = vec![0.0; expected_dim];
            }
            let mut ok = true;
            for (a, v) in acc.iter_mut().zip(arr.iter()) {
                match v {
                    Some(x) if x.is_finite() => *a += f64::from(*x),
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                n += 1;
            }
        }
    });
    if n == 0 {
        return None;
    }
    Some(acc.iter().map(|a| (*a / n as f64) as f32).collect())
}

/// Infer the vector dimension of `vec_col` on a parent/child by reading one
/// non-null row. Returns 0 if the table is empty (caller handles).
fn infer_dim(parent: pg_sys::Oid, vec_col: &str) -> usize {
    let qualified: Option<String> = Spi::get_one_with_args(
        "SELECT format('%I.%I', n.nspname, c.relname) \
         FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.oid = $1",
        &[parent.into()],
    )
    .ok()
    .flatten();
    let Some(qualified) = qualified else {
        return 0;
    };
    let vec_q = pgrx::spi::quote_identifier(vec_col);
    let sql = format!(
        "SELECT turbovec.vector_dims({vec_q}) FROM {qualified} \
         WHERE ({vec_q}) IS NOT NULL LIMIT 1"
    );
    Spi::get_one::<i32>(&sql).ok().flatten().unwrap_or(0) as usize
}

/// `turbovec.refresh_partition_summary(parent regclass, vec_col text)` —
/// (re)compute the per-partition summary (mean vector in original space) for
/// every child partition of `parent` and upsert it into
/// `turbovec.partition_summary`. Returns the number of partitions summarized.
///
/// Call this once after bulk-loading + building the per-partition indexes,
/// and again after a partition's contents change materially (the summary is
/// derived data, like an IVF codebook — a stale summary only affects which
/// partitions get *probed*, never correctness within a probed partition).
///
/// ```ignore
/// SELECT turbovec.refresh_partition_summary('docs'::regclass, 'emb');
/// ```
#[pg_extern]
fn refresh_partition_summary(parent: pg_sys::Oid, vec_col: &str) -> i64 {
    let dim = infer_dim(parent, vec_col);
    if dim == 0 {
        error!(
            "turbovec.refresh_partition_summary: parent table is empty or has no non-null {vec_col} rows"
        );
    }
    let children = child_partitions(parent);
    if children.is_empty() {
        error!(
            "turbovec.refresh_partition_summary: relation {parent:?} has no partitions (not a partitioned table?)"
        );
    }
    // Clear this parent's existing summary rows first, so removed partitions
    // don't linger.
    Spi::run_with_args(
        "DELETE FROM turbovec.partition_summary WHERE parent = $1",
        &[parent.into()],
    )
    .unwrap_or_else(|e| error!("turbovec.refresh_partition_summary: delete failed: {e}"));

    let mut count: i64 = 0;
    for (child_oid, child_qualified) in &children {
        let Some(mean) = partition_mean(child_qualified, vec_col, dim) else {
            continue; // empty partition — no summary
        };
        let centroid = Vector::from_vec(mean);
        Spi::run_with_args(
            "INSERT INTO turbovec.partition_summary (parent, partition, centroid) \
             VALUES ($1, $2, $3)",
            &[parent.into(), (*child_oid).into(), centroid.into()],
        )
        .unwrap_or_else(|e| error!("turbovec.refresh_partition_summary: insert failed: {e}"));
        count += 1;
    }
    count
}

/// Load `(partition_oid, centroid)` summary rows for a parent from the
/// catalog table.
fn load_summaries(parent: pg_sys::Oid) -> Vec<(pg_sys::Oid, Vec<f32>)> {
    let mut out: Vec<(pg_sys::Oid, Vec<f32>)> = Vec::new();
    Spi::connect(|client| {
        let rows = client
            .select(
                "SELECT partition::oid, centroid::real[] \
                 FROM turbovec.partition_summary WHERE parent = $1",
                None,
                &[parent.into()],
            )
            .unwrap_or_else(|e| error!("turbovec.partition: summary load failed: {e}"));
        for row in rows {
            let oid: Option<pg_sys::Oid> = row.get(1).ok().flatten();
            let arr: Option<Vec<Option<f32>>> = row.get(2).ok().flatten();
            if let (Some(oid), Some(arr)) = (oid, arr) {
                let c: Vec<f32> = arr.into_iter().map(|v| v.unwrap_or(f32::NAN)).collect();
                out.push((oid, c));
            }
        }
    });
    out
}

/// `turbovec.nearest_partitions(parent regclass, query vector, k_partitions int,
/// metric text DEFAULT '<=>') RETURNS SETOF regclass` — the partition-level
/// coarse selector. Scores `query` against each partition's summary and
/// returns the `k_partitions` nearest partitions (as `regclass`, nearest
/// first). Fan the user's kNN out to ONLY these partitions.
///
/// `metric` must match the operator the kNN query uses (`<=>` cosine,
/// `<->` L2, `<#>` inner product); default cosine.
///
/// ```ignore
/// SELECT turbovec.nearest_partitions('docs'::regclass, $q, 8);
/// -- -> docs_0007, docs_0003, ...  (the 8 partitions whose mean is
/// --    nearest $q, nearest first)
/// ```
#[pg_extern(stable)]
fn nearest_partitions(
    parent: pg_sys::Oid,
    query: Vector,
    k_partitions: i32,
    metric: default!(&str, "'<=>'"),
) -> TableIterator<'static, (name!(partition, pg_sys::Oid),)> {
    if k_partitions <= 0 {
        error!("turbovec.nearest_partitions: k_partitions must be positive");
    }
    let metric = Metric::parse(metric).unwrap_or_else(|| {
        error!(
            "turbovec.nearest_partitions: unknown metric '{metric}' (use '<=>', '<->', or '<#>')"
        )
    });
    let summaries = load_summaries(parent);
    if summaries.is_empty() {
        error!(
            "turbovec.nearest_partitions: no summaries for {parent:?} — call turbovec.refresh_partition_summary(parent, vec_col) first"
        );
    }
    // Dim guard: a query against summaries of a different dim is a user
    // error; catch it rather than scoring garbage.
    let sdim = summaries[0].1.len();
    if query.dim() != sdim {
        error!(
            "turbovec.nearest_partitions: query dim {} != summary dim {sdim}",
            query.dim()
        );
    }
    let chosen = rank_nearest(query.as_slice(), &summaries, k_partitions as usize, metric);
    TableIterator::new(chosen.into_iter().map(|oid| (oid,)).collect::<Vec<_>>())
}

extension_sql!(
    r#"
    -- Phase S-1: per-partition summary catalog (partition pruning).
    -- Derived data (partition means in ORIGINAL space), NOT index wire
    -- format. Safe to TRUNCATE/rebuild any time via
    -- turbovec.refresh_partition_summary(). One row per non-empty partition.
    CREATE TABLE turbovec.partition_summary (
        parent    regclass NOT NULL,
        partition regclass NOT NULL,
        centroid  turbovec.vector NOT NULL,
        PRIMARY KEY (parent, partition)
    );
    "#,
    name = "partition_summary_catalog",
    requires = [Vector],
);

#[cfg(any(test, feature = "pg_test"))]
mod tests {
    use super::*;

    // Three well-separated summaries in 2-D. Query near summary #1.
    fn fixture() -> Vec<(i32, Vec<f32>)> {
        vec![
            (10, vec![0.0, 0.0]),
            (11, vec![10.0, 0.0]),
            (12, vec![0.0, 10.0]),
            (13, vec![10.0, 10.0]),
        ]
    }

    #[test]
    fn rank_l2_picks_nearest_in_order() {
        let s = fixture();
        // Query at (9,1). dist^2: (10,0)=2 #11; (0,0)=82 #10; (10,10)=82 #13;
        // (0,10)=162 #12. The 82-tie between #10 (input idx 0) and #13 (idx 3)
        // breaks toward the lower input index -> #10 before #13.
        let q = [9.0f32, 1.0];
        let got = rank_nearest(&q, &s, 4, Metric::L2);
        assert_eq!(got, vec![11, 10, 13, 12]);
    }

    #[test]
    fn rank_respects_k() {
        let s = fixture();
        let q = [9.0f32, 1.0];
        assert_eq!(rank_nearest(&q, &s, 2, Metric::L2), vec![11, 10]);
        assert_eq!(rank_nearest(&q, &s, 1, Metric::L2), vec![11]);
    }

    #[test]
    fn rank_edge_cases() {
        let s = fixture();
        let q = [0.0f32, 0.0];
        assert!(rank_nearest(&q, &s, 0, Metric::L2).is_empty());
        // k larger than N clamps to N, no panic.
        assert_eq!(rank_nearest(&q, &s, 99, Metric::L2).len(), 4);
        // Empty summaries -> empty.
        let empty: Vec<(i32, Vec<f32>)> = Vec::new();
        assert!(rank_nearest(&q, &empty, 3, Metric::L2).is_empty());
    }

    #[test]
    fn rank_cosine_ignores_magnitude() {
        // Cosine cares only about direction: (5,5) and (1,1) are the same
        // direction, so a query along (1,1) is equidistant (cos dist 0) to
        // both; the tie breaks toward the lower input index (id 20).
        let s = vec![(20, vec![1.0f32, 1.0]), (21, vec![5.0f32, 5.0])];
        let q = [2.0f32, 2.0];
        assert_eq!(rank_nearest(&q, &s, 1, Metric::Cosine), vec![20]);
    }

    #[test]
    fn rank_neg_ip_prefers_larger_dot() {
        // Negative inner product: larger dot -> smaller (more negative) key
        // -> ranked first.
        let s = vec![(30, vec![1.0f32, 0.0]), (31, vec![3.0f32, 0.0])];
        let q = [1.0f32, 0.0];
        assert_eq!(rank_nearest(&q, &s, 1, Metric::NegInnerProduct), vec![31]);
    }

    #[test]
    fn rank_nan_sorts_last() {
        // Cosine against a zero centroid is NaN; it must sort AFTER finite
        // distances, never crowd out a real neighbour.
        let s = vec![(40, vec![0.0f32, 0.0]), (41, vec![1.0f32, 1.0])];
        let q = [1.0f32, 1.0];
        assert_eq!(rank_nearest(&q, &s, 2, Metric::Cosine), vec![41, 40]);
    }

    #[test]
    fn metric_parse_forms() {
        assert_eq!(Metric::parse("<=>"), Some(Metric::Cosine));
        assert_eq!(Metric::parse(" Cosine "), Some(Metric::Cosine));
        assert_eq!(Metric::parse("<->"), Some(Metric::L2));
        assert_eq!(Metric::parse("L2"), Some(Metric::L2));
        assert_eq!(Metric::parse("<#>"), Some(Metric::NegInnerProduct));
        assert_eq!(Metric::parse("ip"), Some(Metric::NegInnerProduct));
        assert_eq!(Metric::parse("bogus"), None);
    }
}
