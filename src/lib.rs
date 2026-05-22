//! `pg_turbovec` — a PostgreSQL extension providing a vector type and
//! (in Phase 2) an approximate nearest-neighbour index access method
//! backed by the [TurboQuant](https://arxiv.org/abs/2504.19874)
//! algorithm via the [`turbovec`](https://crates.io/crates/turbovec)
//! crate.
//!
//! The public SQL surface mirrors `pgvector` so existing applications
//! and ORMs work with minimal changes:
//!
//! - The `tvector` type (variable dimension `f32` vectors).
//! - Distance operators: `<->` (L2), `<#>` (negative inner product),
//!   `<=>` (cosine), `<+>` (L1).
//! - Helper functions: `l2_distance`, `inner_product`,
//!   `cosine_distance`, `l1_distance`, `vector_dims`, `vector_norm`.
//! - Aggregates: `avg(tvector)`, `sum(tvector)`.
//!
//! See `docs/ARCHITECTURE.md` for the full design and Phase 2/3
//! roadmap (index access method, filtered search, WAL).

use pgrx::prelude::*;

pub mod aggregate;
pub mod cache;
pub mod cast;
pub mod distance;
pub mod extras;
pub mod guc;

#[cfg(feature = "experimental_index_am")]
pub mod index;

pub mod kernels;
pub mod knn;
pub mod normalize;
pub mod tvector;

pgrx::pg_module_magic!();

/// Extension initialization — called when the shared library is loaded.
#[allow(non_snake_case)]
#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    guc::register_gucs();
}

/// Returns the version string for the extension.
#[pg_extern(immutable, parallel_safe)]
fn turbovec_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Set up the search_path so unqualified operator and function
    /// references resolve against the `turbovec` schema. Called at
    /// the top of each test that uses bare operators.
    fn use_turbovec() {
        Spi::run("SET search_path = turbovec, public").unwrap();
    }

    #[pg_test]
    fn version_string() {
        let v: Option<String> = Spi::get_one("SELECT turbovec.turbovec_version()").unwrap();
        assert_eq!(v.as_deref(), Some(env!("CARGO_PKG_VERSION")));
    }

    #[pg_test]
    fn parse_and_render() {
        let out: Option<String> = Spi::get_one(
            "SELECT '[1, 2, 3]'::turbovec.tvector::text",
        )
        .unwrap();
        // Round-trip through CBOR may reorder spacing but preserves values.
        assert!(out.unwrap().contains('1'));
    }

    #[pg_test]
    fn dims_and_norm() {
        let dim: Option<i32> =
            Spi::get_one("SELECT turbovec.vector_dims('[1,2,3]'::turbovec.tvector)").unwrap();
        assert_eq!(dim, Some(3));

        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm('[3,4]'::turbovec.tvector)").unwrap();
        assert!((n.unwrap() - 5.0).abs() < 1e-6);
    }

    #[pg_test]
    fn l2_and_l1() {
        use_turbovec();
        let d: Option<f64> = Spi::get_one(
            "SELECT '[1,2,3]'::tvector <-> '[4,6,3]'::tvector",
        )
        .unwrap();
        assert!((d.unwrap() - 5.0).abs() < 1e-6); // sqrt(9+16+0) = 5

        let l1: Option<f64> = Spi::get_one(
            "SELECT '[1,2,3]'::tvector <+> '[4,6,3]'::tvector",
        )
        .unwrap();
        assert!((l1.unwrap() - 7.0).abs() < 1e-6); // 3 + 4 + 0
    }

    #[pg_test]
    fn inner_product_and_cosine() {
        use_turbovec();
        let neg_ip: Option<f64> = Spi::get_one(
            "SELECT '[1,0,0]'::tvector <#> '[1,0,0]'::tvector",
        )
        .unwrap();
        // <#> = -dot = -1
        assert!((neg_ip.unwrap() + 1.0).abs() < 1e-6);

        let cos: Option<f64> = Spi::get_one(
            "SELECT '[1,0]'::tvector <=> '[0,1]'::tvector",
        )
        .unwrap();
        // perpendicular -> cosine distance = 1.0
        assert!((cos.unwrap() - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn rejects_dim_mismatch() {
        use_turbovec();
        let res = std::panic::catch_unwind(|| {
            Spi::get_one::<f64>(
                "SELECT '[1,2,3]'::tvector <-> '[1,2]'::tvector",
            )
        });
        assert!(res.is_err(), "expected dim-mismatch ERROR");
    }

    #[pg_test]
    fn aggregate_avg() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE t (v tvector)").unwrap();
        Spi::run("INSERT INTO t VALUES ('[1,2,3]'),('[3,4,5]'),('[5,6,7]')").unwrap();
        let avg: Option<String> =
            Spi::get_one("SELECT avg(v)::text FROM t").unwrap();
        let s = avg.unwrap();
        assert!(s.contains("3"));
        assert!(s.contains("4"));
        assert!(s.contains("5"));
    }

    #[pg_test]
    fn array_casts() {
        let v: Option<String> = Spi::get_one(
            "SELECT (ARRAY[1,2,3]::real[])::turbovec.tvector::text",
        )
        .unwrap();
        assert!(v.unwrap().contains('1'));

        let v: Option<String> = Spi::get_one(
            "SELECT '[1.5, 2.5, 3.5]'::turbovec.tvector::real[]::text",
        )
        .unwrap();
        let s = v.unwrap();
        assert!(s.contains("1.5") && s.contains("2.5") && s.contains("3.5"));
    }

    #[pg_test]
    fn normalize_unit_norm() {
        let n: Option<f64> = Spi::get_one(
            "SELECT turbovec.vector_norm(turbovec.tvector_normalize('[3, 4]'::turbovec.tvector))",
        )
        .unwrap();
        assert!((n.unwrap() - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn turbovec_self_score_smoke() {
        let s: Option<f64> = Spi::get_one(
            "SELECT turbovec.turbovec_self_score(\
               turbovec.tvector_normalize('[1,0,0,0,0,0,0,0]'::turbovec.tvector), 4)",
        )
        .unwrap();
        let v = s.unwrap();
        assert!(v.is_finite(), "score not finite: {}", v);
        assert!(v > 0.5, "turbovec self-score should be high, got {}", v);
    }

    #[cfg(feature = "experimental_index_am")]
    #[pg_test]
    fn index_am_create_and_query() {
        use_turbovec();
        // 8-dim test corpus with one obvious nearest neighbour.
        Spi::run(
            "CREATE TABLE t_ann (\
                 id  bigint PRIMARY KEY, \
                 emb tvector)",
        )
        .unwrap();
        Spi::run(
            "INSERT INTO t_ann VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0.9,0.1,0,0,0,0,0,0]'), \
                 (3, '[0,1,0,0,0,0,0,0]'), \
                 (4, '[-1,0,0,0,0,0,0,0]')",
        )
        .unwrap();

        // Build the index. WITH (bit_width = 4) is the default; we
        // pass it explicitly so a future change to the GUC default
        // doesn't silently change behaviour.
        Spi::run(
            "CREATE INDEX t_ann_emb_idx \
             ON t_ann USING turbovec (emb tvector_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Confirm the side-table row was created.
        let n_rows: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM turbovec.am_storage \
             WHERE indexrelid = 't_ann_emb_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_rows, Some(1));

        // Confirm the index actually contains the heap rows. Pull the
        // n_vectors column.
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_ann_emb_idx'::regclass",
        )
        .unwrap();
        assert_eq!(
            n_vec,
            Some(4),
            "expected 4 indexed vectors, got {:?}",
            n_vec
        );

        // ORDER BY <=> with the index in place. Even if the planner
        // doesn't pick the index (cost estimate, small table, etc.)
        // the result must still be correct.
        let first: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ann \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::tvector \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            first,
            Some(1),
            "nearest neighbour to e1 should be row 1"
        );

        // Drop the index — should leave the heap intact.
        Spi::run("DROP INDEX t_ann_emb_idx").unwrap();
        let n_remaining: Option<i64> =
            Spi::get_one("SELECT count(*) FROM t_ann").unwrap();
        assert_eq!(n_remaining, Some(4));
    }

    /// Exercises `aminsert`: build an index over a small corpus,
    /// then INSERT new rows and verify the side-table state and
    /// the search results reflect the additions.
    #[cfg(feature = "experimental_index_am")]
    #[pg_test]
    fn index_am_aminsert_path() {
        use_turbovec();
        Spi::run("CREATE TABLE t_ins (id bigint PRIMARY KEY, emb tvector)").unwrap();
        Spi::run(
            "INSERT INTO t_ins VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_ins_emb_idx \
             ON t_ins USING turbovec (emb tvector_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Insert two more rows AFTER the index exists — these go
        // through aminsert.
        Spi::run(
            "INSERT INTO t_ins VALUES \
                 (3, '[0,0,1,0,0,0,0,0]'), \
                 (4, '[0,0,0,1,0,0,0,0]')",
        )
        .unwrap();

        let n_vec: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_ins_emb_idx'::regclass",
        )
        .unwrap();
        assert_eq!(
            n_vec,
            Some(4),
            "aminsert should have grown the index to 4 vectors, got {:?}",
            n_vec
        );

        // Query for one of the late-inserted rows.
        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ins \
             ORDER BY emb <=> '[0,0,0,1,0,0,0,0]'::tvector \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            nearest,
            Some(4),
            "row 4 (inserted via aminsert) should be nearest to e4"
        );
    }

    /// 64 random-but-deterministic 16-dim vectors. Verifies the AM
    /// agrees with the brute-force kernel on a meaningful recall
    /// measure. dim=8 was too lossy at 4-bit; dim=16 gives the
    /// quantiser enough room.
    #[cfg(feature = "experimental_index_am")]
    #[pg_test]
    fn index_am_recall_64_rows() {
        use_turbovec();
        Spi::run("CREATE TABLE t_64 (id bigint PRIMARY KEY, emb tvector)").unwrap();
        Spi::run(
            "INSERT INTO t_64 \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::tvector \
             FROM generate_series(1, 64) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();

        Spi::run(
            "CREATE INDEX t_64_emb_idx \
             ON t_64 USING turbovec (emb tvector_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Side-table populated.
        let n_indexed: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_64_emb_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_indexed, Some(64));

        // Self-query: the row's own embedding queried back. With
        // 16 dims and 4-bit quantisation the self-score should be
        // among the top-10. (R@10 == 1.0 is the minimum bar; if
        // this fails the kernel is broken.)
        let target_in_top10: Option<bool> = Spi::get_one(
            "WITH q AS (SELECT emb FROM t_64 WHERE id = 17), \
             top10 AS ( \
                 SELECT t.id FROM t_64 t, q \
                 ORDER BY t.emb <=> q.emb \
                 LIMIT 10 \
             ) \
             SELECT EXISTS (SELECT 1 FROM top10 WHERE id = 17)",
        )
        .unwrap();
        assert_eq!(
            target_in_top10,
            Some(true),
            "row 17 should appear in the top-10 nearest to itself"
        );

        // The index's top-5 set should overlap with the brute-force
        // top-10 by at least 3 entries on a fresh random query —
        // a soft recall assertion that catches catastrophic drift
        // without being flaky on quantiser tie-breaks.
        Spi::run(
            "CREATE TEMP TABLE q_64 AS \
             SELECT ('[' || string_agg( \
                 ((hashtext('query:' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::tvector AS q \
             FROM generate_series(1, 16) AS sub(k)",
        )
        .unwrap();
        let overlap: Option<i64> = Spi::get_one(
            "WITH q AS (SELECT q FROM q_64), \
             indexed AS ( \
                 SELECT t.id FROM t_64 t, q \
                 ORDER BY t.emb <=> q.q LIMIT 5 \
             ), \
             exact AS ( \
                 SELECT t.id FROM t_64 t, q \
                 ORDER BY (1.0 - turbovec.inner_product(t.emb, q.q) / \
                                 (turbovec.vector_norm(t.emb) * turbovec.vector_norm(q.q))) \
                 LIMIT 10 \
             ) \
             SELECT count(*) FROM indexed WHERE id IN (SELECT id FROM exact)",
        )
        .unwrap();
        // With dim=16 / 4-bit, expect at least 3/5 overlap with the
        // brute-force top-10. (Tighter recall measurement is in
        // benches/, not unit tests.)
        let overlap = overlap.unwrap_or(0);
        assert!(
            overlap >= 3,
            "index top-5 should overlap brute-force top-10 by >= 3 \
             entries, got {}",
            overlap
        );
    }

    /// Index can be rebuilt via REINDEX without errors.
    #[cfg(feature = "experimental_index_am")]
    #[pg_test]
    fn index_am_reindex() {
        use_turbovec();
        Spi::run("CREATE TABLE t_re (id bigint PRIMARY KEY, emb tvector)").unwrap();
        Spi::run(
            "INSERT INTO t_re VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_re_emb_idx \
             ON t_re USING turbovec (emb tvector_cosine_ops) \
             WITH (bit_width = 2)",
        )
        .unwrap();
        Spi::run("REINDEX INDEX t_re_emb_idx").unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_re_emb_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_vec, Some(2));
    }

    /// `bit_width` reloption out-of-range is rejected at CREATE INDEX.
    #[cfg(feature = "experimental_index_am")]
    #[pg_test]
    fn index_am_rejects_bad_bit_width() {
        use_turbovec();
        Spi::run("CREATE TABLE t_bad (id bigint, emb tvector)").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::run(
                "CREATE INDEX ON t_bad USING turbovec (emb tvector_cosine_ops) \
                 WITH (bit_width = 5)",
            )
        });
        assert!(
            bad.is_err(),
            "bit_width = 5 should be rejected by amoptions"
        );
    }

    #[pg_test]
    fn knn_returns_nearest_first() {
        Spi::run(
            "CREATE TEMP TABLE pgtv_items (\
                 id  bigint PRIMARY KEY, \
                 emb turbovec.tvector)",
        )
        .unwrap();
        Spi::run(
            "INSERT INTO pgtv_items VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0.9,0.1,0,0,0,0,0,0]'), \
                 (3, '[0,1,0,0,0,0,0,0]'), \
                 (4, '[-1,0,0,0,0,0,0,0]')",
        )
        .unwrap();

        let first: Option<i64> = Spi::get_one(
            "SELECT id FROM turbovec.knn(\
                 'pgtv_items'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.tvector, 3) \
             ORDER BY score DESC LIMIT 1",
        )
        .unwrap();
        assert_eq!(first, Some(1));
    }

    #[pg_test]
    fn knn_cache_hit_after_first_call() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE cache_t (id bigint PRIMARY KEY, emb tvector)")
            .unwrap();
        Spi::run(
            "INSERT INTO cache_t VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        let q = "'[1,0,0,0,0,0,0,0]'::turbovec.tvector";
        let first: Option<i64> = Spi::get_one(&format!(
            "SELECT id FROM turbovec.knn(\
                 'cache_t'::regclass, 'id', 'emb', {q}, 1)"
        ))
        .unwrap();
        assert_eq!(first, Some(1));
        let second: Option<i64> = Spi::get_one(&format!(
            "SELECT id FROM turbovec.knn(\
                 'cache_t'::regclass, 'id', 'emb', {q}, 1)"
        ))
        .unwrap();
        assert_eq!(second, Some(1));
        assert!(
            crate::cache::len() >= 1,
            "cache should be populated after lookups"
        );
    }

    #[pg_test]
    fn knn_cache_invalidates_on_insert() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE cache_inv (id bigint PRIMARY KEY, emb tvector)")
            .unwrap();
        Spi::run(
            "INSERT INTO cache_inv VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        let q = "'[0,0,1,0,0,0,0,0]'::turbovec.tvector";
        let _warmup: Option<i64> = Spi::get_one(&format!(
            "SELECT id FROM turbovec.knn(\
                 'cache_inv'::regclass, 'id', 'emb', {q}, 5) \
             ORDER BY score DESC LIMIT 1"
        ))
        .unwrap();
        Spi::run("INSERT INTO cache_inv VALUES (3, '[0,0,1,0,0,0,0,0]')").unwrap();
        let after: Option<i64> = Spi::get_one(&format!(
            "SELECT id FROM turbovec.knn(\
                 'cache_inv'::regclass, 'id', 'emb', {q}, 5) \
             ORDER BY score DESC LIMIT 1"
        ))
        .unwrap();
        assert_eq!(
            after,
            Some(3),
            "newly-inserted closer row should win after cache invalidation"
        );
    }

    #[pg_test]
    fn knn_rejects_bad_k() {
        Spi::run("CREATE TEMP TABLE pgtv_empty (id bigint, emb turbovec.tvector)").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i64>(
                "SELECT count(*) FROM turbovec.knn(\
                     'pgtv_empty'::regclass, 'id', 'emb', \
                     '[1,2,3,4,5,6,7,8]'::turbovec.tvector, 0)",
            )
        });
        assert!(bad.is_err(), "expected ERROR for k=0");
    }

    #[pg_test]
    fn subvector_basic() {
        let s: Option<String> = Spi::get_one(
            "SELECT turbovec.subvector('[10,20,30,40]'::turbovec.tvector, 2, 2)::text",
        )
        .unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("20") && txt.contains("30"));
        assert!(!txt.contains("10") && !txt.contains("40"));
    }

    #[pg_test]
    fn subvector_out_of_bounds() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT turbovec.subvector('[1,2,3]'::turbovec.tvector, 2, 5)::text",
            )
        });
        assert!(bad.is_err(), "expected ERROR for out-of-bounds");
    }

    #[pg_test]
    fn jsonb_round_trip() {
        let txt: Option<String> = Spi::get_one(
            "SELECT '[1, 2.5, -3]'::turbovec.tvector::jsonb::turbovec.tvector::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1") && s.contains("2.5") && s.contains("-3"));
    }

    #[pg_test]
    fn check_dim_passes_and_fails() {
        let ok: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(\
                turbovec.tvector_check_dim('[1,2,3]'::turbovec.tvector, 3))",
        )
        .unwrap();
        assert_eq!(ok, Some(3));

        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>(
                "SELECT turbovec.vector_dims(\
                    turbovec.tvector_check_dim('[1,2,3]'::turbovec.tvector, 4))",
            )
        });
        assert!(bad.is_err(), "expected ERROR for dim mismatch");
    }

    #[pg_test]
    fn zeros_helper() {
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(turbovec.tvector_zeros(8))",
        )
        .unwrap();
        assert_eq!(dim, Some(8));
        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm(turbovec.tvector_zeros(8))")
                .unwrap();
        assert_eq!(n, Some(0.0));
    }
}

/// PGRX test runner harness.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
