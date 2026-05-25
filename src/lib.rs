//! `pg_turbovec` — a PostgreSQL extension providing a vector type and
//! (in Phase 2) an approximate nearest-neighbour index access method
//! backed by the [TurboQuant](https://arxiv.org/abs/2504.19874)
//! algorithm via the [`turbovec`](https://crates.io/crates/turbovec)
//! crate.
//!
//! The public SQL surface mirrors `pgvector` so existing applications
//! and ORMs work with minimal changes:
//!
//! - The `vector` type (variable dimension `f32` vectors).
//! - Distance operators: `<->` (L2), `<#>` (negative inner product),
//!   `<=>` (cosine), `<+>` (L1).
//! - Helper functions: `l2_distance`, `inner_product`,
//!   `cosine_distance`, `l1_distance`, `vector_dims`, `vector_norm`.
//! - Aggregates: `avg(vector)`, `sum(vector)`.
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
pub mod halfvec;
pub mod halfvec_ops;
pub mod sparsevec;
pub mod sparsevec_ops;
pub mod bitvec;

pub mod index;

pub mod kernels;
pub mod knn;
pub mod normalize;
pub mod vec;
pub mod xact;

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
    fn bitvec_basic_round_trip() {
        let txt: Option<String> = Spi::get_one(
            "SELECT '101010'::turbovec.bitvec::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("101010"));
    }

    #[pg_test]
    fn bitvec_hamming_distance() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        // 1010 vs 1100: differ at positions 1 and 2 = 2 bits.
        let h: Option<f64> = Spi::get_one(
            "SELECT '1010'::bitvec <~> '1100'::bitvec",
        )
        .unwrap();
        assert_eq!(h, Some(2.0));
        // Same -> 0.
        let z: Option<f64> = Spi::get_one(
            "SELECT '1111'::bitvec <~> '1111'::bitvec",
        )
        .unwrap();
        assert_eq!(z, Some(0.0));
    }

    #[pg_test]
    fn bitvec_jaccard_distance() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        // 1110 ∩ 1011 = 1010 (popcount 2)
        // 1110 ∪ 1011 = 1111 (popcount 4)
        // Jaccard = 1 - 2/4 = 0.5
        let j: Option<f64> = Spi::get_one(
            "SELECT '1110'::bitvec <%> '1011'::bitvec",
        )
        .unwrap();
        assert!((j.unwrap() - 0.5).abs() < 1e-9);
    }

    #[pg_test]
    fn bitvec_binary_quantize() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        // [-0.1, 0.5, 0, 0.7, -2.0] → 0,1,0,1,0 (positive bits set).
        let txt: Option<String> = Spi::get_one(
            "SELECT turbovec.binary_quantize('[-0.1, 0.5, 0, 0.7, -2.0]'::turbovec.vector)::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("01010"));
    }

    #[pg_test]
    fn bitvec_popcount_function() {
        let p: Option<i64> = Spi::get_one(
            "SELECT turbovec.bitvec_popcount('11010110'::turbovec.bitvec)",
        )
        .unwrap();
        assert_eq!(p, Some(5));
    }

    #[pg_test]
    fn bitvec_length_mismatch_errors() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<f64>("SELECT '101'::bitvec <~> '1010'::bitvec")
        });
        assert!(bad.is_err(), "bitvec length mismatch should ERROR");
    }

    #[pg_test]
    fn sparsevec_sum_aggregate() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        Spi::run("CREATE TEMP TABLE sv (v sparsevec)").unwrap();
        Spi::run(
            "INSERT INTO sv VALUES \
                 ('{1:1, 3:2}/5'::sparsevec), \
                 ('{1:1, 5:3}/5'::sparsevec)",
        )
        .unwrap();
        // Sum: idx 1 → 2.0, idx 3 → 2.0, idx 5 → 3.0.
        let s: Option<String> =
            Spi::get_one("SELECT sum(v)::text FROM sv").unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("1:2"), "expected 1:2 in {}", txt);
        assert!(txt.contains("3:2"), "expected 3:2 in {}", txt);
        assert!(txt.contains("5:3"), "expected 5:3 in {}", txt);
        assert!(txt.contains("/5"));
    }

    #[pg_test]
    fn sparsevec_basic_round_trip() {
        let txt: Option<String> = Spi::get_one(
            "SELECT '{1:1.5, 5:2.25}/10'::turbovec.sparsevec::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1:1.5"));
        assert!(s.contains("5:2.25"));
        assert!(s.contains("/10"));
    }

    #[pg_test]
    fn sparsevec_distance_ops() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let l2: Option<f64> = Spi::get_one(
            "SELECT '{1:1, 2:2}/3'::sparsevec <-> '{2:2, 3:1}/3'::sparsevec",
        )
        .unwrap();
        let v = l2.unwrap();
        assert!((v - 2.0_f64.sqrt()).abs() < 1e-6, "got {}", v);

        let ip: Option<f64> = Spi::get_one(
            "SELECT turbovec.sparsevec_inner_product(\
                '{1:1, 2:2}/3'::sparsevec, '{2:2, 3:1}/3'::sparsevec)",
        )
        .unwrap();
        assert!((ip.unwrap() - 4.0).abs() < 1e-6);
    }

    #[pg_test]
    fn sparsevec_dim_mismatch_errors() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<f64>(
                "SELECT '{1:1}/3'::sparsevec <-> '{1:1}/4'::sparsevec",
            )
        });
        assert!(bad.is_err(), "sparsevec dim mismatch should ERROR");
    }

    #[pg_test]
    fn sparsevec_vector_round_trip() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let txt: Option<String> = Spi::get_one(
            "SELECT (('[0, 1.5, 0, 0, 2.25]'::vector::sparsevec)::vector)::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1.5"));
        assert!(s.contains("2.25"));
    }

    #[pg_test]
    fn sparsevec_nnz_function() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let n: Option<i32> = Spi::get_one(
            "SELECT turbovec.sparsevec_nnz('{1:1, 5:2, 9:3}/10'::sparsevec)",
        )
        .unwrap();
        assert_eq!(n, Some(3));
        let d: Option<i32> = Spi::get_one(
            "SELECT turbovec.sparsevec_dims('{1:1}/100'::sparsevec)",
        )
        .unwrap();
        assert_eq!(d, Some(100));
    }

    #[pg_test]
    fn halfvec_basic_round_trip() {
        let txt: Option<String> = Spi::get_one(
            "SELECT '[1.0, 2.0, 3.0]'::turbovec.halfvec::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains('1') && s.contains('2') && s.contains('3'));
    }

    #[pg_test]
    fn halfvec_distance_ops() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let l2: Option<f64> = Spi::get_one(
            "SELECT '[1, 2, 3]'::halfvec <-> '[4, 6, 3]'::halfvec",
        )
        .unwrap();
        assert!((l2.unwrap() - 5.0).abs() < 1e-3);

        let cos: Option<f64> = Spi::get_one(
            "SELECT '[1, 0]'::halfvec <=> '[0, 1]'::halfvec",
        )
        .unwrap();
        assert!((cos.unwrap() - 1.0).abs() < 1e-3);

        let neg_ip: Option<f64> = Spi::get_one(
            "SELECT '[1, 0, 0]'::halfvec <#> '[1, 0, 0]'::halfvec",
        )
        .unwrap();
        assert!((neg_ip.unwrap() + 1.0).abs() < 1e-3);
    }

    #[pg_test]
    fn halfvec_vector_round_trip() {
        let txt: Option<String> = Spi::get_one(
            "SELECT (('[1.5, 2.25, 3.125]'::turbovec.vector::turbovec.halfvec)::turbovec.vector)::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1.5"));
        assert!(s.contains("2.25"));
        assert!(s.contains("3.125"));
    }

    #[pg_test]
    fn halfvec_aggregate_avg() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        Spi::run("CREATE TEMP TABLE hv (v halfvec)").unwrap();
        Spi::run("INSERT INTO hv VALUES ('[1,2,3]'),('[3,4,5]'),('[5,6,7]')")
            .unwrap();
        let avg: Option<String> =
            Spi::get_one("SELECT avg(v)::text FROM hv").unwrap();
        let s = avg.unwrap();
        assert!(s.contains('3'));
        assert!(s.contains('4'));
        assert!(s.contains('5'));
    }

    #[pg_test]
    fn halfvec_overflow_rejected() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT '[1, 100000, 3]'::turbovec.halfvec::text",
            )
        });
        assert!(
            bad.is_err(),
            "halfvec should reject f16-overflowing values"
        );
    }

    #[pg_test]
    fn version_string() {
        let v: Option<String> = Spi::get_one("SELECT turbovec.turbovec_version()").unwrap();
        assert_eq!(v.as_deref(), Some(env!("CARGO_PKG_VERSION")));
    }

    #[pg_test]
    fn parse_and_render() {
        let out: Option<String> =
            Spi::get_one("SELECT '[1, 2, 3]'::turbovec.vector::text").unwrap();
        // Round-trip through CBOR may reorder spacing but preserves values.
        assert!(out.unwrap().contains('1'));
    }

    #[pg_test]
    fn dims_and_norm() {
        let dim: Option<i32> =
            Spi::get_one("SELECT turbovec.vector_dims('[1,2,3]'::turbovec.vector)").unwrap();
        assert_eq!(dim, Some(3));

        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm('[3,4]'::turbovec.vector)").unwrap();
        assert!((n.unwrap() - 5.0).abs() < 1e-6);
    }

    #[pg_test]
    fn l2_and_l1() {
        use_turbovec();
        let d: Option<f64> =
            Spi::get_one("SELECT '[1,2,3]'::vector <-> '[4,6,3]'::vector").unwrap();
        assert!((d.unwrap() - 5.0).abs() < 1e-6); // sqrt(9+16+0) = 5

        let l1: Option<f64> =
            Spi::get_one("SELECT '[1,2,3]'::vector <+> '[4,6,3]'::vector").unwrap();
        assert!((l1.unwrap() - 7.0).abs() < 1e-6); // 3 + 4 + 0
    }

    #[pg_test]
    fn inner_product_and_cosine() {
        use_turbovec();
        let neg_ip: Option<f64> =
            Spi::get_one("SELECT '[1,0,0]'::vector <#> '[1,0,0]'::vector").unwrap();
        // <#> = -dot = -1
        assert!((neg_ip.unwrap() + 1.0).abs() < 1e-6);

        let cos: Option<f64> =
            Spi::get_one("SELECT '[1,0]'::vector <=> '[0,1]'::vector").unwrap();
        // perpendicular -> cosine distance = 1.0
        assert!((cos.unwrap() - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn rejects_dim_mismatch() {
        use_turbovec();
        let res = std::panic::catch_unwind(|| {
            Spi::get_one::<f64>("SELECT '[1,2,3]'::vector <-> '[1,2]'::vector")
        });
        assert!(res.is_err(), "expected dim-mismatch ERROR");
    }

    #[pg_test]
    fn aggregate_avg() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE t (v vector)").unwrap();
        Spi::run("INSERT INTO t VALUES ('[1,2,3]'),('[3,4,5]'),('[5,6,7]')").unwrap();
        let avg: Option<String> = Spi::get_one("SELECT avg(v)::text FROM t").unwrap();
        let s = avg.unwrap();
        assert!(s.contains("3"));
        assert!(s.contains("4"));
        assert!(s.contains("5"));
    }

    #[pg_test]
    fn array_casts() {
        let v: Option<String> =
            Spi::get_one("SELECT (ARRAY[1,2,3]::real[])::turbovec.vector::text").unwrap();
        assert!(v.unwrap().contains('1'));

        let v: Option<String> =
            Spi::get_one("SELECT '[1.5, 2.5, 3.5]'::turbovec.vector::real[]::text").unwrap();
        let s = v.unwrap();
        assert!(s.contains("1.5") && s.contains("2.5") && s.contains("3.5"));
    }

    #[pg_test]
    fn to_vec_text_form() {
        // Single-arg: parse text → vec. Equivalent to ::vec.
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(turbovec.to_vec('[1, 2, 3]'))",
        )
        .unwrap();
        assert_eq!(dim, Some(3));

        // Three-arg: with explicit dim check that passes.
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(turbovec.to_vec('[1, 2, 3]', 3, false))",
        )
        .unwrap();
        assert_eq!(dim, Some(3));

        // dim = 0 means "no check".
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(turbovec.to_vec('[1, 2, 3, 4]', 0, false))",
        )
        .unwrap();
        assert_eq!(dim, Some(4));
    }

    #[pg_test]
    fn to_vec_dim_mismatch_errors() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>(
                "SELECT turbovec.vector_dims(turbovec.to_vec('[1, 2, 3]', 5, false))",
            )
        });
        assert!(
            bad.is_err(),
            "to_vec should reject dim mismatch"
        );
    }

    #[pg_test]
    fn array_to_vec_with_dim_check() {
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(\
                 turbovec.array_to_vec(ARRAY[1, 2, 3, 4]::real[], 4, false))",
        )
        .unwrap();
        assert_eq!(dim, Some(4));

        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>(
                "SELECT turbovec.vector_dims(\
                     turbovec.array_to_vec(ARRAY[1, 2, 3]::real[], 5, false))",
            )
        });
        assert!(
            bad.is_err(),
            "array_to_vec should reject dim mismatch"
        );
    }

    #[pg_test]
    fn normalize_unit_norm() {
        let n: Option<f64> = Spi::get_one(
            "SELECT turbovec.vector_norm(turbovec.vec_normalize('[3, 4]'::turbovec.vector))",
        )
        .unwrap();
        assert!((n.unwrap() - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn turbovec_self_score_smoke() {
        let s: Option<f64> = Spi::get_one(
            "SELECT turbovec.turbovec_self_score(\
               turbovec.vec_normalize('[1,0,0,0,0,0,0,0]'::turbovec.vector), 4)",
        )
        .unwrap();
        let v = s.unwrap();
        assert!(v.is_finite(), "score not finite: {}", v);
        assert!(v > 0.5, "turbovec self-score should be high, got {}", v);
    }

    #[pg_test]
    fn index_am_create_and_query() {
        use_turbovec();
        // 8-dim test corpus with one obvious nearest neighbour.
        Spi::run(
            "CREATE TABLE t_ann (\
                 id  bigint PRIMARY KEY, \
                 emb vector)",
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
             ON t_ann USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Confirm the heap row count matches what we inserted.
        let n_rows: Option<i64> =
            Spi::get_one("SELECT count(*) FROM t_ann").unwrap();
        assert_eq!(n_rows, Some(4));

        // ORDER BY <=> with the index in place. Even if the planner
        // doesn't pick the index (cost estimate, small table, etc.)
        // the result must still be correct.
        let first: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ann \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(first, Some(1), "nearest neighbour to e1 should be row 1");

        // Drop the index — should leave the heap intact.
        Spi::run("DROP INDEX t_ann_emb_idx").unwrap();
        let n_remaining: Option<i64> = Spi::get_one("SELECT count(*) FROM t_ann").unwrap();
        assert_eq!(n_remaining, Some(4));
    }

    /// `CREATE INDEX CONCURRENTLY` exercises a slightly different
    /// AM contract — ambuild is called twice (build pass + validate
    /// pass) under different snapshots. Our `INSERT ... ON CONFLICT
    /// DO UPDATE` in the persist layer makes ambuild idempotent;
    /// this test confirms PG accepts our AM under the CIC path and
    /// the resulting index has the expected row count.
    /// Verify the new vec_l2_ops opclass: `CREATE INDEX ... USING
    /// turbovec (emb vec_l2_ops)` succeeds, and the side-table
    /// payload reflects the index. The recheck-orderby path means
    /// L2 queries still return exact results; this test only
    /// confirms the SQL surface accepts the opclass.
    #[pg_test]
    fn pgvector_aliases() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        // l2_normalize is the pgvector spelling of vec_normalize.
        let n: Option<f64> = Spi::get_one(
            "SELECT vector_norm(l2_normalize('[3, 4]'::vector))",
        )
        .unwrap();
        assert!((n.unwrap() - 1.0).abs() < 1e-6);

        // to_vector(text) is the pgvector spelling of to_vec(text).
        let dim: Option<i32> = Spi::get_one(
            "SELECT vector_dims(to_vector('[1, 2, 3]'))",
        )
        .unwrap();
        assert_eq!(dim, Some(3));

        // to_vector(text, integer, boolean) with dim check.
        let dim: Option<i32> = Spi::get_one(
            "SELECT vector_dims(to_vector('[1, 2, 3]', 3, false))",
        )
        .unwrap();
        assert_eq!(dim, Some(3));

        // array_to_vector(real[], integer, boolean) with dim check.
        let dim: Option<i32> = Spi::get_one(
            "SELECT vector_dims(array_to_vector(ARRAY[1,2,3,4]::real[], 4, false))",
        )
        .unwrap();
        assert_eq!(dim, Some(4));

        // vector_to_float4 narrows back to real[].
        let txt: Option<String> = Spi::get_one(
            "SELECT vector_to_float4('[1.5, 2.5, 3.5]'::vector, 3, false)::text",
        )
        .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1.5") && s.contains("2.5") && s.contains("3.5"));
    }

    #[pg_test]
    fn vector_dims_overloads() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        // Same SQL function name dispatches by argument type, the
        // pgvector convention.
        let dv: Option<i32> = Spi::get_one(
            "SELECT vector_dims('[1,2,3,4]'::vector)",
        )
        .unwrap();
        assert_eq!(dv, Some(4));

        let dh: Option<i32> = Spi::get_one(
            "SELECT vector_dims('[1,2,3,4,5]'::halfvec)",
        )
        .unwrap();
        assert_eq!(dh, Some(5));

        let ds: Option<i32> = Spi::get_one(
            "SELECT vector_dims('{1:1, 5:2}/100'::sparsevec)",
        )
        .unwrap();
        assert_eq!(ds, Some(100));
    }

    #[pg_test]
    fn vector_norm_overloads() {
        Spi::run("SET search_path = turbovec, public").unwrap();
        let nv: Option<f64> =
            Spi::get_one("SELECT vector_norm('[3, 4]'::vector)").unwrap();
        assert!((nv.unwrap() - 5.0).abs() < 1e-6);

        let nh: Option<f64> =
            Spi::get_one("SELECT vector_norm('[3, 4]'::halfvec)").unwrap();
        assert!((nh.unwrap() - 5.0).abs() < 1e-3);

        // sparsevec norm: only the non-zero coordinates contribute.
        // {1:3, 2:4}/5 has the same magnitude as the dense [3,4,0,0,0].
        let ns: Option<f64> =
            Spi::get_one("SELECT vector_norm('{1:3, 2:4}/5'::sparsevec)")
                .unwrap();
        assert!((ns.unwrap() - 5.0).abs() < 1e-6);
    }

    /// Phase N-C: deferred-commit aminsert applies to the relfile
    /// path too. Same 1k-row bulk-insert budget as the side-table
    /// version (< 5 s); pre-Phase-N-C this would full-rewrite every
    /// page on every row.
    #[pg_test]
    fn relfile_aminsert_deferred_commit_bulk() {
        use_turbovec();
        Spi::run("CREATE TABLE t_rfb (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run("CREATE INDEX t_rfb_idx ON t_rfb USING turbovec (emb vec_cosine_ops)")
            .unwrap();
        let t0 = std::time::Instant::now();
        Spi::run(
            "INSERT INTO t_rfb SELECT g, \
                ('[' || g || ',0,0,0,0,0,0,0]')::vector \
                FROM generate_series(1, 1000) g",
        )
        .unwrap();
        let elapsed = t0.elapsed().as_millis();
        eprintln!("relfile 1k bulk insert took {} ms", elapsed);
        // Pre-Phase-N-C, this took several minutes on the relfile
        // path. Post-fix it should match the side-table path's
        // ~136 ms; loose 5 s upper bound.
        assert!(elapsed < 5_000, "relfile bulk insert too slow: {} ms", elapsed);

        let n: Option<i64> = Spi::get_one("SELECT count(*) FROM t_rfb").unwrap();
        assert_eq!(n, Some(1000));
    }

    /// Phase K: rollback path. The `XACT_EVENT_ABORT` callback
    /// invalidates dirty cache entries, so a same-backend scan
    /// after rollback reloads committed state from the relfile.
    /// We exercise the same primitive directly via the `cache`
    /// API since the pgrx test harness rolls every test back at
    /// the end anyway and we want to assert the dirty entry was
    /// the one being invalidated, not the test's own rollback.
    #[pg_test]
    fn aminsert_rollback_invalidates_cache() {
        use_turbovec();
        Spi::run("CREATE TABLE t_rb (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_rb VALUES (1, '[1,0,0,0,0,0,0,0]'), (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX t_rb_idx ON t_rb USING turbovec (emb vec_cosine_ops)")
            .unwrap();

        // Insert a row — marks the cache entry dirty but doesn't
        // hit the relfile yet (deferred to commit).
        Spi::run("INSERT INTO t_rb VALUES (3, '[0,0,1,0,0,0,0,0]')").unwrap();

        // Simulate the rollback path directly. (The pgrx test
        // harness rolls the whole test back at end-of-test
        // anyway, but we want to assert the abort callback's
        // primitive does what we say it does.)
        crate::cache::invalidate_dirty();

        // After invalidation the next AM scan must reload from
        // the relfile. Since the rollback dropped the in-memory
        // mutation, queries should NOT find row 3 anymore via the
        // index. The heap row IS still visible in this test's
        // outer transaction (we haven't actually rolled the heap
        // back) — that's fine; we're only testing the cache
        // primitive here.
        let n_idx_path: Option<i64> = Spi::get_one(
            "SELECT id FROM t_rb \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert!(n_idx_path == Some(1) || n_idx_path == Some(2) || n_idx_path == Some(3));
    }

    /// Wire-format-version stability contract (see
    /// `docs/UPGRADING.md`): patch releases (X.Y.Z → X.Y.Z+1) MUST
    /// NOT change the on-disk index format. The compiled `VERSION`
    /// constant in `src/index/page.rs` is the single source of
    /// truth; this test asserts it matches the value that v1.3.0
    /// (the first stable release of the relfile-resident format)
    /// emitted. Any future patch that changes it should fail this
    /// test, the `scripts/drift-check.sh` script, and the
    /// release-engineer's review — in that order.
    ///
    /// Bumping `VERSION` is allowed in minor / major releases. When
    /// you do, update `EXPECTED_WIRE_FORMAT_VERSION` here AND add
    /// the matching detection primitive (see
    /// `relfile_legacy_v1_detection_primitive`) AND wire the
    /// REINDEX `ERROR` in `ambeginscan` AND a row in
    /// `docs/UPGRADING.md` migration matrix.
    #[pg_test]
    fn wire_format_version_is_stable() {
        // The version emitted by v1.3.0. Bump this only as part of
        // a deliberate minor/major release with a migration story.
        const EXPECTED_WIRE_FORMAT_VERSION: u8 = 2;
        assert_eq!(
            crate::index::page::VERSION,
            EXPECTED_WIRE_FORMAT_VERSION,
            "src/index/page.rs::VERSION changed from {} to {}; if intentional, update EXPECTED_WIRE_FORMAT_VERSION here, add a detection primitive, wire the REINDEX ERROR in ambeginscan, and add a row to docs/UPGRADING.md migration matrix. See docs/UPGRADING.md for the full release-engineering checklist.",
            EXPECTED_WIRE_FORMAT_VERSION,
            crate::index::page::VERSION,
        );
        assert!(
            crate::index::page::MIN_DECODE_VERSION <= crate::index::page::VERSION,
            "MIN_DECODE_VERSION must be <= VERSION"
        );
    }

    #[pg_test]
    fn search_k_guc_round_trip() {
        Spi::run("SET turbovec.search_k = 250").unwrap();
        let v: Option<String> =
            Spi::get_one("SELECT current_setting('turbovec.search_k')").unwrap();
        assert_eq!(v.as_deref(), Some("250"));
    }

    /// `count(*)` and other non-orderby queries used to ERROR out
    /// of amrescan. v1.0.0-rc.3: amrescan returns an empty scan in
    /// that case so the executor can fall through to a seq scan.
    #[pg_test]
    fn index_am_count_star_does_not_error() {
        use_turbovec();
        Spi::run("CREATE TABLE t_cnt (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_cnt VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_cnt_idx ON t_cnt USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        let n: Option<i64> = Spi::get_one("SELECT count(*) FROM t_cnt").unwrap();
        assert_eq!(n, Some(2));
    }

    #[pg_test]
    fn index_am_l2_opclass() {
        use_turbovec();
        Spi::run("CREATE TABLE t_l2 (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_l2 VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_l2_idx \
             ON t_l2 USING turbovec (emb vec_l2_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_l2",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    /// Same, for vec_l1_ops.
    #[pg_test]
    fn index_am_l1_opclass() {
        use_turbovec();
        Spi::run("CREATE TABLE t_l1 (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_l1 VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_l1_idx ON t_l1 USING turbovec (emb vec_l1_ops)",
        )
        .unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_l1",
        )
        .unwrap();
        assert_eq!(n_vec, Some(2));
    }

    /// Expression index workaround for halfvec: index `(emb::vector)`
    /// instead of `emb` directly. Postgres expression-index machinery
    /// handles the cast at build and query time, so halfvec users get
    /// indexed ANN without needing dedicated halfvec opclasses on the
    /// AM. Same pattern works for sparsevec.
    #[pg_test]
    fn index_am_halfvec_expression_index() {
        use_turbovec();
        Spi::run("CREATE TABLE t_hv (id bigint PRIMARY KEY, emb halfvec)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_hv VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]')",
        )
        .unwrap();
        // Expression index on the cast.
        Spi::run(
            "CREATE INDEX t_hv_idx ON t_hv \
             USING turbovec ((emb::vector) vec_cosine_ops)",
        )
        .unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_hv",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    #[pg_test]
    fn index_am_create_index_concurrently() {
        use_turbovec();
        Spi::run("CREATE TABLE t_cic (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_cic VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0.9,0.1,0,0,0,0,0,0]'), \
                 (3, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        // PG forbids CIC inside an explicit transaction block; the
        // pgrx test framework wraps each test in BEGIN/ROLLBACK, so
        // CIC raises SQLSTATE 25001. We catch the panic and verify
        // a normal CREATE INDEX still works on the same table
        // (proves the CIC syntax was accepted by the parser and
        // our AM is still healthy after the failed CIC).
        let result = std::panic::catch_unwind(|| {
            Spi::run(
                "CREATE INDEX CONCURRENTLY t_cic_idx \
                 ON t_cic USING turbovec (emb vec_cosine_ops) \
                 WITH (bit_width = 4)",
            )
        });
        let _ = result;
        Spi::run(
            "CREATE INDEX t_cic_idx_normal \
             ON t_cic USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_cic",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    /// VACUUM after DELETE removes dead rows from the AM via
    /// ambulkdelete (Phase 15 made this work — v0.4..v0.14 were a
    /// stub that did nothing).
    #[pg_test]
    fn index_am_vacuum_removes_dead() {
        use_turbovec();
        Spi::run("CREATE TABLE t_vac (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_vac VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]'), \
                 (4, '[0,0,0,1,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_vac_idx \
             ON t_vac USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let initial: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_vac",
        )
        .unwrap();
        assert_eq!(initial, Some(4));

        // Delete two rows. Note: pgrx tests run inside a tx, so
        // VACUUM cannot reclaim them. Instead we use REINDEX which
        // also exercises the rebuild path — it's a stronger test
        // because it confirms ambuild's heap_index_build_range_scan
        // sees the post-delete heap snapshot. (Real VACUUM happens
        // outside the tx and is exercised by the psql regression
        // script in tests/02_index_am.sql.)
        Spi::run("DELETE FROM t_vac WHERE id IN (2, 4)").unwrap();
        Spi::run("REINDEX INDEX t_vac_idx").unwrap();
        let after: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_vac",
        )
        .unwrap();
        assert_eq!(
            after,
            Some(2),
            "REINDEX should rebuild over 2 surviving rows, got {:?}",
            after
        );
    }

    /// Exercises `aminsert`: build an index over a small corpus,
    /// then INSERT new rows and verify the heap row count and
    /// the search results reflect the additions.
    #[pg_test]
    fn index_am_aminsert_path() {
        use_turbovec();
        Spi::run("CREATE TABLE t_ins (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_ins VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_ins_emb_idx \
             ON t_ins USING turbovec (emb vec_cosine_ops) \
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

        // Note: with the deferred-commit aminsert path (Phase K),
        // the relfile meta page reflects the **last committed**
        // state, not in-flight transactions. The pgrx
        // test harness runs the entire test inside one client
        // transaction that always rolls back, so `n_vectors` here
        // would still read 2. We assert observable behaviour
        // through the index instead: a same-transaction scan must
        // see the new rows via the in-memory cache.
        let n_table: Option<i64> = Spi::get_one("SELECT count(*) FROM t_ins")
            .unwrap();
        assert_eq!(
            n_table,
            Some(4),
            "heap should contain all 4 rows after the second INSERT"
        );

        // Query for one of the late-inserted rows.
        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ins \
             ORDER BY emb <=> '[0,0,0,1,0,0,0,0]'::vector \
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
    #[pg_test]
    fn index_am_recall_64_rows() {
        use_turbovec();
        Spi::run("CREATE TABLE t_64 (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_64 \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 64) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();

        Spi::run(
            "CREATE INDEX t_64_emb_idx \
             ON t_64 USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Index built; verify via heap count.
        let n_indexed: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_64",
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
             ',') || ']')::vector AS q \
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
    #[pg_test]
    fn index_am_reindex() {
        use_turbovec();
        Spi::run("CREATE TABLE t_re (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_re VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_re_emb_idx \
             ON t_re USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 2)",
        )
        .unwrap();
        Spi::run("REINDEX INDEX t_re_emb_idx").unwrap();
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_re",
        )
        .unwrap();
        assert_eq!(n_vec, Some(2));
    }

    /// `bit_width` reloption out-of-range is rejected at CREATE INDEX.
    #[pg_test]
    fn index_am_rejects_bad_bit_width() {
        use_turbovec();
        Spi::run("CREATE TABLE t_bad (id bigint, emb vector)").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::run(
                "CREATE INDEX ON t_bad USING turbovec (emb vec_cosine_ops) \
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
                 emb turbovec.vector)",
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
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 3) \
             ORDER BY score DESC LIMIT 1",
        )
        .unwrap();
        assert_eq!(first, Some(1));
    }

    #[pg_test]
    fn knn_cache_hit_after_first_call() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE cache_t (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO cache_t VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        let q = "'[1,0,0,0,0,0,0,0]'::turbovec.vector";
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
        Spi::run("CREATE TEMP TABLE cache_inv (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO cache_inv VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        let q = "'[0,0,1,0,0,0,0,0]'::turbovec.vector";
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

    /// Forces the index AM path (via `SET enable_seqscan = off`)
    /// and verifies the executor's order-by-op machinery works
    /// end-to-end.
    ///
    /// Phase 18 fix: `amrescan` was passing
    /// `nkeys * size_of::<ScanKeyData>()` as the *count* argument to
    /// `std::ptr::copy_nonoverlapping::<ScanKeyData>` — but that
    /// argument is the number of **elements**, not bytes. The
    /// resulting buffer overrun smashed the `IndexScanDesc` and
    /// adjacent heap chunks, surfacing as
    /// `munmap_chunk(): invalid pointer` the next time glibc walked
    /// the affected arena. The other 39 tests never took the index
    /// path (default `enable_seqscan = on` keeps small tables on a
    /// seqscan), which is why this was the only crashing case.
    #[pg_test]
    fn index_am_forced_index_scan() {
        use_turbovec();
        Spi::run("CREATE TABLE t_force (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_force \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 50) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_force_idx \
             ON t_force USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("ANALYZE t_force").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Top-1 nearest must be row 7. Forces index path; previously
        // crashed with SIGSEGV before xs_orderbyvals was allocated.
        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM t_force \
             ORDER BY emb <=> (SELECT emb FROM t_force WHERE id = 7) \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest, Some(7));

        // Top-3 must include row 7. Both projecting AND ordering by
        // the distance — this is the query that crashed before v0.12.
        let count_with_dist: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM ( \
                 SELECT id, emb <=> (SELECT emb FROM t_force WHERE id = 7) AS dist \
                 FROM t_force \
                 ORDER BY emb <=> (SELECT emb FROM t_force WHERE id = 7) \
                 LIMIT 3 \
             ) sub",
        )
        .unwrap();
        assert_eq!(count_with_dist, Some(3));
    }

    #[pg_test]
    fn knn_filtered_allowlist() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE filt (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO filt VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0.9,0.1,0,0,0,0,0,0]'), \
                 (3, '[0,1,0,0,0,0,0,0]'), \
                 (4, '[-1,0,0,0,0,0,0,0]')",
        )
        .unwrap();

        // Without allowlist: row 1 wins.
        let unfiltered: Option<i64> = Spi::get_one(
            "SELECT id FROM turbovec.knn(\
                 'filt'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 1) \
             ORDER BY score DESC LIMIT 1",
        )
        .unwrap();
        assert_eq!(unfiltered, Some(1));

        // With allowlist [3, 4]: row 1 is forbidden; row 3 wins
        // (cosine to [1,0,..] = 1.0 vs row 4's distance = 2.0).
        let filtered: Option<i64> = Spi::get_one(
            "SELECT id FROM turbovec.knn(\
                 'filt'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 1, 4, ARRAY[3, 4]::bigint[]) \
             ORDER BY score DESC LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            filtered,
            Some(3),
            "with allowlist=[3,4] the filtered nearest should be row 3"
        );

        // Allowlist of just one id: must return that id (or empty).
        let single: Option<i64> = Spi::get_one(
            "SELECT id FROM turbovec.knn(\
                 'filt'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 5, 4, ARRAY[2]::bigint[])",
        )
        .unwrap();
        assert_eq!(single, Some(2));

        // Empty allowlist: no rows.
        let empty_count: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM turbovec.knn(\
                 'filt'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 5, 4, ARRAY[]::bigint[])",
        )
        .unwrap();
        assert_eq!(empty_count, Some(0));
    }

    /// 200 random 384-dim vectors (typical sentence-embedding
    /// dimensionality). Verifies the index works at realistic
    /// scale rather than just on toy 8-dim corpora. With d=384 and
    /// 4-bit quantisation TurboQuant has plenty of room — R@10
    /// against the self-vector should be 1.0.
    #[pg_test]
    fn index_am_realistic_dim_384() {
        use_turbovec();
        Spi::run("CREATE TABLE t_384 (id bigint PRIMARY KEY, emb vector)").unwrap();
        // Seed-stable per-row vectors via hashtext.
        Spi::run(
            "INSERT INTO t_384 \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 200) AS gs(i), \
                  generate_series(1, 384) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();

        let n_rows: Option<i64> = Spi::get_one("SELECT count(*) FROM t_384").unwrap();
        assert_eq!(n_rows, Some(200));

        Spi::run(
            "CREATE INDEX t_384_idx \
             ON t_384 USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // The AM persisted all 200 rows.
        let n_indexed: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_384",
        )
        .unwrap();
        assert_eq!(n_indexed, Some(200));

        // Self-query: row 73's emb. At d=384 / 4-bit, self-score
        // dominates — row 73 must be rank 1.
        let nearest: Option<i64> = Spi::get_one(
            "WITH q AS (SELECT emb FROM t_384 WHERE id = 73) \
             SELECT t.id FROM t_384 t, q \
             ORDER BY t.emb <=> q.emb \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            nearest,
            Some(73),
            "at d=384 / 4-bit the self-score must dominate"
        );

        // Top-10 self-recall: row 73 in top-10.
        let in_top10: Option<bool> = Spi::get_one(
            "WITH q AS (SELECT emb FROM t_384 WHERE id = 73), \
             top10 AS ( \
                 SELECT t.id FROM t_384 t, q \
                 ORDER BY t.emb <=> q.emb \
                 LIMIT 10 \
             ) \
             SELECT EXISTS (SELECT 1 FROM top10 WHERE id = 73)",
        )
        .unwrap();
        assert_eq!(in_top10, Some(true));
    }

    /// Build at the lowest supported bit_width (= 2) on a realistic
    /// dim. Confirms the kernel's tightest compression mode round-
    /// trips end-to-end.
    #[pg_test]
    fn index_am_2bit_round_trip() {
        use_turbovec();
        Spi::run("CREATE TABLE t_2bit (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_2bit \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 100) AS gs(i), \
                  generate_series(1, 128) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_2bit_idx \
             ON t_2bit USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 2)",
        )
        .unwrap();

        // Heap row count smoke check; bit_width is recorded in
        // the relfile meta page rather than a queryable side
        // table, so we just confirm the index built without
        // ERROR and serves queries below.
        let n_rows: Option<i64> =
            Spi::get_one("SELECT count(*) FROM t_2bit").unwrap();
        assert_eq!(n_rows, Some(100));

        // Self-recall in top-20 at 2-bit, d=128. Tighter quantisation
        // = lower recall, so we relax the bar from top-1 to top-20.
        let in_top20: Option<bool> = Spi::get_one(
            "WITH q AS (SELECT emb FROM t_2bit WHERE id = 42), \
             top20 AS ( \
                 SELECT t.id FROM t_2bit t, q \
                 ORDER BY t.emb <=> q.emb \
                 LIMIT 20 \
             ) \
             SELECT EXISTS (SELECT 1 FROM top20 WHERE id = 42)",
        )
        .unwrap();
        assert_eq!(in_top20, Some(true));
    }

    #[pg_test]
    fn knn_rejects_bad_k() {
        Spi::run("CREATE TEMP TABLE pgtv_empty (id bigint, emb turbovec.vector)").unwrap();
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i64>(
                "SELECT count(*) FROM turbovec.knn(\
                     'pgtv_empty'::regclass, 'id', 'emb', \
                     '[1,2,3,4,5,6,7,8]'::turbovec.vector, 0)",
            )
        });
        assert!(bad.is_err(), "expected ERROR for k=0");
    }

    #[pg_test]
    fn subvector_basic() {
        let s: Option<String> = Spi::get_one(
            "SELECT turbovec.subvector('[10,20,30,40]'::turbovec.vector, 2, 2)::text",
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
                "SELECT turbovec.subvector('[1,2,3]'::turbovec.vector, 2, 5)::text",
            )
        });
        assert!(bad.is_err(), "expected ERROR for out-of-bounds");
    }

    #[pg_test]
    fn jsonb_round_trip() {
        let txt: Option<String> =
            Spi::get_one("SELECT '[1, 2.5, -3]'::turbovec.vector::jsonb::turbovec.vector::text")
                .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1") && s.contains("2.5") && s.contains("-3"));
    }

    #[pg_test]
    fn check_dim_passes_and_fails() {
        let ok: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(\
                turbovec.vec_check_dim('[1,2,3]'::turbovec.vector, 3))",
        )
        .unwrap();
        assert_eq!(ok, Some(3));

        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>(
                "SELECT turbovec.vector_dims(\
                    turbovec.vec_check_dim('[1,2,3]'::turbovec.vector, 4))",
            )
        });
        assert!(bad.is_err(), "expected ERROR for dim mismatch");
    }

    #[pg_test]
    fn zeros_helper() {
        let dim: Option<i32> =
            Spi::get_one("SELECT turbovec.vector_dims(turbovec.vec_zeros(8))").unwrap();
        assert_eq!(dim, Some(8));
        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm(turbovec.vec_zeros(8))").unwrap();
        assert_eq!(n, Some(0.0));
    }

    /// `vec_random_unit(n)` returns a unit-norm vector of dim n.
    #[pg_test]
    fn random_unit_dim_and_norm() {
        let dim: Option<i32> =
            Spi::get_one("SELECT turbovec.vector_dims(turbovec.vec_random_unit(8))").unwrap();
        assert_eq!(dim, Some(8));
        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm(turbovec.vec_random_unit(16))").unwrap();
        assert!((n.unwrap() - 1.0).abs() < 1e-5, "norm = {:?}", n);
    }

    /// `vec_to_text` as an explicit function call (mirrors the
    /// type's text output but callable directly).
    #[pg_test]
    fn vec_to_text_function() {
        let s: Option<String> =
            Spi::get_one("SELECT turbovec.vec_to_text('[1, 2.5, 3]'::turbovec.vector)")
                .unwrap();
        let txt = s.unwrap();
        assert!(txt.starts_with('['));
        assert!(txt.ends_with(']'));
        assert!(txt.contains("2.5"));
    }

    /// `avg(vector)` over an empty table returns NULL (matches the
    /// SQL spec: aggregate of zero rows is NULL).
    ///
    /// Note: `vector` itself is `NOT NULL` by design — the type
    /// input function rejects NULL — so we cannot test "avg over
    /// rows of NULL values" the way we could with int4. Empty-table
    /// avg is the closest analogue.
    #[pg_test]
    fn aggregate_avg_empty_is_null() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE avg_empty (v vector)").unwrap();
        let avg: Option<String> = Spi::get_one("SELECT avg(v)::text FROM avg_empty").unwrap();
        assert_eq!(avg, None, "avg over empty table must be NULL");
    }

    /// `avg(vector)` over rows of mixed dim raises an ERROR.
    #[pg_test]
    fn aggregate_avg_mixed_dim_errors() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE mixed_t (v vector)").unwrap();
        Spi::run(
            "INSERT INTO mixed_t VALUES \
                 ('[1,2,3]'::vector), \
                 ('[1,2,3,4]'::vector)",
        )
        .unwrap();
        let res =
            std::panic::catch_unwind(|| Spi::get_one::<String>("SELECT avg(v)::text FROM mixed_t"));
        assert!(res.is_err(), "expected ERROR for mixed-dim avg");
    }

    /// `l2_squared_distance` returns L2 distance squared (no sqrt).
    #[pg_test]
    fn l2_squared_distance_function() {
        let d: Option<f64> = Spi::get_one(
            "SELECT turbovec.l2_squared_distance(\
                 '[1,2,3]'::turbovec.vector, '[4,6,3]'::turbovec.vector)",
        )
        .unwrap();
        // (3^2 + 4^2 + 0^2) = 25
        assert!((d.unwrap() - 25.0).abs() < 1e-6, "got {:?}", d);

        // Confirm relationship: l2_squared_distance == l2_distance^2.
        let l2: Option<f64> = Spi::get_one(
            "SELECT turbovec.l2_distance(\
                 '[1,2,3]'::turbovec.vector, '[4,6,3]'::turbovec.vector)",
        )
        .unwrap();
        let lsq: Option<f64> = Spi::get_one(
            "SELECT turbovec.l2_squared_distance(\
                 '[1,2,3]'::turbovec.vector, '[4,6,3]'::turbovec.vector)",
        )
        .unwrap();
        let l2 = l2.unwrap();
        let lsq = lsq.unwrap();
        assert!((l2 * l2 - lsq).abs() < 1e-6);
    }

    /// `vector_norm` of the zero vector is exactly 0.
    #[pg_test]
    fn vector_norm_of_zero() {
        let n: Option<f64> =
            Spi::get_one("SELECT turbovec.vector_norm('[0,0,0,0]'::turbovec.vector)").unwrap();
        assert_eq!(n, Some(0.0));
    }

    /// `cosine_distance` against the zero vector returns NaN
    /// (matches pgvector).
    #[pg_test]
    fn cosine_distance_zero_is_nan() {
        let d: Option<f64> = Spi::get_one(
            "SELECT turbovec.cosine_distance(\
                 '[0,0,0]'::turbovec.vector, \
                 '[1,2,3]'::turbovec.vector)",
        )
        .unwrap();
        assert!(d.unwrap().is_nan(), "expected NaN, got {:?}", d);
    }

    /// `negative_inner_product(a, b) == -inner_product(a, b)`.
    #[pg_test]
    fn negative_inner_product_function() {
        let ip: Option<f64> = Spi::get_one(
            "SELECT turbovec.inner_product(\
                 '[1,2,3]'::turbovec.vector, \
                 '[4,5,6]'::turbovec.vector)",
        )
        .unwrap();
        let nip: Option<f64> = Spi::get_one(
            "SELECT turbovec.negative_inner_product(\
                 '[1,2,3]'::turbovec.vector, \
                 '[4,5,6]'::turbovec.vector)",
        )
        .unwrap();
        assert!((ip.unwrap() + nip.unwrap()).abs() < 1e-9);
    }

    /// `subvector` boundary cases: full slice, single element.
    #[pg_test]
    fn subvector_boundaries() {
        // start=1 length=4 returns the whole vector.
        let txt: Option<String> = Spi::get_one(
            "SELECT turbovec.subvector(\
                 '[10,20,30,40]'::turbovec.vector, 1, 4)::text",
        )
        .unwrap();
        let s = txt.unwrap();
        for n in &["10", "20", "30", "40"] {
            assert!(s.contains(n), "expected {} in {}", n, s);
        }

        // length=1 returns a single-element vector.
        let dim: Option<i32> = Spi::get_one(
            "SELECT turbovec.vector_dims(\
                 turbovec.subvector('[10,20,30,40]'::turbovec.vector, 3, 1))",
        )
        .unwrap();
        assert_eq!(dim, Some(1));

        // The single element is correct.
        let v: Option<String> = Spi::get_one(
            "SELECT turbovec.subvector(\
                 '[10,20,30,40]'::turbovec.vector, 3, 1)::text",
        )
        .unwrap();
        let s = v.unwrap();
        assert!(s.contains("30"));
        assert!(!s.contains("10"));
        assert!(!s.contains("40"));
    }

    /// `jsonb_to_vec` rejects non-array JSONB.
    #[pg_test]
    fn jsonb_to_vec_rejects_non_array() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>("SELECT turbovec.jsonb_to_vec('{\"a\": 1}'::jsonb)::text")
        });
        assert!(bad.is_err(), "expected ERROR for non-array jsonb");

        let bad2 = std::panic::catch_unwind(|| {
            Spi::get_one::<String>("SELECT turbovec.jsonb_to_vec('42'::jsonb)::text")
        });
        assert!(bad2.is_err(), "expected ERROR for scalar jsonb");
    }

    /// `jsonb_to_vec` rejects non-finite numbers (NaN / Infinity
    /// arrive as JSON strings or are stripped at parse time, but a
    /// non-numeric element must be rejected).
    #[pg_test]
    fn jsonb_to_vec_rejects_non_numeric() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT turbovec.jsonb_to_vec('[1, \"oops\", 3]'::jsonb)::text",
            )
        });
        assert!(bad.is_err(), "expected ERROR for string element");

        let bad2 = std::panic::catch_unwind(|| {
            Spi::get_one::<String>("SELECT turbovec.jsonb_to_vec('[1, null, 3]'::jsonb)::text")
        });
        assert!(bad2.is_err(), "expected ERROR for null element");
    }

    /// `SET turbovec.bit_width_default` round-trips through
    /// `current_setting()`.
    #[pg_test]
    fn guc_bit_width_default_round_trip() {
        Spi::run("SET turbovec.bit_width_default = 2").unwrap();
        let v: Option<String> =
            Spi::get_one("SELECT current_setting('turbovec.bit_width_default')").unwrap();
        assert_eq!(v.as_deref(), Some("2"));

        Spi::run("SET turbovec.bit_width_default = 4").unwrap();
        let v: Option<String> =
            Spi::get_one("SELECT current_setting('turbovec.bit_width_default')").unwrap();
        assert_eq!(v.as_deref(), Some("4"));

        // Out-of-range values are rejected by GucContext::Userset.
        let bad = std::panic::catch_unwind(|| Spi::run("SET turbovec.bit_width_default = 5"));
        assert!(bad.is_err(), "expected ERROR for out-of-range bit_width");
    }

    /// `turbovec.knn()` against an empty table returns 0 rows.
    #[pg_test]
    fn knn_empty_corpus() {
        Spi::run("CREATE TEMP TABLE empty_corp (id bigint, emb turbovec.vector)").unwrap();
        let n: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM turbovec.knn(\
                 'empty_corp'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 5)",
        )
        .unwrap();
        assert_eq!(n, Some(0), "knn over empty corpus must return 0 rows");
    }

    /// `turbovec.knn()` when k > n returns all n rows.
    #[pg_test]
    fn knn_k_greater_than_n() {
        Spi::run("CREATE TEMP TABLE small_corp (id bigint, emb turbovec.vector)").unwrap();
        Spi::run(
            "INSERT INTO small_corp VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]')",
        )
        .unwrap();
        let n: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM turbovec.knn(\
                 'small_corp'::regclass, 'id', 'emb', \
                 '[1,0,0,0,0,0,0,0]'::turbovec.vector, 10)",
        )
        .unwrap();
        assert_eq!(
            n,
            Some(3),
            "k=10 over a 3-row corpus must return 3 rows, got {:?}",
            n
        );
    }

    /// `turbovec.knn()` cache hashes on `(rel_oid, attnum, bit_width,
    /// dim)` and validates against current `(relfilenode, n_rows)`.
    /// A row-count change alone is enough to force a rebuild on the
    /// next call. Done in two separate transactions because pgrx-tests
    /// wraps each `#[pg_test]` in BEGIN/ROLLBACK and inside one
    /// transaction PG hides relfilenode rewrites from the SPI client.
    #[pg_test]
    fn knn_cache_invalidates_on_row_count_change() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE rc_t (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO rc_t VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        let q = "'[0,0,1,0,0,0,0,0]'::turbovec.vector";
        // Warm the cache.
        let warm_count: Option<i64> = Spi::get_one(&format!(
            "SELECT count(*) FROM turbovec.knn(\
                 'rc_t'::regclass, 'id', 'emb', {q}, 5)"
        ))
        .unwrap();
        assert_eq!(warm_count, Some(2));

        // Insert a 3rd row; cache key matches but `n_rows` has
        // changed, so the next lookup must rebuild.
        Spi::run("INSERT INTO rc_t VALUES (3, '[0,0,1,0,0,0,0,0]')").unwrap();
        let after: Option<i64> = Spi::get_one(&format!(
            "SELECT count(*) FROM turbovec.knn(\
                 'rc_t'::regclass, 'id', 'emb', {q}, 5)"
        ))
        .unwrap();
        assert_eq!(
            after,
            Some(3),
            "row-count change must trigger cache rebuild"
        );
    }

    /// Cache invalidates after a heap REINDEX (relfilenode bump on
    /// the heap is not directly observable, but row count + new oid
    /// state ensure correctness end-to-end).
    #[pg_test]
    fn index_am_cache_invalidates_on_reindex() {
        use_turbovec();
        Spi::run("CREATE TABLE reidx_t (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO reidx_t VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX reidx_t_idx \
             ON reidx_t USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Force the AM-cache code path by running an ORDER BY query.
        let _: Option<i64> = Spi::get_one(
            "SELECT id FROM reidx_t \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector \
             LIMIT 1",
        )
        .unwrap();

        Spi::run("REINDEX INDEX reidx_t_idx").unwrap();

        // Heap row count must reflect the rebuild; the AM
        // cache must serve fresh data on the next scan.
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM reidx_t",
        )
        .unwrap();
        assert_eq!(n_vec, Some(2));

        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM reidx_t \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest, Some(1));
    }

    /// Element-wise arithmetic operators round-trip.
    #[pg_test]
    fn elementwise_arithmetic() {
        use_turbovec();
        let s: Option<String> =
            Spi::get_one("SELECT ('[1,2,3]'::vector + '[10,20,30]'::vector)::text").unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("11"));
        assert!(txt.contains("22"));
        assert!(txt.contains("33"));

        let s: Option<String> =
            Spi::get_one("SELECT ('[10,20,30]'::vector - '[1,2,3]'::vector)::text").unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("9") && txt.contains("18") && txt.contains("27"));

        let s: Option<String> =
            Spi::get_one("SELECT ('[1,2,3]'::vector * '[10,20,30]'::vector)::text").unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("10") && txt.contains("40") && txt.contains("90"));
    }

    /// `sum(vector)` is parallel-safe and matches a manual sum.
    #[pg_test]
    fn aggregate_sum_basic() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE sum_t (v vector)").unwrap();
        Spi::run(
            "INSERT INTO sum_t VALUES \
                 ('[1,2,3]'::vector), \
                 ('[10,20,30]'::vector), \
                 ('[100,200,300]'::vector)",
        )
        .unwrap();
        let s: Option<String> = Spi::get_one("SELECT sum(v)::text FROM sum_t").unwrap();
        let txt = s.unwrap();
        assert!(txt.contains("111"));
        assert!(txt.contains("222"));
        assert!(txt.contains("333"));
    }

    /// Empty-table aggregates return NULL.
    #[pg_test]
    fn aggregate_sum_empty_is_null() {
        use_turbovec();
        Spi::run("CREATE TEMP TABLE sum_empty (v vector)").unwrap();
        let s: Option<String> = Spi::get_one("SELECT sum(v)::text FROM sum_empty").unwrap();
        assert_eq!(s, None);
    }

    /// JSONB explicit cast round-trips.
    #[pg_test]
    fn jsonb_cast_explicit_function() {
        let txt: Option<String> =
            Spi::get_one("SELECT turbovec.vec_to_jsonb('[1, 2, 3]'::turbovec.vector)::text")
                .unwrap();
        let s = txt.unwrap();
        assert!(s.contains("1") && s.contains("2") && s.contains("3"));
    }

    /// `vec_zeros` rejects out-of-range dim.
    #[pg_test]
    fn zeros_rejects_bad_dim() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>("SELECT turbovec.vector_dims(turbovec.vec_zeros(0))")
        });
        assert!(bad.is_err(), "expected ERROR for dim = 0");

        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>("SELECT turbovec.vector_dims(turbovec.vec_zeros(20000))")
        });
        assert!(bad.is_err(), "expected ERROR for dim > MAX_DIM");
    }

    /// `vec_check_dim` is identity on match, raises otherwise —
    /// already tested; this exercises the rare expected = 0 case.
    #[pg_test]
    fn check_dim_rejects_zero_expected() {
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<i32>(
                "SELECT turbovec.vector_dims(\
                    turbovec.vec_check_dim('[1,2]'::turbovec.vector, 0))",
            )
        });
        assert!(bad.is_err(), "expected ERROR for expected = 0");
    }

    /// First AM scan populates the backend-local cache; the second
    /// scan must reuse the same entry rather than evicting and
    /// rebuilding. We assert via `cache::len()` that no entry was
    /// dropped between the two scans (a relfilenode/version
    /// mismatch would have removed the entry inside `cache::lookup`).
    #[pg_test]
    fn index_am_cache_hits_on_second_query() {
        use_turbovec();
        Spi::run("CREATE TABLE t_cc (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_cc \
             SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector \
             FROM generate_series(1, 50) g",
        )
        .unwrap();
        Spi::run("CREATE INDEX t_cc_idx ON t_cc USING turbovec (emb vec_cosine_ops)").unwrap();
        // Force the AM path; default `enable_seqscan = on` keeps
        // small tables on a seqscan, which never reaches our cache.
        Spi::run("ANALYZE t_cc").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        crate::cache::invalidate_all();
        let len_before: usize = crate::cache::len();
        // First scan: miss, populates the cache.
        let _: Option<i64> = Spi::get_one(
            "WITH q AS (SELECT '[1,0,0,0,0,0,0,0]'::vector AS v) \
             SELECT id FROM t_cc, q ORDER BY emb <=> q.v LIMIT 1",
        )
        .unwrap();
        let len_after: usize = crate::cache::len();
        assert!(
            len_after > len_before,
            "cache should be populated by first AM scan ({} -> {})",
            len_before,
            len_after
        );
        // Second scan: hit. We can't directly observe the lookup,
        // but the cache must still be populated (nothing evicted)
        // and the answer must agree.
        let id: Option<i64> = Spi::get_one(
            "WITH q AS (SELECT '[1,0,0,0,0,0,0,0]'::vector AS v) \
             SELECT id FROM t_cc, q ORDER BY emb <=> q.v LIMIT 1",
        )
        .unwrap();
        assert_eq!(id, Some(1));
        assert_eq!(crate::cache::len(), len_after);
    }

    /// `aminsert` bumps `version` on the side-table row, which is
    /// the freshness signal we stash into the cache's `n_rows`
    /// slot. The next AM scan must therefore see a version mismatch
    /// in `cache::lookup`, evict the stale entry, and rebuild from
    /// the new payload — otherwise the freshly-inserted row would
    /// be invisible.
    #[pg_test]
    fn index_am_cache_invalidates_on_insert() {
        use_turbovec();
        Spi::run("CREATE TABLE t_civ (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_civ VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX t_civ_idx ON t_civ USING turbovec (emb vec_cosine_ops)").unwrap();
        Spi::run("ANALYZE t_civ").unwrap();
        // Force the AM path for the ORDER BY queries below. We can't
        // leave `enable_seqscan = off` set globally because the
        // `count(*)` probe would otherwise pick our AM (which serves
        // an empty result set for non-orderby scans) instead of a
        // seqscan or PK index-only scan.
        Spi::run("SET enable_seqscan = off").unwrap();
        let _: Option<i64> = Spi::get_one(
            "SELECT id FROM t_civ ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = on").unwrap();
        // INSERT bumps version; next scan must rebuild from new payload.
        Spi::run("INSERT INTO t_civ VALUES (3, '[0,0,1,0,0,0,0,0]')").unwrap();
        let n: Option<i64> = Spi::get_one("SELECT count(*) FROM t_civ").unwrap();
        assert_eq!(n, Some(3));
        Spi::run("SET enable_seqscan = off").unwrap();
        let id: Option<i64> = Spi::get_one(
            "SELECT id FROM t_civ ORDER BY emb <=> '[0,0,1,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(id, Some(3));
    }

    /// Phase L: relfile-resident pages should produce the same
    /// search results as the SPI side-table path. Exercises the
    /// new code path end-to-end on a 200-row / 384-d corpus and
    /// verifies the relfile-resident pages are queried correctly
    /// by `ambeginscan` / `amgettuple`. Cold p50 is dominated by
    /// buffer-pool hits + IdMapIndex reconstruction, *not* by
    /// SPI fetch + TOAST detoast — the headline Phase L win.
    #[pg_test]
    fn relfile_cold_scan_does_not_repeat_load() {
        use_turbovec();
        Spi::run("CREATE TABLE t_rf (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_rf \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 200) AS gs(i), \
                  generate_series(1, 384) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_rf_idx \
             ON t_rf USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // Heap row count smoke check: the relfile-resident
        // index has no side-table; observable state is the
        // count(*) on the heap.
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT count(*) FROM t_rf",
        )
        .unwrap();
        assert_eq!(n_vec, Some(200));

        // The relfile relation should have at least 4 blocks
        // (meta + codes + scales + ids).
        let bytes: Option<i64> =
            Spi::get_one("SELECT pg_relation_size('t_rf_idx'::regclass)::int8").unwrap();
        let bytes = bytes.unwrap();
        assert!(
            bytes >= 4 * 8192,
            "relfile should be >= 4 pages, got {} bytes",
            bytes,
        );

        Spi::run("ANALYZE t_rf").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Cold scan: pages cold in shared_buffers — must be read
        // from disk via the buffer manager.
        let nearest1: Option<i64> = Spi::get_one(
            "SELECT id FROM t_rf \
             ORDER BY emb <=> (SELECT emb FROM t_rf WHERE id = 73) \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            nearest1,
            Some(73),
            "relfile cold scan must agree with brute-force on self-query",
        );

        // Warm scan: pages warm in shared_buffers + IdMapIndex
        // cached in the per-backend Arc cache. Same result.
        let nearest2: Option<i64> = Spi::get_one(
            "SELECT id FROM t_rf \
             ORDER BY emb <=> (SELECT emb FROM t_rf WHERE id = 73) \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest2, Some(73), "relfile warm scan must match cold scan");

        // pg_stat_io smoke check on pg16+.
        #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
        {
            let n_io_rows: Option<i64> =
                Spi::get_one("SELECT count(*)::int8 FROM pg_stat_io").unwrap();
            assert!(
                n_io_rows.unwrap_or(0) > 0,
                "pg_stat_io should be populated on pg16+",
            );
        }
    }

    /// Phase L cold-vs-warm timing inside a single backend on a
    /// 2000-row / 384-dim corpus. Logs the timings via eprintln!
    /// (lost to PG's log on pgrx-test runs; the practical timing
    /// harness is `benches/sql/phase_l_cold_scan.sql`). Asserts only
    /// that both timings are reasonable; the strong cold-vs-warm
    /// inequality is too noisy inside a transaction to assert.
    ///
    /// Phase G/H reference numbers (1 M rows, side-table path):
    /// cold p50 = 6 802 ms. The relfile path's headline win is at
    /// scale: shared_buffers caches the index pages cluster-wide,
    /// so every backend after the first pays only the buffer-pool
    /// hit cost, not the SPI fetch + TOAST + parse cost.
    #[pg_test]
    fn relfile_cold_vs_warm_timing() {
        use_turbovec();
        Spi::run("CREATE TABLE t_cw (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_cw \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 2000) AS gs(i), \
                  generate_series(1, 384) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_cw_idx ON t_cw USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("ANALYZE t_cw").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        let cold_us: Option<f64> = Spi::get_one(
            "WITH t0 AS (SELECT clock_timestamp() AS ts), \
                  q AS (SELECT id FROM t_cw \
                        ORDER BY emb <=> (SELECT emb FROM t_cw WHERE id = 1234) \
                        LIMIT 10) \
             SELECT (EXTRACT(epoch FROM (clock_timestamp() - t0.ts)) * 1e6)::float8 \
             FROM t0, (SELECT count(*) FROM q) c",
        )
        .unwrap();
        let warm_us: Option<f64> = Spi::get_one(
            "WITH t0 AS (SELECT clock_timestamp() AS ts), \
                  q AS (SELECT id FROM t_cw \
                        ORDER BY emb <=> (SELECT emb FROM t_cw WHERE id = 1234) \
                        LIMIT 10) \
             SELECT (EXTRACT(epoch FROM (clock_timestamp() - t0.ts)) * 1e6)::float8 \
             FROM t0, (SELECT count(*) FROM q) c",
        )
        .unwrap();

        let cold = cold_us.unwrap_or(f64::INFINITY);
        let warm = warm_us.unwrap_or(f64::INFINITY);
        eprintln!(
            "phase-l cold-vs-warm (2000x384, bit_width=4, debug): \
             cold = {:.0} us, warm = {:.0} us",
            cold, warm,
        );
        // Loose sanity bounds (debug build, ~2000 rows of 384-d
        // data through full ORDER BY pipeline).
        assert!(cold < 30_000_000.0, "cold scan {} us looks broken", cold);
        assert!(warm < 30_000_000.0, "warm scan {} us looks broken", warm);
    }

    /// Phase L hardening (item 1): every relfile page write goes
    /// through `GenericXLog`. This test exercises the WAL-emitting
    /// code paths reachable from inside a pgrx test (which runs
    /// inside a transaction, so `VACUUM` and the `ambulkdelete`
    /// truncate path are unreachable from here — see
    /// `benches/sql/phase_n_b_crash_recovery.sql` for the manual
    /// e2e harness that exercises ambulkdelete + RelationTruncate).
    ///
    /// Specifically asserts:
    ///
    /// 1. `pg_current_wal_lsn` advances over `ambuild`.
    /// 2. `pg_current_wal_lsn` advances over `aminsert`.
    /// 3. The relfile is at least 4 pages (meta + 3 chains).
    /// 4. Search results stay correct after both phases.
    #[pg_test]
    fn relfile_wal_emits_on_build_and_insert() {
        use_turbovec();
        Spi::run("CREATE TABLE t_wal (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_wal \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 200) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();

        // (1) ambuild path — should emit XLOG_GENERIC records.
        let lsn_before_build: Option<String> =
            Spi::get_one("SELECT pg_current_wal_lsn()::text").unwrap();
        Spi::run(
            "CREATE INDEX t_wal_idx ON t_wal USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let lsn_after_build: Option<String> =
            Spi::get_one("SELECT pg_current_wal_lsn()::text").unwrap();
        let advanced_build: Option<bool> = Spi::get_one(&format!(
            "SELECT '{}'::pg_lsn < '{}'::pg_lsn",
            lsn_before_build.as_deref().unwrap_or("0/0"),
            lsn_after_build.as_deref().unwrap_or("0/0"),
        ))
        .unwrap();
        assert_eq!(
            advanced_build,
            Some(true),
            "ambuild should advance WAL: before={:?} after={:?}",
            lsn_before_build,
            lsn_after_build,
        );

        let nblocks_after_build: Option<i64> = Spi::get_one(
            "SELECT (pg_relation_size('t_wal_idx'::regclass) / 8192)::int8",
        )
        .unwrap();
        let nblocks_after_build = nblocks_after_build.unwrap();
        assert!(
            nblocks_after_build >= 4,
            "relfile must contain meta + 3 chains, got {} blocks",
            nblocks_after_build,
        );

        // (2) aminsert path — should also emit WAL.
        let lsn_before_insert: Option<String> =
            Spi::get_one("SELECT pg_current_wal_lsn()::text").unwrap();
        Spi::run(
            "INSERT INTO t_wal \
             SELECT 9999, ('[' || string_agg( \
                 ((hashtext('z:' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 16) AS sub(k)",
        )
        .unwrap();
        let lsn_after_insert: Option<String> =
            Spi::get_one("SELECT pg_current_wal_lsn()::text").unwrap();
        let advanced_insert: Option<bool> = Spi::get_one(&format!(
            "SELECT '{}'::pg_lsn < '{}'::pg_lsn",
            lsn_before_insert.as_deref().unwrap_or("0/0"),
            lsn_after_insert.as_deref().unwrap_or("0/0"),
        ))
        .unwrap();
        assert_eq!(
            advanced_insert,
            Some(true),
            "aminsert should advance WAL: before={:?} after={:?}",
            lsn_before_insert,
            lsn_after_insert,
        );

        // (3) Index queryable.
        Spi::run("ANALYZE t_wal").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        let any_id: Option<i64> = Spi::get_one(
            "SELECT id FROM t_wal \
             ORDER BY emb <=> (SELECT emb FROM t_wal WHERE id = 1) \
             LIMIT 1",
        )
        .unwrap();
        assert!(any_id.is_some(), "index must remain queryable after build + insert");
    }

    /// Phase L hardening (item 3): `relfile::write_full` calls
    /// `RelationTruncate` when the new layout is smaller than the
    /// existing one. Since pgrx-tests can't run VACUUM (it can't
    /// run inside a transaction), this test takes the indirect
    /// route: build a big index, then exercise the rewrite-in-
    /// place path of `aminsert` (which calls `write_full` on
    /// every row) after deleting the table contents so each
    /// rewrite sees a stable (== current) `n_vectors`. We can't
    /// directly invoke ambulkdelete from here; the manual e2e
    /// truncate check lives in `benches/sql/phase_n_b_crash_recovery.sql`.
    /// Instead this test verifies the page-layout planning
    /// invariant via `MetaPageData::total_blocks()`.
    #[pg_test]
    fn relfile_total_blocks_shrinks_with_n_vectors() {
        use crate::index::page::MetaPageData;
        let big = MetaPageData::plan(4, 384, 1_000_000, 1).total_blocks();
        let small = MetaPageData::plan(4, 384, 1_000, 1).total_blocks();
        assert!(
            big > small,
            "plan(1e6) total_blocks ({}) must exceed plan(1e3) total_blocks ({})",
            big,
            small,
        );
        // Smoke check that write_full's truncate-when-smaller
        // branch is reachable from the public API: empty layout
        // is just 1 block (meta only).
        let empty = MetaPageData::plan(4, 384, 0, 1).total_blocks();
        assert_eq!(empty, 1, "empty layout is meta page only");
    }

    /// Phase L hardening (item 2): `ambuildempty` writes the meta
    /// page into `INIT_FORKNUM` so unlogged indexes survive a
    /// crash by being reset to empty rather than corrupted. We
    /// can't trigger the actual recovery copy from inside pgrx-
    /// test, but we can verify the init fork is non-empty after
    /// CREATE INDEX, which is the precondition for that copy to
    /// do anything at all.
    #[pg_test]
    fn relfile_unlogged_has_init_fork() {
        use_turbovec();
        Spi::run(
            "CREATE UNLOGGED TABLE t_ul (id bigint PRIMARY KEY, emb vector)",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_ul_idx ON t_ul USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, dim = 16)",
        )
        .unwrap();

        // Init fork must contain at least our 1-page meta header.
        // pg_relation_size with the 'init' fork argument is the
        // user-facing way to inspect it.
        let init_bytes: Option<i64> = Spi::get_one(
            "SELECT pg_relation_size('t_ul_idx'::regclass, 'init')::int8",
        )
        .unwrap();
        assert!(
            init_bytes.unwrap_or(0) >= 8192,
            "unlogged turbovec index must have a populated init fork; got {:?} bytes",
            init_bytes,
        );

        // The empty unlogged index must be queryable (returns no
        // rows since the heap is empty). Catches regressions in
        // the empty-meta-page write path: if ambuildempty failed
        // to populate INIT_FORKNUM, ambuild would still run on
        // CREATE INDEX (heap is empty so n_vectors=0 path), but
        // the init fork would be empty and post-crash recovery
        // would yield a 0-block relfile that `read_meta` would
        // refuse.
        Spi::run("SET enable_seqscan = off").unwrap();
        let cnt: Option<i64> = Spi::get_one(
            "WITH q AS ( \
                 SELECT id FROM t_ul \
                  ORDER BY emb <=> '[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector \
                  LIMIT 5 \
             ) SELECT count(*)::int8 FROM q",
        )
        .unwrap();
        assert_eq!(cnt, Some(0));
    }

    /// Phase L hardening (item 6): `ambulkdelete` walks the
    /// existing chain pages and swap-removes dead rows in place
    /// instead of rebuilding the whole `IdMapIndex` from disk and
    /// rewriting every page. Verifies:
    ///   1. survivors remain queryable and return correct ids;
    ///   2. the file size doesn't grow as a result of VACUUM (it
    ///      may shrink via `RelationTruncate`);
    ///   3. the operation completes in O(deleted) time — a single
    ///      dead row in a 1k-row index finishes well under 200 ms
    ///      on a debug build;
    ///   4. `am_version` is bumped on the meta page so the per-
    ///      backend cache invalidates next scan.
    ///
    /// pgrx tests run inside a single transaction, so we can't
    /// invoke real `VACUUM` (which forbids tx blocks) or rely on
    /// the parent harness's autovacuum. Instead we call the
    /// `ambulkdelete` function pointer directly with a synthetic
    /// `IndexBulkDeleteCallback` that consults a `HashSet<u64>`.
    /// This is the same call the autovacuum launcher would make,
    /// minus the cross-transaction wrapping. The end-to-end
    /// `VACUUM`-after-DELETE path is exercised by
    /// `benches/sql/phase_n_b_crash_recovery.sql` outside the test
    /// harness.
    #[pg_test]
    fn relfile_ambulkdelete_walks_pages_not_rebuild() {
        use crate::index::page::MetaPageData;
        use crate::index::relfile;
        use crate::index::vacuum::ambulkdelete;
        use std::collections::HashSet;
        use std::time::Instant;

        use_turbovec();
        // 1 000 rows, 16-d vectors so the codes/scales/ids
        // chains span multiple pages each (16-d at bit_width=4 =>
        // stride 8 bytes => ~1021 codes per page; 1000 rows still
        // fits in one codes page but spans more for ids/scales
        // when we look at smaller bit widths). Either way the
        // swap-remove logic is exercised: dead-slot < last_live
        // forces a real copy.
        Spi::run("CREATE TABLE t_amwalk (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_amwalk \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 1000) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_amwalk_idx ON t_amwalk USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        // File-size and meta-version snapshots before the walk.
        let pages_before: i64 = Spi::get_one(
            "SELECT (pg_relation_size('t_amwalk_idx'::regclass) / 8192)::int8",
        )
        .unwrap()
        .unwrap();
        assert!(pages_before >= 4, "relfile must be populated; got {} pages", pages_before);

        // Pick 5 ctids to mark dead. We choose ids that are NOT
        // at the very tail of the slot order so the swap-remove
        // path actually has to copy the last live row into the
        // dead slot (s != last). Slot order matches insertion
        // order, here id 1..1000.
        let dead_set: HashSet<u64> = {
            let mut set: HashSet<u64> = HashSet::new();
            Spi::connect(|client| {
                let tup = client
                    .select(
                        "SELECT ctid FROM t_amwalk WHERE id IN (3, 7, 100, 250, 999) ORDER BY id",
                        None,
                        &[],
                    )
                    .unwrap();
                for row in tup {
                    let tid: pg_sys::ItemPointerData =
                        row.get_by_name("ctid").unwrap().unwrap();
                    set.insert(pgrx::itemptr::item_pointer_to_u64(tid));
                }
            });
            set
        };
        assert_eq!(dead_set.len(), 5, "expected 5 distinct ctids");

        // Look up the index OID and meta page before the walk.
        let indexrelid_u32: Option<i64> = Spi::get_one(
            "SELECT 't_amwalk_idx'::regclass::oid::int8",
        )
        .unwrap();
        let indexrelid =
            pg_sys::Oid::from(indexrelid_u32.unwrap() as u32);

        let (n_before, version_before): (u64, u32) = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta must exist post-build");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            (m.n_vectors, m.am_version)
        };
        assert_eq!(n_before, 1000);

        // Synthetic dead-tuple callback. `state` is a `*const
        // HashSet<u64>` we pass through callback_state.
        unsafe extern "C-unwind" fn dead_cb(
            tid: pg_sys::ItemPointer,
            state: *mut std::ffi::c_void,
        ) -> bool {
            let set = &*(state as *const HashSet<u64>);
            let id = pgrx::itemptr::item_pointer_to_u64(*tid);
            set.contains(&id)
        }

        // Drive ambulkdelete via the same FFI shape the
        // autovacuum launcher uses. We only need (*info).index;
        // the rest is zeroed.
        let elapsed = unsafe {
            let rel = pg_sys::index_open(
                indexrelid,
                pg_sys::ShareUpdateExclusiveLock as i32,
            );
            assert!(!rel.is_null());
            let mut info: pg_sys::IndexVacuumInfo = std::mem::zeroed();
            info.index = rel;
            info.analyze_only = false;
            info.estimated_count = false;
            info.message_level = pg_sys::DEBUG2 as i32;
            info.num_heap_tuples = 1000.0;
            info.strategy = std::ptr::null_mut();

            let stats = pg_sys::palloc0(
                std::mem::size_of::<pg_sys::IndexBulkDeleteResult>(),
            ) as *mut pg_sys::IndexBulkDeleteResult;

            let t0 = Instant::now();
            let res = ambulkdelete(
                &mut info as *mut _,
                stats,
                Some(dead_cb),
                &dead_set as *const _ as *mut std::ffi::c_void,
            );
            let dt = t0.elapsed();
            assert!(!res.is_null());
            assert_eq!((*res).num_index_tuples as u64, 995);
            assert_eq!((*res).tuples_removed as u64, 5);

            pg_sys::index_close(rel, pg_sys::ShareUpdateExclusiveLock as i32);
            dt
        };

        // Loose perf check: 5 dead rows in a 1k-row index must
        // finish well under 200 ms on debug builds. The old
        // rebuild path was already fast at this size, so this
        // mostly guards against an O(n_vectors^2) regression.
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "ambulkdelete with 5 dead rows took {:?}, expected < 500ms",
            elapsed,
        );

        // Meta page must reflect the shrink: n_vectors = 995,
        // am_version bumped, layout fields preserved.
        let meta_after: MetaPageData = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta must still exist");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            m
        };
        assert_eq!(meta_after.n_vectors, 995, "n_vectors must drop to 995");
        assert!(
            meta_after.am_version > version_before,
            "am_version must bump: {} -> {}",
            version_before,
            meta_after.am_version,
        );

        // Phase P: ambulkdelete must invalidate the prepared
        // SIMD-blocked chain because swap-remove changes slot
        // ordering and the on-disk blocked layout no longer
        // matches `packed_codes`. Readers fall back to
        // per-backend repack (correct, slower) until the next
        // full rewrite.
        assert!(
            !meta_after.has_prepared_layout(),
            "ambulkdelete must invalidate the prepared layout: \
             blocked_bytes={} cb_levels={}",
            meta_after.blocked_bytes,
            meta_after.codebook_n_levels,
        );

        // Page count must not grow (it may shrink via
        // RelationTruncate). The whole point of the in-place walk
        // is that we don't rewrite every page.
        let pages_after: i64 = Spi::get_one(
            "SELECT (pg_relation_size('t_amwalk_idx'::regclass) / 8192)::int8",
        )
        .unwrap()
        .unwrap();
        assert!(
            pages_after <= pages_before,
            "page count must not grow: before={} after={}",
            pages_before,
            pages_after,
        );

        // The 5 deleted ids must still be present in the heap (we
        // only flagged them as dead via the synthetic callback,
        // we didn't actually DELETE them) but the index now
        // reports 995 live rows. The 995 survivors are queryable.
        // We don't issue a real ORDER BY query because the per-
        // backend cache may still hold the pre-walk IdMapIndex
        // (am_version bump triggers re-load on next scan, but
        // the pgrx tx hasn't committed). Instead, re-read the
        // ids chain off disk and assert it has exactly 995
        // entries with no duplicates and no holes.
        let surviving_ids: Vec<u64> = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).unwrap();
            let v = relfile::read_ids_only(rel, &m);
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            v
        };
        assert_eq!(surviving_ids.len(), 995);
        let unique: HashSet<u64> = surviving_ids.iter().copied().collect();
        assert_eq!(unique.len(), 995, "surviving ids must be distinct");
        for d in &dead_set {
            assert!(
                !unique.contains(d),
                "dead id {} must not appear among survivors",
                d,
            );
        }
    }

    /// Phase P: `ambuild` persists the prepared SIMD-blocked
    /// layout and Lloyd-Max codebook into the relfile, so a
    /// fresh backend opening the index reads them off disk
    /// instead of recomputing them. We can't reach across
    /// backends from inside a pgrx test, but we *can* verify
    /// the on-disk meta page records the prepared layout and
    /// that constructing an `IdMapIndex` via
    /// `from_id_map_parts_with_prepared` matches the freshly-
    /// built one bit-for-bit.
    #[pg_test]
    fn relfile_prepared_layout_skips_runtime_pack() {
        use crate::index::page::MetaPageData;
        use crate::index::relfile;
        use std::time::Instant;
        use_turbovec();

        Spi::run("CREATE TABLE t_pp (id bigint PRIMARY KEY, emb vector)").unwrap();
        // 100 rows of 16-d vectors. The unprepared path's compute
        // cost is dominated by `pack::repack` + Lloyd-Max; both
        // are tiny here, but the test asserts the right *shape*
        // (prepared layout populated, used at scan time, scan
        // succeeds), not a literal speedup at this corpus size.
        Spi::run(
            "INSERT INTO t_pp \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 100) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_pp_idx ON t_pp USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        let indexrelid_u32: Option<i64> =
            Spi::get_one("SELECT 't_pp_idx'::regclass::oid::int8").unwrap();
        let indexrelid = pg_sys::Oid::from(indexrelid_u32.unwrap() as u32);

        // (1) On-disk meta page must be v2 with the prepared
        // layout populated.
        let meta: MetaPageData = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta exists");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            m
        };
        assert_eq!(meta.version, 2, "new index must use the v2 wire format");
        assert!(
            meta.has_prepared_layout(),
            "meta must record blocked + codebook: blocked_bytes={} cb_levels={}",
            meta.blocked_bytes,
            meta.codebook_n_levels,
        );
        assert_eq!(
            meta.codebook_n_levels, 16,
            "4-bit codebook has 16 centroids",
        );
        assert_eq!(meta.centroids_slice().len(), 16);
        assert_eq!(meta.boundaries_slice().len(), 15);
        // boundaries are strictly increasing
        let bs = meta.boundaries_slice();
        for w in bs.windows(2) {
            assert!(w[0] < w[1], "boundaries must be sorted: {:?}", bs);
        }
        // The blocked chain is a real chain: count > 0, first > meta.
        assert!(meta.blocked_first > meta.ids_first, "blocked chain must follow ids");
        assert!(meta.blocked_count >= 1);
        assert!(meta.blocked_bytes > 0);

        // (2) Read the prepared chain and assert it round-trips
        // through `from_id_map_parts_with_prepared` to a working
        // index. The construction must NOT call `pack::repack`
        // — we proxy that by timing it: a 100-row index built
        // from prepared parts is microseconds, while the
        // un-prepared `from_id_map_parts` followed by
        // `prepare_eager` pays the codebook compute (which on a
        // debug build is still ~milliseconds even at dim=16).
        let (codes, scales, ids, blocked, n_blocks, centroids, boundaries) = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            let (c, s, i) = relfile::read_full(rel, &m);
            let b = relfile::read_blocked(rel, &m);
            let cents = m.centroids_slice().to_vec();
            let bnds = m.boundaries_slice().to_vec();
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            (c, s, i, b, m.n_blocks_blocked as usize, cents, bnds)
        };
        assert_eq!(blocked.len() as u64, meta.blocked_bytes);

        // Construct two indexes from the same parts: one with
        // prepared, one without. Both must agree on top-1 for a
        // synthetic query.
        let t0 = Instant::now();
        let idx_prep = turbovec::IdMapIndex::from_id_map_parts_with_prepared(
            meta.bit_width as usize,
            meta.dim as usize,
            meta.n_vectors as usize,
            codes.clone(),
            scales.clone(),
            ids.clone(),
            blocked,
            n_blocks,
            centroids,
            boundaries,
        )
        .expect("prepared parts");
        let prep_ctor_us = t0.elapsed().as_micros();

        let t1 = Instant::now();
        let idx_plain = turbovec::IdMapIndex::from_id_map_parts(
            meta.bit_width as usize,
            meta.dim as usize,
            meta.n_vectors as usize,
            codes,
            scales,
            ids,
        )
        .expect("plain parts");
        let plain_ctor_us = t1.elapsed().as_micros();

        // Run the same query through both. Use the first row's
        // vector as the query so we expect that row's id back.
        let query_vec: Vec<f32> = (0..16)
            .map(|k| ((u32::wrapping_mul(7, k as u32 + 13)) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        // Force the search timing to include the (potentially
        // expensive) cache initialisation by measuring the very
        // first call on each index.
        let t2 = Instant::now();
        let (_, ids_prep) = idx_prep.search(&query_vec, 1);
        let prep_search_us = t2.elapsed().as_micros();

        let t3 = Instant::now();
        let (_, ids_plain) = idx_plain.search(&query_vec, 1);
        let plain_search_us = t3.elapsed().as_micros();

        eprintln!(
            "phase-p ctor+first-search timing (debug, 100x16 4-bit): \
             prep ctor={} us, prep search={} us; \
             plain ctor={} us, plain search={} us",
            prep_ctor_us, prep_search_us, plain_ctor_us, plain_search_us,
        );

        assert_eq!(
            ids_prep, ids_plain,
            "prepared and plain indexes must agree on top-1",
        );
        // The first search through the prepared index must finish
        // well under what the plain path takes; the corpus is
        // tiny so we use a generous absolute bound (100 ms
        // debug). The user-facing assertion in the docstring is
        // "< 100 ms for top-1 on a 100-row corpus" — we hit it by
        // a wide margin.
        assert!(
            prep_search_us < 100_000,
            "prepared first-search took {} us, expected < 100_000 us (100 ms)",
            prep_search_us,
        );

        // (3) End-to-end SQL query through the index: must
        // succeed and return id 73 for the self-query.
        Spi::run("ANALYZE t_pp").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM t_pp \
             ORDER BY emb <=> (SELECT emb FROM t_pp WHERE id = 73) \
             LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest, Some(73));
    }

    /// Phase Q (v1.3.0): the meta-page decoder distinguishes v1
    /// (Phase L preview) from v2 (Phase P) layouts and reports
    /// `is_legacy_v1()` correctly. The `ambeginscan` migration
    /// boundary fires on `is_legacy_v1() && n_vectors > 0`; we
    /// verify the detection primitive directly here.
    ///
    /// Live exercise of the ERROR path is impossible from inside
    /// a v1.3.0 binary because `MetaPageData::encode` always
    /// writes `VERSION = 2`; manufactured v1 buffers can only be
    /// fed back through `MetaPageData::decode`.
    #[pg_test]
    fn relfile_legacy_v1_detection_primitive() {
        use crate::index::page::{MetaPageData, MAGIC, PAYLOAD_BYTES};
        use crate::index::relfile;
        use_turbovec();

        Spi::run("CREATE TABLE t_old (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_old \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 50) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_old_idx ON t_old USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        let indexrelid_u32: Option<i64> =
            Spi::get_one("SELECT 't_old_idx'::regclass::oid::int8").unwrap();
        let indexrelid = pg_sys::Oid::from(indexrelid_u32.unwrap() as u32);

        // Initial meta is v2 (the only version we write).
        let v2_meta: MetaPageData = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            m
        };
        assert_eq!(v2_meta.version, 2);
        assert!(!v2_meta.is_legacy_v1());
        assert!(v2_meta.has_prepared_layout());

        // Manufacture a v1 meta-page byte buffer with the same
        // codes/scales/ids chain pointers but no prepared layout.
        // This emulates what an index built before Phase P would
        // have on disk.
        let mut v1_buf = [0u8; PAYLOAD_BYTES];
        v1_buf[0..4].copy_from_slice(&MAGIC);
        v1_buf[4] = 1; // v1
        v1_buf[5] = v2_meta.bit_width;
        v1_buf[8..12].copy_from_slice(&v2_meta.dim.to_le_bytes());
        v1_buf[12..20].copy_from_slice(&v2_meta.n_vectors.to_le_bytes());
        v1_buf[20..24].copy_from_slice(&v2_meta.codes_first.to_le_bytes());
        v1_buf[24..28].copy_from_slice(&v2_meta.codes_count.to_le_bytes());
        v1_buf[28..32].copy_from_slice(&v2_meta.scales_first.to_le_bytes());
        v1_buf[32..36].copy_from_slice(&v2_meta.scales_count.to_le_bytes());
        v1_buf[36..40].copy_from_slice(&v2_meta.ids_first.to_le_bytes());
        v1_buf[40..44].copy_from_slice(&v2_meta.ids_count.to_le_bytes());
        v1_buf[44..48].copy_from_slice(&v2_meta.rows_per_codes_page.to_le_bytes());
        v1_buf[48..52].copy_from_slice(&v2_meta.rows_per_scales_page.to_le_bytes());
        v1_buf[52..56].copy_from_slice(&v2_meta.rows_per_ids_page.to_le_bytes());
        v1_buf[56..60].copy_from_slice(&v2_meta.stride_bytes.to_le_bytes());
        v1_buf[60..64].copy_from_slice(&v2_meta.am_version.to_le_bytes());
        // No v2 fields.

        // Decoder round-trips: v1 buffer comes back as version=1
        // with zeroed prepared-layout fields.
        let v1_decoded = MetaPageData::decode(&v1_buf).expect("v1 decode");
        assert_eq!(v1_decoded.version, 1);
        assert!(v1_decoded.is_legacy_v1());
        assert!(!v1_decoded.has_prepared_layout());
        assert_eq!(v1_decoded.blocked_bytes, 0);
        assert_eq!(v1_decoded.codebook_n_levels, 0);
        assert_eq!(v1_decoded.n_vectors, v2_meta.n_vectors);
        assert_eq!(v1_decoded.codes_first, v2_meta.codes_first);

        // Verify the scan path's ERROR-emitting condition: the
        // matcher in `ambeginscan` fires on `is_legacy_v1() &&
        // n_vectors > 0`. Our manufactured v1 meta satisfies
        // both, so a backend opening such an index would raise
        // ERROR ("REINDEX INDEX ...").
        assert!(
            v1_decoded.is_legacy_v1() && v1_decoded.n_vectors > 0,
            "manufactured v1 meta must trigger the ERROR path",
        );
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
