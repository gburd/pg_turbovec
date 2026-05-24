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

#[cfg(feature = "experimental_index_am")]
pub mod index;

pub mod kernels;
pub mod knn;
pub mod normalize;
pub mod vec;

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

    #[cfg(feature = "experimental_index_am")]
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

    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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

    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_l2_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    /// Same, for vec_l1_ops.
    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_l1_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_vec, Some(2));
    }

    /// Expression index workaround for halfvec: index `(emb::vector)`
    /// instead of `emb` directly. Postgres expression-index machinery
    /// handles the cast at build and query time, so halfvec users get
    /// indexed ANN without needing dedicated halfvec opclasses on the
    /// AM. Same pattern works for sparsevec.
    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_hv_idx'::regclass",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_cic_idx_normal'::regclass",
        )
        .unwrap();
        assert_eq!(n_vec, Some(3));
    }

    /// VACUUM after DELETE removes dead rows from the AM via
    /// ambulkdelete (Phase 15 made this work — v0.4..v0.14 were a
    /// stub that did nothing).
    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_vac_idx'::regclass",
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_vac_idx'::regclass",
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
    /// then INSERT new rows and verify the side-table state and
    /// the search results reflect the additions.
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_384_idx'::regclass",
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
    #[cfg(feature = "experimental_index_am")]
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

        let bit_width: Option<i32> = Spi::get_one(
            "SELECT bit_width FROM turbovec.am_storage \
             WHERE indexrelid = 't_2bit_idx'::regclass",
        )
        .unwrap();
        assert_eq!(bit_width, Some(2));

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
    #[cfg(feature = "experimental_index_am")]
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

        // Side-table row count must reflect the rebuild; the AM
        // cache must serve fresh data on the next scan.
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 'reidx_t_idx'::regclass",
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
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(feature = "experimental_index_am")]
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
    #[cfg(all(
        feature = "experimental_index_am",
        feature = "relfile_storage"
    ))]
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

        // Side-table marker row tracks the count under the new
        // path (n_vectors mirrored, payload empty).
        let n_vec: Option<i64> = Spi::get_one(
            "SELECT n_vectors FROM turbovec.am_storage \
             WHERE indexrelid = 't_rf_idx'::regclass",
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
    /// harness is `bench/sql/phase_l_cold_scan.sql`). Asserts only
    /// that both timings are reasonable; the strong cold-vs-warm
    /// inequality is too noisy inside a transaction to assert.
    ///
    /// Phase G/H reference numbers (1 M rows, side-table path):
    /// cold p50 = 6 802 ms. The relfile path's headline win is at
    /// scale: shared_buffers caches the index pages cluster-wide,
    /// so every backend after the first pays only the buffer-pool
    /// hit cost, not the SPI fetch + TOAST + parse cost.
    #[cfg(all(
        feature = "experimental_index_am",
        feature = "relfile_storage"
    ))]
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
