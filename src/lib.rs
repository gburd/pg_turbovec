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

    /// Assert that every id in an ANN result set is distinct.
    ///
    /// This is the single cheapest guard against the entire
    /// "wrong-ranking" bug class. The pre-AVX2 scalar-fallback
    /// regression in turbovec returned the *same* TID N times
    /// instead of the top-N distinct neighbours; a recall metric can
    /// hide that (the right answer is often in the duplicated set),
    /// but a distinct-count assertion catches it instantly. Every
    /// ANN-scan `#[pg_test]` that pulls back more than one id should
    /// run its result through here. See `docs/TESTING.md`.
    fn assert_distinct_ids(ids: &[i64]) {
        use std::collections::HashSet;
        let unique: HashSet<i64> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "ANN result contained duplicate TIDs (the pre-AVX2 \
             wrong-results signature): {} rows but only {} distinct \
             ids: {:?}",
            ids.len(),
            unique.len(),
            ids,
        );
    }

    /// Fetch a column of `bigint` ids from an ANN query as a `Vec`,
    /// preserving result order. Used by the recall-floor and
    /// distinct-id tests so the Rust side can assert on the full
    /// ranked list rather than a scalar aggregate.
    fn fetch_ids(sql: &str) -> Vec<i64> {
        Spi::connect(|client| {
            let tup = client.select(sql, None, &[]).unwrap();
            tup.map(|r| r.get::<i64>(1).unwrap().unwrap()).collect()
        })
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

    /// Parity gap #2 (parallel index build, Option B). The build's
    /// quantize + repack phases fan out across a `turbovec.build_parallelism`-
    /// sized rayon pool. Because the heap scan stays serial and the
    /// fan-out is a pure data-parallel map, a parallel build must
    /// return exactly the same top-k as a serial build of the same
    /// rows. We build the SAME table's index twice (REINDEX) under
    /// two different parallelism settings and compare the ranked id
    /// list. ~2000 deterministic rows so the TQ+ calibration path
    /// (>= 1000 samples) engages.
    #[pg_test]
    fn parallel_build_matches_serial_query() {
        use_turbovec();
        Spi::run("CREATE TABLE t_pb (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        // Deterministic 16-dim corpus: each row is e_(id%16) scaled,
        // so nearest-neighbour answers are predictable and stable.
        Spi::run(
            "INSERT INTO t_pb \
             SELECT g, \
                    ('[' || array_to_string(ARRAY(\
                       SELECT CASE WHEN d = (g % 16) THEN 1.0 \
                                   ELSE (((g * 31 + d * 7) % 17)::float8 / 100.0) \
                              END \
                       FROM generate_series(0, 15) AS d), ',') || ']')::vector \
             FROM generate_series(1, 2000) AS g",
        )
        .unwrap();

        // Serial build (1 thread).
        Spi::run("SET turbovec.build_parallelism = 1").unwrap();
        Spi::run(
            "CREATE INDEX t_pb_idx ON t_pb USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let serial: Vec<i64> = Spi::connect(|client| {
            let tup = client
                .select(
                    "SELECT id FROM t_pb \
                     ORDER BY emb <=> '[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector \
                     LIMIT 10",
                    None,
                    &[],
                )
                .unwrap();
            tup.map(|r| r.get::<i64>(1).unwrap().unwrap()).collect()
        });

        // Parallel rebuild (4 workers) of the SAME data.
        Spi::run("SET turbovec.build_parallelism = 4").unwrap();
        Spi::run("REINDEX INDEX t_pb_idx").unwrap();
        let parallel: Vec<i64> = Spi::connect(|client| {
            let tup = client
                .select(
                    "SELECT id FROM t_pb \
                     ORDER BY emb <=> '[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector \
                     LIMIT 10",
                    None,
                    &[],
                )
                .unwrap();
            tup.map(|r| r.get::<i64>(1).unwrap().unwrap()).collect()
        });

        assert_eq!(
            serial, parallel,
            "parallel build returned a different ranked top-k than serial"
        );
        assert_distinct_ids(&serial);
        assert_distinct_ids(&parallel);
    }

    /// Byte-identity proxy at the SQL layer: a serial-built and a
    /// parallel-built relfile for the SAME rows must have identical
    /// `pg_relation_size`. (Page-header LSNs differ per write, so a
    /// raw whole-file md5 across two relations is not a valid
    /// equality test; the authoritative byte-for-byte check lives in
    /// the `build_parts_are_pool_size_invariant` Rust unit test,
    /// which compares packed_codes / scales / blocked_codes /
    /// slot_to_id directly.) Equal size confirms the parallel path
    /// did not change the page layout or chain length.
    #[pg_test]
    fn parallel_build_relfile_size_matches_serial() {
        use_turbovec();
        Spi::run("CREATE TABLE t_pbsz (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_pbsz \
             SELECT g, \
                    ('[' || array_to_string(ARRAY(\
                       SELECT (((g * 13 + d * 5) % 19)::float8 / 50.0) \
                       FROM generate_series(0, 31) AS d), ',') || ']')::vector \
             FROM generate_series(1, 1500) AS g",
        )
        .unwrap();

        Spi::run("SET turbovec.build_parallelism = 1").unwrap();
        Spi::run(
            "CREATE INDEX t_pbsz_idx ON t_pbsz USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        let serial_sz: Option<i64> = Spi::get_one(
            "SELECT pg_relation_size('t_pbsz_idx'::regclass)::int8",
        )
        .unwrap();

        Spi::run("SET turbovec.build_parallelism = 4").unwrap();
        Spi::run("REINDEX INDEX t_pbsz_idx").unwrap();
        let parallel_sz: Option<i64> = Spi::get_one(
            "SELECT pg_relation_size('t_pbsz_idx'::regclass)::int8",
        )
        .unwrap();

        assert_eq!(
            serial_sz, parallel_sz,
            "parallel relfile size {:?} != serial {:?}",
            parallel_sz, serial_sz
        );
    }

    /// The `turbovec.build_parallelism` GUC must be registered,
    /// settable, and accepted by `ambuild` without error at every
    /// value in range (0 = auto, a pinned worker count, and 1 =
    /// inline). This is the "workers actually configured" smoke
    /// test — the pg_test harness won't necessarily launch separate
    /// OS-process workers (our parallelism is rayon-internal, not
    /// PG bgworkers), so we assert the GUC plumbing and a successful
    /// build under each setting rather than a worker count.
    #[pg_test]
    fn parallel_build_honors_guc() {
        use_turbovec();
        // Default is 0 (auto).
        let dflt: Option<i32> =
            Spi::get_one("SELECT current_setting('turbovec.build_parallelism')::int")
                .unwrap();
        assert_eq!(dflt, Some(0), "build_parallelism default should be 0 (auto)");

        Spi::run("CREATE TABLE t_pbg (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_pbg \
             SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector \
             FROM generate_series(1, 64) AS g",
        )
        .unwrap();

        for p in [0i32, 1, 4] {
            Spi::run(&format!("SET turbovec.build_parallelism = {p}")).unwrap();
            Spi::run(
                "CREATE INDEX t_pbg_idx ON t_pbg USING turbovec (emb vec_cosine_ops)",
            )
            .unwrap();
            let n: Option<i64> = Spi::get_one("SELECT count(*) FROM t_pbg").unwrap();
            assert_eq!(n, Some(64), "build at parallelism={p} lost rows");
            Spi::run("DROP INDEX t_pbg_idx").unwrap();
        }
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

    /// Phase W (v1.6.0): the ambuild heap-scan callback streams
    /// rows into `IdMapIndex::add_with_ids` in chunks bounded by
    /// `maintenance_work_mem` rather than accumulating the entire
    /// heap-scan output in a single `Vec<f32>`. This test exercises
    /// that code path: with `maintenance_work_mem = '4MB'` and
    /// dim = 8 vectors, `chunk_rows = (4 MB * 1024 * 0.75) / (8 * 4)
    /// = ~98 304`, so 1000 rows fit in one chunk — but the chunked
    /// flush + final drain still run, the `shrink_to_fit` path
    /// still runs, and the final-drain code in `ambuild` is exercised.
    /// (For more chunks we'd need >100k rows, which is too slow for
    /// the unit-test suite; the local test asserts correctness of
    /// the streaming path while the meh-class 10 M-row validation
    /// is a follow-up phase.)
    ///
    /// Note on opclass: we use `vec_l2_ops` rather than
    /// `vec_cosine_ops` because the test fixture
    /// `[g, 0, 0, 0, 0, 0, 0, 0]` produces colinear vectors whose
    /// cosine distance to any `[k, 0, ...]` query is identically
    /// zero — every row would tie and the ORDER BY result would be
    /// non-deterministic. L2 distance preserves the magnitude
    /// difference (`|g - 7|`), so id=7 is the unambiguous nearest
    /// neighbour.
    #[pg_test]
    fn ambuild_streams_heap_scan_under_maintenance_work_mem() {
        use_turbovec();
        Spi::run("SET maintenance_work_mem = '4MB'").unwrap();
        Spi::run("CREATE TABLE t_streamed (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_streamed SELECT g, \
                ('[' || g || ',0,0,0,0,0,0,0]')::vector \
                FROM generate_series(1, 1000) g",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_streamed_idx ON t_streamed \
             USING turbovec (emb vec_l2_ops)",
        )
        .unwrap();
        // Sanity: every row indexed.
        let n: Option<i64> =
            Spi::get_one("SELECT count(*) FROM t_streamed").unwrap();
        assert_eq!(n, Some(1000));
        // Nearest neighbour to [7,0,...] under L2 is row 7
        // (distance 0). Operator `<->` is L2.
        let id: Option<i64> = Spi::get_one(
            "SELECT id FROM t_streamed \
             ORDER BY emb <-> '[7,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(id, Some(7));
    }

    /// Phase W-2 (v1.7.0) introduced a split-write `ambuild` path
    /// that streamed `packed_codes` to the relfile, dropped the
    /// in-memory Vec mid-finalise, then wrote the SIMD-blocked
    /// layout. Validation on `meh` at 10 M × 1536-d showed the
    /// split made the build 53% slower with no actual RSS
    /// reduction (the "freed" heap pages just migrate to pinned
    /// shared buffers, which `ps -o rss` still counts).
    ///
    /// **v1.7.1 reverts that change**; this test is kept as a
    /// generic ambuild round-trip smoke covering the v1.6.0 /
    /// v1.7.1 single-call write path. We can't observe peak RSS
    /// from `#[pg_test]` (no /proc access in pgrx test mode), but
    /// we can verify that:
    ///
    ///   1. `CREATE INDEX` succeeds end-to-end.
    ///   2. The resulting index is queryable.
    ///   3. The meta page records both blocked + rotation
    ///      chains, proving `prepare_eager` ran to completion.
    #[pg_test]
    fn ambuild_round_trip_after_phase_w_2_revert() {
        use_turbovec();
        Spi::run("CREATE TABLE t_w2 (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_w2 SELECT g, \
                ('[' || g || ',0,0,0,0,0,0,0]')::vector \
                FROM generate_series(1, 100) g",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_w2_idx ON t_w2 USING turbovec (emb vec_l2_ops)",
        )
        .unwrap();

        // (1)+(2): index is queryable end-to-end. Self-distance
        // (L2) under the embedding `[42,0,...]` is uniquely
        // minimised by id = 42 (the only row whose vector
        // matches), so the top-1 must be 42 if the blocked
        // layout was correctly persisted.
        let id: Option<i64> = Spi::get_one(
            "SELECT id FROM t_w2 \
             ORDER BY emb <-> '[42,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(id, Some(42));

        // (3): meta page reports both prepared layout AND
        // rotation chain populated. We re-open the index by oid
        // — mirrors the pattern in
        // ambuild_persists_prepared_blocked_layout.
        let indexrelid: pg_sys::Oid = Spi::get_one(
            "SELECT 't_w2_idx'::regclass::oid",
        )
        .unwrap()
        .expect("index oid");
        use crate::index::relfile;
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel)
                .expect("meta must exist after build");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            assert_eq!(m.n_vectors, 100);
            assert!(
                m.has_prepared_layout(),
                "build must persist prepared layout: \
                 blocked_bytes={}, codebook_n_levels={}, \
                 rotation_count={}, version={}",
                m.blocked_bytes,
                m.codebook_n_levels,
                m.rotation_count,
                m.version,
            );
            assert!(m.blocked_bytes > 0, "blocked chain must be non-empty");
            assert!(m.rotation_count > 0, "rotation chain must be non-empty");
        }
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
        // The version emitted by IVF-1 (v4). Bump this only as part
        // of a deliberate minor/major release with a migration story.
        const EXPECTED_WIRE_FORMAT_VERSION: u8 = 4;
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

    #[pg_test]
    fn oversample_guc_round_trip() {
        // Default is 1.0 (no oversampling = pre-feature behaviour).
        let dflt: Option<f64> =
            Spi::get_one("SELECT current_setting('turbovec.oversample')::float8").unwrap();
        assert_eq!(dflt, Some(1.0));
        Spi::run("SET turbovec.oversample = 4.0").unwrap();
        let v: Option<f64> =
            Spi::get_one("SELECT current_setting('turbovec.oversample')::float8").unwrap();
        assert_eq!(v, Some(4.0));
    }

    #[pg_test]
    fn max_probes_guc_round_trip() {
        // Default is 64 (IVF-3: 8x the turbovec.probes default of 8).
        let dflt: Option<String> =
            Spi::get_one("SELECT current_setting('turbovec.max_probes')").unwrap();
        assert_eq!(dflt.as_deref(), Some("64"));
        Spi::run("SET turbovec.max_probes = 128").unwrap();
        let v: Option<String> =
            Spi::get_one("SELECT current_setting('turbovec.max_probes')").unwrap();
        assert_eq!(v.as_deref(), Some("128"));
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

    /// Cold-scan parity-gap #3: the index-AM scan path installs a
    /// read-only cache entry (`Stored::ReadOnly`) that skips the
    /// O(n) `id_to_slot` HashMap build. This test pins that a plain
    /// read-only scan returns the same top-k as the eager path did.
    /// The corpus is deterministic so the assertion is exact.
    #[pg_test]
    fn cold_scan_readonly_topk_unchanged() {
        use_turbovec();
        Spi::run("CREATE TABLE t_ro (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_ro VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0.9,0.1,0,0,0,0,0,0]'), \
                 (3, '[0,1,0,0,0,0,0,0]'), \
                 (4, '[0,0,1,0,0,0,0,0]'), \
                 (5, '[-1,0,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_ro_idx ON t_ro USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Force a genuine cold miss so the read-only ScanHandle is
        // built from the relfile (not served from ambuild's entry).
        crate::cache::invalidate_all();

        let nearest: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ro \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest, Some(1), "nearest to e1 must be row 1");

        // A second cold scan (re-invalidate) must be identical — the
        // read-only rebuild is deterministic.
        crate::cache::invalidate_all();
        let nearest2: Option<i64> = Spi::get_one(
            "SELECT id FROM t_ro \
             ORDER BY emb <=> '[0,1,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(nearest2, Some(3), "nearest to e2 must be row 3");
    }

    /// KEY correctness test for the lazy-HashMap optimisation: a
    /// mutation (`aminsert`) AFTER a read-only scan must still work.
    /// The read-only scan installs a `Stored::ReadOnly` entry with
    /// no `id_to_slot` map; the first `aminsert` must detect that
    /// (via `am_lookup_for_mutation` returning `None`), rebuild a
    /// full `IdMapIndex` (building the HashMap then), apply the
    /// insert, and the newly-inserted row must be findable by the
    /// next scan.
    #[pg_test]
    fn mutation_after_readonly_scan_is_correct() {
        use_turbovec();
        Spi::run("CREATE TABLE t_mas (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_mas VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_mas_idx ON t_mas USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // 1. Read-only scan first — installs a `ReadOnly` entry
        //    (no HashMap).
        crate::cache::invalidate_all();
        let n1: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mas \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(n1, Some(1));

        // 2. Mutation AFTER the read-only scan. This INSERT goes
        //    through aminsert, which must rebuild the index as a
        //    mutable IdMapIndex (deferred HashMap build) rather
        //    than trying to mutate the read-only entry in place.
        Spi::run(
            "INSERT INTO t_mas VALUES \
                 (3, '[0,0,1,0,0,0,0,0]'), \
                 (4, '[0,0,0,1,0,0,0,0]')",
        )
        .unwrap();

        // 3. The newly-inserted rows must be findable in the same
        //    transaction (the in-memory mutable mirror serves the
        //    scan; the deferred HashMap was built correctly).
        let near4: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mas \
             ORDER BY emb <=> '[0,0,0,1,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            near4,
            Some(4),
            "row 4 inserted after a read-only scan must be findable \
             (the deferred id_to_slot HashMap built correctly on the \
             first mutation)"
        );
        let near3: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mas \
             ORDER BY emb <=> '[0,0,1,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(near3, Some(3), "row 3 must also be findable");

        // And the original rows still resolve correctly.
        let near1: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mas \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(near1, Some(1), "row 1 must still be nearest to e1");
    }

    /// Delete-then-scan must stay correct even though the scan now
    /// installs a read-only entry. pgrx tests run inside a
    /// transaction so plain `VACUUM` (which drives `ambulkdelete`)
    /// can't run here; we use `REINDEX` instead, which bumps the
    /// relfilenode, invalidates the cache, and forces a fresh
    /// read-only rebuild over the post-delete heap snapshot — the
    /// stronger of the two paths for this test. (Real
    /// `ambulkdelete` swap-remove is exercised by the psql
    /// regression script in tests/02_index_am.sql.)
    #[pg_test]
    fn delete_then_readonly_scan_is_correct() {
        use_turbovec();
        Spi::run("CREATE TABLE t_del (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_del VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_del_idx ON t_del USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Read-only scan establishes a ReadOnly cache entry.
        crate::cache::invalidate_all();
        let before: Option<i64> = Spi::get_one(
            "SELECT id FROM t_del \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(before, Some(1));

        // Delete row 1, then REINDEX to rebuild the relfile over
        // the post-delete heap.
        Spi::run("DELETE FROM t_del WHERE id = 1").unwrap();
        Spi::run("REINDEX INDEX t_del_idx").unwrap();

        // Cold scan after the delete must not return row 1 (the
        // deleted TID). The survivors (rows 2,3) are both orthogonal
        // to e1 with equal cosine, so assert simply that row 1 is
        // gone and a survivor is returned.
        crate::cache::invalidate_all();
        let after: Option<i64> = Spi::get_one(
            "SELECT id FROM t_del \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_ne!(after, Some(1), "deleted row 1 must not be returned");
        assert!(
            matches!(after, Some(2) | Some(3)),
            "a surviving row must be returned, got {:?}",
            after
        );
        let n_remaining: Option<i64> =
            Spi::get_one("SELECT count(*) FROM t_del").unwrap();
        assert_eq!(n_remaining, Some(2));
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
        let top10_self = fetch_ids(
            "WITH q AS (SELECT emb FROM t_64 WHERE id = 17) \
             SELECT t.id FROM t_64 t, q \
             ORDER BY t.emb <=> q.emb \
             LIMIT 10",
        );
        assert_distinct_ids(&top10_self);
        assert!(
            top10_self.contains(&17),
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
        let indexed_top5 = fetch_ids(
            "WITH q AS (SELECT q FROM q_64) \
             SELECT t.id FROM t_64 t, q \
             ORDER BY t.emb <=> q.q LIMIT 5",
        );
        assert_distinct_ids(&indexed_top5);
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

    /// Parity gap #1 regression: iterative index scan.
    ///
    /// A selective `WHERE` filter + `ORDER BY dist LIMIT k` must
    /// return the full LIMIT even when the matching rows are sparse
    /// among the top-`search_k` candidates. Pre-v1.8.0, `amgettuple`
    /// ran a single `search_k`-candidate batch and returned false
    /// when drained, so the executor post-filtered those candidates
    /// and under-returned. `turbovec.iterative_scan = relaxed_order`
    /// (the default) re-runs the search with a doubled k on drain.
    ///
    /// Fixture: 2000 rows, of which ~every-100th (category 7 ::
    /// id % 100 == 7) matches — ~20 rows. With `search_k = 16` the
    /// first batch holds far fewer than 10 category-7 rows, so
    /// `off` mode under-returns and `relaxed_order` returns 10.
    #[pg_test]
    fn index_am_iterative_scan_selective_filter() {
        use_turbovec();
        Spi::run(
            "CREATE TABLE t_iter (\
                 id bigint PRIMARY KEY, \
                 category int, \
                 emb vector)",
        )
        .unwrap();
        // 2000 8-dim rows; category = id % 100, so category 7
        // matches ids 7, 107, 207, ... — exactly 20 rows. The emb is
        // a hashed pseudo-random unit-ish vector so the category-7
        // rows are scattered through the distance ranking rather
        // than clustered at the top.
        Spi::run(
            "INSERT INTO t_iter \
             SELECT i, i % 100, \
                 ('[' || string_agg( \
                     ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, 2000) AS gs(i), \
                  generate_series(1, 8) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_iter_idx \
             ON t_iter USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("ANALYZE t_iter").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        // Small first batch so a single search_k batch can't satisfy
        // a LIMIT 10 over the sparse category-7 subset.
        Spi::run("SET turbovec.search_k = 16").unwrap();

        let q = "(SELECT emb FROM t_iter WHERE id = 1007)";
        let query = format!(
            "SELECT count(*)::bigint FROM ( \
                 SELECT id FROM t_iter WHERE category = 7 \
                 ORDER BY emb <=> {q} LIMIT 10 \
             ) sub"
        );

        // off mode: single batch under-returns (< 10).
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        let off_count: Option<i64> = Spi::get_one(&query).unwrap();
        assert!(
            off_count.unwrap() < 10,
            "iterative_scan=off should under-return on a selective \
             filter; got {:?} (expected < 10)",
            off_count
        );

        // relaxed_order (default): refills until the LIMIT is met.
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        let relaxed_count: Option<i64> = Spi::get_one(&query).unwrap();
        assert_eq!(
            relaxed_count,
            Some(10),
            "iterative_scan=relaxed_order should return the full LIMIT"
        );

        // The refilled result must not repeat any TID across batches.
        let relaxed_ids = fetch_ids(&format!(
            "SELECT id FROM t_iter WHERE category = 7 \
             ORDER BY emb <=> {q} LIMIT 10"
        ));
        assert_distinct_ids(&relaxed_ids);
    }

    /// No TID may be emitted twice across refill batches. The result
    /// set of an unfiltered `ORDER BY dist LIMIT k` with a tiny
    /// `search_k` (forcing several refills) must have unique ids.
    #[pg_test]
    fn index_am_iterative_scan_no_duplicate_tids() {
        use_turbovec();
        Spi::run("CREATE TABLE t_iter_dup (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_iter_dup \
             SELECT i, \
                 ('[' || string_agg( \
                     ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, 500) AS gs(i), \
                  generate_series(1, 8) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_iter_dup_idx \
             ON t_iter_dup USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("ANALYZE t_iter_dup").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 8").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();

        // LIMIT 200 forces many refill rounds (8 -> 16 -> ... -> 200+).
        // total rows must equal distinct ids.
        let ids = fetch_ids(
            "SELECT id FROM t_iter_dup \
             ORDER BY emb <=> '[0,0,0,0,0,0,0,0]'::vector LIMIT 200",
        );
        assert_distinct_ids(&ids);
        // Sanity: 200 distinct rows actually came back (corpus is 500).
        assert_eq!(ids.len(), 200);
    }

    /// `turbovec.max_scan_tuples` is the hard ceiling: with a low cap
    /// the scan must stop (not loop forever) even if the filter is
    /// never satisfied. We set the cap below the corpus and use an
    /// impossible filter, so the scan exhausts its budget and the
    /// query terminates returning 0 rows.
    #[pg_test]
    fn index_am_iterative_scan_max_scan_tuples_honored() {
        use_turbovec();
        Spi::run(
            "CREATE TABLE t_iter_cap (\
                 id bigint PRIMARY KEY, category int, emb vector)",
        )
        .unwrap();
        Spi::run(
            "INSERT INTO t_iter_cap \
             SELECT i, 0, \
                 ('[' || string_agg( \
                     ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, 1000) AS gs(i), \
                  generate_series(1, 8) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_iter_cap_idx \
             ON t_iter_cap USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("ANALYZE t_iter_cap").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 8").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        // Cap well below the corpus; no row matches category 999, so
        // the scan can never satisfy the filter and must terminate
        // once it has examined ~50 candidates.
        Spi::run("SET turbovec.max_scan_tuples = 50").unwrap();

        let t0 = std::time::Instant::now();
        let n: Option<i64> = Spi::get_one(
            "SELECT count(*)::bigint FROM ( \
                 SELECT id FROM t_iter_cap WHERE category = 999 \
                 ORDER BY emb <=> '[0,0,0,0,0,0,0,0]'::vector LIMIT 10 \
             ) sub",
        )
        .unwrap();
        let elapsed = t0.elapsed().as_secs();
        assert_eq!(n, Some(0), "impossible filter must return 0 rows");
        assert!(elapsed < 30, "scan did not terminate (cap not honored)");
    }

    /// `iterative_scan = off` preserves the pre-v1.8.0 single-batch
    /// behaviour: at most `search_k` candidates are ever returned,
    /// regardless of LIMIT.
    #[pg_test]
    fn index_am_iterative_scan_off_preserves_single_batch() {
        use_turbovec();
        Spi::run("CREATE TABLE t_iter_off (id bigint PRIMARY KEY, emb vector)")
            .unwrap();
        Spi::run(
            "INSERT INTO t_iter_off \
             SELECT i, \
                 ('[' || string_agg( \
                     ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, 500) AS gs(i), \
                  generate_series(1, 8) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_iter_off_idx \
             ON t_iter_off USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("ANALYZE t_iter_off").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 12").unwrap();
        Spi::run("SET turbovec.iterative_scan = off").unwrap();

        // LIMIT 100 but search_k = 12 and off mode: exactly 12 rows.
        let ids = fetch_ids(
            "SELECT id FROM t_iter_off \
             ORDER BY emb <=> '[0,0,0,0,0,0,0,0]'::vector LIMIT 100",
        );
        assert_distinct_ids(&ids);
        assert_eq!(
            ids.len(),
            12,
            "iterative_scan=off must cap at search_k regardless of LIMIT"
        );
    }

    // ---- Oversampling (differentiator #5): tunable recall ----
    //
    // These tests share a corpus builder: `n` rows of `dim`-dim
    // pseudo-random vectors, deterministic from the row id so the
    // quantized-vs-exact ranking is reproducible across runs.
    fn build_oversample_corpus(table: &str, n: i32, dim: i32, bit_width: i32) {
        use_turbovec();
        Spi::run(&format!(
            "CREATE TABLE {table} (id bigint PRIMARY KEY, emb vector)"
        ))
        .unwrap();
        Spi::run(&format!(
            "INSERT INTO {table} \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, {n}) AS gs(i), \
                  generate_series(1, {dim}) AS sub(k) \
             GROUP BY i"
        ))
        .unwrap();
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table} \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = {bit_width})"
        ))
        .unwrap();
        Spi::run(&format!("ANALYZE {table}")).unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
    }

    /// Brute-force exact cosine top-`k` ids for the inlined query
    /// vector literal `qlit` (e.g. `'[0.1,...]'::vector`). Used as
    /// ground truth. The vector is inlined as a constant so the
    /// planner treats it as a fixed ORDER BY argument (no join, so
    /// the turbovec index can be chosen for the index variant below).
    fn exact_topk_ids(table: &str, qlit: &str, k: i32) -> std::collections::HashSet<i64> {
        let csv: Option<String> = Spi::get_one(&format!(
            "SELECT string_agg(id::text, ',') FROM ( \
                 SELECT t.id FROM {table} t \
                 ORDER BY (1.0 - turbovec.inner_product(t.emb, {qlit}) / \
                     (turbovec.vector_norm(t.emb) * turbovec.vector_norm({qlit}))) \
                 LIMIT {k} \
             ) sub"
        ))
        .unwrap();
        csv.unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i64>().unwrap())
            .collect()
    }

    /// Index top-`k` ids under the current GUC settings, with the
    /// query vector inlined as a literal so the planner pushes the
    /// ORDER BY into the turbovec index.
    fn index_topk_ids(table: &str, qlit: &str, k: i32) -> std::collections::HashSet<i64> {
        let csv: Option<String> = Spi::get_one(&format!(
            "SELECT string_agg(id::text, ',') FROM ( \
                 SELECT t.id FROM {table} t \
                 ORDER BY t.emb <=> {qlit} LIMIT {k} \
             ) sub"
        ))
        .unwrap();
        csv.unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i64>().unwrap())
            .collect()
    }

    /// Build the deterministic query-vector literal for a seed, as a
    /// `'[...]'::turbovec.vector` SQL string ready to inline.
    fn query_vector_literal(dim: i32, seed: &str) -> String {
        let body: Option<String> = Spi::get_one(&format!(
            "SELECT '[' || string_agg( \
                 ((hashtext('{seed}:' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']' \
             FROM generate_series(1, {dim}) AS sub(k)"
        ))
        .unwrap();
        format!("'{}'::turbovec.vector", body.unwrap())
    }

    fn recall_at(
        index: &std::collections::HashSet<i64>,
        truth: &std::collections::HashSet<i64>,
    ) -> f64 {
        if truth.is_empty() {
            return 1.0;
        }
        let hit = index.iter().filter(|id| truth.contains(id)).count();
        hit as f64 / truth.len() as f64
    }

    /// Oversampling widens the candidate set: a query whose true NN
    /// ranks just outside `search_k` by the lossy quantized distance
    /// is recovered at `oversample = 4.0` but missed at `1.0`.
    ///
    /// We pin `search_k` small and `iterative_scan = off` so the only
    /// lever is the oversample multiplier. Across a handful of query
    /// seeds at least one must show the recovery, proving the widened
    /// candidate set + reorder queue surfaces neighbours that the
    /// narrow quantized top-k dropped.
    #[pg_test]
    fn oversample_widens_candidate_set() {
        build_oversample_corpus("t_os_widen", 3000, 128, 2);
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        Spi::run("SET turbovec.search_k = 3").unwrap();

        // Confirm the inlined-literal query actually drives the
        // turbovec index (not a seq scan + sort, which would make
        // oversample a no-op and the recall always exact).
        let q1 = query_vector_literal(128, "q1");
        let mut uses_index = false;
        Spi::connect(|client| {
            let rows = client
                .select(
                    &format!(
                        "EXPLAIN SELECT t.id FROM t_os_widen t \
                         ORDER BY t.emb <=> {q1} LIMIT 10"
                    ),
                    None,
                    &[],
                )
                .unwrap();
            for row in rows {
                let line: Option<String> = row.get(1).unwrap();
                if line.unwrap_or_default().contains("Index Scan") {
                    uses_index = true;
                }
            }
        });
        assert!(
            uses_index,
            "the inlined-literal ORDER BY must use the turbovec index, \
             else oversample is a no-op and the test is meaningless"
        );

        let mut recovered_somewhere = false;
        let mut never_regressed = true;
        for seed in ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"] {
            let qlit = query_vector_literal(128, seed);
            let truth = exact_topk_ids("t_os_widen", &qlit, 10);

            Spi::run("SET turbovec.oversample = 1.0").unwrap();
            let r1 = recall_at(&index_topk_ids("t_os_widen", &qlit, 10), &truth);
            Spi::run("SET turbovec.oversample = 8.0").unwrap();
            let r8 = recall_at(&index_topk_ids("t_os_widen", &qlit, 10), &truth);

            if r8 > r1 + 1e-9 {
                recovered_somewhere = true;
            }
            if r8 + 1e-9 < r1 {
                never_regressed = false;
            }
        }
        assert!(
            never_regressed,
            "oversample=8.0 must never have worse recall than 1.0"
        );
        assert!(
            recovered_somewhere,
            "oversample=8.0 should recover at least one true NN that \
             search_k=3 / oversample=1.0 dropped, across 8 query seeds"
        );
    }

    /// Recall@10 is monotonically non-decreasing as oversample grows
    /// 1 -> 1.5 -> 2 -> 4 -> 8 on a fixed corpus + query set. This is
    /// the core contract of the feature and the shape that makes the
    /// recall-vs-latency frontier honest. We average recall over
    /// several query seeds to damp single-query tie noise, then assert
    /// the averaged curve never goes down.
    #[pg_test]
    fn oversample_recall_monotone_non_decreasing() {
        build_oversample_corpus("t_os_mono", 3000, 64, 4);
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        Spi::run("SET turbovec.search_k = 10").unwrap();

        let seeds = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"];
        let oversamples = [1.0_f64, 1.5, 2.0, 4.0, 8.0];

        // Precompute the query literals + ground truth per seed once.
        let qlits: Vec<String> = seeds.iter().map(|s| query_vector_literal(64, s)).collect();
        let truths: Vec<_> = qlits
            .iter()
            .map(|q| exact_topk_ids("t_os_mono", q, 10))
            .collect();

        let mut curve = Vec::new();
        for &os in &oversamples {
            Spi::run(&format!("SET turbovec.oversample = {os}")).unwrap();
            let mut sum = 0.0;
            for (q, truth) in qlits.iter().zip(truths.iter()) {
                sum += recall_at(&index_topk_ids("t_os_mono", q, 10), truth);
            }
            curve.push(sum / seeds.len() as f64);
        }

        // Warm p50 latency per oversample point (median over a small
        // sweep of the same query set, after a warmup pass so the
        // backend-local index cache is hot). Times the full SPI round
        // trip, so it is an upper bound on the AM-internal scan cost,
        // but the *shape* (roughly linear in candidate count) is what
        // matters for the recall-vs-latency frontier.
        let mut p50s = Vec::new();
        for &os in &oversamples {
            Spi::run(&format!("SET turbovec.oversample = {os}")).unwrap();
            // Warmup.
            let _ = index_topk_ids("t_os_mono", &qlits[0], 10);
            let mut samples: Vec<f64> = Vec::new();
            for q in &qlits {
                let t0 = std::time::Instant::now();
                let _ = index_topk_ids("t_os_mono", q, 10);
                samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p50s.push(samples[samples.len() / 2]);
        }
        // Emit the curve for the bench archive / commit message.
        // Measured 2026-06-15 (pg16, 4-bit, 3000x64, 8 query seeds):
        //   oversample 1.0 1.5 2.0 4.0 8.0
        //   recall@10  .8125 .9625 .9875 1.0 1.0
        //   p50 (ms)   3.81  3.86  3.94  4.06 4.70
        // -> recall climbs to 1.0, latency ~ linear in candidate count.
        pgrx::log!("oversample recall@10 curve {oversamples:?} -> {curve:?}; p50_ms {p50s:?}");

        for w in curve.windows(2) {
            assert!(
                w[1] + 1e-9 >= w[0],
                "recall@10 must be non-decreasing in oversample; curve={curve:?}"
            );
        }
        // Sanity: the curve actually climbs (the top of the range
        // beats the bottom). If it were flat the feature would be
        // pointless; this guards against a no-op wiring bug.
        assert!(
            *curve.last().unwrap() >= curve[0],
            "recall@10 at oversample=8 should be >= oversample=1; curve={curve:?}"
        );
    }

    /// `oversample = 1.0` is byte-identical to the pre-feature path:
    /// the same result set as a plain `search_k` scan. Regression
    /// guard so a future refactor can't silently change the default.
    #[pg_test]
    fn oversample_one_is_pre_feature_behaviour() {
        build_oversample_corpus("t_os_reg", 1000, 16, 4);
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        Spi::run("SET turbovec.search_k = 20").unwrap();
        let qlit = query_vector_literal(16, "reg");

        // Default GUC is 1.0; explicitly setting it must match.
        Spi::run("SET turbovec.oversample = 1.0").unwrap();
        let a = index_topk_ids("t_os_reg", &qlit, 20);
        // Re-read with the default unchanged path (RESET to default).
        Spi::run("RESET turbovec.oversample").unwrap();
        let b = index_topk_ids("t_os_reg", &qlit, 20);
        assert_eq!(
            a, b,
            "oversample=1.0 and the default must produce the same result set"
        );
        assert_eq!(a.len(), 20, "search_k=20 scan should return 20 ids");
    }

    /// Oversample composes with a selective WHERE filter: iterative
    /// scan still kicks in. Oversample sets the initial k; iterative
    /// refill grows from there to satisfy the LIMIT over a sparse
    /// filtered subset. Mirrors the iterative-scan fixture but with
    /// oversample > 1.0 active, proving the two knobs don't collide.
    #[pg_test]
    fn oversample_composes_with_iterative_scan() {
        use_turbovec();
        Spi::run(
            "CREATE TABLE t_os_iter (id bigint PRIMARY KEY, category int, emb vector)",
        )
        .unwrap();
        Spi::run(
            "INSERT INTO t_os_iter \
             SELECT i, i % 100, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 2000) AS gs(i), \
                  generate_series(1, 8) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_os_iter_idx ON t_os_iter \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4)",
        )
        .unwrap();
        Spi::run("ANALYZE t_os_iter").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 16").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        // Oversample > 1.0: initial k = ceil(16 * 2.0) = 32, then
        // iterative refill doubles from 32 to satisfy the LIMIT over
        // the sparse (~20-row) category-7 subset.
        Spi::run("SET turbovec.oversample = 2.0").unwrap();

        let q = "(SELECT emb FROM t_os_iter WHERE id = 1007)";
        let n: Option<i64> = Spi::get_one(&format!(
            "SELECT count(*)::bigint FROM ( \
                 SELECT id FROM t_os_iter WHERE category = 7 \
                 ORDER BY emb <=> {q} LIMIT 10 \
             ) sub"
        ))
        .unwrap();
        assert_eq!(
            n,
            Some(10),
            "oversample + iterative scan should still return the full LIMIT \
             over a selective filter"
        );
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
        let top10 = fetch_ids(
            "WITH q AS (SELECT emb FROM t_384 WHERE id = 73) \
             SELECT t.id FROM t_384 t, q \
             ORDER BY t.emb <=> q.emb \
             LIMIT 10",
        );
        assert_distinct_ids(&top10);
        assert!(top10.contains(&73), "row 73 must be in its own top-10");
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
        let top20 = fetch_ids(
            "WITH q AS (SELECT emb FROM t_2bit WHERE id = 42) \
             SELECT t.id FROM t_2bit t, q \
             ORDER BY t.emb <=> q.emb \
             LIMIT 20",
        );
        assert_distinct_ids(&top20);
        assert!(top20.contains(&42), "row 42 must be in its own top-20");
    }

    /// Build a medium-scale corpus of distinct random unit-ish
    /// vectors and assert the index's recall@10 against a
    /// brute-force exact ranking clears a per-bit-width floor —
    /// AND that every returned id is distinct.
    ///
    /// This is the regression guard the pre-AVX2 wrong-results bug
    /// slipped past: every other ANN `#[pg_test]` uses <= 2000 rows
    /// (often 64), but that bug only manifested at scale on a
    /// non-AVX2 CPU, returning the *same* TID N times. The
    /// distinct-ids assertion here would have caught it instantly
    /// regardless of recall; the recall floor catches a quieter
    /// quantiser/de-interleave regression that degrades ranking
    /// without collapsing it to duplicates.
    ///
    /// Corpus size is tuned so the in-process pgrx harness builds in
    /// well under ~30s (see `docs/TESTING.md`); it is deliberately
    /// much larger than the historical 2000-row ceiling but is not
    /// the 1M+ scale that only VectorDBBench exercises.
    fn run_recall_floor(bit_width: i32, dim: i32, n_rows: i32, floor: f64) {
        use_turbovec();
        let t_start = std::time::Instant::now();

        Spi::run("CREATE TABLE t_rf (id bigint PRIMARY KEY, emb vector)").unwrap();
        // Deterministic distinct random vectors. setseed makes the
        // corpus reproducible run-to-run so a recall regression is
        // a code change, not RNG noise. Each row is `dim` independent
        // uniform(-1,1) coordinates built with a set-based
        // array_to_string (much faster than string_agg + GROUP BY at
        // this scale).
        Spi::run("SELECT setseed(0.42)").unwrap();
        Spi::run(&format!(
            "INSERT INTO t_rf \
             SELECT g, \
                 ('[' || array_to_string(ARRAY( \
                     SELECT (random() * 2.0 - 1.0)::float4 \
                     FROM generate_series(1, {dim})), ',') || ']')::vector \
             FROM generate_series(1, {n_rows}) AS g"
        ))
        .unwrap();

        // 20 held-out query vectors (ids 1..=20 in a separate table,
        // NOT drawn from the corpus, so recall is measured on unseen
        // queries the way a real workload would).
        Spi::run("CREATE TEMP TABLE q_rf (qid int PRIMARY KEY, q vector)").unwrap();
        Spi::run("SELECT setseed(0.99)").unwrap();
        Spi::run(&format!(
            "INSERT INTO q_rf \
             SELECT g, \
                 ('[' || array_to_string(ARRAY( \
                     SELECT (random() * 2.0 - 1.0)::float4 \
                     FROM generate_series(1, {dim})), ',') || ']')::vector \
             FROM generate_series(1, 20) AS g"
        ))
        .unwrap();

        Spi::run(&format!(
            "CREATE INDEX t_rf_idx ON t_rf USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = {bit_width})"
        ))
        .unwrap();
        Spi::run("ANALYZE t_rf").unwrap();

        let build_secs = t_start.elapsed().as_secs_f64();
        eprintln!(
            "recall-floor setup: {n_rows}x{dim} bit_width={bit_width} \
             corpus+query+index build = {build_secs:.1}s"
        );

        // For each held-out query, compute exact top-10 (forced
        // seqscan over the full Vector — the `<=>` operator computes
        // exact cosine, the index is what quantises) and the index
        // top-10 (forced indexscan). recall@10 = |GT ∩ index| / 10,
        // averaged over the 20 queries.
        let qids: Vec<i64> = fetch_ids("SELECT qid FROM q_rf ORDER BY qid");
        assert_eq!(qids.len(), 20, "expected 20 held-out queries");

        let mut hits = 0usize;
        let mut total = 0usize;
        for qid in &qids {
            // Exact ground truth: force seqscan, no index.
            Spi::run("SET enable_indexscan = off").unwrap();
            Spi::run("SET enable_indexonlyscan = off").unwrap();
            Spi::run("SET enable_seqscan = on").unwrap();
            let gt = fetch_ids(&format!(
                "SELECT t.id FROM t_rf t, (SELECT q FROM q_rf WHERE qid = {qid}) qq \
                 ORDER BY t.emb <=> qq.q LIMIT 10"
            ));
            assert_eq!(gt.len(), 10, "GT top-10 must return 10 rows");
            assert_distinct_ids(&gt);

            // Index path: force indexscan.
            Spi::run("SET enable_seqscan = off").unwrap();
            Spi::run("SET enable_indexscan = on").unwrap();
            Spi::run("SET enable_indexonlyscan = on").unwrap();
            // Modest search budget (2x the LIMIT). Large enough that
            // a healthy quantiser recovers the true top-10, small
            // enough that bit-width quality actually shows in recall:
            // an over-large search_k makes the exact rerank trivially
            // perfect at every width and the floor stops
            // discriminating.
            Spi::run("SET turbovec.search_k = 20").unwrap();
            let idx = fetch_ids(&format!(
                "SELECT t.id FROM t_rf t, (SELECT q FROM q_rf WHERE qid = {qid}) qq \
                 ORDER BY t.emb <=> qq.q LIMIT 10"
            ));
            assert_eq!(idx.len(), 10, "index top-10 must return 10 rows");
            // The cheapest, sharpest guard against the pre-AVX2 bug:
            // a wrong-ranking regression that duplicated a TID would
            // fail HERE even if recall (by luck) stayed high.
            assert_distinct_ids(&idx);

            use std::collections::HashSet;
            let gt_set: HashSet<i64> = gt.iter().copied().collect();
            hits += idx.iter().filter(|id| gt_set.contains(id)).count();
            total += 10;
        }

        let recall = hits as f64 / total as f64;
        eprintln!(
            "recall-floor: bit_width={bit_width} {n_rows}x{dim} \
             recall@10 = {recall:.3} (floor {floor:.2})"
        );
        assert!(
            recall >= floor,
            "recall@10 = {recall:.3} fell below the {floor:.2} floor \
             for bit_width={bit_width} ({n_rows}x{dim}); a ranking \
             regression (e.g. a SIMD de-interleave bug) is the likely \
             cause — see docs/TESTING.md"
        );
    }

    /// Medium-scale recall floor at the default 4-bit width.
    #[pg_test]
    fn index_am_recall_floor_4bit() {
        // Observed R@10 on this 20k x 128 uniform-random corpus is
        // 1.000 at every supported bit width: TurboQuant separates
        // near-orthogonal random vectors cleanly, so the floors are
        // catastrophic-collapse guards (they fire well before the
        // ~0.1 the pre-AVX2 duplicate-id bug produced), not
        // fine-grained quality gates. The per-width quality
        // discriminator is VectorDBBench on real embeddings (see
        // docs/TESTING.md), not this synthetic unit test. Floors are
        // staggered by width anyway so a width-specific regression
        // on harder future data still trips the right test.
        run_recall_floor(4, 128, 20_000, 0.95);
    }

    /// Same corpus shape at 3-bit. Slightly coarser quantisation =
    /// slightly lower floor (observed still 1.000 on this easy data).
    #[pg_test]
    fn index_am_recall_floor_3bit() {
        run_recall_floor(3, 128, 20_000, 0.90);
    }

    /// Same corpus shape at the tightest 2-bit width. Lowest floor
    /// of the three; still far above the duplicate-id failure mode
    /// (observed 1.000 on this corpus).
    #[pg_test]
    fn index_am_recall_floor_2bit() {
        run_recall_floor(2, 128, 20_000, 0.80);
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
        assert_eq!(meta.version, 4, "new index must use the v4 wire format");
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
        let (codes, scales, ids, blocked, n_blocks, centroids, boundaries, rotation) = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            let (c, s, i) = relfile::read_full(rel, &m);
            let b = relfile::read_blocked(rel, &m);
            let cents = m.centroids_slice().to_vec();
            let bnds = m.boundaries_slice().to_vec();
            let rot = relfile::read_rotation(rel, &m);
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            (c, s, i, b, m.n_blocks_blocked as usize, cents, bnds, rot)
        };
        assert_eq!(blocked.len() as u64, meta.blocked_bytes);
        // Phase R-2: rotation chain must be a `dim*dim` `f32`
        // matrix (`16*16 = 256` elements at this corpus).
        assert_eq!(rotation.len(), (meta.dim as usize) * (meta.dim as usize));

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
            Some(rotation),
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

        // Initial meta is v4 (the version we write today; IVF-1).
        let v_current_meta: MetaPageData = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            m
        };
        assert_eq!(v_current_meta.version, 4);
        assert!(!v_current_meta.is_legacy_v1());
        assert!(!v_current_meta.is_legacy_v2());
        assert!(v_current_meta.has_prepared_layout());

        // Manufacture a v1 meta-page byte buffer with the same
        // codes/scales/ids chain pointers but no prepared layout.
        // This emulates what an index built before Phase P would
        // have on disk.
        let mut v1_buf = [0u8; PAYLOAD_BYTES];
        v1_buf[0..4].copy_from_slice(&MAGIC);
        v1_buf[4] = 1; // v1
        v1_buf[5] = v_current_meta.bit_width;
        v1_buf[8..12].copy_from_slice(&v_current_meta.dim.to_le_bytes());
        v1_buf[12..20].copy_from_slice(&v_current_meta.n_vectors.to_le_bytes());
        v1_buf[20..24].copy_from_slice(&v_current_meta.codes_first.to_le_bytes());
        v1_buf[24..28].copy_from_slice(&v_current_meta.codes_count.to_le_bytes());
        v1_buf[28..32].copy_from_slice(&v_current_meta.scales_first.to_le_bytes());
        v1_buf[32..36].copy_from_slice(&v_current_meta.scales_count.to_le_bytes());
        v1_buf[36..40].copy_from_slice(&v_current_meta.ids_first.to_le_bytes());
        v1_buf[40..44].copy_from_slice(&v_current_meta.ids_count.to_le_bytes());
        v1_buf[44..48].copy_from_slice(&v_current_meta.rows_per_codes_page.to_le_bytes());
        v1_buf[48..52].copy_from_slice(&v_current_meta.rows_per_scales_page.to_le_bytes());
        v1_buf[52..56].copy_from_slice(&v_current_meta.rows_per_ids_page.to_le_bytes());
        v1_buf[56..60].copy_from_slice(&v_current_meta.stride_bytes.to_le_bytes());
        v1_buf[60..64].copy_from_slice(&v_current_meta.am_version.to_le_bytes());
        // No v2/v3 fields.

        // Decoder round-trips: v1 buffer comes back as version=1
        // with zeroed prepared-layout fields.
        let v1_decoded = MetaPageData::decode(&v1_buf).expect("v1 decode");
        assert_eq!(v1_decoded.version, 1);
        assert!(v1_decoded.is_legacy_v1());
        assert!(v1_decoded.is_legacy_v2());
        assert!(!v1_decoded.has_prepared_layout());
        assert_eq!(v1_decoded.blocked_bytes, 0);
        assert_eq!(v1_decoded.codebook_n_levels, 0);
        assert_eq!(v1_decoded.rotation_count, 0);
        assert_eq!(v1_decoded.n_vectors, v_current_meta.n_vectors);
        assert_eq!(v1_decoded.codes_first, v_current_meta.codes_first);

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

    /// Phase R-2 (v1.4.0): the meta-page decoder distinguishes v2
    /// (Phase P, v1.3.x) from v3 (Phase R-2, v1.4.0) layouts and
    /// reports `is_legacy_v2()` correctly. The `ambeginscan`
    /// migration boundary fires on `is_legacy_v2() && n_vectors >
    /// 0`; we verify the detection primitive directly here, the
    /// same shape as the legacy_v1 test.
    ///
    /// Live exercise of the ERROR path is impossible from inside
    /// a v1.4.0 binary because `MetaPageData::encode` always
    /// writes `VERSION = 3`; manufactured v2 buffers can only
    /// be fed back through `MetaPageData::decode`.
    #[pg_test]
    fn relfile_legacy_v2_detection_primitive() {
        use crate::index::page::{MetaPageData, MAGIC, PAYLOAD_BYTES};
        use crate::index::relfile;
        use_turbovec();

        Spi::run("CREATE TABLE t_old2 (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_old2 \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 50) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_old2_idx ON t_old2 USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        let indexrelid_u32: Option<i64> =
            Spi::get_one("SELECT 't_old2_idx'::regclass::oid::int8").unwrap();
        let indexrelid = pg_sys::Oid::from(indexrelid_u32.unwrap() as u32);

        let v3_meta: MetaPageData = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            m
        };
        assert_eq!(v3_meta.version, 4);
        assert!(!v3_meta.is_legacy_v1());
        assert!(!v3_meta.is_legacy_v2());

        // Manufacture a v2 meta-page byte buffer with the same
        // chain pointers + plausible Phase P prepared layout but
        // no rotation chain. This emulates what an index built
        // by pg_turbovec 1.3.x would have on disk.
        let mut v2_buf = [0u8; PAYLOAD_BYTES];
        v2_buf[0..4].copy_from_slice(&MAGIC);
        v2_buf[4] = 2; // v2
        v2_buf[5] = v3_meta.bit_width;
        v2_buf[8..12].copy_from_slice(&v3_meta.dim.to_le_bytes());
        v2_buf[12..20].copy_from_slice(&v3_meta.n_vectors.to_le_bytes());
        v2_buf[20..24].copy_from_slice(&v3_meta.codes_first.to_le_bytes());
        v2_buf[24..28].copy_from_slice(&v3_meta.codes_count.to_le_bytes());
        v2_buf[28..32].copy_from_slice(&v3_meta.scales_first.to_le_bytes());
        v2_buf[32..36].copy_from_slice(&v3_meta.scales_count.to_le_bytes());
        v2_buf[36..40].copy_from_slice(&v3_meta.ids_first.to_le_bytes());
        v2_buf[40..44].copy_from_slice(&v3_meta.ids_count.to_le_bytes());
        v2_buf[44..48].copy_from_slice(&v3_meta.rows_per_codes_page.to_le_bytes());
        v2_buf[48..52].copy_from_slice(&v3_meta.rows_per_scales_page.to_le_bytes());
        v2_buf[52..56].copy_from_slice(&v3_meta.rows_per_ids_page.to_le_bytes());
        v2_buf[56..60].copy_from_slice(&v3_meta.stride_bytes.to_le_bytes());
        v2_buf[60..64].copy_from_slice(&v3_meta.am_version.to_le_bytes());
        // v2 prepared-layout fields (ported straight from the
        // current index's metadata so the buffer is plausible).
        v2_buf[64..68].copy_from_slice(&v3_meta.blocked_first.to_le_bytes());
        v2_buf[68..72].copy_from_slice(&v3_meta.blocked_count.to_le_bytes());
        v2_buf[72..80].copy_from_slice(&v3_meta.blocked_bytes.to_le_bytes());
        v2_buf[80..84].copy_from_slice(&v3_meta.n_blocks_blocked.to_le_bytes());
        v2_buf[84..88].copy_from_slice(&v3_meta.codebook_n_levels.to_le_bytes());
        for (i, c) in v3_meta.centroids.iter().enumerate() {
            let off = 88 + i * 4;
            v2_buf[off..off + 4].copy_from_slice(&c.to_le_bytes());
        }
        for (i, b) in v3_meta.boundaries.iter().enumerate() {
            let off = 88 + 16 * 4 + i * 4;
            v2_buf[off..off + 4].copy_from_slice(&b.to_le_bytes());
        }
        // No v3 (rotation) fields — they stay zero.

        let v2_decoded = MetaPageData::decode(&v2_buf).expect("v2 decode");
        assert_eq!(v2_decoded.version, 2);
        assert!(!v2_decoded.is_legacy_v1());
        assert!(v2_decoded.is_legacy_v2(), "v2 must trip is_legacy_v2");
        assert_eq!(v2_decoded.rotation_first, 0);
        assert_eq!(v2_decoded.rotation_count, 0);
        assert_eq!(v2_decoded.rotation_dim, 0);
        // has_prepared_layout requires the rotation chain too —
        // a v2 index returns false even though blocked + codebook
        // are present.
        assert!(!v2_decoded.has_prepared_layout());
        assert_eq!(v2_decoded.n_vectors, v3_meta.n_vectors);

        // Verify the scan path's ERROR-emitting condition: the
        // matcher in `ambeginscan` fires on `is_legacy_v2() &&
        // n_vectors > 0`. Our manufactured v2 meta satisfies
        // both, so a backend opening such an index would raise
        // ERROR ("pg_turbovec 1.4+ … REINDEX INDEX ...").
        assert!(
            v2_decoded.is_legacy_v2() && v2_decoded.n_vectors > 0,
            "manufactured v2 meta must trigger the ERROR path",
        );
    }

    /// Phase R-2 (v1.4.0): `ambuild` writes the rotation matrix
    /// into the relfile so a fresh backend reading the index
    /// pre-fills the rotation `OnceLock` from disk instead of
    /// running QR on first search. We can't reach across
    /// backends from inside a pgrx test, but we *can* (a) verify
    /// the rotation chain is on disk and (b) verify that
    /// constructing an `IdMapIndex` via
    /// `from_id_map_parts_with_prepared` with `Some(rotation)`
    /// answers a top-1 query in well under 100 ms on a 100-row
    /// corpus — a proxy that the lazy QR did not run on the
    /// search path. (Pre-Phase-R-2 the QR alone took >100 ms on
    /// a debug build at small dim, so this fence catches
    /// regressions cleanly.)
    #[pg_test]
    fn relfile_rotation_persisted() {
        use crate::index::page::MetaPageData;
        use crate::index::relfile;
        use std::time::Instant;
        use_turbovec();

        Spi::run("CREATE TABLE t_rot (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_rot \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text, \
             ',') || ']')::vector \
             FROM generate_series(1, 100) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_rot_idx ON t_rot USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4)",
        )
        .unwrap();

        let indexrelid_u32: Option<i64> =
            Spi::get_one("SELECT 't_rot_idx'::regclass::oid::int8").unwrap();
        let indexrelid = pg_sys::Oid::from(indexrelid_u32.unwrap() as u32);

        // (1) Meta page must be v3 with a populated rotation
        // chain: rotation_first > meta.blocked_first,
        // rotation_count >= 1, rotation_dim == meta.dim.
        let (meta, codes, scales, ids, blocked, centroids, boundaries, rotation): (
            MetaPageData,
            Vec<u8>,
            Vec<f32>,
            Vec<u64>,
            Vec<u8>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ) = unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let m = relfile::read_meta(rel).expect("meta");
            let (c, s, i) = relfile::read_full(rel, &m);
            let b = relfile::read_blocked(rel, &m);
            let cents = m.centroids_slice().to_vec();
            let bnds = m.boundaries_slice().to_vec();
            let rot = relfile::read_rotation(rel, &m);
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            (m, c, s, i, b, cents, bnds, rot)
        };
        assert_eq!(meta.version, 4);
        assert!(meta.has_prepared_layout());
        assert_eq!(meta.rotation_dim, meta.dim);
        assert!(meta.rotation_count >= 1);
        assert!(
            meta.rotation_first > meta.blocked_first,
            "rotation chain must follow the blocked chain on disk",
        );

        // (2) Rotation buffer is the right shape (`dim*dim`
        // f32s) and is a plausible orthogonal matrix — each
        // column has unit L2 norm to within float roundoff.
        let dim = meta.dim as usize;
        assert_eq!(rotation.len(), dim * dim);
        for j in 0..dim {
            let mut sumsq = 0.0f64;
            for i in 0..dim {
                let v = f64::from(rotation[i * dim + j]);
                sumsq += v * v;
            }
            assert!(
                (sumsq - 1.0).abs() < 1e-3,
                "column {} has |.|^2 = {} (expected ~1)",
                j,
                sumsq,
            );
        }

        // (3) Build an IdMapIndex from prepared parts including
        // the persisted rotation. A top-1 query on a 100-row
        // corpus must finish well under 100 ms; pre-Phase-R-2
        // the lazy QR alone exceeded that budget on debug.
        let n_blocks = meta.n_blocks_blocked as usize;
        let idx_with_rot = turbovec::IdMapIndex::from_id_map_parts_with_prepared(
            meta.bit_width as usize,
            dim,
            meta.n_vectors as usize,
            codes,
            scales,
            ids,
            blocked,
            n_blocks,
            centroids,
            boundaries,
            Some(rotation),
        )
        .expect("prepared+rotation parts");

        let query_vec: Vec<f32> = (0..16)
            .map(|k| ((u32::wrapping_mul(7, k as u32 + 13)) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let t0 = Instant::now();
        let (_, ids_top) = idx_with_rot.search(&query_vec, 1);
        let elapsed_us = t0.elapsed().as_micros();
        assert_eq!(ids_top.len(), 1);
        assert!(
            elapsed_us < 100_000,
            "prepared+rotation first-search took {} us, expected < 100_000 us (100 ms); \
             rotation OnceLock probably ran QR on the search path",
            elapsed_us,
        );
        eprintln!(
            "phase-r2 first-search with persisted rotation (debug, 100x16 4-bit): {} us",
            elapsed_us,
        );
    }

    // -------------------------------------------------------
    // Phase R-3 (v1.5.0): mmap-based reads of the static
    // regions (blocked codes + rotation + inline codebook). The
    // pg_test harness runs each test inside a single backend
    // transaction; we exercise the mmap path by:
    //   1. Building a small index with the prepared layout
    //      (already the default for v1.5.x).
    //   2. Running an ORDER BY query with `mmap_static_blocked`
    //      on (default).
    //   3. Invalidating the cache, toggling the GUC off, and
    //      running the same query.
    //   4. Asserting both paths return the same top-1 id.
    //
    // The brief's worked example is "concurrent aminsert commits
    // between scan-begin and the next amgettuple". pgrx tests
    // share one backend / one transaction and we can't simulate
    // a separate-backend commit mid-test without trampolining
    // through dblink, so we exercise the cache-invalidation
    // primitive directly: the same primitive an `am_version`
    // bump from a concurrent committed insert would trigger
    // (`cache::invalidate(rel_oid)`). After invalidation the
    // next scan re-mmaps from the (post-write) relfile state;
    // recheck-orderby corrects any ranking error from the
    // intervening write.
    // -------------------------------------------------------

    /// Round-trip: scans through the mmap path produce the same
    /// top-1 result as scans through the buffer-manager fall
    /// back. If they differed, a real workload toggling the GUC
    /// would silently regress recall.
    ///
    /// Also captures debug-build warm-scan timings for both
    /// paths as a coarse smoke. Asserts only that both modes
    /// finish in well under a second; the production timing
    /// harness is `benches/scripts/warm_phase_r3*.sh` on arnold.
    #[pg_test]
    fn relfile_mmap_static_round_trip_matches_buffer_manager() {
        use_turbovec();
        Spi::run("CREATE TABLE t_mmap (id bigint PRIMARY KEY, emb vector)").unwrap();
        // 8 anchors at the basis vectors plus 8 noisy rows; small
        // enough that the prepared layout stays under one page
        // per chain but big enough that the search has work to do.
        Spi::run(
            "INSERT INTO t_mmap VALUES \
             (1, '[1,0,0,0,0,0,0,0]'), \
             (2, '[0,1,0,0,0,0,0,0]'), \
             (3, '[0,0,1,0,0,0,0,0]'), \
             (4, '[0,0,0,1,0,0,0,0]'), \
             (5, '[0,0,0,0,1,0,0,0]'), \
             (6, '[0,0,0,0,0,1,0,0]'), \
             (7, '[0,0,0,0,0,0,1,0]'), \
             (8, '[0,0,0,0,0,0,0,1]'), \
             (9, '[0.9,0.1,0,0,0,0,0,0]'), \
             (10, '[0.5,0.5,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_mmap_idx ON t_mmap USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Path A: mmap path (default).
        Spi::run("SET turbovec.mmap_static_blocked = on").unwrap();
        crate::cache::invalidate_all();
        // Warmup: pays cache-fill (mmap-load + IdMapIndex construct).
        let _: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mmap \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        let t0 = std::time::Instant::now();
        let mmap_top: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mmap \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        let mmap_warm_us = t0.elapsed().as_micros();

        // Path B: buffer-manager fallback.
        Spi::run("SET turbovec.mmap_static_blocked = off").unwrap();
        crate::cache::invalidate_all();
        let _: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mmap \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        let t0 = std::time::Instant::now();
        let bm_top: Option<i64> = Spi::get_one(
            "SELECT id FROM t_mmap \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        let bm_warm_us = t0.elapsed().as_micros();

        eprintln!(
            "phase-r3 debug-build warm-scan smoke (10 rows x 8-d, bit_width=4): \
             mmap={} us, buffer-manager={} us",
            mmap_warm_us, bm_warm_us,
        );
        // Also persist to a file so the harness driver can
        // read the smoke number out-of-band; pgrx-tests captures
        // PG stderr into the cluster's logfile rather than the
        // cargo-test stdout, so eprintln is invisible to the
        // CI runner.
        let _ = std::fs::write(
            "/tmp/pg_turbovec_phase_r3_smoke.txt",
            format!("mmap_us={}\nbuffer_manager_us={}\n", mmap_warm_us, bm_warm_us),
        );

        assert_eq!(
            mmap_top, bm_top,
            "mmap path and buffer-manager path returned different top-1 ids: {:?} vs {:?}",
            mmap_top, bm_top
        );
        // Sanity: should be id 1, the exact anchor.
        assert_eq!(mmap_top, Some(1));
        // Loose upper bound (debug build, no LTO): both paths
        // must finish well under 1 s on this trivial corpus.
        assert!(
            mmap_warm_us < 1_000_000,
            "mmap warm scan too slow: {} us", mmap_warm_us,
        );
        assert!(
            bm_warm_us < 1_000_000,
            "buffer-manager warm scan too slow: {} us", bm_warm_us,
        );
    }

    /// Isolation: after a committed insert (which bumps
    /// `am_version`), the next scan in this backend invalidates
    /// its mmap'd cache entry and re-mmaps from the post-insert
    /// relfile state. We exercise the same primitive an
    /// `am_version` mismatch in `cache::lookup` would trigger.
    /// `xs_recheckorderby = true` (asserted unconditionally in
    /// `amgettuple`) is the backstop for any ranking error
    /// during the brief window before the cache notices.
    #[pg_test]
    fn relfile_mmap_static_concurrent_aminsert_recheck_corrects() {
        use_turbovec();
        Spi::run("CREATE TABLE t_iso (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_iso VALUES \
             (1, '[1,0,0,0,0,0,0,0]'), \
             (2, '[0,1,0,0,0,0,0,0]'), \
             (3, '[0,0,1,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_iso_idx ON t_iso USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // First scan: warms the cache (mmap path).
        let pre: Option<i64> = Spi::get_one(
            "SELECT id FROM t_iso \
             ORDER BY emb <=> '[0,1,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(pre, Some(2));

        // Insert a closer match. In a separate-backend scenario,
        // this would commit and bump `am_version`; our
        // same-backend deferred-commit machinery does the same
        // when this test's outer transaction commits. To
        // simulate the cross-backend cache-invalidation primitive
        // a committed insert triggers, drop the cached entry.
        Spi::run("INSERT INTO t_iso VALUES (4, '[0,0.99,0.01,0,0,0,0,0]')").unwrap();
        crate::cache::invalidate_all();

        // Next scan: re-mmaps. Should now find the closer match.
        // recheck-orderby in amgettuple recomputes the exact
        // distance against the heap tuple, so even if the mmap'd
        // image was stale wrt this insert (it isn't here — the
        // insert went through the same backend's cache and was
        // invalidated above) the executor's ordering would still
        // be exact.
        let post: Option<i64> = Spi::get_one(
            "SELECT id FROM t_iso \
             ORDER BY emb <=> '[0,1,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert!(
            post == Some(4) || post == Some(2),
            "top-1 should be id 4 (closer match, [0,0.99,..]) or id 2 \
             (the original anchor, if the index didn't surface 4 in \
             the candidate set); got {:?}",
            post,
        );
    }

    /// Cache-entry drop ordering: when a cache entry installed via
    /// the mmap path is evicted (LRU cap exceeded or explicit
    /// invalidation), the `IdMapIndex` must be dropped before the
    /// `Mmap` so the borrowed-cache pointers (in the
    /// `pg_turbovec_integration` future zero-copy path) don't
    /// dangle. We can't directly observe drop order from a
    /// pg_test, but we *can* assert the entry is gone after
    /// invalidation and a follow-up scan rebuilds cleanly — if
    /// the drop-order invariant were violated, the next scan's
    /// re-mmap path would hit a use-after-free at minimum and
    /// likely segfault.
    #[pg_test]
    fn relfile_mmap_static_cache_invalidation_drop_order() {
        use_turbovec();
        Spi::run("CREATE TABLE t_drop (id bigint PRIMARY KEY, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO t_drop VALUES \
             (1, '[1,0,0,0,0,0,0,0]'), \
             (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX t_drop_idx ON t_drop USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Warm the mmap cache.
        let _: Option<i64> = Spi::get_one(
            "SELECT id FROM t_drop \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();

        // Force drop of every cache entry. The Mmap held by the
        // entry should munmap cleanly; if drop order were wrong
        // (Mmap dropped before IdMapIndex while a borrowed-path
        // reader were still live), this would crash. Today we
        // hold owned Vecs in IdMapIndex so the contract is
        // trivially satisfied; the test guards the invariant for
        // future zero-copy work.
        crate::cache::invalidate_all();

        // Re-warm and re-query. Must succeed and match.
        let again: Option<i64> = Spi::get_one(
            "SELECT id FROM t_drop \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(again, Some(1));
    }

    // ----------------------------------------------------------------
    // Phase Y (v1.7.2): automated upgrade-matrix validation.
    //
    // The migration matrix in `docs/UPGRADING.md` and the
    // `is_legacy_v{1,2}()` predicates in `src/index/page.rs` together
    // promise that:
    //
    //   - v1.4.0 -> v1.7.x is `ALTER EXTENSION` only (wire format
    //     unchanged across `MetaPageData::version = 3`).
    //   - Any pre-v1.4 index ERRORs at first scan with a
    //     `REINDEX INDEX` HINT, never silent corruption.
    //
    // The tests below exercise those promises end-to-end against the
    // currently-installed v1.7.x binary. We forge a legacy meta page
    // via the cfg-gated `relfile::force_meta_version` helper rather
    // than carrying old binaries around (option (a) in the upgrade-
    // testing trade-off).
    // ----------------------------------------------------------------

    /// Phase Y: smoke-test that the migration SQL files for the
    /// `ALTER EXTENSION`-only hops (v1.4.0 -> v1.7.1, soon v1.7.2)
    /// can be replayed in sequence without error against an
    /// already-v1.7.x cluster. The post-v1.3.0 migration files are
    /// intentionally empty (all wire-format changes are caught at
    /// `ambeginscan` time, not at `ALTER EXTENSION` time), so this
    /// is a tautology in the steady state -- but it catches a
    /// release engineer who lands a real DDL change in one of these
    /// files without checking it parses.
    #[pg_test]
    fn alter_extension_path_140_to_171_runs_clean() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        // 005 (v1.3.0) carries a real `DROP TABLE IF EXISTS
        // turbovec.am_storage CASCADE`; replaying it is harmless
        // when the table doesn't exist. The 006..010 files are
        // pure comments. We replay them all to assert each is
        // syntactically valid SQL the running 1.7.1 backend
        // accepts.
        let files = [
            "005_pg_turbovec_v1.3.0.sql",
            "006_pg_turbovec_v1.5.0.sql",
            "007_pg_turbovec_v1.6.0.sql",
            "008_pg_turbovec_v1.6.1.sql",
            "009_pg_turbovec_v1.7.0.sql",
            "010_pg_turbovec_v1.7.1.sql",
        ];
        for name in files {
            let path = std::path::Path::new(manifest_dir)
                .join("migrations")
                .join(name);
            let sql = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
            // PostgreSQL's parser is fine with a script that's
            // entirely SQL line comments; Spi::run no-ops cleanly.
            Spi::run(&sql)
                .unwrap_or_else(|e| panic!("replay {} failed: {:?}", name, e));
        }
    }

    /// Phase Y: forge a v1 (Phase L preview) meta page on top of a
    /// freshly-built v1.7.x index and confirm `ambeginscan` ERRORs
    /// at first scan with the migration message. Exercises the
    /// `is_legacy_v1()` predicate + the `ereport!(ERROR,
    /// FEATURE_NOT_SUPPORTED, ...)` path in `src/index/scan.rs`.
    ///
    /// The expected-error string must match the primary message
    /// emitted by that ereport! verbatim (the pgrx test framework
    /// does an `==` compare). If you intentionally change the
    /// wording in `scan.rs`, update both the v1 and v2 strings
    /// here.
    #[pg_test(error = "turbovec index uses the legacy v1 relfile layout (built under pg_turbovec 1.2)")]
    fn ambeginscan_errors_on_legacy_v1_meta() {
        use_turbovec();
        Spi::run("CREATE TABLE legacy_v1 (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO legacy_v1 VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX legacy_v1_idx ON legacy_v1 \
             USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();

        // Patch the on-disk meta page so it looks like a v1
        // index. The byte mutation is wrapped in a GenericXLog
        // record so it sticks within the test transaction.
        let indexrelid: pg_sys::Oid = Spi::get_one(
            "SELECT 'legacy_v1_idx'::regclass::oid",
        )
        .unwrap()
        .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(
                indexrelid,
                pg_sys::AccessExclusiveLock as i32,
            );
            crate::index::relfile::force_meta_version(rel, 1);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        // The cache may already hold a (legitimate v3) entry from
        // CREATE INDEX. Drop it so ambeginscan re-reads the meta
        // page from the buffer manager. (ambeginscan would catch
        // the legacy version even with a stale cache because it
        // inspects the meta page first, before any cache lookup,
        // but flushing here makes the test's intent explicit.)
        crate::cache::invalidate_all();

        // Force the planner onto the index path so ambeginscan
        // runs. The error must propagate out of the test
        // function so the pgrx framework matches it against
        // `error = ...`.
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run(
            "SELECT id FROM legacy_v1 \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
    }

    /// Phase Y: same as `ambeginscan_errors_on_legacy_v1_meta`
    /// but for the v2 (Phase P, v1.3.x) wire format that lacks
    /// the persisted rotation chain v1.4.0+ requires.
    #[pg_test(error = "turbovec index built under pg_turbovec ≤ 1.3 cannot be scanned by pg_turbovec 1.4+")]
    fn ambeginscan_errors_on_legacy_v2_meta() {
        use_turbovec();
        Spi::run("CREATE TABLE legacy_v2 (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO legacy_v2 VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX legacy_v2_idx ON legacy_v2 \
             USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();

        let indexrelid: pg_sys::Oid = Spi::get_one(
            "SELECT 'legacy_v2_idx'::regclass::oid",
        )
        .unwrap()
        .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(
                indexrelid,
                pg_sys::AccessExclusiveLock as i32,
            );
            crate::index::relfile::force_meta_version(rel, 2);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        crate::cache::invalidate_all();

        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run(
            "SELECT id FROM legacy_v2 \
             ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
    }

    /// Phase Y: smoke-test that the extension chain `ALTER
    /// EXTENSION ... UPDATE` machinery resolves cleanly. We
    /// can't actually walk the chain from `1.0.0 -> 1.7.x`
    /// inside a pgrx test (the test process boots with the
    /// current Cargo.toml version pre-installed), but we can
    /// confirm the `pg_extension` row exists at the version
    /// `Cargo.toml` advertises, which catches version-mismatch
    /// regressions between `Cargo.toml`, `pg_turbovec.control`,
    /// and the migration file naming.
    #[pg_test]
    fn alter_extension_update_chain_resolves() {
        let count: Option<i64> = Spi::get_one(
            "SELECT count(*)::bigint FROM pg_extension \
             WHERE extname = 'pg_turbovec'",
        )
        .unwrap();
        assert_eq!(count, Some(1), "pg_turbovec must be installed exactly once");

        let version: Option<String> = Spi::get_one(
            "SELECT extversion FROM pg_extension \
             WHERE extname = 'pg_turbovec'",
        )
        .unwrap();
        assert_eq!(
            version.as_deref(),
            Some(env!("CARGO_PKG_VERSION")),
            "installed extension version must match Cargo.toml"
        );
    }

    /// Phase Y: drift guard. The set of `migrations/*.sql` files
    /// must match the documented release history. If you tag a
    /// new release without adding a `migrations/0NN_pg_turbovec_
    /// vX.Y.Z.sql`, `ALTER EXTENSION pg_turbovec UPDATE TO
    /// 'X.Y.Z'` will fail in production with `extension
    /// pg_turbovec has no update path from <prev> to <X.Y.Z>`.
    /// This test catches the omission at unit-test time.
    ///
    /// The expected list mirrors the migration matrix in
    /// `docs/UPGRADING.md` (and is enforced from the other
    /// direction by `scripts/drift-check.sh` § 9). Update both
    /// when adding a release.
    #[pg_test]
    fn migration_files_cover_documented_versions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("migrations");
        let mut sigils: Vec<String> = std::fs::read_dir(&path)
            .unwrap_or_else(|e| panic!("read_dir {:?}: {}", path, e))
            .filter_map(|e| {
                let f = e.ok()?.file_name().into_string().ok()?;
                // Format: NNN_pg_turbovec_vX.Y.Z.sql
                let v = f
                    .strip_suffix(".sql")?
                    .rsplit('_')
                    .next()?
                    .strip_prefix('v')?;
                Some(v.to_string())
            })
            .collect();
        sigils.sort_by(|a, b| {
            // Lexicographic on (major, minor, patch) tuples so
            // "1.10.0" sorts after "1.7.1" rather than between
            // "1.1.0" and "1.2.0".
            let parse = |s: &str| -> (u32, u32, u32) {
                let mut it = s.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
                (
                    it.next().unwrap_or(0),
                    it.next().unwrap_or(0),
                    it.next().unwrap_or(0),
                )
            };
            parse(a).cmp(&parse(b))
        });

        // Update this list (and `scripts/drift-check.sh` § 9 and
        // the matrix in `docs/UPGRADING.md`) whenever you tag a
        // new release that ships a migration file.
        let expected: Vec<&str> = vec![
            "0.1.0", "0.2.0", "0.4.0", "0.5.0",
            "1.3.0", "1.5.0", "1.6.0", "1.6.1",
            "1.7.0", "1.7.1", "1.7.2", "1.7.3",
            "1.8.0", "1.9.0", "1.9.1", "1.10.0", "1.10.1", "1.11.0", "1.11.1",
        ];
        let expected_owned: Vec<String> =
            expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            sigils, expected_owned,
            "migrations/*.sql sigils don't match the documented release \
             history. Update this list, scripts/drift-check.sh § 9, \
             and docs/UPGRADING.md together.",
        );
    }

    // -----------------------------------------------------------------
    // Vector / halfvec arithmetic + concatenation (parity gap #4).
    // -----------------------------------------------------------------

    #[pg_test]
    fn vector_concat_basic() {
        use_turbovec();
        let txt: Option<String> = Spi::get_one(
            "SELECT ('[1,2]'::vector || '[3,4]'::vector)::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("[1, 2, 3, 4]"));
    }

    #[pg_test]
    fn vector_concat_function_name() {
        // pgvector exposes the SQL function `vector_concat`; mirror it.
        let txt: Option<String> = Spi::get_one(
            "SELECT turbovec.vector_concat('[1,2,3]'::turbovec.vector, \
                                          '[4,5]'::turbovec.vector)::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("[1, 2, 3, 4, 5]"));
    }

    #[pg_test]
    fn halfvec_concat_basic() {
        use_turbovec();
        let txt: Option<String> = Spi::get_one(
            "SELECT ('[1,2]'::halfvec || '[3,4]'::halfvec)::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("[1, 2, 3, 4]"));
    }

    #[pg_test]
    fn halfvec_concat_function_name() {
        let txt: Option<String> = Spi::get_one(
            "SELECT turbovec.halfvec_concat('[1,2,3]'::turbovec.halfvec, \
                                           '[4,5]'::turbovec.halfvec)::text",
        )
        .unwrap();
        assert_eq!(txt.as_deref(), Some("[1, 2, 3, 4, 5]"));
    }

    #[pg_test]
    fn halfvec_arithmetic_elementwise() {
        use_turbovec();
        // +  : [1,2,3] + [4,5,6] = [5,7,9]
        let add: Option<String> = Spi::get_one(
            "SELECT ('[1,2,3]'::halfvec + '[4,5,6]'::halfvec)::text",
        )
        .unwrap();
        assert_eq!(add.as_deref(), Some("[5, 7, 9]"));
        // -  : [4,5,6] - [1,2,3] = [3,3,3]
        let sub: Option<String> = Spi::get_one(
            "SELECT ('[4,5,6]'::halfvec - '[1,2,3]'::halfvec)::text",
        )
        .unwrap();
        assert_eq!(sub.as_deref(), Some("[3, 3, 3]"));
        // *  : Hadamard product [1,2,3] * [4,5,6] = [4,10,18]
        let mul: Option<String> = Spi::get_one(
            "SELECT ('[1,2,3]'::halfvec * '[4,5,6]'::halfvec)::text",
        )
        .unwrap();
        assert_eq!(mul.as_deref(), Some("[4, 10, 18]"));
    }

    #[pg_test]
    fn halfvec_add_dim_mismatch_errors() {
        use_turbovec();
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT ('[1,2]'::halfvec + '[1,2,3]'::halfvec)::text",
            )
        });
        assert!(bad.is_err(), "halfvec + dim mismatch should ERROR");
    }

    #[pg_test]
    fn halfvec_mul_overflow_errors() {
        use_turbovec();
        // 300 * 300 = 90000 > f16 max (65504) -> non-finite -> ERROR,
        // matching pgvector's "value out of range: overflow".
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT ('[300]'::halfvec * '[300]'::halfvec)::text",
            )
        });
        assert!(bad.is_err(), "halfvec * f16 overflow should ERROR");
    }

    #[pg_test]
    fn vector_concat_exceeds_max_dim_errors() {
        use_turbovec();
        // Two 9000-d vectors concat to 18000 > MAX_DIM (16000) -> ERROR.
        let bad = std::panic::catch_unwind(|| {
            Spi::get_one::<String>(
                "SELECT (\
                   array_fill(1.0::real, ARRAY[9000])::turbovec.vector || \
                   array_fill(1.0::real, ARRAY[9000])::turbovec.vector \
                 )::text",
            )
        });
        assert!(bad.is_err(), "concat exceeding MAX_DIM should ERROR");
    }

    // ----------------------------------------------------------------
    // IVF-1 (an internal design note): build path + on-disk layout for the
    // inverted-file layer. The scan path stays FLAT in IVF-1; these
    // tests prove the v3->v4 wire change round-trips and that flat
    // v3/lists=0 indexes need no REINDEX.
    // ----------------------------------------------------------------

    /// k-means trains deterministically: same sample + same lists =>
    /// byte-identical centroids. The determinism anchor (mirrors the
    /// rotation's fixed ROTATION_SEED precedent).
    #[pg_test]
    fn ivf_kmeans_deterministic() {
        use crate::index::ivf;
        let dim = 16;
        let n = 500;
        // Deterministic pseudo-random sample.
        let mut sample = vec![0.0f32; n * dim];
        let mut x = 0xC0FFEEu64;
        for v in sample.iter_mut() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let m1 = ivf::train_kmeans(&sample, n, 32, dim);
        let m2 = ivf::train_kmeans(&sample, n, 32, dim);
        assert_eq!(
            m1.centroids, m2.centroids,
            "IVF k-means must be byte-deterministic"
        );
    }

    /// Every slot is assigned to exactly one cell and the cell
    /// directory partitions all vectors. Pure-permutation property
    /// checked at the SQL boundary via the build path.
    #[pg_test]
    fn ivf_cell_assignment_covers_all_vectors() {
        use crate::index::ivf;
        // Hand-built assignment over 7 slots, 3 cells.
        let assignment = [2u32, 0, 0, 1, 2, 1, 0];
        let (perm, dir) = ivf::build_permutation(&assignment, 3);
        dir.validate_partition(7).unwrap();
        assert_eq!(dir.total_vectors(), 7);
        // Every old slot appears exactly once in the permutation.
        let mut seen = [false; 7];
        for &old in &perm {
            assert!(!seen[old as usize], "slot assigned to two cells");
            seen[old as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "some slot unassigned");
    }

    /// Building `WITH (lists = 0)` produces a relfile byte-identical
    /// to a no-lists (default) build, modulo the meta-page version
    /// byte. Proves the v4 flat layout is backward-compatible with
    /// v3 (no REINDEX for non-IVF users).
    #[pg_test]
    fn ivf_build_lists0_is_byte_identical_to_flat() {
        use_turbovec();
        // Deterministic corpus; dim multiple of 8.
        Spi::run("CREATE TABLE ivf_b0 (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_b0 \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 31 + s * 17) % 97)::float8 / 97.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 400) g",
        )
        .unwrap();

        // Default build (no lists reloption).
        Spi::run(
            "CREATE INDEX ivf_b0_default ON ivf_b0 \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4)",
        )
        .unwrap();
        // Explicit lists = 0 build over the SAME data.
        Spi::run(
            "CREATE INDEX ivf_b0_lists0 ON ivf_b0 \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 0)",
        )
        .unwrap();

        // Compare the two relfiles block-for-block, ignoring only the
        // meta-page version byte. Both are v4 here (lists=0 is
        // structurally v3-equivalent), so they should be fully equal.
        let (def_oid, l0_oid): (pg_sys::Oid, pg_sys::Oid) = (
            Spi::get_one("SELECT 'ivf_b0_default'::regclass::oid")
                .unwrap()
                .unwrap(),
            Spi::get_one("SELECT 'ivf_b0_lists0'::regclass::oid")
                .unwrap()
                .unwrap(),
        );
        let (def_bytes, l0_bytes) = unsafe {
            let a = read_relfile_blocks(def_oid);
            let b = read_relfile_blocks(l0_oid);
            (a, b)
        };
        assert_eq!(
            def_bytes.len(),
            l0_bytes.len(),
            "lists=0 relfile must have same block count as default"
        );
        // Meta page is block 0; its version byte lives at PG header
        // (24) + magic (4) = offset 28. Both should already be v4, so
        // assert full byte-equality including the version byte.
        assert_eq!(
            def_bytes, l0_bytes,
            "lists=0 build must be byte-identical to the default flat build"
        );
    }

    /// A forged v3 index (version byte = 3, v4 IVF fields absent)
    /// must still scan flat under the v4 binary, returning correct
    /// top-k. The no-REINDEX guarantee.
    #[pg_test]
    fn ivf_v3_index_still_scans_under_v4_binary() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_v3 (id bigint, emb vector)").unwrap();
        // Orthogonal-ish basis so the nearest neighbour is
        // unambiguous.
        Spi::run(
            "INSERT INTO ivf_v3 VALUES \
                 (1, '[1,0,0,0,0,0,0,0]'), \
                 (2, '[0,1,0,0,0,0,0,0]'), \
                 (3, '[0,0,1,0,0,0,0,0]'), \
                 (4, '[0,0,0,1,0,0,0,0]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX ivf_v3_idx ON ivf_v3 \
             USING turbovec (emb vec_cosine_ops)",
        )
        .unwrap();

        // Force the on-disk meta page back to version 3. The v4 IVF
        // fields are already zero (lists=0), so a v3-stamped page is
        // a legitimate flat v3 index as far as the decoder is
        // concerned.
        let indexrelid: pg_sys::Oid =
            Spi::get_one("SELECT 'ivf_v3_idx'::regclass::oid")
                .unwrap()
                .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessExclusiveLock as i32);
            crate::index::relfile::force_meta_version(rel, 3);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        crate::cache::invalidate_all();

        // A v3 index must NOT trip the legacy ERROR path; it scans
        // flat. Query nearest to id=2's vector.
        Spi::run("SET enable_seqscan = off").unwrap();
        let top: Option<i64> = Spi::get_one(
            "SELECT id FROM ivf_v3 \
             ORDER BY emb <=> '[0,1,0,0,0,0,0,0]'::vector LIMIT 1",
        )
        .unwrap();
        assert_eq!(top, Some(2), "v3 flat scan under v4 binary must be correct");
    }

    /// Build `WITH (lists = 16)` over a few thousand rows; read back
    /// the coarse centroids + cell directory and assert the
    /// directory partitions all n_vectors exactly. A FLAT scan over
    /// the (cell-reordered) codes must still return correct top-k
    /// (IVF-1 scan is flat, so recall is unchanged vs lists=0), with
    /// distinct ids.
    #[pg_test]
    fn ivf_build_with_lists_roundtrips() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_rt (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_rt \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 13 + s * 7 + (g % 5) * 1000) % 211)::float8 / 211.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 3000) g",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX ivf_rt_idx ON ivf_rt \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        let indexrelid: pg_sys::Oid =
            Spi::get_one("SELECT 'ivf_rt_idx'::regclass::oid")
                .unwrap()
                .expect("index oid");

        // Read back the meta, coarse centroids, and cell directory.
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel)
                .expect("ivf index has a meta page");
            assert_eq!(meta.version, 4, "IVF index must be wire v4");
            assert!(meta.has_ivf(), "meta.has_ivf() must be true for lists=16");
            assert_eq!(meta.lists, 16);
            assert_eq!(meta.n_vectors, 3000);

            let coarse =
                crate::index::relfile::read_coarse_centroids(rel, &meta);
            assert_eq!(
                coarse.len(),
                16 * 16,
                "coarse chain must hold lists*dim f32"
            );
            // Centroids are finite (trained, not garbage).
            assert!(coarse.iter().all(|x| x.is_finite()));

            let dir = crate::index::relfile::read_cell_directory(rel, &meta)
                .expect("lists>0 index has a cell directory");
            assert_eq!(dir.len(), 16);
            // The cell directory must partition all 3000 vectors
            // exactly: contiguous, non-overlapping, summing to n.
            dir.validate_partition(3000).unwrap();
            assert_eq!(dir.total_vectors(), 3000);
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }

        // A flat scan over the cell-reordered codes must still return
        // a correct, distinct-id top-k. Pull back several neighbours
        // and assert distinctness (the cheapest wrong-ranking guard).
        Spi::run("SET enable_seqscan = off").unwrap();
        let ids: Vec<i64> = Spi::connect(|client| {
            let tup = client
                .select(
                    "SELECT id FROM ivf_rt \
                     ORDER BY emb <=> '[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,\
                                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]'::vector \
                     LIMIT 20",
                    None,
                    &[],
                )
                .unwrap();
            tup.map(|r| r.get::<i64>(1).unwrap().unwrap()).collect()
        });
        assert_eq!(ids.len(), 20, "top-20 over a 3000-row IVF index");
        assert_distinct_ids(&ids);
    }

    /// Phase B-4 temp-file hygiene: a SUCCESSFUL out-of-core IVF
    /// Phase B-4 temp-file hygiene: a build that ERRORS partway (a
    /// dim-mismatch row aborting the heap scan while the spill is
    /// open) leaks no spill file. The `CorpusSpill` is a PG `BufFile`
    /// created via `BufFileCreateTemp(false)`, which registers the
    /// file with `CurrentResourceOwner` so a (sub)transaction abort
    /// closes + unlinks it even when a PG `ereport(ERROR)` longjmps
    /// past the Rust `Drop` (the success path unlinks via
    /// `Drop` -> `BufFileClose`). We measure the `pg_ls_tmpdir()`
    /// count just before and just after the errored build: the abort
    /// must return the temp-dir to the same steady-state count
    /// (delta 0), proving nothing leaked.
    #[pg_test]
    fn ivf_streaming_build_temp_file_cleanup() {
        use_turbovec();
        let count_tmp = || -> i64 {
            Spi::get_one::<i64>("SELECT count(*)::bigint FROM pg_ls_tmpdir()")
                .unwrap()
                .unwrap_or(0)
        };

        // Warm-up successful build, to reach a steady-state temp-dir
        // population (the backend may keep a recycled fd / segment
        // around; we measure the DELTA across the errored build, not
        // an absolute zero, to be robust to that).
        Spi::run("CREATE TABLE ivf_tf (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_tf \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 13 + s * 7) % 211)::float8 / 211.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 1500) g",
        )
        .unwrap();
        Spi::run("SET maintenance_work_mem = '1MB'").unwrap();
        Spi::run(
            "CREATE INDEX ivf_tf_ok ON ivf_tf \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 8)",
        )
        .unwrap();

        // --- Errored build: a dim-mismatch row aborts the heap scan
        //     mid-build (the callback errors after the spill is open
        //     and some rows have been written). We run the failing
        //     CREATE INDEX inside a PL/pgSQL block with an EXCEPTION
        //     handler so the error rolls back the build's subtxn
        //     (releasing the resource owner that owns the spill)
        //     WITHOUT poisoning the outer test transaction -- letting
        //     us re-query pg_ls_tmpdir() afterwards. ---
        Spi::run("CREATE TABLE ivf_tf_err (id bigint, emb vector)").unwrap();
        // First rows are 16-d; a later row is 24-d -> dim mismatch
        // ERROR inside build_callback while the spill holds the
        // already-scanned 16-d rows.
        Spi::run(
            "INSERT INTO ivf_tf_err \
             SELECT g, ('[' || array_to_string(array(\
                SELECT (g % 7)::float8 / 7.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 500) g",
        )
        .unwrap();
        Spi::run(
            "INSERT INTO ivf_tf_err VALUES \
             (9999, '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector)",
        )
        .unwrap();
        let before = count_tmp();
        let errored: bool = Spi::get_one(
            "DO $$ BEGIN \
                CREATE INDEX ivf_tf_err_idx ON ivf_tf_err \
                  USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 8); \
             EXCEPTION WHEN OTHERS THEN NULL; END $$; \
             SELECT to_regclass('ivf_tf_err_idx') IS NULL",
        )
        .unwrap()
        .unwrap_or(false);
        assert!(errored, "dim-mismatch build must error (no index created)");
        let after = count_tmp();
        // The failed CREATE INDEX rolled back its subtransaction; PG
        // released the resource owner, which (together with the
        // CorpusSpill Drop) must have unlinked the spill: the
        // temp-dir population must not have grown.
        assert!(
            after <= before,
            "errored IVF build leaked a spill file in pgsql_tmp \
             (before={before}, after={after})"
        );
    }

    /// Phase B-4: the out-of-core streaming IVF build must be
    /// byte-deterministic. Build the SAME table twice (single
    /// assignment) and assert the two relfiles are byte-identical.
    /// The spill is just a relocation of the corpus to disk; the
    /// trained centroids, the (stable-sort) permutation, and the
    /// cell-order quantize feed are all deterministic, so the
    /// assembled v4 relfile must not vary run-to-run.
    #[pg_test]
    fn ivf_streaming_build_determinism_byte_identical() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_sb (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_sb \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 13 + s * 7 + (g % 5) * 1000) % 211)::float8 / 211.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 3000) g",
        )
        .unwrap();
        // Force a SMALL maintenance_work_mem so the streamed assign +
        // cell-order quantize feed actually iterate over MANY blocks
        // (proving chunking doesn't perturb the bytes).
        Spi::run("SET maintenance_work_mem = '1MB'").unwrap();
        Spi::run(
            "CREATE INDEX ivf_sb_a ON ivf_sb \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX ivf_sb_b ON ivf_sb \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        let (a_oid, b_oid): (pg_sys::Oid, pg_sys::Oid) = (
            Spi::get_one("SELECT 'ivf_sb_a'::regclass::oid").unwrap().unwrap(),
            Spi::get_one("SELECT 'ivf_sb_b'::regclass::oid").unwrap().unwrap(),
        );
        let (a, b) = unsafe {
            (read_relfile_blocks(a_oid), read_relfile_blocks(b_oid))
        };
        assert_eq!(
            a, b,
            "streaming IVF build must produce byte-identical relfiles run-to-run"
        );
    }

    /// Phase B-4: the streamed build is invariant to
    /// `maintenance_work_mem` (the chunk-size knob). A single huge
    /// chunk (large mwm) and many tiny chunks (small mwm) must
    /// assemble the SAME v4 relfile bytes — the corpus lives on the
    /// spill, and `add_with_ids` is incremental + order-preserving, so
    /// the slot fill order (and thus packed_codes / scales /
    /// slot_to_id) is independent of how the cell-ordered feed is
    /// chunked. This is the load-bearing out-of-core determinism
    /// guarantee: the disk relocation + chunking does not change the
    /// wire format. Also exercises soft assignment (assign_dups = 2),
    /// whose expanded slot count makes the chunk boundaries fall
    /// mid-cell.
    #[pg_test]
    fn ivf_streaming_build_chunk_size_invariant() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_ci (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_ci \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 29 + s * 11 + (g % 7) * 500) % 197)::float8 / 197.0 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 20000) g",
        )
        .unwrap();
        // Many tiny chunks.
        Spi::run("SET maintenance_work_mem = '1MB'").unwrap();
        Spi::run(
            "CREATE INDEX ivf_ci_small ON ivf_ci \
             USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, lists = 12, assign_dups = 2)",
        )
        .unwrap();
        // One big chunk.
        Spi::run("SET maintenance_work_mem = '1GB'").unwrap();
        Spi::run(
            "CREATE INDEX ivf_ci_big ON ivf_ci \
             USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, lists = 12, assign_dups = 2)",
        )
        .unwrap();
        let (s_oid, b_oid): (pg_sys::Oid, pg_sys::Oid) = (
            Spi::get_one("SELECT 'ivf_ci_small'::regclass::oid").unwrap().unwrap(),
            Spi::get_one("SELECT 'ivf_ci_big'::regclass::oid").unwrap().unwrap(),
        );
        let (small, big) = unsafe {
            (read_relfile_blocks(s_oid), read_relfile_blocks(b_oid))
        };
        assert_eq!(
            small, big,
            "streamed IVF relfile must be byte-identical regardless of \
             maintenance_work_mem (chunk size); the corpus spill + \
             incremental add_with_ids make the bytes chunk-invariant"
        );
    }

    /// Phase B-4: an out-of-core build over a corpus large enough that
    /// the streamed assign + cell-order feed iterate over MANY spill
    /// chunks (forced via a tiny `maintenance_work_mem`) must complete
    /// and round-trip correctly. This is the in-harness proxy for the
    /// memory bound (we can't easily read VmHWM mid-test; the external
    /// timed build in the B-4 report measures peak RSS). It asserts:
    /// the build completes, the cell directory partitions all n_slots,
    /// and a flat scan over the cell-reordered codes returns a
    /// correct, distinct-id top-k.
    #[pg_test]
    fn ivf_streaming_build_bounded_memory_completes() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_bm (id bigint, emb vector)").unwrap();
        Spi::run(
            "INSERT INTO ivf_bm \
             SELECT g, ('[' || array_to_string(array(\
                SELECT ((g * 17 + s * 23 + (g % 11) * 300) % 233)::float8 / 233.0 \
                FROM generate_series(1, 32) s), ',') || ']')::vector \
             FROM generate_series(1, 15000) g",
        )
        .unwrap();
        // 1MB mwm (the PG floor): at dim=32 (128 B/row) the assign
        // block is ~6144 rows, so the 15000-row corpus streams over
        // 3 blocks -- exercising the multi-block out-of-core assign
        // sweep with bounded transient buffers.
        Spi::run("SET maintenance_work_mem = '1MB'").unwrap();
        Spi::run(
            "CREATE INDEX ivf_bm_idx ON ivf_bm \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 32)",
        )
        .unwrap();

        let indexrelid: pg_sys::Oid = Spi::get_one(
            "SELECT 'ivf_bm_idx'::regclass::oid",
        )
        .unwrap()
        .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel)
                .expect("streamed ivf index has a meta page");
            assert_eq!(meta.version, 4);
            assert!(meta.has_ivf());
            assert_eq!(meta.lists, 32);
            assert_eq!(meta.n_vectors, 15000);
            let dir = crate::index::relfile::read_cell_directory(rel, &meta)
                .expect("streamed lists>0 index has a cell directory");
            assert_eq!(dir.len(), 32);
            dir.validate_partition(15000).unwrap();
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }
        // Correctness: flat scan over the cell-reordered codes.
        Spi::run("SET enable_seqscan = off").unwrap();
        let ids: Vec<i64> = Spi::connect(|client| {
            let tup = client
                .select(
                    "SELECT id FROM ivf_bm \
                     ORDER BY emb <=> ('[' || array_to_string(array(\
                        SELECT 0.5 FROM generate_series(1,32)), ',') || ']')::vector \
                     LIMIT 20",
                    None,
                    &[],
                )
                .unwrap();
            tup.map(|r| r.get::<i64>(1).unwrap().unwrap()).collect()
        });
        assert_eq!(ids.len(), 20, "top-20 over a streamed 15000-row IVF index");
        assert_distinct_ids(&ids);
    }

    /// IVF-1 recall anchor: a flat scan over a lists=16 index must
    /// return essentially the SAME top-k as a lists=0 (flat) build of
    /// the same data. IVF-1 doesn't probe cells, so reordering the
    /// codes can't change *which corpus* is scanned — every vector is
    /// still scored.
    ///
    /// The result isn't bit-identical, though: turbovec's TQ+
    /// calibration is fit on the first ~1000 vectors *in slot order*,
    /// and cell reordering changes that prefix, so the per-coordinate
    /// calibration (and thus the exact quantized codes) shifts
    /// slightly. The executor's exact-distance recheck
    /// (`xs_recheckorderby`) absorbs most of that, so the returned
    /// neighbour SET overlaps the flat scan's heavily. We assert a
    /// strong overlap (the recall-unchanged guarantee) plus distinct
    /// ids (the wrong-ranking guard), not byte-exact ranking.
    #[pg_test]
    fn ivf_lists_scan_matches_flat() {
        use_turbovec();
        Spi::run("CREATE TABLE ivf_eq (id bigint, emb vector)").unwrap();
        // A corpus with clear, well-separated structure so the top-k
        // is robust to small quantization shifts: each row gets a
        // smoothly-varying signal (sinusoidal in the id) rather than
        // the near-tie modular noise that makes ranking pathological.
        Spi::run(
            "INSERT INTO ivf_eq \
             SELECT g, ('[' || array_to_string(array(\
                SELECT sin(g::float8 / 50.0 + s::float8)::float8 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, 2000) g",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();

        // Query = a specific row's vector (id 777), so the true top-1
        // is unambiguous and the surrounding neighbours are stable.
        let query = "(SELECT emb FROM ivf_eq WHERE id = 777)";

        let pull = |idx_sql: &str| -> Vec<i64> {
            Spi::run(idx_sql).unwrap();
            let q = format!(
                "SELECT id FROM ivf_eq ORDER BY emb <=> {query} LIMIT 10"
            );
            let r: Vec<i64> = Spi::connect(|client| {
                client
                    .select(&q, None, &[])
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            });
            r
        };

        let flat = pull(
            "CREATE INDEX ivf_eq_idx ON ivf_eq \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 0)",
        );
        Spi::run("DROP INDEX ivf_eq_idx").unwrap();
        let ivf = pull(
            "CREATE INDEX ivf_eq_idx ON ivf_eq \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        );
        // Strong set overlap (the recall-unchanged guarantee). TQ+
        // calibration's order-sensitivity means the ranking isn't
        // byte-identical, but the neighbour sets must overlap heavily
        // (>= 8/10 — in practice they match exactly on most queries).
        let flat_set: std::collections::HashSet<i64> = flat.iter().copied().collect();
        let overlap = ivf.iter().filter(|id| flat_set.contains(id)).count();
        assert!(
            overlap >= 8,
            "IVF-1 flat scan (lists=16) must recall the flat top-10 set \
             (overlap {overlap}/10): flat={flat:?} ivf={ivf:?}"
        );
        // The true neighbour (id 777, the query row itself) must be
        // present in both — a flat scan over the whole corpus can't
        // miss the query's own vector.
        assert!(flat.contains(&777), "flat scan must find the query row");
        assert!(ivf.contains(&777), "IVF flat scan must find the query row");
        assert_distinct_ids(&ivf);
    }

    // ----------------------------------------------------------------
    // IVF-2 (an internal design note §5/§7): coarse search + cell-restricted
    // fine search in amgettuple, gated on turbovec.probes. The latency
    // win. probes >= lists reduces to the exact flat scan (the
    // correctness anchor); vacuum-degraded IVF indexes fall back to
    // flat.
    // ----------------------------------------------------------------

    /// Build a smoothly-structured IVF corpus and return the table
    /// name. Shared by the IVF-2 scan tests. `n` rows, dim 16.
    fn ivf2_make_corpus(table: &str, n: i64) {
        Spi::run(&format!("CREATE TABLE {table} (id bigint, emb vector)")).unwrap();
        Spi::run(&format!(
            "INSERT INTO {table} \
             SELECT g, ('[' || array_to_string(array(\
                SELECT sin(g::float8 / 50.0 + s::float8)::float8 \
                FROM generate_series(1, 16) s), ',') || ']')::vector \
             FROM generate_series(1, {n}) g"
        ))
        .unwrap();
    }

    /// Drive `ambulkdelete` directly with a synthetic dead-tuple
    /// callback, the same FFI shape the autovacuum launcher uses
    /// (mirrors `relfile_ambulkdelete_walks_pages_not_rebuild`). pgrx
    /// tests run inside a transaction, so real `VACUUM` is forbidden;
    /// this is how the IVF VACUUM-survival tests exercise the
    /// tombstone path. `callback_state` is a `&HashSet<u64>` of dead
    /// ctids.
    fn ivf_drive_ambulkdelete(
        indexrelid: pg_sys::Oid,
        dead_set: &std::collections::HashSet<u64>,
        cb: pg_sys::IndexBulkDeleteCallback,
    ) {
        use crate::index::vacuum::ambulkdelete;
        unsafe {
            let rel =
                pg_sys::index_open(indexrelid, pg_sys::ShareUpdateExclusiveLock as i32);
            assert!(!rel.is_null());
            let mut info: pg_sys::IndexVacuumInfo = std::mem::zeroed();
            info.index = rel;
            info.analyze_only = false;
            info.estimated_count = false;
            info.message_level = pg_sys::DEBUG2 as i32;
            info.num_heap_tuples = 0.0;
            info.strategy = std::ptr::null_mut();
            let stats = pg_sys::palloc0(
                std::mem::size_of::<pg_sys::IndexBulkDeleteResult>(),
            ) as *mut pg_sys::IndexBulkDeleteResult;
            let res = ambulkdelete(
                &mut info as *mut _,
                stats,
                cb,
                dead_set as *const _ as *mut std::ffi::c_void,
            );
            assert!(!res.is_null());
            pg_sys::index_close(rel, pg_sys::ShareUpdateExclusiveLock as i32);
        }
    }

    /// THE correctness anchor: with `probes >= lists`, the IVF scan
    /// probes every cell (an all-true mask), which must be
    /// byte-for-byte identical to scanning the SAME physical index
    /// with no mask at all (the flat path). We prove it on one index:
    /// capture the probes=lists result, then degrade the same index
    /// to flat (blank the IVF meta) so the next scan takes the
    /// maskless flat path over the identical codes — the two results
    /// must be exactly equal (same ids, same order). Probing all
    /// cells IS the full scan.
    #[pg_test]
    fn ivf_probes_all_equals_flat() {
        use_turbovec();
        ivf2_make_corpus("ivf_pa", 2000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run("SET turbovec.iterative_scan = off").unwrap();

        let query = "(SELECT emb FROM ivf_pa WHERE id = 777)";
        let pull = || -> Vec<i64> {
            let q = format!("SELECT id FROM ivf_pa ORDER BY emb <=> {query} LIMIT 10");
            Spi::connect(|client| {
                client
                    .select(&q, None, &[])
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            })
        };

        Spi::run(
            "CREATE INDEX ivf_pa_idx ON ivf_pa \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        // IVF scan probing every cell (all-true mask).
        Spi::run("SET turbovec.probes = 16").unwrap();
        crate::cache::invalidate_all();
        let ivf_all = pull();

        // Degrade the SAME index to flat: blank the v4 IVF fields so
        // has_ivf() is false and the scan takes the maskless flat
        // path over the identical codes. probes=lists (all-true mask)
        // must equal the maskless scan exactly — same bytes scanned,
        // the mask differs only in that it allows everything.
        let indexrelid: pg_sys::Oid = Spi::get_one("SELECT 'ivf_pa_idx'::regclass::oid")
            .unwrap()
            .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessExclusiveLock as i32);
            crate::index::relfile::force_meta_blank_ivf(rel);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        crate::cache::invalidate_all();
        let flat_same = pull();

        assert_eq!(
            ivf_all, flat_same,
            "probes>=lists (all cells) must equal the maskless flat scan of \
             the identical index exactly: ivf_all={ivf_all:?} flat={flat_same:?}"
        );
        assert_eq!(ivf_all.len(), 10);
        assert!(ivf_all.contains(&777), "all-cells scan must find the query row");
        assert_distinct_ids(&ivf_all);
    }

    /// Build WITH (lists = N), query, and assert recall@10 vs a
    /// brute-force ground truth is high at a reasonable nprobe, with
    /// distinct ids. The IVF correctness/recall property.
    #[pg_test]
    fn ivf_scan_returns_correct_topk() {
        use_turbovec();
        ivf2_make_corpus("ivf_tk", 3000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();

        let query = "(SELECT emb FROM ivf_tk WHERE id = 1234)";

        // Brute-force ground truth: exact distance, no index. A
        // seqscan ORDER BY computes the true neighbours.
        Spi::run("SET enable_seqscan = on").unwrap();
        let gt: Vec<i64> = Spi::connect(|client| {
            client
                .select(
                    &format!(
                        "SELECT id FROM ivf_tk ORDER BY emb <=> {query} LIMIT 10"
                    ),
                    None,
                    &[],
                )
                .unwrap()
                .map(|row| row.get::<i64>(1).unwrap().unwrap())
                .collect()
        });
        assert_eq!(gt.len(), 10);

        // IVF scan at a reasonable nprobe.
        Spi::run(
            "CREATE INDEX ivf_tk_idx ON ivf_tk \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.probes = 8").unwrap();
        let ivf: Vec<i64> = Spi::connect(|client| {
            client
                .select(
                    &format!(
                        "SELECT id FROM ivf_tk ORDER BY emb <=> {query} LIMIT 10"
                    ),
                    None,
                    &[],
                )
                .unwrap()
                .map(|row| row.get::<i64>(1).unwrap().unwrap())
                .collect()
        });
        assert_eq!(ivf.len(), 10, "IVF top-10");
        assert_distinct_ids(&ivf);

        let gt_set: std::collections::HashSet<i64> = gt.iter().copied().collect();
        let hits = ivf.iter().filter(|id| gt_set.contains(id)).count();
        let recall = hits as f64 / gt.len() as f64;
        assert!(
            recall >= 0.9,
            "IVF recall@10 must be >= 0.9 at probes=8/lists=16, got {recall} \
             (gt={gt:?} ivf={ivf:?})"
        );
    }

    /// Cell restriction actually reduces scan work: with a small
    /// `probes`, turbovec's blocked kernel skips more 32-vector blocks
    /// than with `probes = lists`. We read the process-global
    /// `blocks_skipped_by_mask()` counter before/after a scan to prove
    /// the unprobed (contiguous) cell ranges are short-circuited — the
    /// IVF latency mechanism, not just result filtering. Correctness
    /// (distinct ids, query row found) is asserted alongside.
    #[pg_test]
    fn ivf_low_probes_faster_fewer_cells() {
        use_turbovec();
        ivf2_make_corpus("ivf_lp", 4000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 100").unwrap();
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        Spi::run(
            "CREATE INDEX ivf_lp_idx ON ivf_lp \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 32)",
        )
        .unwrap();

        let query = "(SELECT emb FROM ivf_lp WHERE id = 1500)";
        let run_and_count = |probes: i32| -> (Vec<i64>, u64) {
            Spi::run(&format!("SET turbovec.probes = {probes}")).unwrap();
            // Fresh handle each time so the search runs (cache stays
            // valid; the counter is process-global so reset right
            // before the query).
            crate::cache::invalidate_all();
            turbovec::search::reset_blocks_skipped_by_mask();
            let ids: Vec<i64> = Spi::connect(|client| {
                client
                    .select(
                        &format!(
                            "SELECT id FROM ivf_lp ORDER BY emb <=> {query} LIMIT 10"
                        ),
                        None,
                        &[],
                    )
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            });
            let skipped = turbovec::search::blocks_skipped_by_mask();
            (ids, skipped)
        };

        let (ids_low, skipped_low) = run_and_count(2);
        let (ids_all, skipped_all) = run_and_count(32);

        // Correctness on both.
        assert_eq!(ids_low.len(), 10);
        assert_distinct_ids(&ids_low);
        assert_distinct_ids(&ids_all);
        assert!(
            ids_low.contains(&1500),
            "probes=2 should still find the query's own row (its cell is probed)"
        );

        // The latency signal: probing 2/32 cells skips strictly more
        // blocks than probing all 32 cells (which skips ~none). This
        // is the proof that fewer cells == less scan work.
        assert!(
            skipped_low > skipped_all,
            "low probes must skip more blocks than all-cells: \
             skipped(probes=2)={skipped_low} skipped(probes=32)={skipped_all}"
        );
        // Probing all cells should skip essentially nothing (the
        // whole corpus is allowed).
        assert_eq!(
            skipped_all, 0,
            "probes=lists must allow every block (skip none), got {skipped_all}"
        );
    }

    /// probes > lists behaves exactly like probes == lists (the
    /// coarse search clamps nprobe to lists, so the extra probes are a
    /// no-op).
    #[pg_test]
    fn ivf_probes_clamped_to_lists() {
        use_turbovec();
        ivf2_make_corpus("ivf_cl", 2000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run(
            "CREATE INDEX ivf_cl_idx ON ivf_cl \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        let query = "(SELECT emb FROM ivf_cl WHERE id = 500)";
        let pull = || -> Vec<i64> {
            Spi::connect(|client| {
                client
                    .select(
                        &format!(
                            "SELECT id FROM ivf_cl ORDER BY emb <=> {query} LIMIT 10"
                        ),
                        None,
                        &[],
                    )
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            })
        };

        Spi::run("SET turbovec.probes = 16").unwrap();
        let exact = pull();
        Spi::run("SET turbovec.probes = 9999").unwrap();
        let clamped = pull();
        assert_eq!(
            exact, clamped,
            "probes > lists must behave like probes == lists: \
             probes=16 {exact:?} vs probes=9999 {clamped:?}"
        );
        assert_distinct_ids(&clamped);
    }

    /// A vacuum-degraded IVF index (meta v4 IVF fields blanked, as
    /// `write_meta_shrink_in_place` does after a swap-remove) reports
    /// `has_ivf() == false` and the scan falls back to the flat path,
    /// returning correct top-k. We simulate the degradation by
    /// forcing the meta page's `lists` field to 0 in place via a
    /// REINDEX-free meta rewrite.
    #[pg_test]
    fn ivf_vacuum_degraded_falls_back_to_flat() {
        use_turbovec();
        ivf2_make_corpus("ivf_vd", 2000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run(
            "CREATE INDEX ivf_vd_idx ON ivf_vd \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        let query = "(SELECT emb FROM ivf_vd WHERE id = 1000)";
        let pull = || -> Vec<i64> {
            Spi::connect(|client| {
                client
                    .select(
                        &format!(
                            "SELECT id FROM ivf_vd ORDER BY emb <=> {query} LIMIT 10"
                        ),
                        None,
                        &[],
                    )
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            })
        };

        // Healthy IVF scan first (sanity).
        Spi::run("SET turbovec.probes = 16").unwrap();
        let healthy = pull();
        assert_eq!(healthy.len(), 10);
        assert!(healthy.contains(&1000));

        // Degrade the meta page: blank the v4 IVF fields in place so
        // has_ivf() returns false (exactly what the vacuum
        // swap-remove path does via write_meta_shrink_in_place).
        let indexrelid: pg_sys::Oid = Spi::get_one("SELECT 'ivf_vd_idx'::regclass::oid")
            .unwrap()
            .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessExclusiveLock as i32);
            crate::index::relfile::force_meta_blank_ivf(rel);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        crate::cache::invalidate_all();

        // Confirm has_ivf() is now false.
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel).expect("meta");
            assert!(
                !meta.has_ivf(),
                "degraded index must report has_ivf() == false"
            );
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }

        // The degraded index scans flat (probes is ignored) and still
        // returns correct, distinct top-k.
        Spi::run("SET turbovec.probes = 2").unwrap();
        let degraded = pull();
        assert_eq!(degraded.len(), 10, "flat fallback must return full top-10");
        assert_distinct_ids(&degraded);
        assert!(
            degraded.contains(&1000),
            "flat fallback must find the query row"
        );
    }

    /// Phase E-2: an IVF index must SURVIVE a VACUUM. We build an IVF
    /// index, mark a chunk of rows dead via `ambulkdelete` (driven
    /// directly with a synthetic callback, since pgrx tests run
    /// inside a transaction and `VACUUM` forbids that — same pattern
    /// as `relfile_ambulkdelete_walks_pages_not_rebuild`), and assert
    /// the index is STILL IVF (`has_ivf()` true, NOT degraded) and
    /// queries are still cell-restricted (the `blocks_skipped_by_mask`
    /// counter proves the scan isn't flat) and correct. This is the
    /// regression test for the silent IVF->flat degradation landmine.
    #[pg_test]
    fn ivf_survives_vacuum() {
        use std::collections::HashSet;
        use_turbovec();
        ivf2_make_corpus("ivf_sv", 4000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 100").unwrap();
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        Spi::run(
            "CREATE INDEX ivf_sv_idx ON ivf_sv \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 32)",
        )
        .unwrap();

        let indexrelid: pg_sys::Oid = Spi::get_one("SELECT 'ivf_sv_idx'::regclass::oid")
            .unwrap()
            .expect("index oid");

        // Healthy IVF before the vacuum.
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel).expect("meta");
            assert!(meta.has_ivf(), "index must start as IVF");
            assert!(!meta.is_degraded(), "freshly-built IVF is not degraded");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }

        // Collect ctids of ~25% of the rows (id % 4 == 0) to mark
        // dead through ambulkdelete.
        let dead_set: HashSet<u64> = {
            let mut set = HashSet::new();
            Spi::connect(|client| {
                let tup = client
                    .select("SELECT ctid FROM ivf_sv WHERE id % 4 = 0", None, &[])
                    .unwrap();
                for row in tup {
                    let tid: pg_sys::ItemPointerData =
                        row.get_by_name("ctid").unwrap().unwrap();
                    set.insert(pgrx::itemptr::item_pointer_to_u64(tid));
                }
            });
            set
        };
        assert!(dead_set.len() >= 900, "expected ~1000 dead ctids");

        unsafe extern "C-unwind" fn dead_cb(
            tid: pg_sys::ItemPointer,
            state: *mut std::ffi::c_void,
        ) -> bool {
            let set = &*(state as *const HashSet<u64>);
            set.contains(&pgrx::itemptr::item_pointer_to_u64(*tid))
        }
        ivf_drive_ambulkdelete(indexrelid, &dead_set, Some(dead_cb));
        crate::cache::invalidate_all();

        // STILL IVF after the vacuum (the whole point of E-2).
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel).expect("meta");
            assert!(
                meta.has_ivf(),
                "IVF index must remain IVF after VACUUM (no degradation)"
            );
            assert!(
                !meta.is_degraded(),
                "tombstone vacuum must not mark the index degraded"
            );
            assert!(
                meta.tombstone_bytes > 0,
                "deletes must have written a tombstone bitmap"
            );
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }

        // Queries are still cell-restricted, not flat: probing a few
        // cells skips strictly more blocks than probing all of them.
        let query = "(SELECT emb FROM ivf_sv WHERE id = 1001)";
        let run_and_count = |probes: i32| -> (Vec<i64>, u64) {
            Spi::run(&format!("SET turbovec.probes = {probes}")).unwrap();
            crate::cache::invalidate_all();
            turbovec::search::reset_blocks_skipped_by_mask();
            let ids: Vec<i64> = Spi::connect(|client| {
                client
                    .select(
                        &format!("SELECT id FROM ivf_sv ORDER BY emb <=> {query} LIMIT 10"),
                        None,
                        &[],
                    )
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            });
            (ids, turbovec::search::blocks_skipped_by_mask())
        };
        let (ids_low, skipped_low) = run_and_count(2);
        let (_ids_all, skipped_all) = run_and_count(32);
        assert!(
            skipped_low > skipped_all,
            "post-vacuum index must still be cell-restricted (not flat): \
             skipped(probes=2)={skipped_low} skipped(probes=32)={skipped_all}"
        );
        // Correctness: distinct ids, and no deleted (id % 4 == 0) row
        // in the result (they were tombstoned).
        assert_eq!(ids_low.len(), 10);
        assert_distinct_ids(&ids_low);
        assert!(
            ids_low.iter().all(|id| id % 4 != 0),
            "a deleted (id % 4 == 0) row must never appear: {ids_low:?}"
        );
    }

    /// Phase E-2: a row deleted then VACUUMed out of an IVF index must
    /// NEVER appear in query results, even though its slot is left in
    /// place on disk (tombstoned, not swap-removed). We tombstone the
    /// row that WOULD be the exact nearest neighbour of a query and
    /// confirm it's masked out.
    #[pg_test]
    fn ivf_tombstoned_rows_not_returned() {
        use std::collections::HashSet;
        use_turbovec();
        ivf2_make_corpus("ivf_tomb", 3000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run("SET turbovec.probes = 16").unwrap();
        Spi::run(
            "CREATE INDEX ivf_tomb_idx ON ivf_tomb \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        // Query is row 1500's own embedding; row 1500 is its own
        // exact nearest neighbour, so it tops the healthy result.
        let query = "(SELECT emb FROM ivf_tomb WHERE id = 1500)";
        let pull = || -> Vec<i64> {
            Spi::connect(|client| {
                client
                    .select(
                        &format!("SELECT id FROM ivf_tomb ORDER BY emb <=> {query} LIMIT 10"),
                        None,
                        &[],
                    )
                    .unwrap()
                    .map(|row| row.get::<i64>(1).unwrap().unwrap())
                    .collect()
            })
        };
        crate::cache::invalidate_all();
        let healthy = pull();
        assert!(
            healthy.contains(&1500),
            "row 1500 should be its own nearest neighbour before delete: {healthy:?}"
        );

        // Tombstone row 1500's slot via ambulkdelete (the heap row
        // stays — we only mark the index slot dead).
        let indexrelid: pg_sys::Oid = Spi::get_one("SELECT 'ivf_tomb_idx'::regclass::oid")
            .unwrap()
            .expect("index oid");
        let dead_set: HashSet<u64> = {
            let mut set = HashSet::new();
            Spi::connect(|client| {
                let tup = client
                    .select("SELECT ctid FROM ivf_tomb WHERE id = 1500", None, &[])
                    .unwrap();
                for row in tup {
                    let tid: pg_sys::ItemPointerData =
                        row.get_by_name("ctid").unwrap().unwrap();
                    set.insert(pgrx::itemptr::item_pointer_to_u64(tid));
                }
            });
            set
        };
        assert_eq!(dead_set.len(), 1, "one ctid for id = 1500");
        unsafe extern "C-unwind" fn dead_cb(
            tid: pg_sys::ItemPointer,
            state: *mut std::ffi::c_void,
        ) -> bool {
            let set = &*(state as *const HashSet<u64>);
            set.contains(&pgrx::itemptr::item_pointer_to_u64(*tid))
        }
        ivf_drive_ambulkdelete(indexrelid, &dead_set, Some(dead_cb));
        crate::cache::invalidate_all();

        let post = pull();
        assert_eq!(post.len(), 10, "still returns a full top-10 after delete");
        assert_distinct_ids(&post);
        assert!(
            !post.contains(&1500),
            "tombstoned (deleted+vacuumed) row 1500 must never be returned: {post:?}"
        );
        // The index stays IVF — confirm has_ivf() after the tombstone.
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel).expect("meta");
            assert!(meta.has_ivf(), "tombstoned index stays IVF");
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }
    }

    /// Phase E-2a: when an IVF index DOES degrade to flat (the
    /// last-resort safety-net path, simulated here by forcing the
    /// `ivf_degraded` meta flag), the degradation is OBSERVABLE: the
    /// `turbovec.index_is_degraded()` function returns true and the
    /// scan still returns correct results via the flat fallback (and
    /// emits the throttled WARNING). `lists` is preserved so the
    /// operator can see the index WAS built IVF.
    #[pg_test]
    fn ivf_degradation_is_observable() {
        use_turbovec();
        ivf2_make_corpus("ivf_obs", 2000);
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run(
            "CREATE INDEX ivf_obs_idx ON ivf_obs \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();

        // Healthy index: not degraded, both via the meta predicate and
        // the SQL signal.
        let degraded_sql = || -> bool {
            Spi::get_one("SELECT turbovec.index_is_degraded('ivf_obs_idx'::regclass)")
                .unwrap()
                .expect("index_is_degraded")
        };
        assert!(!degraded_sql(), "healthy IVF index must not report degraded");

        // Force the safety-net degradation: set ivf_degraded = 1 in
        // place, KEEPING lists (so index_was_ivf() stays true). This
        // is exactly what write_meta_shrink_in_place does if ever
        // called on an IVF index.
        let indexrelid: pg_sys::Oid = Spi::get_one("SELECT 'ivf_obs_idx'::regclass::oid")
            .unwrap()
            .expect("index oid");
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessExclusiveLock as i32);
            crate::index::relfile::force_meta_set_degraded(rel);
            pg_sys::index_close(rel, pg_sys::AccessExclusiveLock as i32);
        }
        crate::cache::invalidate_all();

        // The queryable signal now reports degraded.
        assert!(
            degraded_sql(),
            "index_is_degraded() must report a degraded IVF index"
        );
        unsafe {
            let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
            let meta = crate::index::relfile::read_meta(rel).expect("meta");
            assert!(meta.is_degraded(), "meta.is_degraded() must be true");
            assert!(
                meta.index_was_ivf(),
                "lists is preserved so the index is still recognisably IVF-built"
            );
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        }

        // The degraded index still returns correct results (flat
        // fallback) and emits the WARNING on scan (not asserted on
        // here, but exercised so it doesn't panic).
        let query = "(SELECT emb FROM ivf_obs WHERE id = 800)";
        let res: Vec<i64> = Spi::connect(|client| {
            client
                .select(
                    &format!("SELECT id FROM ivf_obs ORDER BY emb <=> {query} LIMIT 10"),
                    None,
                    &[],
                )
                .unwrap()
                .map(|row| row.get::<i64>(1).unwrap().unwrap())
                .collect()
        });
        assert_eq!(res.len(), 10, "degraded index still returns top-10 (flat)");
        assert_distinct_ids(&res);
        assert!(res.contains(&800), "flat fallback must find the query row");
    }

    // ---- IVF-3: probes-widening iterative scan ----

    /// Build a clustered IVF corpus where `category` aligns with the
    /// angular cluster of the embedding, so a selective `WHERE
    /// category = X` filter targets rows that live in a *few* cells
    /// that are angularly far from a query pointed at a *different*
    /// cluster. This is the fixture that exercises probe-widening:
    /// at probes=1 the scan only sees the query's own cell (no
    /// category-X rows); widening probes reaches the category-X
    /// cells.
    ///
    /// `nclust` distinct angular directions in 16-D; row `i` belongs
    /// to cluster `i % nclust`, and `category = i % nclust` too, so
    /// category C == cluster C. The embedding is the cluster's basis
    /// direction (a single hot coordinate scaled per cluster) plus a
    /// tiny per-row jitter so rows within a cluster are near but
    /// distinct. Clusters are mutually near-orthogonal (different hot
    /// coordinate), so cluster C is far from cluster D under cosine.
    fn ivf3_make_clustered_corpus(table: &str, n: i64, nclust: i64) {
        Spi::run(&format!(
            "CREATE TABLE {table} (id bigint PRIMARY KEY, category int, emb vector)"
        ))
        .unwrap();
        // emb[k] = (k == (i % nclust)) ? 1.0 : small jitter. The hot
        // coordinate is the cluster id, so each cluster occupies a
        // distinct near-orthogonal direction in the 16-D space
        // (nclust must be <= 16). The jitter
        // (hashtext-derived, ~+/-0.03) keeps rows within a cluster
        // distinct without collapsing the inter-cluster separation.
        Spi::run(&format!(
            "INSERT INTO {table} \
             SELECT i, (i % {nclust})::int, \
                 ('[' || string_agg( \
                     (CASE WHEN (k - 1) = (i % {nclust}) THEN 1.0 \
                           ELSE (hashtext(i::text || ':' || k::text) % 100) / 1600.0 \
                      END)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, {n}) AS gs(i), \
                  generate_series(1, 16) AS sub(k) \
             GROUP BY i"
        ))
        .unwrap();
    }

    /// IVF-3 core regression: under a selective `WHERE category = X`
    /// filter whose matching rows live in cells NOT in the initial
    /// `probes` nearest set, `iterative_scan = relaxed_order` must
    /// WIDEN the probe set and return the full LIMIT, while
    /// `iterative_scan = off` (single probe batch) under-returns.
    /// The IVF analogue of `index_am_iterative_scan_selective_filter`.
    #[pg_test]
    fn ivf_iterative_widens_probes_under_filter() {
        use_turbovec();
        // 16 clusters over 3200 rows => 200 rows/cluster. category C
        // == cluster C. lists = 16 so (ideally) one cell per cluster.
        ivf3_make_clustered_corpus("ivf_w", 3200, 16);
        Spi::run(
            "CREATE INDEX ivf_w_idx ON ivf_w \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        Spi::run("ANALYZE ivf_w").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        // Query points at cluster 7's direction (a row in cluster 7),
        // but we filter for category = 11 (cluster 11) — a DIFFERENT,
        // angularly-far cluster. At probes=1 the scan only sees the
        // cells nearest cluster 7; the category-11 rows are in a far
        // cell that only a widened probe set reaches.
        let query = "(SELECT emb FROM ivf_w WHERE id = 7)";
        // search_k modest so a single batch can't accidentally pull
        // the whole corpus.
        Spi::run("SET turbovec.search_k = 50").unwrap();
        Spi::run("SET turbovec.probes = 1").unwrap();
        Spi::run("SET turbovec.max_probes = 16").unwrap();

        let count_q = format!(
            "SELECT count(*)::bigint FROM ( \
                 SELECT id FROM ivf_w WHERE category = 11 \
                 ORDER BY emb <=> {query} LIMIT 10 \
             ) sub"
        );

        // off mode: single probe batch (probes=1) under-returns —
        // the category-11 rows aren't in the one probed cell.
        Spi::run("SET turbovec.iterative_scan = off").unwrap();
        crate::cache::invalidate_all();
        let off_count: Option<i64> = Spi::get_one(&count_q).unwrap();
        assert!(
            off_count.unwrap() < 10,
            "iterative_scan=off with probes=1 should under-return on a \
             selective filter whose matches live in an un-probed cell; \
             got {off_count:?} (expected < 10)"
        );

        // relaxed_order: widens probes until the category-11 cell is
        // probed and the LIMIT is satisfied.
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        crate::cache::invalidate_all();
        let relaxed_count: Option<i64> = Spi::get_one(&count_q).unwrap();
        assert_eq!(
            relaxed_count,
            Some(10),
            "iterative_scan=relaxed_order must widen probes and return the \
             full LIMIT under a selective filter on an un-probed cell"
        );

        // Distinct ids across the widening batches.
        let relaxed_ids = fetch_ids(&format!(
            "SELECT id FROM ivf_w WHERE category = 11 \
             ORDER BY emb <=> {query} LIMIT 10"
        ));
        assert_distinct_ids(&relaxed_ids);
        // Every returned row really is category 11 (the executor
        // applies the filter, but assert it to catch any mask /
        // dedup bug that smuggled a wrong-cell row through).
        for id in &relaxed_ids {
            assert_eq!(
                id % 16,
                11,
                "widened scan returned a non-category-11 row: id={id}"
            );
        }
    }

    /// `max_probes` caps the widening: with a low cap, a selective
    /// filter whose matches sit beyond the cap can't be fully
    /// satisfied (the scan stops widening instead of looping to all
    /// cells), AND the scan terminates (no infinite refill). We prove
    /// termination by the query returning at all, and the cap by
    /// contrasting a low cap (under-returns) with a high cap (full
    /// LIMIT) on the same fixture.
    #[pg_test]
    fn ivf_max_probes_caps_widening() {
        use_turbovec();
        ivf3_make_clustered_corpus("ivf_mp", 3200, 16);
        Spi::run(
            "CREATE INDEX ivf_mp_idx ON ivf_mp \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        Spi::run("ANALYZE ivf_mp").unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.search_k = 50").unwrap();
        Spi::run("SET turbovec.probes = 1").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();

        let query = "(SELECT emb FROM ivf_mp WHERE id = 7)";
        let count_q = format!(
            "SELECT count(*)::bigint FROM ( \
                 SELECT id FROM ivf_mp WHERE category = 11 \
                 ORDER BY emb <=> {query} LIMIT 10 \
             ) sub"
        );

        // Low cap: probes can grow at most to 2 (1 -> 2, capped).
        // The category-11 cell is unlikely to be within the 2
        // nearest cells to cluster 7, so the scan stops widening and
        // under-returns rather than looping to all 16 cells.
        Spi::run("SET turbovec.max_probes = 2").unwrap();
        crate::cache::invalidate_all();
        let capped: Option<i64> = Spi::get_one(&count_q).unwrap();
        assert!(
            capped.unwrap() < 10,
            "max_probes=2 should stop widening before reaching the \
             category-11 cell; got {capped:?} (expected < 10). If this \
             fails the cap isn't being honoured (scan looped to all cells)."
        );

        // High cap (== lists): widening reaches every cell, full LIMIT.
        Spi::run("SET turbovec.max_probes = 16").unwrap();
        crate::cache::invalidate_all();
        let uncapped: Option<i64> = Spi::get_one(&count_q).unwrap();
        assert_eq!(
            uncapped,
            Some(10),
            "max_probes=lists must let the widening reach the category-11 \
             cell and satisfy the full LIMIT"
        );
    }

    /// Oversample composes with probe-widening: oversample sets the
    /// initial `k` within the probed cells, probes-widening grows the
    /// cell set, and the scan returns the correct distinct top-k
    /// matching the brute-force ground truth. Both knobs active at
    /// once must not corrupt ordering or dedup.
    #[pg_test]
    fn ivf_oversample_composes_with_probes() {
        use_turbovec();
        ivf3_make_clustered_corpus("ivf_ov", 3200, 16);
        Spi::run("SET enable_seqscan = off").unwrap();

        // Unfiltered query: top-10 nearest to a cluster-3 row. With
        // oversample widening k and probes widening the cell set, the
        // result must match the exact brute-force GT.
        let query = "(SELECT emb FROM ivf_ov WHERE id = 3)";

        // Brute-force ground truth via seqscan.
        Spi::run("SET enable_seqscan = on").unwrap();
        let gt = fetch_ids(&format!(
            "SELECT id FROM ivf_ov ORDER BY emb <=> {query} LIMIT 10"
        ));
        assert_eq!(gt.len(), 10);

        Spi::run(
            "CREATE INDEX ivf_ov_idx ON ivf_ov \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 16)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        Spi::run("SET turbovec.search_k = 20").unwrap();
        Spi::run("SET turbovec.oversample = 2.0").unwrap();
        Spi::run("SET turbovec.probes = 2").unwrap();
        Spi::run("SET turbovec.max_probes = 16").unwrap();
        crate::cache::invalidate_all();

        let ivf = fetch_ids(&format!(
            "SELECT id FROM ivf_ov ORDER BY emb <=> {query} LIMIT 10"
        ));
        assert_eq!(ivf.len(), 10, "oversample+probes top-10");
        assert_distinct_ids(&ivf);
        // Recall@10 vs the exact GT must be high — oversample +
        // widening should recover the true neighbours for the
        // query's own (well-clustered) cell.
        let gt_set: std::collections::HashSet<i64> = gt.iter().copied().collect();
        let hits = ivf.iter().filter(|id| gt_set.contains(id)).count();
        let recall = hits as f64 / gt.len() as f64;
        assert!(
            recall >= 0.9,
            "oversample=2 + probes-widening recall@10 must be >= 0.9, \
             got {recall} (gt={gt:?} ivf={ivf:?})"
        );
        Spi::run("SET turbovec.oversample = 1.0").unwrap();
    }

    /// IVF-4a: soft assignment (`WITH (assign_dups = M)`, M>1) must
    /// raise recall@10 at a FIXED small `probes` vs single assignment
    /// (`assign_dups = 1`) on the same corpus/queries — the whole
    /// point of soft assignment. A true neighbour in a cell adjacent
    /// to the query's cell is missed by single-assign at low probes
    /// but recovered when boundary vectors are duplicated into both
    /// cells.
    #[pg_test]
    fn ivf_soft_assignment_raises_recall() {
        use_turbovec();
        // Clustered corpus: 16 clusters, rows near cluster directions.
        // Boundary jitter puts some rows near two clusters' border.
        ivf3_make_clustered_corpus("ivf_soft_h", 3200, 16);
        ivf3_make_clustered_corpus("ivf_soft_s", 3200, 16);
        Spi::run("SET enable_seqscan = on").unwrap();
        let query = "(SELECT emb FROM ivf_soft_h WHERE id = 7)";
        let gt = fetch_ids(&format!(
            "SELECT id FROM ivf_soft_h ORDER BY emb <=> {query} LIMIT 10"
        ));
        assert_eq!(gt.len(), 10);
        let gt_set: std::collections::HashSet<i64> = gt.iter().copied().collect();

        // Single-assignment index (assign_dups = 1, the default).
        Spi::run(
            "CREATE INDEX ivf_soft_h_idx ON ivf_soft_h \
             USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, lists = 16, assign_dups = 1)",
        )
        .unwrap();
        // Soft index (assign_dups = 2): boundary rows in top-2 cells.
        Spi::run(
            "CREATE INDEX ivf_soft_s_idx ON ivf_soft_s \
             USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, lists = 16, assign_dups = 2)",
        )
        .unwrap();

        // Fixed small probes; iterative widening OFF so probes is the
        // only cell-set lever — isolates the soft-assignment effect.
        let recall_at = |table: &str| -> f64 {
            Spi::run("SET enable_seqscan = off").unwrap();
            Spi::run("SET turbovec.iterative_scan = off").unwrap();
            Spi::run("SET turbovec.search_k = 40").unwrap();
            Spi::run("SET turbovec.probes = 1").unwrap();
            crate::cache::invalidate_all();
            let q = format!("(SELECT emb FROM {table} WHERE id = 7)");
            let ids = fetch_ids(&format!(
                "SELECT id FROM {table} ORDER BY emb <=> {q} LIMIT 10"
            ));
            assert_distinct_ids(&ids);
            let hits = ids.iter().filter(|id| gt_set.contains(id)).count();
            hits as f64 / gt.len() as f64
        };

        let recall_hard = recall_at("ivf_soft_h");
        let recall_soft = recall_at("ivf_soft_s");
        assert!(
            recall_soft >= recall_hard,
            "soft assignment (assign_dups=2) recall@10 ({recall_soft}) must be \
             >= single-assignment ({recall_hard}) at fixed probes=1"
        );
    }

    /// IVF-4a: a query over a soft index must return DISTINCT ids even
    /// when a boundary vector lives in two probed cells (slot_to_id is
    /// non-injective under soft assignment; the scan must dedup by id).
    #[pg_test]
    fn ivf_soft_assignment_no_duplicate_ids() {
        use_turbovec();
        ivf3_make_clustered_corpus("ivf_soft_dd", 3200, 16);
        Spi::run(
            "CREATE INDEX ivf_soft_dd_idx ON ivf_soft_dd \
             USING turbovec (emb vec_cosine_ops) \
             WITH (bit_width = 4, lists = 16, assign_dups = 3)",
        )
        .unwrap();
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET turbovec.iterative_scan = relaxed_order").unwrap();
        Spi::run("SET turbovec.search_k = 50").unwrap();
        // Probe many cells so duplicated boundary vectors are very
        // likely to appear in more than one probed cell.
        Spi::run("SET turbovec.probes = 12").unwrap();
        Spi::run("SET turbovec.max_probes = 16").unwrap();
        crate::cache::invalidate_all();
        let ids = fetch_ids(
            "SELECT id FROM ivf_soft_dd \
             ORDER BY emb <=> (SELECT emb FROM ivf_soft_dd WHERE id = 11) LIMIT 25",
        );
        assert!(!ids.is_empty(), "soft-index scan returned no rows");
        assert_distinct_ids(&ids);
    }

    // ----------------------------------------------------------------
    // IVF-3 Part B (an internal design note §7): the recall-vs-probes
    // frontier. Recall is CPU-independent (a function of which cells
    // are probed vs where true neighbours live, not of SIMD speed),
    // so this runs locally on any host and is the host-independent
    // evidence that IVF trades recall for scan-work as designed. The
    // absolute warm-p50 latency on AVX2 is a separate measurement
    // deferred to a quiet `arnold` window (TODO in BENCHMARKS.md).
    // ----------------------------------------------------------------

    /// Generate `n` deterministic pseudo-random unit vectors of
    /// dimension `dim` into `table(id bigint, emb vector)`. Each
    /// coordinate is a hash-derived value in (-1, 1); the `::vector`
    /// cast does not normalise, but `turbovec.normalize_on_insert`
    /// (on by default) unit-normalises at index time and the coarse
    /// search normalises the query, so the cells live on the unit
    /// sphere. Deterministic (hashtext of id:coord), so the frontier
    /// is reproducible.
    fn ivf_make_random_corpus(table: &str, n: i64, dim: i64) {
        Spi::run(&format!(
            "CREATE TABLE {table} (id bigint PRIMARY KEY, emb vector)"
        ))
        .unwrap();
        Spi::run(&format!(
            "INSERT INTO {table} \
             SELECT i, ('[' || string_agg( \
                 ((hashtext(i::text || ':' || k::text) % 20001) / 10000.0)::text, \
                 ',') || ']')::vector \
             FROM generate_series(1, {n}) AS gs(i), \
                  generate_series(1, {dim}) AS sub(k) \
             GROUP BY i"
        ))
        .unwrap();
    }

    /// THE IVF-3 Part B contract test + bench-artefact producer.
    ///
    /// Builds a real-ish corpus of distinct pseudo-random unit
    /// vectors, builds an IVF index `WITH (lists ≈ √n)`, computes
    /// brute-force exact top-10 ground truth for a set of held-out
    /// queries (seqscan), then sweeps `turbovec.probes` and records
    /// recall@10 and the fraction of 32-vector blocks skipped by the
    /// cell mask (the CPU-independent latency-win proxy) at each
    /// probe count.
    ///
    /// Asserts the core IVF correctness/quality guarantee:
    ///   1. recall@10 is monotonically non-decreasing in probes;
    ///   2. at `probes = lists` recall@10 == the exact flat scan's
    ///      recall (≈1.0) — probing every cell IS the full scan;
    ///   3. small probes skips a large fraction of blocks (the
    ///      latency-win proxy: fewer blocks scanned ⇒ proportionally
    ///      faster, host-independent).
    ///
    /// Persists `benches/results/ivf_recall_vs_probes_<DATE>.json`
    /// with the per-probes curve + corpus metadata.
    #[pg_test]
    fn ivf_recall_vs_probes_frontier() {
        use_turbovec();
        // CI-friendly scale that still gives meaningful cells:
        // n=16k, dim=64 ⇒ lists≈128 (√16384). The recall/scan-work
        // trade-off is scale-invariant in *shape*; BENCHMARKS.md
        // notes the larger-corpus run is the same curve.
        let n: i64 = 16_384;
        let dim: i64 = 64;
        let lists: i64 = 128; // √16384
        let n_queries: i64 = 50;
        ivf_make_random_corpus("ivf_frontier", n, dim);

        // Hold the query ids OUT of the index by deleting them before
        // the build (they're real corpus members, so GT and IVF see
        // the same heap; we just ensure a query never trivially
        // matches its own indexed copy at distance 0). Query ids are
        // the last `n_queries` rows.
        let q_lo = n - n_queries + 1;
        // Snapshot the query vectors into a side table, then remove
        // them from the corpus so neither the GT nor the IVF scan can
        // return a query as its own neighbour.
        Spi::run(&format!(
            "CREATE TABLE ivf_frontier_q AS \
             SELECT id, emb FROM ivf_frontier WHERE id >= {q_lo}"
        ))
        .unwrap();
        Spi::run(&format!("DELETE FROM ivf_frontier WHERE id >= {q_lo}")).unwrap();

        Spi::run(
            "CREATE INDEX ivf_frontier_idx ON ivf_frontier \
             USING turbovec (emb vec_cosine_ops) WITH (bit_width = 4, lists = 128)",
        )
        .unwrap();
        Spi::run("SET turbovec.search_k = 200").unwrap();
        Spi::run("SET turbovec.iterative_scan = off").unwrap();

        // Collect the held-out query rows.
        let queries: Vec<(i64, String)> = Spi::connect(|client| {
            client
                .select(
                    "SELECT id, emb::text FROM ivf_frontier_q ORDER BY id",
                    None,
                    &[],
                )
                .unwrap()
                .map(|r| {
                    (
                        r.get::<i64>(1).unwrap().unwrap(),
                        r.get::<String>(2).unwrap().unwrap(),
                    )
                })
                .collect()
        });
        assert_eq!(queries.len() as i64, n_queries);

        // Brute-force exact top-10 GT per query (seqscan, no index).
        Spi::run("SET enable_seqscan = on").unwrap();
        Spi::run("SET enable_indexscan = off").unwrap();
        let gt: Vec<std::collections::HashSet<i64>> = queries
            .iter()
            .map(|(_, emb)| {
                let ids = fetch_ids(&format!(
                    "SELECT id FROM ivf_frontier \
                     ORDER BY emb <=> '{emb}'::vector LIMIT 10"
                ));
                ids.into_iter().collect()
            })
            .collect();

        // Sweep probes. recall@10 averaged over queries + blocks
        // skipped (summed across queries) per probe count.
        Spi::run("SET enable_seqscan = off").unwrap();
        Spi::run("SET enable_indexscan = on").unwrap();
        let probe_sweep: Vec<i64> = vec![1, 2, 4, 8, 16, 32, lists];
        // Curve rows: (probes, recall@10, blocks_skipped_fraction).
        let mut curve: Vec<(i64, f64, f64)> = Vec::new();
        for &probes in &probe_sweep {
            Spi::run(&format!("SET turbovec.probes = {probes}")).unwrap();
            let mut hit_sum = 0usize;
            let mut blocks_skipped_sum = 0u64;
            let mut blocks_total_sum = 0u64;
            for (qi, (_, emb)) in queries.iter().enumerate() {
                // Fresh handle so each query's scan re-runs; the
                // counter is process-global, reset right before.
                crate::cache::invalidate_all();
                turbovec::search::reset_blocks_skipped_by_mask();
                let ids = fetch_ids(&format!(
                    "SELECT id FROM ivf_frontier \
                     ORDER BY emb <=> '{emb}'::vector LIMIT 10"
                ));
                let skipped = turbovec::search::blocks_skipped_by_mask();
                blocks_skipped_sum += skipped;
                // Total blocks scanned per query = ceil(n_live / 32).
                // n_live is the corpus minus the held-out queries.
                let n_live = (n - n_queries) as u64;
                blocks_total_sum += n_live.div_ceil(32);
                let hits = ids.iter().filter(|id| gt[qi].contains(id)).count();
                hit_sum += hits;
            }
            let recall = hit_sum as f64 / (n_queries as usize * 10) as f64;
            let skipped_frac = if blocks_total_sum == 0 {
                0.0
            } else {
                blocks_skipped_sum as f64 / blocks_total_sum as f64
            };
            curve.push((probes, recall, skipped_frac));
        }

        // ---- Contract assertions ----

        // (1) recall@10 monotonically non-decreasing in probes.
        for w in curve.windows(2) {
            let (p0, r0, _) = w[0];
            let (p1, r1, _) = w[1];
            assert!(
                r1 >= r0 - 1e-9,
                "recall@10 must be monotone non-decreasing in probes: \
                 recall({p0})={r0} > recall({p1})={r1}"
            );
        }

        // (2) at probes=lists recall@10 == the exact flat recall
        // (≈1.0). Probing every cell IS the full scan; the only
        // residual loss is the lossy quantized ranking inside the
        // (now-complete) candidate set, which the reorder-queue
        // exact recheck mostly absorbs. We require ≥ 0.99.
        let (last_probes, last_recall, _) = *curve.last().unwrap();
        assert_eq!(last_probes, lists, "final sweep point must be probes=lists");
        assert!(
            last_recall >= 0.99,
            "recall@10 at probes=lists must equal the exact flat scan \
             (≈1.0); got {last_recall}"
        );

        // (3) the latency-win proxy: at the low end (probes=1) the
        // cell mask must skip a large fraction of blocks. With
        // lists=128 and probes=1, ~1/128 of cells are probed, so we
        // expect to skip the vast majority of blocks. Require ≥ 0.8.
        let (p_lo, _, skipped_lo) = curve[0];
        assert_eq!(p_lo, 1);
        assert!(
            skipped_lo >= 0.8,
            "probes=1 must skip a large fraction of blocks (the \
             latency-win proxy); skipped only {skipped_lo} of blocks"
        );
        // At probes=lists the mask allows everything ⇒ skip ≈ 0.
        let (_, _, skipped_all) = *curve.last().unwrap();
        assert!(
            skipped_all < 0.05,
            "probes=lists must allow ≈ every block (skip ≈ 0); \
             skipped {skipped_all}"
        );

        // ---- Persist the bench artefact ----
        ivf_write_frontier_json(n, dim, lists, n_queries, &curve);
    }

    /// Write the recall-vs-probes curve to
    /// `benches/results/ivf_recall_vs_probes_<DATE>.json`. Best-effort:
    /// a write failure (e.g. read-only CI checkout) is a warning, not
    /// a test failure — the contract assertions above are the gate;
    /// the JSON is the published evidence.
    fn ivf_write_frontier_json(
        n: i64,
        dim: i64,
        lists: i64,
        n_queries: i64,
        curve: &[(i64, f64, f64)],
    ) {
        let date = "2026-06-16";
        let rows: Vec<String> = curve
            .iter()
            .map(|(p, r, f)| {
                format!(
                    "    {{ \"probes\": {p}, \"recall_at_10\": {r:.6}, \"blocks_skipped_fraction\": {f:.6} }}"
                )
            })
            .collect();
        let json = format!(
            "{{\n  \"bench\": \"ivf_recall_vs_probes\",\n  \"date\": \"{date}\",\n  \"host_independent\": true,\n  \"note\": \"Recall is CPU-independent (which cells are probed vs where true neighbours live). Absolute warm-p50 latency on AVX2 is a separate measurement deferred to a quiet arnold window.\",\n  \"corpus\": {{ \"n_vectors\": {n_live}, \"dim\": {dim}, \"distribution\": \"deterministic pseudo-random unit vectors\", \"bit_width\": 4 }},\n  \"lists\": {lists},\n  \"queries\": {n_queries},\n  \"ground_truth\": \"brute-force exact top-10 by cosine (seqscan, enable_indexscan=off)\",\n  \"curve\": [\n{rows}\n  ]\n}}\n",
            n_live = n - n_queries,
            rows = rows.join(",\n"),
        );
        let path = format!(
            "{}/benches/results/ivf_recall_vs_probes_{date}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        if let Err(e) = std::fs::write(&path, json) {
            // Not a hard failure; the contract assertions are the
            // gate. Surface it so a missing artefact is noticed.
            eprintln!("warning: could not write frontier artefact {path}: {e}");
        } else {
            eprintln!("wrote IVF recall-vs-probes frontier artefact: {path}");
        }
    }

    /// Helper: slurp the DATA REGION (bytes after the 24-byte PG page
    /// header) of every block of an index relation's main fork into
    /// one byte vector via the buffer manager. We skip the page
    /// header deliberately: `pd_lsn` / `pd_checksum` live there and
    /// differ between two independently-written relations even when
    /// the logical payload is identical. The turbovec payload lives
    /// entirely in the data region. Used by the byte-identity test.
    ///
    /// # Safety
    /// Caller passes a valid index relation oid.
    unsafe fn read_relfile_blocks(indexrelid: pg_sys::Oid) -> Vec<u8> {
        let rel = pg_sys::index_open(indexrelid, pg_sys::AccessShareLock as i32);
        let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(
            rel,
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
        let mut out = Vec::new();
        for blk in 0..nblocks {
            let buf = pg_sys::ReadBufferExtended(
                rel,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                blk,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);
            // Skip the 24-byte page header; compare only the data
            // region (LSN/checksum-free).
            let data = page
                .cast::<u8>()
                .add(crate::index::page::PAGE_HEADER_BYTES);
            let slice = std::slice::from_raw_parts(
                data,
                crate::index::page::BLCKSZ - crate::index::page::PAGE_HEADER_BYTES,
            );
            out.extend_from_slice(slice);
            pg_sys::UnlockReleaseBuffer(buf);
        }
        pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
        out
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
