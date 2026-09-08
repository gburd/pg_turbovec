-- Plumbing smoke test for benches/scripts/bq/bq_frontier.py.
--
-- THIS MEASURES NOTHING. It contains no turbovec index, no quantization, and
-- no SIMD kernel. Its only job is to prove the DRIVER executes: that the
-- ground-truth LATERAL query parses, the EXPLAIN (ANALYZE) Execution Time
-- parse works, recall is computed, the iso-recall derivation runs, and the
-- JSON artefact is written in the archive shape. Any number it produces is a
-- property of this stub's SQL cosine function, not of pg_turbovec.
--
-- It fakes exactly the surface the driver touches:
--   * a `turbovec` schema holding a `vector` domain over float8[]
--   * a `<=>` operator (cosine distance in plain SQL)
--   * a tiny synthetic corpus + query set
-- and is driven with --skip-build, so no CREATE INDEX is attempted. The
-- driver will correctly WARN in every row's `plan` block that the plan is a
-- seq scan and not an Index Scan -- that warning firing IS part of what this
-- test proves.
--
-- Run (own cluster, non-default port, never the shared pgrx one):
--   initdb -D /tmp/bqsmoke -U bq -A trust
--   pg_ctl -D /tmp/bqsmoke -o "-p 54329 -k /tmp -c listen_addresses=''" start
--   psql -h /tmp -p 54329 -U bq -d postgres -f smoke_stub.sql
--   python3 bq_frontier.py --dsn "host=/tmp port=54329 user=bq dbname=postgres" \
--       --table docs --vec-expr emb --dim 16 --query-provenance held_out \
--       --bit-widths 1 --search-k-sweep 10,50 --k 10 --k-deep 10 \
--       --gt-depth 10 --n-warm 1 --skip-build --out /tmp/bq_smoke.json
--   pg_ctl -D /tmp/bqsmoke stop -m fast      # NEVER kill -9 a postmaster

CREATE SCHEMA IF NOT EXISTS turbovec;

CREATE DOMAIN turbovec.vector AS float8[];

CREATE FUNCTION turbovec.cosine_distance(a float8[], b float8[])
RETURNS float8 LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT 1.0 - (
        SELECT sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))
        FROM unnest(a, b) AS t(x, y)
    )
$$;

CREATE OPERATOR turbovec.<=> (
    LEFTARG = float8[], RIGHTARG = float8[],
    FUNCTION = turbovec.cosine_distance
);

-- Deterministic pseudo-random corpus, 16-d, mixed signs so the sign bits a
-- real BQ index would take are not degenerate. Correlated generate_series
-- (the v1.24.0 test-harness lesson: an uncorrelated random() subquery gets
-- hoisted and every row comes out identical).
DROP TABLE IF EXISTS docs;
CREATE TABLE docs (id bigint PRIMARY KEY, emb turbovec.vector NOT NULL);
INSERT INTO docs
SELECT g,
       (SELECT array_agg((((g * 7919 + s * 104729) % 2000)::float8 / 1000.0) - 1.0
                         ORDER BY s)
        FROM generate_series(1, 16) s)::turbovec.vector
FROM generate_series(1, 2000) g;

-- Held-out query source: same generator, disjoint ids.
DROP TABLE IF EXISTS docs_heldout;
CREATE TABLE docs_heldout (id bigint PRIMARY KEY, emb turbovec.vector NOT NULL);
INSERT INTO docs_heldout
SELECT g,
       (SELECT array_agg((((g * 6551 + s * 65537) % 2000)::float8 / 1000.0) - 1.0
                         ORDER BY s)
        FROM generate_series(1, 16) s)::turbovec.vector
FROM generate_series(100001, 100020) g;

DROP TABLE IF EXISTS bq_query_set;
CREATE TABLE bq_query_set AS
SELECT row_number() OVER (ORDER BY id)::int AS qid, emb AS qvec
FROM docs_heldout ORDER BY id;
CREATE INDEX ON bq_query_set (qid);

-- A stub index NAMED as the driver expects (bqbench_b<bw>), so
-- pg_relation_size / --skip-build have something to report. It is a plain
-- btree on `id` -- it does NOT and cannot serve the `<=>` ORDER BY, which is
-- exactly why the driver's plan check must WARN "not an Index Scan" on every
-- row of this smoke run. That warning firing is part of the test.
DROP INDEX IF EXISTS bqbench_b1;
CREATE INDEX bqbench_b1 ON docs (id);

-- Sanity: the corpus must not be degenerate (the footgun a real BQ build
-- rejects). n_distinct on the first coordinate stands in for it.
SELECT count(DISTINCT emb[1]) AS distinct_first_coord, count(*) AS rows FROM docs;
