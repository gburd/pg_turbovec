-- Scatter-gather PoC for pg_turbovec at 1T-scale partitioning.
-- Proves: a partitioned table + per-partition turbovec index, queried by
-- partition-wise (ORDER BY emb <=> q LIMIT k) UNION ALL then an outer
-- (ORDER BY dist LIMIT k) merge, returns the CORRECT global top-k.
-- Verified against a single-table brute-force exact top-k.

\set ON_ERROR_STOP on
SET turbovec.normalize_on_insert = on;
SET search_path TO public, turbovec;

-- ---------------------------------------------------------------------------
-- 1. Reference (unpartitioned) table: the ground-truth corpus.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS ref_docs CASCADE;
CREATE TABLE ref_docs (
    id  bigint PRIMARY KEY,
    emb turbovec.vector
);

-- 3000 deterministic pseudo-random 64-d unit vectors.
-- Correlated inner generate_series (per AGENTS.md graph-test bug note:
-- an uncorrelated random() subquery gets hoisted -> identical rows).
INSERT INTO ref_docs (id, emb)
SELECT g,
       (SELECT ('[' || string_agg((sin(g*0.7 + d*1.3) + cos(g*d*0.011))::text, ',') || ']')::turbovec.vector
        FROM generate_series(1, 64) d)
FROM generate_series(1, 3000) g;

-- ---------------------------------------------------------------------------
-- 2. Partitioned table: hash into 4 partitions, per-partition turbovec index.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS part_docs CASCADE;
CREATE TABLE part_docs (
    id  bigint,
    emb turbovec.vector
) PARTITION BY HASH (id);

CREATE TABLE part_docs_0 PARTITION OF part_docs FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE part_docs_1 PARTITION OF part_docs FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE part_docs_2 PARTITION OF part_docs FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE part_docs_3 PARTITION OF part_docs FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- Same corpus, routed by hash(id) into the 4 partitions.
INSERT INTO part_docs SELECT id, emb FROM ref_docs;

-- Per-partition turbovec index (flat, 4-bit). Each is independent.
CREATE INDEX ON part_docs_0 USING turbovec (emb turbovec.vec_cosine_ops) WITH (bit_width = 4);
CREATE INDEX ON part_docs_1 USING turbovec (emb turbovec.vec_cosine_ops) WITH (bit_width = 4);
CREATE INDEX ON part_docs_2 USING turbovec (emb turbovec.vec_cosine_ops) WITH (bit_width = 4);
CREATE INDEX ON part_docs_3 USING turbovec (emb turbovec.vec_cosine_ops) WITH (bit_width = 4);

-- Confirm the rows landed spread across partitions (not all in one).
SELECT 'partition row counts' AS check,
       (SELECT count(*) FROM part_docs_0) AS p0,
       (SELECT count(*) FROM part_docs_1) AS p1,
       (SELECT count(*) FROM part_docs_2) AS p2,
       (SELECT count(*) FROM part_docs_3) AS p3,
       (SELECT count(*) FROM part_docs)   AS total;

-- ---------------------------------------------------------------------------
-- 3. Pick a query vector = the embedding of an existing row (id=1234), so the
--    exact top-1 is deterministic (id 1234 itself, dist 0).
-- ---------------------------------------------------------------------------
\set k 10
SELECT emb AS qemb FROM ref_docs WHERE id = 1234 \gset

-- Ground truth: exact global top-k by cosine over the WHOLE reference table.
SET turbovec.search_k = 3000;   -- force effectively-exact candidate set
SELECT 'GROUND TRUTH (single-table exact top-k)' AS label;
SELECT id, round((emb <=> :'qemb'::turbovec.vector)::numeric, 6) AS dist
FROM ref_docs
ORDER BY emb <=> :'qemb'::turbovec.vector
LIMIT :k;

-- ---------------------------------------------------------------------------
-- 4. SCATTER-GATHER: per-partition ORDER BY..LIMIT k, UNION ALL, outer merge.
--    This is the pattern a query would fan out across N partitions.
-- ---------------------------------------------------------------------------
SET turbovec.search_k = 100;
SELECT 'SCATTER-GATHER (per-partition top-k UNION ALL -> outer top-k)' AS label;
WITH per_partition AS (
    (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_0 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
    UNION ALL
    (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_1 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
    UNION ALL
    (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_2 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
    UNION ALL
    (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_3 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
)
SELECT id, round(dist::numeric, 6) AS dist
FROM per_partition
ORDER BY dist
LIMIT :k;

-- ---------------------------------------------------------------------------
-- 5. VERDICT: do the two top-k ID SETS match exactly?
-- ---------------------------------------------------------------------------
SET turbovec.search_k = 100;
WITH gt AS (
    SELECT id FROM ref_docs ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k
),
sg AS (
    WITH per_partition AS (
        (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_0 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
        UNION ALL
        (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_1 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
        UNION ALL
        (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_2 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
        UNION ALL
        (SELECT id, emb <=> :'qemb'::turbovec.vector AS dist FROM part_docs_3 ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k)
    )
    SELECT id FROM per_partition ORDER BY dist LIMIT :k
)
SELECT
    (SELECT count(*) FROM gt) AS gt_n,
    (SELECT count(*) FROM sg) AS sg_n,
    (SELECT count(*) FROM (SELECT id FROM gt INTERSECT SELECT id FROM sg) x) AS overlap,
    CASE WHEN (SELECT count(*) FROM (SELECT id FROM gt INTERSECT SELECT id FROM sg) x) = :k
         THEN 'PASS: scatter-gather top-k == exact global top-k'
         ELSE 'FAIL: mismatch' END AS verdict;

-- ---------------------------------------------------------------------------
-- 6. Show PG's NATIVE partition-wise plan over the parent (Append of 4
--    per-partition index scans) for the same query -- proves the planner
--    already fans out ORDER BY..LIMIT across partitions without new code.
-- ---------------------------------------------------------------------------
SET enable_partitionwise_join = on;
SELECT 'NATIVE parent-table plan (Append over per-partition index scans)' AS label;
EXPLAIN (COSTS OFF)
SELECT id FROM part_docs ORDER BY emb <=> :'qemb'::turbovec.vector LIMIT :k;

-- Native parent query result, for comparison with scatter-gather + GT.
SELECT 'NATIVE parent-table result' AS label;
SELECT id, round((emb <=> :'qemb'::turbovec.vector)::numeric, 6) AS dist
FROM part_docs
ORDER BY emb <=> :'qemb'::turbovec.vector
LIMIT :k;
