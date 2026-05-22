-- Phase 4+ exercise of the `turbovec` index access method.
-- Run with `cargo pgrx run pg16` then `\i tests/02_index_am.sql`.

\echo === Phase 4: turbovec index AM ===
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = turbovec, public;

BEGIN;

-- 1. Create + populate a small corpus.
CREATE TEMP TABLE ix_demo (id bigint PRIMARY KEY, emb tvector);
INSERT INTO ix_demo VALUES
  (1, '[1, 0, 0, 0, 0, 0, 0, 0]'),
  (2, '[0.9, 0.1, 0, 0, 0, 0, 0, 0]'),
  (3, '[0, 1, 0, 0, 0, 0, 0, 0]'),
  (4, '[-1, 0, 0, 0, 0, 0, 0, 0]');

-- 2. Build the index.
CREATE INDEX ix_demo_idx
  ON ix_demo USING turbovec (emb tvector_cosine_ops)
  WITH (bit_width = 4);

-- 3. The side-table row exists.
SELECT bit_width, dim, n_vectors
FROM   turbovec.am_storage
WHERE  indexrelid = 'ix_demo_idx'::regclass;

-- 4. Top-1 nearest \u2014 must be row 1 (the self-vector).
SELECT id, emb <=> '[1,0,0,0,0,0,0,0]'::tvector AS dist
FROM   ix_demo
ORDER  BY emb <=> '[1,0,0,0,0,0,0,0]'::tvector
LIMIT  3;

-- 5. EXPLAIN: the planner picks the index.
EXPLAIN (COSTS OFF)
SELECT id FROM ix_demo
ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::tvector
LIMIT 3;

-- 6. INSERT after CREATE INDEX exercises aminsert.
INSERT INTO ix_demo VALUES (5, '[0.95, 0.05, 0, 0, 0, 0, 0, 0]');
SELECT n_vectors FROM turbovec.am_storage WHERE indexrelid = 'ix_demo_idx'::regclass;
SELECT id FROM ix_demo
ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::tvector
LIMIT 1;
-- Expected: 1 (still the closest), 5 should be in top-3.

-- 7. REINDEX rebuilds.
REINDEX INDEX ix_demo_idx;
SELECT n_vectors FROM turbovec.am_storage WHERE indexrelid = 'ix_demo_idx'::regclass;

-- 8. DROP INDEX cleans up the heap is intact.
DROP INDEX ix_demo_idx;
SELECT count(*) FROM ix_demo;

ROLLBACK;

\echo === Phase 8: filtered turbovec.knn() ===

BEGIN;
CREATE TEMP TABLE filt_demo (id bigint PRIMARY KEY, emb tvector, tag text);
INSERT INTO filt_demo VALUES
  (1, '[1, 0, 0, 0, 0, 0, 0, 0]', 'A'),
  (2, '[0.9, 0.1, 0, 0, 0, 0, 0, 0]', 'B'),
  (3, '[0, 1, 0, 0, 0, 0, 0, 0]', 'A'),
  (4, '[-1, 0, 0, 0, 0, 0, 0, 0]', 'B');

-- Hybrid retrieval pattern: use SQL to narrow to candidates by tag,
-- then push the resulting id-set into the SIMD kernel as an
-- allowlist.
SELECT k.id, k.score
FROM   turbovec.knn(
         'filt_demo'::regclass, 'id', 'emb',
         '[1, 0, 0, 0, 0, 0, 0, 0]'::tvector,
         5, 4,
         ARRAY(SELECT id FROM filt_demo WHERE tag = 'A')::bigint[]
       ) k
ORDER BY k.score DESC;
-- Expected: row 1 first, row 3 second, row 2/4 absent.

ROLLBACK;
