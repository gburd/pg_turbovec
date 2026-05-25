-- bench/sql/phase_n_b_crash_recovery.sql
--
-- Phase L hardening (items 1, 2, 3) — manual crash-recovery
-- verification harness. The pgrx-test harness can't perform a real
-- immediate-shutdown crash inside a single test (it restarts the
-- cluster between tests), so this script exercises the WAL replay
-- path on a real running cluster.
--
-- Usage:
--   1. Build with the relfile feature on:
--        cargo pgrx install --release \
--            --no-default-features \
--            --features "pg16 experimental_index_am relfile_storage"
--   2. Start a cluster:
--        cargo pgrx run pg16
--   3. Phase A — populate, BEFORE crash:
--        psql -h /tmp -p 28816 postgres -f phase_n_b_crash_recovery.sql
--   4. Crash the cluster:
--        pg_ctl -D ~/.pgrx/data-16 stop -m immediate
--   5. Restart the cluster:
--        pg_ctl -D ~/.pgrx/data-16 start
--   6. Phase B — verify, AFTER crash:
--        psql -h /tmp -p 28816 postgres \
--            -c "SET enable_seqscan = off;" \
--            -c "SELECT id FROM phase_n_b_corpus
--                 ORDER BY emb <=> (SELECT emb FROM phase_n_b_corpus
--                                   WHERE id = 73) LIMIT 1;"
--      The expected answer is 73 (self-query); a different answer
--      or empty result means the relfile pages were lost on crash
--      and Phase L hardening item 1 is regressed.
--
-- Coverage:
--   * `ambuild` WAL emission         — verified by phase A populate
--                                      surviving the crash.
--   * `aminsert` WAL emission        — verified by the post-build
--                                      INSERT in phase A surviving.
--   * `ambulkdelete` + RelationTruncate
--                                    — verified by the VACUUM in
--                                      phase A, after which the
--                                      index file should be smaller
--                                      and still queryable.
--   * `ambuildempty` (init fork)     — verified by the unlogged
--                                      table at the bottom: after
--                                      crash, PG must reset the
--                                      unlogged relation from its
--                                      init fork and the index
--                                      must still respond (with
--                                      0 rows; the heap is
--                                      truncated by recovery).

-- ---- Phase A: populate before crash ----
CREATE EXTENSION IF NOT EXISTS pg_turbovec;

DROP TABLE IF EXISTS phase_n_b_corpus CASCADE;
CREATE TABLE phase_n_b_corpus (id bigint PRIMARY KEY, emb vector);

INSERT INTO phase_n_b_corpus
SELECT i,
       ('[' || string_agg(
           ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text,
       ',') || ']')::vector
FROM generate_series(1, 1000) AS gs(i),
     generate_series(1, 64)   AS sub(k)
GROUP BY i;

CREATE INDEX phase_n_b_idx
    ON phase_n_b_corpus USING turbovec (emb vec_cosine_ops)
    WITH (bit_width = 4);

-- aminsert path (a row inserted post-build).
INSERT INTO phase_n_b_corpus
SELECT 9999, ('[' || string_agg(
    ((hashtext('z:' || k::text) % 2000) / 1000.0 - 1)::text,
',') || ']')::vector
FROM generate_series(1, 64) AS sub(k);

-- Force a checkpoint before the test row but not after, so the
-- post-checkpoint changes (the DELETE + VACUUM below) only survive
-- via WAL replay.
CHECKPOINT;

-- ambulkdelete + RelationTruncate path.
DELETE FROM phase_n_b_corpus WHERE id <= 200;
VACUUM phase_n_b_corpus;

-- Record the index size pre-crash so we can compare post-crash.
SELECT pg_relation_size('phase_n_b_idx'::regclass) AS bytes_before_crash
\gset

-- Sanity: pre-crash query must work.
SET enable_seqscan = off;
SELECT id AS expected_after_crash
FROM phase_n_b_corpus
ORDER BY emb <=> (SELECT emb FROM phase_n_b_corpus WHERE id = 500)
LIMIT 1
\gset

-- ---- Phase A2: unlogged table for ambuildempty / init fork test ----

DROP TABLE IF EXISTS phase_n_b_unlogged CASCADE;
CREATE UNLOGGED TABLE phase_n_b_unlogged (id bigint PRIMARY KEY, emb vector);

INSERT INTO phase_n_b_unlogged
SELECT i,
       ('[' || string_agg(
           ((hashtext('u:' || i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text,
       ',') || ']')::vector
FROM generate_series(1, 100) AS gs(i),
     generate_series(1, 16)  AS sub(k)
GROUP BY i;

CREATE INDEX phase_n_b_unlogged_idx
    ON phase_n_b_unlogged USING turbovec (emb vec_cosine_ops)
    WITH (bit_width = 4);

-- Init fork must be populated for the post-crash recovery copy.
SELECT pg_relation_size('phase_n_b_unlogged_idx'::regclass, 'init') > 0
    AS init_fork_populated;

\echo
\echo '=== Phase A complete. Now: ==='
\echo '  pg_ctl -D ~/.pgrx/data-16 stop -m immediate'
\echo '  pg_ctl -D ~/.pgrx/data-16 start'
\echo '  psql ... -f phase_n_b_crash_recovery_verify.sql'
\echo

-- ---- Phase B: verification post-crash ----
-- (Run this part after the crash + restart. Lives in the same
-- file but \gexec'd verification is in this trailing block; the
-- pre-crash gset/echo above prints the expected values.)
--
-- After restart, run:
--   SET enable_seqscan = off;
--   SELECT id FROM phase_n_b_corpus
--    ORDER BY emb <=> (SELECT emb FROM phase_n_b_corpus WHERE id = 500)
--    LIMIT 1;
--   -- expected: same as :expected_after_crash printed above
--
--   SELECT pg_relation_size('phase_n_b_idx'::regclass);
--   -- expected: == :bytes_before_crash printed above (truncated
--   --           pages don't come back via WAL replay)
--
--   SELECT count(*) FROM phase_n_b_unlogged;
--   -- expected: 0 (unlogged table reset to init fork)
--
--   -- Querying the unlogged index must succeed (returns 0 rows
--   -- since the heap was reset). This is the actual init-fork
--   -- recovery test \u2014 if ambuildempty didn't write the meta
--   -- page, the index would be empty (0 blocks) post-recovery
--   -- and our amgettuple would error.
--   SELECT id FROM phase_n_b_unlogged
--    ORDER BY emb <=> '[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
--    LIMIT 1;
