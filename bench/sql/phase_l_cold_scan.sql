-- Phase L cold-scan measurement: 2000-row, 384-dim corpus,
-- side-table path vs relfile-resident path.
--
-- Run with:
--   # install both feature configurations into separate clusters
--   cargo pgrx install --no-default-features --features "pg16 experimental_index_am" --release
--   cargo pgrx run pg16 -- -c "\\i bench/sql/phase_l_cold_scan.sql"
--   # then under relfile:
--   cargo pgrx install --no-default-features --features "pg16 experimental_index_am relfile_storage" --release
--   cargo pgrx run pg16 -- -c "\\i bench/sql/phase_l_cold_scan.sql"
--
-- The script:
--   1. (re)builds the corpus + index in a fresh schema.
--   2. measures the FIRST scan in this backend (cold path).
--   3. measures the SECOND scan (warm cache).
--
-- pg_stat_io is consulted on PG 16+ to confirm the cold scan
-- actually went through shared_buffers reads.

CREATE EXTENSION IF NOT EXISTS pg_turbovec;
SET search_path = turbovec, public;

DROP TABLE IF EXISTS phase_l_corpus CASCADE;
CREATE TABLE phase_l_corpus (id bigint PRIMARY KEY, emb vector);

INSERT INTO phase_l_corpus
SELECT i,
       ('[' || string_agg(
            ((hashtext(i::text || ':' || k::text) % 2000) / 1000.0 - 1)::text,
        ',') || ']')::vector
FROM generate_series(1, 2000) AS gs(i),
     generate_series(1, 384)  AS sub(k)
GROUP BY i;

CREATE INDEX phase_l_idx
    ON phase_l_corpus USING turbovec (emb vec_cosine_ops)
    WITH (bit_width = 4);

ANALYZE phase_l_corpus;

\echo '--- relfile pages --'
SELECT pg_size_pretty(pg_relation_size('phase_l_idx'::regclass))
   AS index_size,
       pg_relation_size('phase_l_idx'::regclass) / 8192
   AS index_blocks;

-- Force the AM path.
SET enable_seqscan = off;

-- Reset stats so the cold scan is the only contributor.
SELECT pg_stat_reset();

\echo '--- COLD scan (first ORDER BY in this backend) ---'
\timing on
SELECT id FROM phase_l_corpus
 ORDER BY emb <=> (SELECT emb FROM phase_l_corpus WHERE id = 1234)
 LIMIT 10;
\timing off

\echo '--- WARM scan (second ORDER BY, cache + shared_buffers hot) ---'
\timing on
SELECT id FROM phase_l_corpus
 ORDER BY emb <=> (SELECT emb FROM phase_l_corpus WHERE id = 1234)
 LIMIT 10;
\timing off

\echo '--- pg_statio_user_indexes (idx_blks_*) ---'
SELECT relname, idx_blks_read, idx_blks_hit
  FROM pg_statio_user_indexes
 WHERE relname = 'phase_l_idx';

\echo '--- heap-side row count (sanity) ---'
SELECT count(*) FROM phase_l_corpus;
