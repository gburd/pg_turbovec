-- Build all three indexes with timing + size capture. Run in bench_wiki.
\set ON_ERROR_STOP on
\timing on

SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 16;

\echo '=== pre-build heap size + row count ==='
SELECT pg_size_pretty(pg_relation_size('public.docs')) AS heap_main,
       pg_size_pretty(pg_table_size('public.docs'))    AS heap_total,
       count(*) AS rows FROM public.docs;

\echo '=== PRIMARY KEY (id) + ANALYZE ==='
ALTER TABLE public.docs ADD CONSTRAINT docs_pkey PRIMARY KEY (id);
ANALYZE public.docs;

\echo '=== HNSW (m=16, ef_construction=64) ==='
DROP INDEX IF EXISTS public.docs_pgv_hnsw;
CREATE INDEX docs_pgv_hnsw ON public.docs USING hnsw (emb vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
SELECT pg_size_pretty(pg_relation_size('public.docs_pgv_hnsw')) AS hnsw_size,
       pg_relation_size('public.docs_pgv_hnsw') AS hnsw_bytes;

\echo '=== pg_turbovec 4-bit ==='
DROP INDEX IF EXISTS public.docs_tv_4bit;
CREATE INDEX docs_tv_4bit ON public.docs
    USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 4);
SELECT pg_size_pretty(pg_relation_size('public.docs_tv_4bit')) AS tv4_size,
       pg_relation_size('public.docs_tv_4bit') AS tv4_bytes;

\echo '=== pg_turbovec 2-bit ==='
DROP INDEX IF EXISTS public.docs_tv_2bit;
CREATE INDEX docs_tv_2bit ON public.docs
    USING turbovec ((emb::real[]::turbovec.vector) turbovec.vec_cosine_ops)
    WITH (bit_width = 2);
SELECT pg_size_pretty(pg_relation_size('public.docs_tv_2bit')) AS tv2_size,
       pg_relation_size('public.docs_tv_2bit') AS tv2_bytes;

\echo '=== all index sizes ==='
SELECT indexrelid::regclass AS idx,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size,
       pg_relation_size(indexrelid) AS bytes
FROM pg_index WHERE indrelid = 'public.docs'::regclass
ORDER BY pg_relation_size(indexrelid) DESC;
