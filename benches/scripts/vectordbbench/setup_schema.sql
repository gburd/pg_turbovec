-- Schema for the 1M wiki bench. Run in db bench_wiki.
\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turbovec;

DROP TABLE IF EXISTS public.docs CASCADE;
CREATE TABLE public.docs (
    id  bigint NOT NULL,
    emb vector(1024) NOT NULL
);

-- Held-out queries (loaded separately from q1000.npy via COPY).
DROP TABLE IF EXISTS public.query_set CASCADE;
CREATE TABLE public.query_set (
    qid int NOT NULL,
    emb vector(1024) NOT NULL
);

-- Ground-truth top-10 (qid, hit_id) computed by brute force.
DROP TABLE IF EXISTS public.gt_top10 CASCADE;
CREATE TABLE public.gt_top10 (
    qid    int    NOT NULL,
    rnk    int    NOT NULL,
    hit_id bigint NOT NULL,
    dist   float8 NOT NULL
);
