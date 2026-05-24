-- One row per (config_label, qid). Single timed pass per query;
-- caller is responsible for any prior warmup.
CREATE TABLE IF NOT EXISTS bench_runs (
    run_id        bigserial primary key,
    config_label  text   not null,
    qid           int    not null,
    exec_ms       float8 not null,
    hits          bigint[] not null,
    ts            timestamptz default now()
);

CREATE INDEX IF NOT EXISTS bench_runs_label ON bench_runs(config_label);

CREATE OR REPLACE FUNCTION bench_one_query_pgv(q_qid int)
RETURNS TABLE(ms float8, hits bigint[])
LANGUAGE plpgsql AS $$
DECLARE
    q_emb vector;
    t0 timestamptz; t1 timestamptz;
    h  bigint[];
BEGIN
    SELECT q.emb INTO q_emb FROM query_set q WHERE q.qid = q_qid;
    t0 := clock_timestamp();
    SELECT array_agg(id) INTO h FROM (
        SELECT id FROM docs ORDER BY emb <=> q_emb LIMIT 10
    ) s;
    t1 := clock_timestamp();
    ms := EXTRACT(EPOCH FROM (t1 - t0)) * 1000.0;
    hits := h;
    RETURN NEXT;
END $$;

CREATE OR REPLACE FUNCTION bench_one_query_tv(q_qid int)
RETURNS TABLE(ms float8, hits bigint[])
LANGUAGE plpgsql AS $$
DECLARE
    q_emb turbovec.vector;
    t0 timestamptz; t1 timestamptz;
    h  bigint[];
BEGIN
    SELECT (q.emb::real[]::turbovec.vector) INTO q_emb
      FROM query_set q WHERE q.qid = q_qid;
    t0 := clock_timestamp();
    SELECT array_agg(id) INTO h FROM (
        SELECT id FROM docs
         ORDER BY (emb::real[]::turbovec.vector)
                  OPERATOR(turbovec.<=>) q_emb
         LIMIT 10
    ) s;
    t1 := clock_timestamp();
    ms := EXTRACT(EPOCH FROM (t1 - t0)) * 1000.0;
    hits := h;
    RETURN NEXT;
END $$;

CREATE OR REPLACE FUNCTION bench_run_config(p_label text, p_engine text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE qid_v int; rec record;
BEGIN
    DELETE FROM bench_runs WHERE config_label = p_label;
    FOR qid_v IN SELECT qid FROM query_set ORDER BY qid LOOP
        IF p_engine = 'pgv' THEN
            FOR rec IN SELECT * FROM bench_one_query_pgv(qid_v) LOOP
                INSERT INTO bench_runs(config_label, qid, exec_ms, hits)
                VALUES (p_label, qid_v, rec.ms, rec.hits);
            END LOOP;
        ELSE
            FOR rec IN SELECT * FROM bench_one_query_tv(qid_v) LOOP
                INSERT INTO bench_runs(config_label, qid, exec_ms, hits)
                VALUES (p_label, qid_v, rec.ms, rec.hits);
            END LOOP;
        END IF;
    END LOOP;
END $$;

-- Compute R@10 for one config, comparing to gt_top10.
CREATE OR REPLACE FUNCTION bench_recall_at_10(p_label text)
RETURNS float8 LANGUAGE sql AS $$
    SELECT round(avg(per_q)::numeric, 4)::float8
    FROM (
        SELECT b.qid,
               (SELECT count(*)::float8 / 10
                  FROM unnest(b.hits) AS h(hit_id)
                 WHERE hit_id IN (SELECT hit_id FROM gt_top10 WHERE qid = b.qid)
               ) AS per_q
          FROM bench_runs b
         WHERE b.config_label = p_label
    ) z;
$$;

-- Per-config summary as text rows (consumed by the JSON dumper).
CREATE OR REPLACE FUNCTION bench_summary(p_label text)
RETURNS TABLE(
    label   text,
    n       bigint,
    min_ms  float8,
    p50_ms  float8,
    p95_ms  float8,
    max_ms  float8,
    mean_ms float8,
    r_at_10 float8
) LANGUAGE sql AS $$
    SELECT p_label,
           count(*)::bigint,
           min(exec_ms),
           percentile_cont(0.5) WITHIN GROUP (ORDER BY exec_ms),
           percentile_cont(0.95) WITHIN GROUP (ORDER BY exec_ms),
           max(exec_ms),
           avg(exec_ms),
           bench_recall_at_10(p_label)
      FROM bench_runs
     WHERE config_label = p_label;
$$;
