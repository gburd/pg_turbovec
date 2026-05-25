-- pgbench script: one transaction = one turbovec.knn call.
\set qid random(0, :pool_max)
SELECT count(*) FROM turbovec.knn(
    'bench_corpus'::regclass::oid,
    'id',
    'embedding',
    (SELECT q FROM bench_queries WHERE qid = :qid),
    :k,
    :bw
) AS r;
