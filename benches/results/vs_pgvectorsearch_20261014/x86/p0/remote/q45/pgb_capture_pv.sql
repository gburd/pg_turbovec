\set qid random(0, 5199)
SELECT q::text AS qv FROM queries WHERE qid = :qid \gset
INSERT INTO pgb_out SELECT :qid, array_agg(id) FROM (SELECT id FROM docs ORDER BY emb <=> :qv::vector LIMIT 10) s;
