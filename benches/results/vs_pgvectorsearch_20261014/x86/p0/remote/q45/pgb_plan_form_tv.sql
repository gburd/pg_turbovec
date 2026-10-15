\set qid random(0, 5199)
SELECT q::text AS qv FROM queries WHERE qid = :qid \gset
SELECT id FROM docs ORDER BY emb <=> :'qv'::turbovec.vector LIMIT 10;
