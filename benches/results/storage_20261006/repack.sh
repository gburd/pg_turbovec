#!/bin/sh
# Online migration with pg_repack: does it honour the new SET STORAGE MAIN,
# and does the turbovec index survive (pg_repack rebuilds indexes from their
# definition on the new table)? 20k x 1024-d, flat index + an IVF index.
B=/work/pg16rel_c/bin; P="$B/psql -X -h /work/fixc -p 55433 -d bench"
$P -q -c "CREATE EXTENSION IF NOT EXISTS pg_repack" <<'SQL'
SQL
$P -q <<'SQL' 2>&1 | grep -v "NOTICE\|DETAIL\|HINT"
DROP TABLE IF EXISTS rp;
CREATE TABLE rp (id bigint PRIMARY KEY, tv turbovec.vector);
COPY rp FROM PROGRAM 'head -n 20000 /work/fixc/data_1024.tsv';
CREATE INDEX rp_tv  ON rp USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4);
CREATE INDEX rp_ivf ON rp USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 64);
VACUUM ANALYZE rp;
ALTER TABLE rp ALTER COLUMN tv SET STORAGE MAIN;
SQL
cat > /tmp/rp_view.sql <<'SQL'
SELECT a.attstorage, round(pg_relation_size('rp') / 1048576.0, 1) AS heap_mb,
       round(coalesce(pg_relation_size(nullif(c.reltoastrelid, 0)), 0) / 1048576.0, 1) AS toast_mb,
       (SELECT n_vectors FROM turbovec.turbovec_check('rp_tv'::regclass)) AS flat_entries,
       (SELECT n_vectors FROM turbovec.turbovec_check('rp_ivf'::regclass)) AS ivf_entries,
       turbovec.index_is_degraded('rp_ivf'::regclass) AS ivf_degraded
  FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'tv' WHERE c.relname = 'rp';
SQL
echo "--- before pg_repack (SET STORAGE MAIN already applied)"; $P -f /tmp/rp_view.sql
$B/pg_repack -h /work/fixc -p 55433 -d bench -t rp 2>&1 | grep -v "built a FLAT\|^DETAIL"
echo "--- after pg_repack"; $P -f /tmp/rp_view.sql
$P -At -c "SELECT count(*) FROM (SELECT id FROM rp ORDER BY tv OPERATOR(turbovec.<=>) (SELECT tv FROM rp WHERE id = 5) LIMIT 10) s" -c "DROP TABLE rp"
