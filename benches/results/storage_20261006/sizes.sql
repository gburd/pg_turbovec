-- Heap / TOAST / index sizes per table, plus where the vector actually lives.
SELECT c.relname,
       a.attstorage                                         AS attstorage,
       (SELECT count(*) FROM pg_class x WHERE x.oid = c.oid) * 0 + c.reltuples::bigint AS rows,
       pg_relation_size(c.oid)                 / 1048576.0  AS heap_mb,
       coalesce(pg_relation_size(c.reltoastrelid),0)  / 1048576.0 AS toast_mb,
       pg_table_size(c.oid)                    / 1048576.0  AS table_mb,
       (SELECT coalesce(sum(pg_relation_size(i.indexrelid)),0) FROM pg_index i JOIN pg_class ic ON ic.oid=i.indexrelid JOIN pg_am am ON am.oid=ic.relam WHERE i.indrelid=c.oid AND am.amname='turbovec') / 1048576.0 AS tv_index_mb,
       pg_relation_size(c.oid) / 8192          AS heap_pages
  FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'tv'
 WHERE c.relkind = 'r' AND c.relname ~ '^t[0-9]+_'
 ORDER BY c.relname;
