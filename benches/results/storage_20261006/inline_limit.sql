-- Where each storage mode stops keeping a vector inline. One single-row table
-- per (storage, dim, extra) shaped (id bigint, [extra NOT NULL bigint
-- columns], tv turbovec.vector); the bigints are fixed-width, so they can't be
-- compressed or TOASTed themselves and just eat 8 bytes each of the budget.
-- Values are (random()*2-1)::real: full 24-bit mantissas, so serde-CBOR
-- writes every element as 5 bytes (it uses 3 for values exactly
-- representable in f16), i.e. the varlena is exactly 13 + 5*dim bytes.
\set ON_ERROR_STOP 0
\pset footer off
CREATE OR REPLACE FUNCTION pg_temp.probe(st text, d int, extra int) RETURNS TABLE(storage text, dim int, extra_bigints int, vec_bytes int, heap_tuple_len int, toast_chunks bigint, result text)
LANGUAGE plpgsql AS $$
DECLARE toast regclass; n bigint; sz int; vb int;
BEGIN
  DROP TABLE IF EXISTS probe_t;
  EXECUTE format('CREATE TABLE probe_t (id bigint %s, tv turbovec.vector)',
                 (SELECT coalesce(string_agg(format(', b%s bigint NOT NULL DEFAULT 0', i), ''), '') FROM generate_series(1, extra) i));
  EXECUTE format('ALTER TABLE probe_t ALTER COLUMN tv SET STORAGE %s', st);
  BEGIN
    INSERT INTO probe_t (id, tv) SELECT 1, array_agg((random() * 2 - 1)::real)::turbovec.vector FROM generate_series(1, d);
  EXCEPTION WHEN others THEN
    RETURN QUERY SELECT st, d, extra, NULL::int, NULL::int, NULL::bigint, 'ERROR: ' || SQLERRM; RETURN;
  END;
  SELECT reltoastrelid::regclass INTO toast FROM pg_class WHERE oid = 'probe_t'::regclass;
  EXECUTE format('SELECT count(*) FROM %s', toast) INTO n;
  SELECT lp_len INTO sz FROM heap_page_items(get_raw_page('probe_t', 0)) WHERE lp = 1;
  SELECT pg_column_size(tv) INTO vb FROM probe_t;
  RETURN QUERY SELECT st, d, extra, vb, sz, n, CASE WHEN n > 0 THEN 'out of line (TOAST)' ELSE 'inline' END;
END $$;
SELECT p.* FROM unnest(ARRAY['EXTENDED','EXTERNAL']) s, unnest(ARRAY[384, 396, 397, 398, 400]) d, pg_temp.probe(s, d, 0) p;
SELECT p.* FROM unnest(ARRAY['MAIN','PLAIN']) s, unnest(ARRAY[1536, 1622, 1623, 1624, 1625, 2048]) d, pg_temp.probe(s, d, 0) p;
-- 12 extra bigint columns (96 bytes): predicted limits 378 (EXTENDED) and 1603 (MAIN/PLAIN)
SELECT p.* FROM unnest(ARRAY['EXTENDED','MAIN','PLAIN']) s, unnest(ARRAY[378, 379, 1603, 1604]) d, pg_temp.probe(s, d, 12) p
 WHERE (s = 'EXTENDED') = (d < 1000);
DROP TABLE probe_t;
