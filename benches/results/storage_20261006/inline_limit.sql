-- Where each storage mode stops keeping a vector inline. One single-row table
-- per (storage, dim, extra) shaped (id bigint, [extra NOT NULL bigint
-- columns], tv turbovec.vector); the bigints are fixed-width, so they can't be
-- compressed or TOASTed themselves and just eat 8 bytes each of the budget.
-- Values are (random()*2-1)::real: full 24-bit mantissas, so serde-CBOR
-- writes every element as 5 bytes (it uses 3 for values exactly
-- representable in f16), i.e. the varlena is exactly 13 + 5*dim bytes.
\set ON_ERROR_STOP 0
\pset footer off
SELECT setseed(0.20261006);
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

-- Text neighbours: (id bigint, title text, body text, tv turbovec.vector),
-- vector storage per row, optionally with the TABLE's toast_tuple_target
-- raised to 8160 instead. The text is random hex (md5), which pglz can't
-- compress. Each column's placement is read from the raw heap tuple
-- (pageinspect): an 18-byte value whose first byte is 0x01 is an on-disk
-- TOAST pointer; low bits 10 = compressed inline; otherwise plain inline.
CREATE OR REPLACE FUNCTION pg_temp.at(b bytea) RETURNS text LANGUAGE sql AS $$
  SELECT CASE WHEN get_byte(b, 0) = 1 THEN 'TOAST'
              WHEN get_byte(b, 0) & 3 = 2 THEN format('compressed (%s B)', length(b))
              ELSE format('inline (%s B)', length(b)) END $$;
CREATE OR REPLACE FUNCTION pg_temp.probe_cols(st text, ttt int, d int, tlen int, blen int)
RETURNS TABLE(tv_storage text, toast_tuple_target int, dim int, title_len int, body_len int, title_at text, body_at text, tv_at text, heap_tuple_len int, toast_chunks bigint)
LANGUAGE plpgsql AS $$
DECLARE toast regclass; n bigint; a bytea[]; sz int;
BEGIN
  DROP TABLE IF EXISTS probe_c;
  EXECUTE 'CREATE TABLE probe_c (id bigint, title text, body text, tv turbovec.vector)';
  EXECUTE format('ALTER TABLE probe_c ALTER COLUMN tv SET STORAGE %s', st);
  IF ttt IS NOT NULL THEN EXECUTE format('ALTER TABLE probe_c SET (toast_tuple_target = %s)', ttt); END IF;
  INSERT INTO probe_c SELECT 1,
    (SELECT left(string_agg(md5(random()::text), ''), tlen) FROM generate_series(1, tlen / 32 + 1)),
    (SELECT left(string_agg(md5(random()::text), ''), blen) FROM generate_series(1, blen / 32 + 1)),
    (SELECT array_agg((random() * 2 - 1)::real)::turbovec.vector FROM generate_series(1, d));
  SELECT reltoastrelid::regclass INTO toast FROM pg_class WHERE oid = 'probe_c'::regclass;
  EXECUTE format('SELECT count(*) FROM %s', toast) INTO n;
  SELECT h.t_attrs, h.lp_len INTO a, sz FROM heap_page_item_attrs(get_raw_page('probe_c', 0), 'probe_c'::regclass) h WHERE h.lp = 1;
  RETURN QUERY SELECT st, ttt, d, tlen, blen, pg_temp.at(a[2]), pg_temp.at(a[3]), pg_temp.at(a[4]), sz, n;
END $$;
-- 40-byte title + 300-byte body next to the vector
SELECT p.* FROM (VALUES ('EXTENDED', NULL::int), ('MAIN', NULL), ('EXTENDED', 8160)) s(st, ttt),
       unnest(ARRAY[384, 1024, 1536, 3072]) d, pg_temp.probe_cols(s.st, s.ttt, d, 40, 300) p;
-- the default-storage lower bound moves too: 384-d next to one 200-byte text
SELECT p.* FROM pg_temp.probe_cols('EXTENDED', NULL, 384, 0, 200) p;
-- a row that overflows 8 KB anyway (1024-d + 3,500-byte body): which value
-- goes out? PostgreSQL externalizes the largest EXTENDED value first, and
-- skips MAIN columns until the last round.
SELECT p.* FROM (VALUES ('EXTENDED', NULL::int), ('MAIN', NULL), ('EXTENDED', 8160), ('MAIN', 8160)) s(st, ttt),
       pg_temp.probe_cols(s.st, s.ttt, 1024, 40, 3500) p;
-- MAIN on the vector AND toast_tuple_target = 8160 together, small neighbours
SELECT p.* FROM pg_temp.probe_cols('MAIN', 8160, 1024, 40, 300) p;
DROP TABLE probe_c;

-- Encoding size depends on the values: serde-CBOR writes an element in 3
-- bytes when it is exactly representable in half precision, else 5.
SELECT 'full float32' AS element_values, pg_column_size(array_agg((random() * 2 - 1)::real)::turbovec.vector) AS vec_bytes_1024d
  FROM generate_series(1, 1024)
UNION ALL
SELECT 'rounded through turbovec.halfvec', pg_column_size((array_agg((random() * 2 - 1)::real)::turbovec.halfvec)::real[]::turbovec.vector)
  FROM generate_series(1, 1024);
