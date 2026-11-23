#!/usr/bin/env bash
# A savepoint rollback of inserts that REUSED TIDs freed by VACUUM, in a
# backend whose cached index copy (warm from an earlier commit) still holds
# the old rows at those TIDs, so the inserts replace them in memory.  The
# rollback must leave none of those TIDs in the index: VACUUM removed the
# old rows from disk, and the new ones are rolled back.  Also checks the
# rows the transaction kept, turbovec_check, and (via run.sh) that no
# backend crashed: this all happens inside subtransaction abort, where an
# ERROR becomes a PANIC.
. "$(dirname "$0")/lib.sh"
mk tr "" "${TR_WITH:-bit_width = 4}"
got=$(q <<SQL
\set ON_ERROR_STOP 1
SET enable_seqscan = off;
-- warm this backend's cache entry with a committed insert
INSERT INTO tr VALUES (7001, $(vec 7001));
SELECT 1 FROM tr ORDER BY emb <=> $(vec 1) LIMIT 1;
-- another session frees the TIDs of rows 1..200
CREATE TEMP TABLE freed AS SELECT ctid AS t FROM tr WHERE id <= 200;
\! psql -X -q -v ON_ERROR_STOP=1 -c "DELETE FROM tr WHERE id <= 200" -c "VACUUM tr"
BEGIN;
INSERT INTO tr VALUES (7002, $(vec 7002));
SAVEPOINT a;
INSERT INTO tr SELECT g, $(vec g) FROM generate_series(8001, 8200) g;
SELECT 'reused:' || count(*) FROM tr WHERE id > 8000 AND ctid IN (SELECT t FROM freed);
ROLLBACK TO SAVEPOINT a;
SELECT 'mid:' || coalesce(string_agg(b.id::text, ' ' ORDER BY b.id), '') FROM tr b
 WHERE b.id IN (7001, 7002) AND (SELECT x.id FROM tr x ORDER BY x.emb <=> b.emb LIMIT 1) = b.id;
INSERT INTO tr VALUES (7003, $(vec 7003));
COMMIT;
SQL
)
reused=$(echo "$got" | sed -n 's/^reused://p')
[ "${reused:-0}" -gt 0 ] || { echo "setup: no inserted row reused a freed TID (got '$reused')"; exit 1; }
expect "this transaction's scan after the rollback" "$(echo "$got" | sed -n 's/^mid://p')" "7001 7002"
expect "heap" "$(q -c "SELECT count(*) FROM tr WHERE id > 8000 OR id <= 200")" "0"
expect "index, new backend" "$(found tr 7001 7002 7003 201 1000 2000)" "201 1000 2000 7001 7002 7003"
expect "no rolled-back or deleted row in the index" \
    "$(q -c "SET enable_seqscan = off" -c "SELECT count(*) FROM (SELECT id FROM tr ORDER BY emb <=> $(vec 8001) LIMIT 50) s WHERE id > 8000 OR id <= 200")" "0"
expect "turbovec_check" "$(q -c "SELECT count_matches AND NOT is_corrupt AND duplicate_id IS NULL FROM turbovec.turbovec_check('tr_idx'::regclass)")" "t"
# A stale entry for a rolled-back row is invisible to queries (the heap
# recheck drops it), so count the index's LIVE entries (an IVF index keeps
# vacuumed slots, tombstoned) against the heap's rows.
expect "live index entries == heap rows" \
    "$(q -c "SELECT round(n_vectors * (1 - tombstone_density)) FROM turbovec.turbovec_check('tr_idx'::regclass)")" \
    "$(q -c "SELECT count(*) FROM tr")"
