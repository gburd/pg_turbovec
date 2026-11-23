#!/usr/bin/env bash
# Rows inserted before ROLLBACK TO SAVEPOINT, or around a PL/pgSQL EXCEPTION
# block, must reach the index; the rolled-back rows must not (2.16.3 and
# earlier lost the pre-savepoint rows until REINDEX).
. "$(dirname "$0")/lib.sh"
mk sp
q -c "BEGIN" -c "INSERT INTO sp VALUES (5001, $(vec 5001))" -c "SAVEPOINT a" \
  -c "INSERT INTO sp VALUES (5002, $(vec 5002))" -c "ROLLBACK TO SAVEPOINT a" \
  -c "INSERT INTO sp VALUES (5003, $(vec 5003))" -c "COMMIT"
q -c "DO \$\$ BEGIN INSERT INTO sp VALUES (5004, $(vec 5004));
        BEGIN INSERT INTO sp VALUES (5005, $(vec 5005)); RAISE EXCEPTION 'x';
        EXCEPTION WHEN OTHERS THEN NULL; END; END \$\$"
# an aborted savepoint that touched only another table
q -c "BEGIN" -c "INSERT INTO sp VALUES (5006, $(vec 5006))" -c "SAVEPOINT a" \
  -c "CREATE TEMP TABLE sp_other (k int)" -c "ROLLBACK TO SAVEPOINT a" -c "COMMIT"
# one backend, two transactions: the first commits, the second rolls back a
# savepoint.  The cache entry carries over from the first, so its pending
# bookkeeping must have been reset at commit.
q -c "INSERT INTO sp VALUES (5009, $(vec 5009))" \
  -c "DO \$\$ BEGIN INSERT INTO sp VALUES (5010, $(vec 5010));
        BEGIN INSERT INTO sp VALUES (5011, $(vec 5011)); RAISE EXCEPTION 'x';
        EXCEPTION WHEN OTHERS THEN NULL; END; END \$\$"
# control: RELEASE keeps both
q -c "BEGIN" -c "INSERT INTO sp VALUES (5007, $(vec 5007))" -c "SAVEPOINT a" \
  -c "INSERT INTO sp VALUES (5008, $(vec 5008))" -c "RELEASE SAVEPOINT a" -c "COMMIT"
expect "heap" "$(q -c "SELECT string_agg(id::text, ' ' ORDER BY id) FROM sp WHERE id > 5000")" "5001 5003 5004 5006 5007 5008 5009 5010"
expect "index, new backend" "$(found sp 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011)" "5001 5003 5004 5006 5007 5008 5009 5010"
expect "index entries == heap rows" \
    "$(q -c "SELECT n_vectors FROM turbovec.turbovec_check('sp_idx'::regclass)")" "$(q -c "SELECT count(*) FROM sp")"
# and the index agrees with a REINDEX (nothing extra, nothing missing)
q -c "REINDEX INDEX sp_idx"
expect "index after REINDEX" "$(found sp 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011)" "5001 5003 5004 5006 5007 5008 5009 5010"
