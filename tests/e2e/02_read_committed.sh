#!/usr/bin/env bash
# READ COMMITTED: once a transaction has inserted, its later statements must
# still see rows other sessions commit (2.16.3 served its own stale copy).
# REPEATABLE READ: a transaction whose first insert reuses this backend's
# cached copy, which predates another session's commit, must still see that
# commit (it precedes the snapshot); a commit made after the snapshot must
# stay invisible.
. "$(dirname "$0")/lib.sh"
mk rc
# A separate-session INSERT, invoked from psql's `\!`.  psql runs `\!` through
# /bin/sh, which is dash on Debian/Ubuntu and drops exported bash functions, so
# this must be a real executable, not an `export -f` function (that failed on
# every Ubuntu CI leg with "other: command not found").  It opens its own libpq
# connection, i.e. a genuinely separate backend, which is the point of the test.
other=$(mktemp)
cat > "$other" <<OTHER
#!/usr/bin/env bash
set -euo pipefail
export PGOPTIONS="-c search_path=public,turbovec"
v="('[' || array_to_string(array(SELECT sin(\$1 * 0.37 + s)::float8 FROM generate_series(1, 32) s), ',') || ']')::turbovec.vector"
psql -X -q -v ON_ERROR_STOP=1 -c "INSERT INTO rc VALUES (\$1, \$v)"
OTHER
chmod +x "$other"
trap 'rm -f "$other"' EXIT
got=$(q <<SQL
\set ON_ERROR_STOP 1
SET enable_seqscan = off;
BEGIN;
INSERT INTO rc VALUES (6001, $(vec 6001));
\! $other 6002
SELECT 'rc:' || coalesce(string_agg(b.id::text, ' ' ORDER BY b.id), '') FROM rc b
 WHERE b.id IN (6001, 6002) AND (SELECT x.id FROM rc x ORDER BY x.emb <=> b.emb LIMIT 1) = b.id;
INSERT INTO rc VALUES (6003, $(vec 6003));
\! $other 6004
SELECT 'rc2:' || coalesce(string_agg(b.id::text, ' ' ORDER BY b.id), '') FROM rc b
 WHERE b.id IN (6001, 6002, 6003, 6004) AND (SELECT x.id FROM rc x ORDER BY x.emb <=> b.emb LIMIT 1) = b.id;
COMMIT;
\! $other 6007
BEGIN ISOLATION LEVEL REPEATABLE READ;
INSERT INTO rc VALUES (6005, $(vec 6005));
\! $other 6006
SELECT 'rr:' || coalesce(string_agg(b.id::text, ' ' ORDER BY b.id), '') FROM rc b
 WHERE b.id IN (6005, 6006, 6007) AND (SELECT x.id FROM rc x ORDER BY x.emb <=> b.emb LIMIT 1) = b.id;
COMMIT;
SQL
)
expect "read committed, after own insert" "$(echo "$got" | sed -n 's/^rc://p')" "6001 6002"
expect "read committed, second round" "$(echo "$got" | sed -n 's/^rc2://p')" "6001 6002 6003 6004"
expect "repeatable read: own row + the commit before its snapshot" "$(echo "$got" | sed -n 's/^rr://p')" "6005 6007"
expect "all seven after commit, new backend" "$(found rc 6001 6002 6003 6004 6005 6006 6007)" "6001 6002 6003 6004 6005 6006 6007"
expect "index entries == heap rows" \
    "$(q -c "SELECT n_vectors FROM turbovec.turbovec_check('rc_idx'::regclass)")" "$(q -c "SELECT count(*) FROM rc")"
