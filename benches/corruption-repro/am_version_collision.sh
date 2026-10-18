#!/usr/bin/env bash
# Repro for the claimed private-cache am_version collision (S2b worker report,
# 2026-10-17). Three real backends against one flat index:
#   A, B = two writers in OVERLAPPING transactions, one INSERT each
#   C    = a persistent reader that caches the index between the two commits
# Claim: A and B both stamp am_version = V+1, so C's cached (relfilenode, V+1)
# entry never invalidates and C never sees B's row.
# Pre-fix (v1.29.2..v2.14.0) 3/3 REPRODUCED (am_version 1 -> 2 -> 2); v2.14.1
# 3/3 NOT REPRODUCED (1 -> 2 -> 3). VACUUM variant: am_version_collision_vacuum.sh.
set -euo pipefail
BIN=$(dirname "${PG_CONFIG:-$(ls -d ~/.pgrx/16.*/pgrx-install/bin/pg_config | head -1)}")  # override: PG_CONFIG=/path/to/pg_config
DATA=/tmp/repro_pgdata; PORT=28977; LOG=/tmp/repro_pg.log
export PGHOST=/tmp PGPORT=$PORT PGDATABASE=postgres
if [ ! -d "$DATA" ]; then
  "$BIN/initdb" -D "$DATA" -A trust >/dev/null
  echo "shared_preload_libraries = 'pg_turbovec'" >> "$DATA/postgresql.conf"
  echo "unix_socket_directories = '/tmp'" >> "$DATA/postgresql.conf"
fi
"$BIN/pg_ctl" -D "$DATA" -o "-p $PORT" -l "$LOG" -w start >/dev/null
trap '"$BIN/pg_ctl" -D "$DATA" -m fast -w stop >/dev/null' EXIT
P="$BIN/psql -X -q -v ON_ERROR_STOP=1"

$P <<'SQL'
DROP TABLE IF EXISTS t; DROP EXTENSION IF EXISTS pg_turbovec CASCADE;
CREATE EXTENSION pg_turbovec;
CREATE EXTENSION IF NOT EXISTS pageinspect;
ALTER DATABASE postgres SET search_path = turbovec, public;
SET search_path = turbovec, public;
CREATE TABLE t (id int, v turbovec.vector);
INSERT INTO t SELECT g, ('[' || array_to_string(ARRAY(SELECT (g*13+k)%97/97.0 FROM generate_series(1,8) k), ',') || ']')::turbovec.vector
  FROM generate_series(1,200) g;
CREATE INDEX t_idx ON t USING turbovec (v vec_cosine_ops);
SQL
meta() { $P -tA -c "SELECT (get_byte(p,84)+get_byte(p,85)*256+get_byte(p,86)*65536+get_byte(p,87)*16777216) FROM get_raw_page('t_idx',0) p"; }

# Fixed probe vectors: row A uses QA, row B uses QB, both far from the corpus
# so each is the unique nearest neighbour of its own probe.
QA="[9,9,9,9,9,9,9,9]"; QB="[-9,-9,-9,-9,-9,-9,-9,-9]"
Q="SET enable_seqscan=off; SELECT id FROM t ORDER BY v <=> '%s' LIMIT 1;"

mkfifo /tmp/repro_A /tmp/repro_B /tmp/repro_C 2>/dev/null || true
exec 3> >($P -tA < /tmp/repro_A > /tmp/repro_A.out 2>&1)
exec 4> >($P -tA < /tmp/repro_B > /tmp/repro_B.out 2>&1)
exec 5> >($P -tA < /tmp/repro_C > /tmp/repro_C.out 2>&1)
exec 3>/tmp/repro_A 4>/tmp/repro_B 5>/tmp/repro_C
step() { sleep 1.5; }

# C warms its cache on the base index.
V0=$(meta)
printf "$Q\n" "$QA" >&5; step
# A and B both open transactions and insert (both load version V).
echo "BEGIN; INSERT INTO t VALUES (100001, '$QA');" >&3; step
echo "BEGIN; INSERT INTO t VALUES (100002, '$QB');" >&4; step
# A commits -> on-disk am_version V+1.
echo "COMMIT;" >&3; step
V_A=$(meta)
# C scans: sees V+1, rebuilds, caches (rfn, V+1). Must find 100001.
printf "SELECT 'C_after_A';\n$Q\n" "$QA" >&5; step
# B commits -> claimed: ALSO am_version V+1, with 100002 merged in.
echo "COMMIT;" >&4; step
V_B=$(meta)
# C scans for B's row. Correct answer: 100002.
printf "SELECT 'C_after_B';\n$Q\n" "$QB" >&5; step
# A brand-new backend (no cache) as the control.
FRESH=$($P -tA -c "$(printf "$Q" "$QB")" | tail -1)
exec 3>&- 4>&- 5>&-; sleep 1
echo "=== C (persistent reader) output:"; cat /tmp/repro_C.out
echo "=== fresh-backend control for QB: $FRESH"
echo "=== am_version: base=$V0  after A commit=$V_A  after B commit=$V_B"
echo "=== heap truth: $($P -tA -c "SELECT count(*) FROM t WHERE id IN (100001,100002)") of 2 rows committed"
C_B=$(grep -A1 C_after_B /tmp/repro_C.out | tail -1)
if [ "$C_B" = "100002" ]; then echo "RESULT: NOT REPRODUCED (C sees B's row)"; else echo "RESULT: REPRODUCED (C returned '$C_B', expected 100002)"; fi
