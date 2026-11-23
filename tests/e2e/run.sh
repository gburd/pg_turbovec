#!/usr/bin/env bash
# End-to-end tests the #[pg_test] harness cannot express: real commits, a
# second session, and a crash.  Each pg_test runs in one transaction that is
# always rolled back, in one backend, on a server that never restarts, so the
# PreCommit flush, cross-session visibility, and crash recovery are all out of
# its reach.  These three bugs (Zishuo's report, 2026-11-20) lived exactly
# there.
#
#   tests/e2e/run.sh <pg_config>
#
# <pg_config> belongs to a PostgreSQL that already has pg_turbovec installed
# (`cargo pgrx install --pg-config <pg_config>`).  The script creates a
# throwaway cluster under $E2E_DIR (default: a mktemp dir), runs every case,
# stops the cluster, and exits non-zero if any case failed.
set -euo pipefail

PG_CONFIG=${1:?usage: run.sh <pg_config>}
BIN=$("$PG_CONFIG" --bindir)
HERE=$(cd "$(dirname "$0")" && pwd)
E2E_DIR=${E2E_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tv-e2e.XXXXXX")}
mkdir -p "$E2E_DIR"
export PGDATA=$E2E_DIR/data PGHOST=$E2E_DIR PGPORT=${E2E_PORT:-5491} PGUSER=postgres PGDATABASE=postgres
export PATH=$BIN:$PATH

"$BIN/initdb" -D "$PGDATA" -U postgres --no-locale -E UTF8 -A trust > "$E2E_DIR/initdb.log"
cat >> "$PGDATA/postgresql.conf" <<CONF
listen_addresses = ''
unix_socket_directories = '$E2E_DIR'
port = $PGPORT
fsync = on
# 03_unlogged_crash needs the crash to come before any checkpoint: a
# checkpoint writes the init fork out and hides the missing WAL record.
checkpoint_timeout = '1h'
max_wal_size = '10GB'
CONF
start() { "$BIN/pg_ctl" -D "$PGDATA" -l "$E2E_DIR/server.log" -w start > /dev/null; }
stop() { "$BIN/pg_ctl" -D "$PGDATA" -w stop -m "${1:-fast}" > /dev/null; }
crash() { "$BIN/pg_ctl" -D "$PGDATA" -w stop -m immediate > /dev/null; }
export -f start stop crash
export BIN E2E_DIR
start
trap 'stop fast 2>/dev/null || true' EXIT
psql -X -q -v ON_ERROR_STOP=1 -c "CREATE EXTENSION pg_turbovec"

fails=0
for t in "$HERE"/[0-9][0-9]_*.sh; do
    name=$(basename "$t" .sh)
    if out=$(bash "$t" 2>&1); then
        echo "ok   $name"
    else
        echo "FAIL $name"
        printf '%s\n' "$out" | sed 's/^/     | /'
        fails=$((fails + 1))
    fi
done
# A backend that PANICs (e.g. an ERROR raised inside subtransaction abort)
# takes the cluster through crash recovery, which can hide behind a case's
# own expectations.  Fail on it explicitly.
if grep -E "PANIC|was terminated by signal|TRAP:" "$E2E_DIR/server.log"; then
    echo "FAIL server log: a backend crashed"
    fails=$((fails + 1))
fi
echo "$(ls "$HERE"/[0-9][0-9]_*.sh | wc -l) cases, $fails failed (cluster: $E2E_DIR)"
[ "$fails" -eq 0 ]
