#!/usr/bin/env bash
# FINDING#2 end-to-end check: a graph `CREATE INDEX` must be cancellable
# from another session, and `pg_ctl stop -m fast` must not hang behind it.
#
# Why this is a script and not a #[pg_test]: the in-tree test
# (`graph_build_polls_for_interrupts`) sets `QueryCancelPending` directly,
# which proves the build POLLS but runs in one backend. Only a second
# session can prove the operator-visible behaviour: `pg_cancel_backend`
# actually returns control, and a `-m fast` shutdown completes.
#
# NOTE on `statement_timeout`: it does NOT work as the cancel mechanism
# here. PG arms the statement timer when a statement starts, so a timeout
# set inside the same session/statement as the build never fires for it.
# An out-of-band `pg_cancel_backend` from a second session is the real
# thing, which is what this does.
#
# FAIL-BEFORE (pg_turbovec 2.0.0, unfixed): zero check_for_interrupts in
# the graph build path -- the cancel is swallowed until the build
# finishes, and `pg_ctl stop -m fast` blocks behind the rayon workers'
# futex wait.
#
# Usage:
#   PGDATA=/path/to/data PGHOST=/tmp PGPORT=5432 ./graph_cancel_check.sh
#   N=200000 D=256 ./graph_cancel_check.sh
set -uo pipefail

PGHOST="${PGHOST:-/tmp}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-postgres}"
export PGHOST PGPORT PGDATABASE
export PGOPTIONS="${PGOPTIONS:--c search_path=turbovec,public}"

N="${N:-200000}"      # rows: big enough that the build takes >> CANCEL_AFTER
D="${D:-256}"
CANCEL_AFTER="${CANCEL_AFTER:-5}"   # seconds to let the build run first
q() { psql -qtAX -c "$1"; }

echo "=== setup: N=$N D=$D cancel after ${CANCEL_AFTER}s"
q "CREATE EXTENSION IF NOT EXISTS pg_turbovec" >/dev/null
q "DROP TABLE IF EXISTS g_cancel CASCADE" >/dev/null
q "CREATE TABLE g_cancel (id serial PRIMARY KEY, v turbovec.vector)" >/dev/null
# Correlated per-row random (AGENTS.md: uncorrelated random() gets hoisted).
q "INSERT INTO g_cancel(v) SELECT (SELECT array_agg((random()-0.5)::real) FROM generate_series(1,$D) x WHERE x+g1*0 = x)::turbovec.vector FROM generate_series(1,$N) g1" >/dev/null
echo "corpus loaded: $(q 'SELECT count(*) FROM g_cancel') rows"

BUILDLOG="$(mktemp -t graph_cancel.XXXXXX.log)"
trap 'rm -f "$BUILDLOG"' EXIT

psql -qtAX -c "SELECT pg_backend_pid()" > "$BUILDLOG.pid" 2>/dev/null || true
(
  psql -qtAX \
    -c "SELECT pg_backend_pid() AS building_pid" \
    -c "CREATE INDEX g_cancel_idx ON g_cancel USING turbovec (v vec_cosine_ops) WITH (graph = true)" \
    > "$BUILDLOG" 2>&1
  echo "BUILD_EXIT=$?" >> "$BUILDLOG"
) &
build_sh=$!

sleep "$CANCEL_AFTER"
pid="$(q "SELECT pid FROM pg_stat_activity WHERE query LIKE 'CREATE INDEX g_cancel_idx%' AND pid <> pg_backend_pid() LIMIT 1")"
if [ -z "$pid" ]; then
  echo "FAIL: no backend found running the CREATE INDEX (did it finish in <${CANCEL_AFTER}s? raise N)"
  wait "$build_sh"; cat "$BUILDLOG"; exit 1
fi
echo "cancelling backend $pid ..."
t0=$(date +%s)
q "SELECT pg_cancel_backend($pid)" >/dev/null

# Bounded wait: how long until the build backend actually gives up?
deadline=$(( t0 + 60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  alive="$(q "SELECT count(*) FROM pg_stat_activity WHERE pid = $pid AND query LIKE 'CREATE INDEX g_cancel_idx%'")"
  [ "$alive" = "0" ] && break
  sleep 1
done
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
wait "$build_sh" 2>/dev/null

echo "--- build session output:"
cat "$BUILDLOG"
echo "--- cancel took ${elapsed}s"

fail=0
if grep -qiE "canceling statement|cancel" "$BUILDLOG"; then
  echo "PASS: the build reported the cancel"
else
  echo "FAIL: the build did NOT report a cancel — it was swallowed (FINDING#2)"
  fail=1
fi
if [ "$elapsed" -ge 60 ]; then
  echo "FAIL: cancel did not take effect within 60s"
  fail=1
fi
# The cancelled build must leave no index behind.
left="$(q "SELECT count(*) FROM pg_class WHERE relname = 'g_cancel_idx'")"
[ "$left" != "0" ] && echo "NOTE: g_cancel_idx still exists (relcount=$left) — a completed build, not a cancelled one"

# Optional: prove `pg_ctl stop -m fast` completes. Only when PGDATA is
# given, since it stops the cluster.
if [ -n "${PGDATA:-}" ] && [ "${TEST_FAST_SHUTDOWN:-0}" = "1" ]; then
  echo "=== -m fast shutdown check (PGDATA=$PGDATA)"
  ( psql -qtAX -c "CREATE INDEX g_cancel_idx2 ON g_cancel USING turbovec (v vec_cosine_ops) WITH (graph = true)" >/dev/null 2>&1 ) &
  sleep "$CANCEL_AFTER"
  t0=$(date +%s)
  # NEVER kill -9 a postmaster (AGENTS.md): crash recovery truncates
  # UNLOGGED tables. -m fast is the supported path and is exactly what
  # FINDING#2 said hangs.
  if timeout 120 pg_ctl -D "$PGDATA" stop -m fast -w; then
    echo "PASS: pg_ctl stop -m fast completed in $(( $(date +%s) - t0 ))s during a graph build"
  else
    echo "FAIL: pg_ctl stop -m fast did not complete within 120s during a graph build"
    fail=1
  fi
  pg_ctl -D "$PGDATA" start -w -t 60 >/dev/null
fi

exit "$fail"
