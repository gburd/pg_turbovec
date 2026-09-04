#!/usr/bin/env bash
# C1 reproduction + regression harness: concurrent INSERTs into a graph
# index (`WITH (graph = true)`).
#
# Why this lives here and not in a #[pg_test]: a #[pg_test] runs in ONE
# backend inside one implicit transaction, so it cannot produce two
# genuinely concurrent `aminsert`s. C1 is a lost-update / torn-write bug
# that only manifests with real concurrency, so the reproduction has to
# be multi-backend.
#
# FAIL-BEFORE (pg_turbovec 2.0.0, unfixed): `insert_graph_row` did an
# UNLOCKED `read_full`, then the expensive quantize + Vamana-insert CPU
# work, then a BLIND `write_full_with_prepared_graph` from that stale
# snapshot. Measured with W=4 ROWS=25 N0=2000 on PG18:
#     8 / 100 inserts died with
#       ERROR: turbovec relfile: corrupt graph adjacency chain:
#              graph offsets[n]=54272 != neighbors.len()=54240
#     heap=2092, expected=2100  (8 rows committed with no index entry)
#   ...and `turbovec_check` still said is_corrupt=false, because the
#   relfile that survived was self-consistent -- the losers' work was
#   simply gone. That silent variant is the more dangerous one.
#
# PASS-AFTER (with the one-lock read-modify-write): W=8 ROWS=40 N0=3000
#     320 / 320 inserts OK, n_vectors == slot_count == 3320,
#     is_corrupt=false, duplicate_id=none, 0 errors.
#
# Usage:
#   PGHOST=/tmp PGPORT=5432 ./graph_concurrent_insert.sh
#   W=8 ROWS=40 N0=3000 D=128 ./graph_concurrent_insert.sh
#   SUSTAIN_SECS=300 ./graph_concurrent_insert.sh   # + a VACUUM loop
#
# Exits non-zero if the index ends up corrupt, loses a row, or any
# inserter errors.
set -uo pipefail

PGHOST="${PGHOST:-/tmp}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-postgres}"
export PGHOST PGPORT PGDATABASE
export PGOPTIONS="${PGOPTIONS:--c search_path=turbovec,public}"

D="${D:-128}"          # vector dim
N0="${N0:-2000}"       # rows in the index before the concurrent phase
W="${W:-4}"            # concurrent inserter backends
ROWS="${ROWS:-25}"     # rows each inserter adds
SUSTAIN_SECS="${SUSTAIN_SECS:-0}"  # >0: also run a VACUUM loop for N secs

ERRLOG="$(mktemp -t graph_concurrent_insert.XXXXXX.err)"
trap 'rm -f "$ERRLOG"' EXIT

q() { psql -qtAX -c "$1"; }

# A correlated per-row random vector. AGENTS.md: an UNCORRELATED
# `random()` subquery gets hoisted by the planner, making every row
# identical (n_distinct=1) -- which silently invalidates any recall or
# graph-structure claim. The `x+g1*0 = x` predicate forces correlation.
rowexpr() {
  local outer="$1"
  echo "(SELECT array_agg((random()-0.5)::real) FROM generate_series(1,$D) x WHERE x+${outer}*0 = x)::turbovec.vector"
}

echo "=== setup: D=$D N0=$N0 W=$W ROWS=$ROWS SUSTAIN_SECS=$SUSTAIN_SECS"
q "CREATE EXTENSION IF NOT EXISTS pg_turbovec" >/dev/null
q "DROP TABLE IF EXISTS g_conc CASCADE" >/dev/null
q "CREATE TABLE g_conc (id serial PRIMARY KEY, v turbovec.vector)" >/dev/null
q "INSERT INTO g_conc(v) SELECT $(rowexpr g1) FROM generate_series(1,$N0) g1" >/dev/null
q "CREATE INDEX g_conc_idx ON g_conc USING turbovec (v vec_cosine_ops) WITH (graph = true)" >/dev/null

check() {
  psql -qtAX -c "SELECT n_vectors||'|'||slot_count||'|'||is_corrupt||'|'||coalesce(duplicate_id::text,'none') FROM turbovec_check('g_conc_idx'::regclass)"
}

before="$(check)"
echo "before: n|slots|corrupt|dup = $before"

heap_before="$(q 'SELECT count(*) FROM g_conc')"

# --- the concurrent phase ---
for w in $(seq 1 "$W"); do
  (
    for _ in $(seq 1 "$ROWS"); do
      psql -qtAX -c "INSERT INTO g_conc(v) SELECT $(rowexpr 0)" >/dev/null 2>>"$ERRLOG"
    done
  ) &
done

# Optional sustained-load leg: a VACUUM loop racing the inserters, and
# a periodic corruption check. This is the "several minutes of
# concurrent inserters + VACUUM stays is_corrupt=false" gate.
vac_pid=""
if [ "$SUSTAIN_SECS" -gt 0 ]; then
  (
    end=$(( $(date +%s) + SUSTAIN_SECS ))
    while [ "$(date +%s)" -lt "$end" ]; do
      psql -qtAX -c "DELETE FROM g_conc WHERE id IN (SELECT id FROM g_conc ORDER BY random() LIMIT 3)" >/dev/null 2>>"$ERRLOG"
      psql -qtAX -c "VACUUM g_conc" >/dev/null 2>>"$ERRLOG"
      c="$(check)"
      case "$c" in
        *'|true|'*) echo "!!! CORRUPTION DETECTED mid-run: $c" ;;
      esac
      sleep 2
    done
  ) &
  vac_pid=$!
  # Keep the inserters going for the whole sustained window.
  (
    end=$(( $(date +%s) + SUSTAIN_SECS ))
    while [ "$(date +%s)" -lt "$end" ]; do
      psql -qtAX -c "INSERT INTO g_conc(v) SELECT $(rowexpr 0)" >/dev/null 2>>"$ERRLOG"
    done
  ) &
fi

wait
[ -n "$vac_pid" ] && wait "$vac_pid" 2>/dev/null

after="$(check)"
heap_after="$(q 'SELECT count(*) FROM g_conc')"
n_err="$(grep -c ERROR "$ERRLOG" 2>/dev/null || echo 0)"

echo "after:  n|slots|corrupt|dup = $after"
echo "heap:   $heap_before -> $heap_after"
echo "insert errors: $n_err"
[ "$n_err" -gt 0 ] && { echo "--- first errors:"; grep ERROR "$ERRLOG" | head -5; }

fail=0
IFS='|' read -r a_n a_slots a_corrupt a_dup <<<"$after"
[ "$a_corrupt" = "true" ]  && { echo "FAIL: turbovec_check reports is_corrupt"; fail=1; }
[ "$a_dup" != "none" ]     && { echo "FAIL: duplicate id $a_dup on disk"; fail=1; }
[ "$a_n" != "$a_slots" ]   && { echo "FAIL: meta n_vectors=$a_n != ids-chain slots=$a_slots"; fail=1; }
[ "$n_err" -gt 0 ]         && { echo "FAIL: $n_err inserter error(s)"; fail=1; }
if [ "$SUSTAIN_SECS" -eq 0 ]; then
  want=$(( heap_before + W * ROWS ))
  [ "$heap_after" != "$want" ] && { echo "FAIL: heap has $heap_after rows, expected $want (lost rows)"; fail=1; }
  [ "$a_n" != "$heap_after" ] && { echo "FAIL: index has $a_n rows but the heap has $heap_after"; fail=1; }
fi

[ "$fail" -eq 0 ] && echo "PASS: graph index consistent after $W concurrent inserters"
exit "$fail"
