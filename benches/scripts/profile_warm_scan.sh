#!/usr/bin/env bash
# benches/scripts/profile_warm_scan.sh
#
# Item 5 from the v1.2.0 audit proposal: profile the warm-scan path
# to understand the 1.3-1.6× warm-p50 multiplier vs HNSW on
# dbpedia-1M (Phase J). This script captures `perf record` output
# for a 50-iteration warm-cache loop hitting the AM through the
# planner's reorder-by path.
#
# Usage (on arnold or another bench host with the bench_dbpedia
# corpus loaded):
#
#   bash benches/scripts/profile_warm_scan.sh
#
# Produces (under $OUT_DIR, default /tmp):
#   <OUT_DIR>/turbovec-warm-scan.perf      # raw perf data
#   <OUT_DIR>/turbovec-warm-scan-symbols.txt # top-50 hot symbols
#   <OUT_DIR>/turbovec-warm-scan-flame.svg  # FlameGraph (if available)
#
# Prereqs:
#   - kernel.perf_event_paranoid <= 1 (no sudo escalation here).
#   - PG cluster with pg_turbovec installed and indexed on docs.emb.
#   - LD_LIBRARY_PATH=/lib64 on Fedora hosts (matches arnold).
#
# Env knobs (with arnold-specific defaults):
#   PG_BIN     PostgreSQL bin dir (default $HOME/.pgrx/17.9/pgrx-install/bin)
#   PG_PORT    cluster port           (default 28815)
#   PG_SOCK    Unix socket directory  (default /scratch/pg_turbovec-bench)
#   PG_DB      database               (default bench_dbpedia)
#   PERF_BIN   `perf` binary          (auto-detected; falls back to nix store)
#   FG_BIN_DIR FlameGraph bin dir     (auto-detected from /nix/store on arnold)
#   OUT_DIR    output directory       (default /tmp)
#   FREQ_HZ    perf sampling rate     (default 999)
#   DURATION_S perf record duration   (default 10)
#   SEARCH_K   turbovec.search_k      (default 100)

set -euo pipefail

PG_BIN=${PG_BIN:-$HOME/.pgrx/17.9/pgrx-install/bin}
PG_PORT=${PG_PORT:-28815}
PG_SOCK=${PG_SOCK:-/scratch/pg_turbovec-bench}
PG_DB=${PG_DB:-bench_dbpedia}
OUT_DIR=${OUT_DIR:-/tmp}
FREQ_HZ=${FREQ_HZ:-999}
DURATION_S=${DURATION_S:-10}
SEARCH_K=${SEARCH_K:-100}

mkdir -p "$OUT_DIR"
PERF_OUT="$OUT_DIR/turbovec-warm-scan.perf"
SYM_OUT="$OUT_DIR/turbovec-warm-scan-symbols.txt"
FLAME_OUT="$OUT_DIR/turbovec-warm-scan-flame.svg"

# --- auto-detect perf -------------------------------------------------
# perf binaries from /nix/store crash with the host's LD_LIBRARY_PATH=/lib64
# (needed for the PG server side). Drop it whenever we shell out to perf.
perf_run() { env -u LD_LIBRARY_PATH "$PERF_BIN" "$@"; }

if [ -z "${PERF_BIN:-}" ]; then
    if command -v perf >/dev/null 2>&1; then
        PERF_BIN=$(command -v perf)
    else
        # arnold: linux-tools isn't packaged on Fedora, but a working
        # perf ships in the user's home-manager nix profile. Prefer
        # the version that matches the running kernel (e.g. 7.0.9 on
        # current arnold) — older perf builds segfault on this host.
        kver=$(uname -r | cut -d- -f1)
        for cand in \
            $(ls -1 /nix/store/*-perf-linux-${kver}/bin/perf 2>/dev/null) \
            $(ls -1 /nix/store/*-perf-linux-*/bin/perf 2>/dev/null \
                | sort -t- -k4,4 -V -r); do
            if [ -x "$cand" ] && env -u LD_LIBRARY_PATH "$cand" --version >/dev/null 2>&1; then
                PERF_BIN=$cand
                break
            fi
        done
    fi
fi
if [ -z "${PERF_BIN:-}" ] || [ ! -x "$PERF_BIN" ]; then
    echo "ERROR: no perf binary found. Install linux-tools or set PERF_BIN." >&2
    exit 3
fi

# --- check perf_event_paranoid ----------------------------------------
PARANOID=$(sysctl -n kernel.perf_event_paranoid 2>/dev/null || echo 'unknown')
echo "kernel.perf_event_paranoid=$PARANOID"
if [ "$PARANOID" != "unknown" ] && [ "$PARANOID" -gt 1 ] 2>/dev/null; then
    echo "WARNING: paranoid=$PARANOID may block userspace perf. Continuing anyway." >&2
fi

# --- auto-detect FlameGraph -------------------------------------------
if [ -z "${FG_BIN_DIR:-}" ]; then
    for cand in /opt/FlameGraph /home/gburd/src/flamegraph $(ls -1d /nix/store/*-FlameGraph-*/bin 2>/dev/null); do
        if [ -x "$cand/flamegraph.pl" ] && [ -x "$cand/stackcollapse-perf.pl" ]; then
            FG_BIN_DIR=$cand
            break
        fi
    done
fi

PSQL="$PG_BIN/psql --no-psqlrc -P pager=off -h $PG_SOCK -p $PG_PORT -d $PG_DB"

echo "=== using PERF_BIN=$PERF_BIN"
echo "=== using PG=$PG_BIN/psql -h $PG_SOCK -p $PG_PORT -d $PG_DB"
echo "=== output dir: $OUT_DIR"

echo "=== warming the cache (run query once) ==="
$PSQL -c "
SET search_path = turbovec, public;
SET enable_seqscan = off;
SET jit = off;
SET turbovec.search_k = $SEARCH_K;
SELECT id FROM docs ORDER BY (emb::real[]::vector) <=>
    (SELECT emb::real[]::vector FROM query_set WHERE qid = 1) LIMIT 10;
" >/dev/null

echo "=== finding the postgres backend pid (long-lived session) ==="
# Open a single backend that we'll attach perf to AND drive the
# 50-query loop through. This is the only way to ensure perf
# samples exactly the backend that's executing the workload — the
# previous version of this script grabbed a pg_backend_pid() from
# one psql session and ran the query loop in another.
FIFO_IN=$(mktemp -u "$OUT_DIR/turbovec-warm.psql-in.XXXXXX")
FIFO_OUT=$(mktemp -u "$OUT_DIR/turbovec-warm.psql-out.XXXXXX")
mkfifo "$FIFO_IN" "$FIFO_OUT"

# Background psql: read commands from FIFO_IN, write echoed PID + DONE
# markers to FIFO_OUT.
$PSQL -At -q < "$FIFO_IN" > "$FIFO_OUT" 2>&1 &
PSQL_PID=$!
exec 7> "$FIFO_IN"
exec 8< "$FIFO_OUT"

cleanup() {
    exec 7>&- 2>/dev/null || true
    exec 8<&- 2>/dev/null || true
    kill "$PSQL_PID" 2>/dev/null || true
    rm -f "$FIFO_IN" "$FIFO_OUT"
}
trap cleanup EXIT

# Ask the backend to print its PID then a sentinel.
printf 'SELECT pg_backend_pid();\n' >&7
printf "SELECT 'PID_DONE';\n" >&7

PG_BACKEND_PID=""
while IFS= read -r -t 10 line <&8; do
    case "$line" in
        PID_DONE) break ;;
        ''|'('*|*' rows)') ;;
        [0-9]*) PG_BACKEND_PID=$line ;;
    esac
done

if [ -z "$PG_BACKEND_PID" ]; then
    echo "ERROR: could not capture pg_backend_pid()" >&2
    exit 4
fi
echo "backend pid: $PG_BACKEND_PID"

# Configure session, no output noise.
printf 'SET search_path = turbovec, public;\n' >&7
printf 'SET enable_seqscan = off;\n' >&7
printf 'SET jit = off;\n' >&7
printf 'SET turbovec.search_k = %s;\n' "$SEARCH_K" >&7

echo "=== launching perf record (50-iteration warm loop, $DURATION_S s @ $FREQ_HZ Hz) ==="
perf_run record -F "$FREQ_HZ" -p "$PG_BACKEND_PID" -g \
    -o "$PERF_OUT" -- sleep "$DURATION_S" &
PERF_PID=$!

# Drive a tight 50-query loop on the same backend. The DO block
# runs server-side so there's no roundtrip per iteration.
printf "DO \$\$ DECLARE q vector; BEGIN
  SELECT emb::real[]::vector INTO q FROM query_set WHERE qid = 1;
  FOR i IN 1..50 LOOP
    PERFORM id FROM docs ORDER BY (emb::real[]::vector) <=> q LIMIT 10;
  END LOOP;
END \$\$;\n" >&7
printf "SELECT 'LOOP_DONE';\n" >&7

# Drain output until we see LOOP_DONE OR perf finishes.
LOOP_DONE=0
while IFS= read -r -t "$((DURATION_S * 4))" line <&8; do
    case "$line" in
        LOOP_DONE) LOOP_DONE=1; break ;;
    esac
done

wait "$PERF_PID" || true
echo "loop_done=$LOOP_DONE"

# Close session cleanly.
printf '\\q\n' >&7
wait "$PSQL_PID" 2>/dev/null || true

if [ ! -s "$PERF_OUT" ]; then
    echo "ERROR: perf output is empty: $PERF_OUT" >&2
    exit 5
fi

echo "=== top-50 symbols ==="
perf_run report -i "$PERF_OUT" --stdio --no-source -g none 2>/dev/null \
    | head -120 \
    | tee "$SYM_OUT"

if [ -n "${FG_BIN_DIR:-}" ]; then
    echo "=== generating flame graph ($FG_BIN_DIR) ==="
    perf_run script -i "$PERF_OUT" 2>/dev/null \
        | "$FG_BIN_DIR/stackcollapse-perf.pl" \
        | "$FG_BIN_DIR/flamegraph.pl" --title "pg_turbovec warm scan (4-bit)" \
        > "$FLAME_OUT"
    echo "flame graph: $FLAME_OUT"
else
    echo "=== FlameGraph not found, skipping SVG ==="
fi

echo "=== done."
echo "    perf data:    $PERF_OUT"
echo "    top symbols:  $SYM_OUT"
[ -f "$FLAME_OUT" ] && echo "    flame graph:  $FLAME_OUT"
