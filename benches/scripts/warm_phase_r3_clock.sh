#!/usr/bin/env bash
# Phase R-3 secondary: same methodology as Phase O-3 -- single
# psql session, 2 untimed warmups, 50 timed via bench_one_query_tv
# (clock_timestamp inside plpgsql, not EXPLAIN ANALYZE). This
# isolates the v1.4.0 vs v1.3.0 delta from EXPLAIN ANALYZE
# instrumentation overhead.
set -euo pipefail
export LD_LIBRARY_PATH=/lib64

PG_BIN=${PG_BIN:-$HOME/.pgrx/17.9/pgrx-install/bin}
PG_PORT=${PG_PORT:-28815}
PG_SOCK=${PG_SOCK:-/scratch/pg_turbovec-bench}
PG_DB=${PG_DB:-bench_dbpedia}
OUT_DIR=${OUT_DIR:-/scratch/pg_turbovec-bench/phase_r3}
SEARCH_K=${SEARCH_K:-100}

mkdir -p "$OUT_DIR"
TSV="$OUT_DIR/warm_phase_r3_clock.tsv"
STATS="$OUT_DIR/warm_phase_r3_clock.stats"
LOG="$OUT_DIR/warm_phase_r3_clock.log"

PSQL="$PG_BIN/psql -X -P pager=off -h $PG_SOCK -p $PG_PORT -d $PG_DB"

SQL_FILE=$(mktemp)
{
    cat <<'SQL'
SET search_path = turbovec, public;
SET enable_seqscan = off;
SET jit = off;
SET turbovec.search_k = 100;
\pset format unaligned
\pset tuples_only on
-- warmup 1 (pays per-backend init, untimed report we discard)
SELECT 'warmup1', ms FROM bench_one_query_tv(1);
-- warmup 2 (cache-warm steady-state)
SELECT 'warmup2', ms FROM bench_one_query_tv(1);
SQL
    for qid in $(seq 1 50); do
        printf "SELECT 'TIMED-%d', ms FROM bench_one_query_tv(%d);\n" "$qid" "$qid"
    done
} > "$SQL_FILE"

echo "=== running warm sweep (bench_one_query_tv) against $PG_DB ==="
$PSQL -f "$SQL_FILE" > "$LOG" 2>&1
rm -f "$SQL_FILE"

# Parse "TIMED-N|ms" lines.
: > "$TSV"
awk -F'|' '/^TIMED-/ { sub(/TIMED-/, "", $1); print $1 "\t" $2 }' "$LOG" > "$TSV"

n=$(wc -l < "$TSV")
echo "=== captured $n timed rows ==="

awk -F'\t' '
    { vals[NR] = $2; sum += $2 }
    END {
        if (NR == 0) { exit 1 }
        n = NR
        for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (vals[i] > vals[j]) { t = vals[i]; vals[i] = vals[j]; vals[j] = t }
        p50_idx = int((n+1) * 0.50); if (p50_idx < 1) p50_idx = 1
        p95_idx = int((n+1) * 0.95); if (p95_idx > n) p95_idx = n
        printf "n=%d\nmin=%.3f\np50=%.3f\np95=%.3f\nmax=%.3f\nmean=%.3f\n", n, vals[1], vals[p50_idx], vals[p95_idx], vals[n], sum/n
    }
' "$TSV" | tee "$STATS"

echo "=== done ==="
echo "tsv:    $TSV"
echo "stats:  $STATS"
echo "log:    $LOG"
