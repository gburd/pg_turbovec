#!/usr/bin/env bash
# Phase R-3: arnold warm-scan p50 re-validation post-Phase-R-2 (v1.4.0).
#
# Methodology mirrors Phase J / Phase O-3 warm-scan measurement:
#   - one persistent psql session (single backend)
#   - 2 untimed warmup queries against qid=1 (re-pays per-backend init cost)
#   - 50 timed queries, qid=1..50, each captured via EXPLAIN ANALYZE,
#     parsing the "Execution Time" line.
#
# Output:
#   $OUT_DIR/warm_phase_r3.tsv     - one row per query: qid TAB ms
#   $OUT_DIR/warm_phase_r3.stats   - min/p50/p95/max/mean
#   $OUT_DIR/warm_phase_r3.log     - full psql output (audit trail)
set -euo pipefail
export LD_LIBRARY_PATH=/lib64

PG_BIN=${PG_BIN:-$HOME/.pgrx/17.9/pgrx-install/bin}
PG_PORT=${PG_PORT:-28815}
PG_SOCK=${PG_SOCK:-/scratch/pg_turbovec-bench}
PG_DB=${PG_DB:-bench_dbpedia}
OUT_DIR=${OUT_DIR:-/scratch/pg_turbovec-bench/phase_r3}
SEARCH_K=${SEARCH_K:-100}

mkdir -p "$OUT_DIR"
TSV="$OUT_DIR/warm_phase_r3.tsv"
STATS="$OUT_DIR/warm_phase_r3.stats"
LOG="$OUT_DIR/warm_phase_r3.log"

PSQL="$PG_BIN/psql -X -P pager=off -h $PG_SOCK -p $PG_PORT -d $PG_DB"

# Build the input SQL stream:
#   1. session setup (search_path, search_k, force planner to use turbovec)
#   2. 2 untimed warmups against qid=1
#   3. 50 timed EXPLAIN ANALYZE queries, qid=1..50
#
# We use a single \echo MARK-<qid> sentinel in front of every timed
# query so the log parser can pair the qid with the Execution Time line
# below it.
SQL_FILE=$(mktemp)
{
    cat <<'SQL'
SET search_path = turbovec, public;
SET enable_seqscan = off;
SET jit = off;
SET turbovec.search_k = 100;
\timing off
-- warmup 1: pays per-backend init cost (Lloyd-Max codebook + blocked layout)
SELECT id FROM docs ORDER BY (emb::real[]::turbovec.vector)
    OPERATOR(turbovec.<=>)
    (SELECT emb::real[]::turbovec.vector FROM query_set WHERE qid = 1)
    LIMIT 10;
-- warmup 2: cache-warm steady-state
SELECT id FROM docs ORDER BY (emb::real[]::turbovec.vector)
    OPERATOR(turbovec.<=>)
    (SELECT emb::real[]::turbovec.vector FROM query_set WHERE qid = 1)
    LIMIT 10;
SQL
    for qid in $(seq 1 50); do
        printf '\\echo MARK-%d\n' "$qid"
        cat <<SQL
EXPLAIN (ANALYZE, COSTS off, BUFFERS off, TIMING on)
SELECT id FROM docs ORDER BY (emb::real[]::turbovec.vector)
    OPERATOR(turbovec.<=>)
    (SELECT emb::real[]::turbovec.vector FROM query_set WHERE qid = ${qid})
    LIMIT 10;
SQL
    done
} > "$SQL_FILE"

echo "=== running warm sweep against $PG_DB ==="
$PSQL -f "$SQL_FILE" > "$LOG" 2>&1
rm -f "$SQL_FILE"

# Parse: walk the log, and for each "MARK-N" capture the next
# "Execution Time:" line.
: > "$TSV"
awk '
    /^MARK-/ { qid = substr($0, 6); next }
    /Execution Time:/ {
        if (qid != "") {
            for (i = 1; i <= NF; i++) {
                if ($i == "Time:") { print qid "\t" $(i+1); break }
            }
            qid = ""
        }
    }
' "$LOG" > "$TSV"

n=$(wc -l < "$TSV")
echo "=== captured $n timed rows ==="

# Stats from the TSV (sort numerically by ms; pick min, p50, p95, max, mean).
awk -F'\t' '
    { vals[NR] = $2; sum += $2 }
    END {
        if (NR == 0) { exit 1 }
        n = NR
        # bubble sort is fine for 50 rows
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
