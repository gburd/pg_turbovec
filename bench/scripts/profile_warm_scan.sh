#!/usr/bin/env bash
# bench/scripts/profile_warm_scan.sh
#
# Item 5 from the v1.2.0 audit proposal: profile the warm-scan path
# to understand the 1.3-1.6× warm-p50 multiplier vs HNSW on
# dbpedia-1M (Phase J). This script captures `perf record` output
# for a single `EXPLAIN ANALYZE … ORDER BY emb <=> q LIMIT 10`
# query running through the AM path with the cache hot.
#
# Usage (on arnold or another bench host with the bench_dbpedia
# corpus loaded):
#
#   bash bench/scripts/profile_warm_scan.sh
#
# Produces:
#   /tmp/turbovec-warm-scan.perf      # raw perf data
#   /tmp/turbovec-warm-scan-flame.svg  # FlameGraph (if installed)
#   /tmp/turbovec-warm-scan-symbols.txt # top-50 hot symbols
#
# Prereqs:
#   - linux-tools-common / perf installed (apt) or
#     `sudo perf record` permissions.
#   - FlameGraph repo cloned at /opt/FlameGraph (optional).
#   - PG cluster running with pg_turbovec installed and indexed.
#   - LD_LIBRARY_PATH=/lib64 if Fedora (matches arnold convention).

set -euo pipefail
PG_BIN=${PG_BIN:-$HOME/.pgrx/17.9/pgrx-install/bin}
PG_PORT=${PG_PORT:-28815}
PG_SOCK=${PG_SOCK:-/scratch/pg_turbovec-bench}
PG_DB=${PG_DB:-bench_dbpedia}

PSQL="$PG_BIN/psql -P pager=off -h $PG_SOCK -p $PG_PORT -d $PG_DB"

echo "=== warming the cache (run query once) ==="
$PSQL -c "
SET search_path = turbovec, public;
SET enable_seqscan = off;
SET turbovec.search_k = 100;
SELECT id FROM docs ORDER BY (emb::real[]::vector) <=> \
    (SELECT emb::real[]::vector FROM query_set WHERE qid = 1) LIMIT 10;
" >/dev/null

echo "=== finding the postgres backend pid ==="
PG_BACKEND_PID=$($PSQL -At -c "SELECT pg_backend_pid();")
echo "backend pid: $PG_BACKEND_PID"

# Race a perf attach with a 50-query loop. perf needs root or
# kernel.perf_event_paranoid <= 1; check with `sysctl
# kernel.perf_event_paranoid`.
echo "=== launching perf record (50-iteration warm loop) ==="
$PSQL -c "
DO \$\$
DECLARE q vector;
BEGIN
  SELECT emb::real[]::vector INTO q FROM query_set WHERE qid = 1;
  FOR i IN 1..50 LOOP
    PERFORM id FROM docs ORDER BY (emb::real[]::vector) <=> q LIMIT 10;
  END LOOP;
END\$\$;
" &
QUERY_PID=$!

# Sample the backend at 999 Hz for ~10 seconds (matches the
# 50-query loop's runtime at ~70 ms/q on dbpedia + relfile).
sudo perf record -F 999 -p $PG_BACKEND_PID -g \
    -o /tmp/turbovec-warm-scan.perf -- sleep 10 || true

wait $QUERY_PID

echo "=== top-50 symbols ==="
sudo perf report -i /tmp/turbovec-warm-scan.perf --stdio --no-source -g none \
    | head -80 \
    > /tmp/turbovec-warm-scan-symbols.txt
cat /tmp/turbovec-warm-scan-symbols.txt

if [ -d /opt/FlameGraph ]; then
    echo "=== generating flame graph ==="
    sudo perf script -i /tmp/turbovec-warm-scan.perf \
        | /opt/FlameGraph/stackcollapse-perf.pl \
        | /opt/FlameGraph/flamegraph.pl \
        > /tmp/turbovec-warm-scan-flame.svg
    echo "flame graph: /tmp/turbovec-warm-scan-flame.svg"
fi

echo "=== done. symbols at /tmp/turbovec-warm-scan-symbols.txt ==="
