#!/usr/bin/env bash
# benches/scripts/pooled/fresh_backend_cost.sh — the fresh-backend cost probe.
#
# pg_turbovec builds its per-backend decoded/SIMD-blocked cache on the FIRST
# query a backend runs against an index (src/cache.rs). Under a transaction
# pooler, server_lifetime / max_db_connections churn recycles backends, so a
# fraction of queries land on a fresh backend and pay that build cost.
# METHODOLOGY.md requires this as its own row.
#
# This probe measures the first-query penalty directly: open N fresh server
# connections, time query #1 (cold cache build) vs query #2..M (warm) on the
# SAME connection. The delta is the per-backend cache-build cost. It then
# estimates how often a pooled workload pays it from server_lifetime and the
# measured warm QPS/backend.
#
# Connects DIRECT to the DB tier (not the pooler) so each psql = a guaranteed
# fresh backend; that is exactly what a recycled pooled backend looks like on
# its first query.
#
# Usage:
#   SAMPLES=20 bash fresh_backend_cost.sh
#
# Env (plus common.sh):
#   SAMPLES    number of fresh backends to probe   [20]
#   WARM_REPS  warm queries after the cold one      [5]

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

SAMPLES="${SAMPLES:-20}"
WARM_REPS="${WARM_REPS:-5}"
OP="$(distop)"
dump_config

# One probe = a fresh psql (new backend). Inside it: timing the first indexed
# scan (cold) then WARM_REPS more (warm). \timing prints ms per statement.
probe_sql() {
    cat <<SQL
\\set qid 0
SET enable_seqscan = off;
-- bind a query vector first (its own txn/statement, not timed as the scan)
SELECT q AS qv FROM bench_queries WHERE qid = 0 \\gset
\\timing on
-- COLD: first index scan on this backend -> builds the per-backend cache
SELECT id FROM docs ORDER BY embedding OPERATOR(turbovec.${OP}) :'qv'::turbovec.vector LIMIT ${K};
SQL
    for _ in $(seq 1 "${WARM_REPS}"); do
        echo "SELECT id FROM docs ORDER BY embedding OPERATOR(turbovec.${OP}) :'qv'::turbovec.vector LIMIT ${K};"
    done
}

ts="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR}/${RUN_TAG}_fresh_backend.json"
mkdir -p "${RESULTS_DIR}"

cold_list=(); warm_list=()
log "probing ${SAMPLES} fresh backends (cold query #1 vs ${WARM_REPS} warm)..."
for s in $(seq 1 "${SAMPLES}"); do
    # fresh connection each iteration; parse "Time: X ms" lines in order.
    times="$(probe_sql | psql_db -At 2>&1 | grep -oE 'Time: [0-9.]+ ms' | grep -oE '[0-9.]+')"
    mapfile -t arr <<<"${times}"
    [ "${#arr[@]}" -ge 2 ] || { log "  sample ${s}: insufficient timings, skip"; continue; }
    cold="${arr[0]}"
    # mean of warm reps
    warm="$(printf '%s\n' "${arr[@]:1}" | awk '{s+=$1;n++} END{if(n)printf "%.3f", s/n; else print 0}')"
    cold_list+=("${cold}"); warm_list+=("${warm}")
done

stats() {  # median p95 of a list
    printf '%s\n' "$@" | sort -n | awk '
        {a[n++]=$1}
        END{ if(n==0){print "0 0 0";exit}
             printf "%.3f %.3f %.3f", a[int(.5*(n-1))], a[int(.95*(n-1))], a[n-1] }'
}
read -r cold_med cold_p95 cold_max <<<"$(stats "${cold_list[@]}")"
read -r warm_med warm_p95 warm_max <<<"$(stats "${warm_list[@]}")"
penalty="$(awk -v c="${cold_med}" -v w="${warm_med}" 'BEGIN{printf "%.3f", c-w}')"

# how often a pooled workload pays it:
#   a backend lives server_lifetime seconds, serves ~ warm_qps_per_backend *
#   server_lifetime queries, pays the cold penalty once => fraction =
#   1 / (warm_qps * server_lifetime). warm_qps ~ 1000/warm_med.
SERVER_LIFETIME="${SERVER_LIFETIME:-600}"
frac="$(awk -v w="${warm_med}" -v L="${SERVER_LIFETIME}" 'BEGIN{
    if(w>0){ qps=1000.0/w; q=qps*L; if(q>0) printf "%.6f", 1.0/q; else print 1 } else print 1 }')"

cat > "${OUT}" <<JSON
{
  "timestamp_utc": "${ts}", "run_tag": "${RUN_TAG}",
  "topology": "${TOPOLOGY:-collapsed-single-host DIAGNOSTIC}",
  "corpus_rows": ${CORPUS_ROWS}, "dim": ${DIM}, "index_kind": "${INDEX_KIND}",
  "lists": ${LISTS}, "bit_width": ${BIT_WIDTH}, "k": ${K},
  "scan_parallelism": "${SCAN_PARALLELISM}", "samples": ${#cold_list[@]},
  "cold_first_query_ms": {"median": ${cold_med}, "p95": ${cold_p95}, "max": ${cold_max}},
  "warm_query_ms":       {"median": ${warm_med}, "p95": ${warm_p95}, "max": ${warm_max}},
  "cache_build_penalty_ms": ${penalty},
  "server_lifetime_s": ${SERVER_LIFETIME},
  "fraction_of_queries_paying_penalty": ${frac},
  "note": "cold = first indexed scan on a fresh backend (builds per-backend cache, src/cache.rs); warm = steady state. Fraction assumes one cold hit per backend lifetime at the measured warm rate."
}
JSON
log "fresh-backend: cold_med=${cold_med}ms warm_med=${warm_med}ms penalty=${penalty}ms (~${frac} of queries pay it at server_lifetime=${SERVER_LIFETIME}s)"
echo "${OUT}"
