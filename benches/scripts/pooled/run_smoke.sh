#!/usr/bin/env bash
# benches/scripts/pooled/run_smoke.sh — collapsed single-host DIAGNOSTIC.
#
# >>> THIS IS A DIAGNOSTIC, NOT A HEADLINE. <<<
# It runs the DB, the pooler, and the load driver on ONE host (no separate
# metal DB, no isolated pooler instance, no load fleet, no >RAM corpus at
# TB scale). Per benches/METHODOLOGY.md a collapsed topology is a diagnostic
# only. Its job: PROVE the harness emits the right numbers and give a
# provisional before-number for Phase S1 — not a publishable result.
#
# It exercises the full chain with the SAME tier-agnostic scripts the real
# rig uses, differing only in that every tier points at localhost.
#
# Usage (defaults = 500k x 768-d flat, clients 1..64, both sp settings):
#   bash run_smoke.sh
#   CORPUS_ROWS=1000000 INDEX_KIND=ivf LISTS=1024 bash run_smoke.sh
#
# Requires: a running pgrx pg18 cluster on DB_PORT with pg_turbovec installed,
#           pgbouncer on PATH or /usr/sbin, and the datadir for restart.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOPOLOGY="collapsed-single-host DIAGNOSTIC (not a headline)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

PG_DATA="${PG_DATA:-$HOME/.pgrx/data-18}"
CLIENTS="${CLIENTS:-1 2 4 8 16 32 64}"
DURATION="${DURATION:-20}"
OPEN_RATES="${OPEN_RATES:-}"
# the two scan_parallelism settings METHODOLOGY.md requires both of.
SP_SETTINGS="${SP_SETTINGS:-auto 1}"

log "=== COLLAPSED SINGLE-HOST SMOKE (DIAGNOSTIC) ==="
dump_config

# 1. tune + load + build (once; same corpus/index for both sp settings).
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    bash "${HERE}/setup_db.sh"
    log "restarting cluster so shared_buffers=${SHARED_BUFFERS} takes effect"
    "${PG_BIN}/pg_ctl" -D "${PG_DATA}" restart -m fast -w >/dev/null
    # re-set turbovec cache after restart isn't needed (ALTER SYSTEM persists)
fi

# 2. pooler up, pointed at the DB tier.
bash "${HERE}/setup_pooler.sh" start

# 3. for each scan_parallelism setting: set the GUC cluster-wide, bounce the
#    pooler's server conns (so new backends pick it up), run the sweep.
JSONS=()
for sp in ${SP_SETTINGS}; do
    gucval="${sp}"; [ "${sp}" = "auto" ] && gucval=0
    log "--- scan_parallelism=${sp} (guc=${gucval}) ---"
    psql_db -d postgres -c "ALTER SYSTEM SET turbovec.scan_parallelism = ${gucval};"
    psql_db -d postgres -c "SELECT pg_reload_conf();" >/dev/null
    # drop pooled server conns so the next client gets a backend that reads
    # the new GUC (RELOAD in pgbouncer kills idle server conns).
    bash "${HERE}/setup_pooler.sh" reload
    sleep 2
    RUN_TAG="smoke_${INDEX_KIND}_sp${sp}_$(date -u +%Y%m%dT%H%M%SZ)"
    export RUN_TAG SCAN_PARALLELISM="${sp}" CLIENTS DURATION OPEN_RATES
    j="$(bash "${HERE}/run_load.sh")"
    JSONS+=("${j}")
    # aggregate per-sp into its own markdown
    python3 "${HERE}/aggregate.py" "${RESULTS_DIR}/${RUN_TAG}.md" "${j}"
done

# 4. fresh-backend cost (sp=1, the pooled setting).
psql_db -d postgres -c "ALTER SYSTEM SET turbovec.scan_parallelism = 1;"
psql_db -d postgres -c "SELECT pg_reload_conf();" >/dev/null
RUN_TAG="smoke_${INDEX_KIND}_fresh_$(date -u +%Y%m%dT%H%M%SZ)" \
    SCAN_PARALLELISM=1 SAMPLES="${FRESH_SAMPLES:-15}" \
    bash "${HERE}/fresh_backend_cost.sh" || log "fresh-backend probe failed (non-fatal)"

log "=== SMOKE DONE ==="
log "per-sp markdown + json under ${RESULTS_DIR}"
printf '%s\n' "${JSONS[@]}"
