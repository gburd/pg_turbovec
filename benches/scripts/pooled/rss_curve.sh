#!/usr/bin/env bash
# benches/scripts/pooled/rss_curve.sh — per-backend RSS vs client count.
#
# >>> The headline S0 datum. <<<
# pg_turbovec holds a PRIVATE decoded/SIMD-blocked copy of the index per
# backend (SCALE_PLAN bottleneck #1, src/cache.rs). N concurrent backends =
# N copies. This probe measures that wall directly: ramp N PERSISTENT
# backends (each builds its own cache on first query), sample every
# backend's VmRSS from /proc/<pid>/status while they are warm and busy, and
# report per-backend mean RSS + total backend RSS at each N — up to the point
# total crosses a safety fraction of host RAM (or a backend OOMs).
#
# Drives DIRECT to the DB (not the pooler) with N persistent pgbench clients
# so there are EXACTLY N backends, each with its own cache — the clean
# measurement. (A transaction pooler would multiplex and hide per-backend
# count.) This is a diagnostic, not a load test; QPS here is incidental.
#
# Repeats for each SCAN_PARALLELISM in SP_SETTINGS so we can confirm/refute
# that sp=1 raises the OOM ceiling only MODESTLY (it cuts threads, not the
# per-backend cache size — the cache is the wall, so S1 alone does not fix
# the memory wall; only S2 does).
#
# Usage:
#   N_LEVELS="1 2 4 8 16 24 32 48 64" SP_SETTINGS="auto 1" bash rss_curve.sh
#
# Env (plus common.sh):
#   N_LEVELS       backend counts to ramp through      [1 2 4 8 16 24 32 48 64]
#   SP_SETTINGS    scan_parallelism values to test     [auto 1]
#   HOLD_S         seconds to hold+sample at each level [8]
#   RAM_SAFE_FRAC  stop a curve when total backend RSS  [0.80]
#                  exceeds this fraction of MemTotal
#   SETTLE_S       warmup seconds before sampling       [4]

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOPOLOGY="${TOPOLOGY:-collapsed-single-host DIAGNOSTIC (not a headline)}"
# shellcheck source=common.sh
source "${HERE}/common.sh"

N_LEVELS="${N_LEVELS:-1 2 4 8 16 24 32 48 64}"
SP_SETTINGS="${SP_SETTINGS:-auto 1}"
HOLD_S="${HOLD_S:-8}"
SETTLE_S="${SETTLE_S:-4}"
RAM_SAFE_FRAC="${RAM_SAFE_FRAC:-0.80}"

MEM_TOTAL_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
SAFE_KB="$(awk -v m="${MEM_TOTAL_KB}" -v f="${RAM_SAFE_FRAC}" 'BEGIN{printf "%d", m*f}')"
OP="$(distop)"
dump_config
log "MemTotal=${MEM_TOTAL_KB}kB safe_ceiling=${SAFE_KB}kB (${RAM_SAFE_FRAC})"

# After a backend crash the postmaster runs crash recovery; wait until a
# trivial direct query succeeds again (bounded).
wait_db_healthy() { local i; for i in $(seq 1 60); do psql_db -tAc 'SELECT 1' >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

ts="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR}/${RUN_TAG:-rsscurve_${ts}}_rss_curve.json"
mkdir -p "${RESULTS_DIR}"

# pgbench script: one persistent client loops the indexed top-k query so its
# backend stays warm (cache built) and busy while we sample RSS.
SQLFILE="$(mktemp /tmp/rss_knn_XXXX.sql)"
cat >"${SQLFILE}" <<SQL
\\set qid random(0, ${QUERY_POOL} - 1)
SELECT id FROM docs
ORDER BY embedding OPERATOR(turbovec.${OP}) (SELECT q FROM bench_queries WHERE qid = :qid)
LIMIT ${K};
SQL

# identify backends serving OUR database (exclude checkpointer/bgwriter/etc).
# A user backend's /proc/<pid>/cmdline shows "postgres: <user> <db> ...".
backend_pids() {
    local p pids=()
    for p in $(pgrep -f "postgres:" 2>/dev/null || true); do
        [ -r "/proc/$p/cmdline" ] || continue
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q " ${PGDATABASE} "; then
            pids+=("$p")
        fi
    done
    echo "${pids[@]:-}"
}

# sum VmRSS (kB) across backend pids; also echo count + mean.
sample_rss() {
    local pids total=0 n=0 p rss
    pids="$(backend_pids)"
    for p in ${pids}; do
        rss="$(awk '/^VmRSS:/{print $2}' "/proc/$p/status" 2>/dev/null || echo 0)"
        total=$(( total + ${rss:-0} )); n=$(( n + 1 ))
    done
    local mean=0; [ "$n" -gt 0 ] && mean=$(( total / n ))
    echo "${n} ${total} ${mean}"   # count total_kb mean_kb
}

run_curve() {   # $1 = sp label
    local sp="$1" gucval="$1"; [ "$sp" = "auto" ] && gucval=0
    log "=== RSS curve: scan_parallelism=${sp} (guc=${gucval}) ==="
    psql_db -d postgres -c "ALTER SYSTEM SET turbovec.scan_parallelism = ${gucval};" >/dev/null
    psql_db -d postgres -c "SELECT pg_reload_conf();" >/dev/null
    local rows="" oomed=0 last_n=0 prev_mean=0
    for N in ${N_LEVELS}; do
        [ "${oomed}" = 1 ] && break
        # PREDICTIVE SAFETY: if the previous level's mean-per-backend implies
        # this N would blow past MemTotal, don't run it (deliberately OOMing
        # a shared host crashes the whole cluster). Record a predicted point.
        if [ "${prev_mean}" -gt 0 ]; then
            local pred=$(( prev_mean * N ))
            if awk -v p="${pred}" -v m="${MEM_TOTAL_KB}" 'BEGIN{exit !(p > m*0.92)}'; then
                local pf; pf="$(awk -v p="${pred}" -v s="${SAFE_KB}" 'BEGIN{printf "%.3f", p/s}')"
                log "    N=${N}: PREDICTED total ~${pred}kB > 92%% MemTotal — not run (would OOM host); projected ${pf} of safe"
                [ -n "${rows}" ] && rows="${rows},"
                rows="${rows}
      {\"clients\":${N},\"predicted_total_backend_rss_kb\":${pred},\"projected_frac_of_safe\":${pf},\"not_run_would_oom\":true}"
                oomed=1; break
            fi
        fi
        log "  ramp ${N} persistent backends, hold ${HOLD_S}s"
        # launch N persistent clients, each looping forever (-T large); we
        # kill pgbench after sampling. --max-tries=1; no -C so backends persist.
        local lp; lp="$(mktemp -u /tmp/rsspgb_XXXX)"
        "${PGBENCH}" -h "${DB_HOST}" -p "${DB_PORT}" -U "${PGUSER}" -d "${PGDATABASE}" \
            -n -f "${SQLFILE}" -c "${N}" -j "$(( N < $(nproc) ? N : $(nproc) ))" \
            -T "$(( SETTLE_S + HOLD_S + 2 ))" --max-tries=1 >"${lp}.out" 2>&1 &
        local pgb=$!
        sleep "${SETTLE_S}"
        # sample a few times across HOLD_S, take the max (peak RSS).
        local peak_total=0 peak_n=0 peak_mean=0 s
        for s in $(seq 1 "${HOLD_S}"); do
            if ! kill -0 "${pgb}" 2>/dev/null; then break; fi
            read -r cn ct cm <<<"$(sample_rss)"
            if [ "${ct:-0}" -gt "${peak_total}" ]; then peak_total=${ct}; peak_n=${cn}; peak_mean=${cm}; fi
            sleep 1
        done
        wait "${pgb}" 2>/dev/null || true
        # did a backend die / OOM during this level?
        local crashed=0
        if grep -qiE 'server closed the connection|crashed|out of memory|terminating connection' "${lp}.out" 2>/dev/null; then crashed=1; oomed=1; fi
        rm -f "${lp}.out" 2>/dev/null || true

        local frac; frac="$(awk -v t="${peak_total}" -v s="${SAFE_KB}" 'BEGIN{if(s>0)printf "%.3f", t/s; else print 0}')"
        log "    N=${N}: backends=${peak_n} total_rss=${peak_total}kB mean=${peak_mean}kB (${frac} of safe); crashed=${crashed}"
        [ -n "${rows}" ] && rows="${rows},"
        rows="${rows}
      {\"clients\":${N},\"backends_sampled\":${peak_n},\"total_backend_rss_kb\":${peak_total},\"mean_backend_rss_kb\":${peak_mean},\"frac_of_safe_ceiling\":${frac},\"crashed\":${crashed}}"
        last_n=${N}
        [ "${peak_mean}" -gt 0 ] && prev_mean=${peak_mean}
        # stop BEFORE deliberately OOMing the host if this level is already
        # near the ceiling (safety for a shared box).
        if awk -v t="${peak_total}" -v s="${SAFE_KB}" 'BEGIN{exit !(t>s)}'; then
            log "    reached safe ceiling at N=${N}; stopping this curve"
            oomed=1
        fi
        wait_db_healthy || true
    done
    cat <<JSON
  {"scan_parallelism":"${sp}","oom_or_ceiling_at_clients":${last_n},"points":[${rows}
  ]}
JSON
}

{
    echo "{"
    echo "  \"timestamp_utc\": \"${ts}\", \"topology\": \"${TOPOLOGY}\","
    echo "  \"host\": \"$(uname -n)\", \"ncpu\": $(nproc), \"mem_total_kb\": ${MEM_TOTAL_KB},"
    echo "  \"ram_safe_frac\": ${RAM_SAFE_FRAC}, \"safe_ceiling_kb\": ${SAFE_KB},"
    echo "  \"corpus_rows\": ${CORPUS_ROWS}, \"dim\": ${DIM}, \"index_kind\": \"${INDEX_KIND}\","
    echo "  \"lists\": ${LISTS}, \"bit_width\": ${BIT_WIDTH}, \"k\": ${K}, \"shared_buffers\": \"${SHARED_BUFFERS}\","
    echo "  \"note\": \"per-backend RSS = PRIVATE decoded cache (SCALE_PLAN #1). N backends x mean_rss crosses host RAM => OOM. sp=1 cuts threads not cache size, so it should raise the ceiling only modestly.\","
    echo "  \"curves\": ["
} > "${OUT}"

first=1
for sp in ${SP_SETTINGS}; do
    [ "${first}" = 1 ] && first=0 || echo "," >> "${OUT}"
    run_curve "${sp}" >> "${OUT}"
done
echo "  ]" >> "${OUT}"
echo "}" >> "${OUT}"
rm -f "${SQLFILE}"
log "wrote ${OUT}"
echo "${OUT}"
