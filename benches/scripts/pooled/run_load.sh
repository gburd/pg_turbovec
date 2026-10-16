#!/usr/bin/env bash
# benches/scripts/pooled/run_load.sh — the load driver.
#
# Drives the pooler (POOLER_HOST:POOLER_PORT) with pgbench, sweeping client
# concurrency past saturation. Two modes per sweep point:
#   * CLOSED loop  — no -R; clients fire as fast as replies arrive (max QPS).
#   * OPEN loop    — pgbench -R <rate> fixed arrival (Poisson) for tail
#                    latency under queueing. Rates default to a fraction of
#                    the measured closed-loop knee.
#
# Captures per run: QPS (tps excluding reconnections), QPS/core, p50/p95/
# p99/p99.9 latency, and DB-backend CPU from /proc (sampled around the run).
# Emits one JSON file aggregating all runs.
#
# On the real rig this script runs on EACH load-fleet instance (same args,
# distinct DRIVER_ID); aggregate.py folds the per-driver JSONs together.
# On the collapsed diagnostic it runs once on the DB host.
#
# Usage:
#   CLIENTS="1 2 4 8 16 32 64" DURATION=30 bash run_load.sh
#   OPEN_RATES="500 1000 2000" bash run_load.sh         # also open-loop
#
# Key env (plus everything in common.sh):
#   CLIENTS       client counts to sweep (closed loop)       [1 2 4 8 16 32 64]
#   DURATION      seconds per run                            [30]
#   WARMUP        seconds discarded at start of each run     [5]
#   JOBS          pgbench threads (<= clients, <= cores-1)   [min(clients,ncpu)]
#   OPEN_RATES    tx/s targets for open-loop runs ("" = skip)[""]
#   OPEN_CLIENTS  client count used for open-loop runs       [64]
#   DRIVER_ID     label for this fleet node                  [local]
#   DB_PID_HINT   grep pattern to find DB backends in /proc  [postgres]

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

CLIENTS="${CLIENTS:-1 2 4 8 16 32 64}"
DURATION="${DURATION:-30}"
WARMUP="${WARMUP:-5}"
OPEN_RATES="${OPEN_RATES:-}"
OPEN_CLIENTS="${OPEN_CLIENTS:-64}"
DRIVER_ID="${DRIVER_ID:-local}"
DB_PID_HINT="${DB_PID_HINT:-postgres}"
NCPU="$(nproc)"

OP="$(distop)"
# Schema-qualified operator so the query resolves through a transaction-mode
# pooler that drops SET search_path. One txn = one indexed top-k ORDER BY.
SQLFILE="$(mktemp /tmp/pooled_knn_XXXX.sql)"
cat >"${SQLFILE}" <<SQL
\\set qid random(0, ${QUERY_POOL} - 1)
SELECT id FROM docs
ORDER BY embedding OPERATOR(turbovec.${OP}) (SELECT q FROM bench_queries WHERE qid = :qid)
LIMIT ${K};
SQL

ts="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR}/${RUN_TAG}_${DRIVER_ID}.json"
mkdir -p "${RESULTS_DIR}"
dump_config

# ---- backend CPU sampling (from /proc; works same-host or via node_exporter
#      on the real rig — here we read /proc of local postgres backends) ------
# Sum utime+stime jiffies across matching backends; delta / wall / CLK_TCK
# / ncpu => mean cores busy. Only meaningful when run on the DB host (the
# collapsed diagnostic); on a remote driver it reports driver CPU and the
# field is labelled driver_cpu_note.
CLK="$(getconf CLK_TCK)"
backend_jiffies() {
    local total=0 p
    for p in $(pgrep -f "${DB_PID_HINT}" 2>/dev/null || true); do
        [ -r "/proc/$p/stat" ] || continue
        read -r -a f < "/proc/$p/stat" 2>/dev/null || continue
        total=$(( total + ${f[13]:-0} + ${f[14]:-0} ))
    done
    echo "$total"
}

# After a backend crash the postmaster runs crash recovery and refuses new
# connections briefly. Wait until a trivial query succeeds again (bounded).
wait_db_healthy() {
    local i
    for i in $(seq 1 60); do
        if psql_db -tAc 'SELECT 1' >/dev/null 2>&1; then
            [ "$i" -gt 1 ] && echo "[run_load] DB healthy again after ${i}s" >&2
            return 0
        fi
        sleep 1
    done
    echo "[run_load] DB did not recover within 60s" >&2
    return 1
}

# run one pgbench invocation; echoes a JSON object for the run.
run_one() {
    local label="$1" clients="$2" jobs="$3" rate="$4"  # rate="" => closed
    local rflag=(); [ -n "${rate}" ] && rflag=(-R "${rate}")
    local j0 j1 w0 w1
    j0="$(backend_jiffies)"; w0="$(date +%s.%N)"
    # -P progress not parsed; we take pgbench's final latency summary +
    # --latency-limit none; percentiles come from per-txn log aggregation.
    local logpfx; logpfx="$(mktemp -u /tmp/pgb_${label}_XXXX)"
    local out
    out="$("${PGBENCH}" -h "${POOLER_HOST}" -p "${POOLER_PORT}" -U "${PGUSER}" \
        -d "${PGDATABASE}" -n -f "${SQLFILE}" \
        -c "${clients}" -j "${jobs}" -T "$((DURATION))" \
        --max-tries=1 "${rflag[@]}" \
        -l --log-prefix="${logpfx}" 2>&1)" || true
    w1="$(date +%s.%N)"; j1="$(backend_jiffies)"

    # A backend crash (OOM, segfault) under high concurrency is itself a
    # RESULT (the memory-per-backend wall), not a harness error. Record it
    # as a failure row and let the sweep continue — do NOT abort.
    if printf '%s\n' "${out}" | grep -qiE 'server conn crashed|terminating connection because of crash|could not connect|out of memory|FATAL'; then
        rm -f ${logpfx}.* 2>/dev/null || true
        local reason="backend_crash_or_oom"
        printf '%s\n' "${out}" | grep -qi 'out of memory' && reason="out_of_memory"
        echo "[run_load] ${label} c=${clients} FAILED (${reason}) — recording failure row, continuing" >&2
        cat <<JSON
    {"mode":"${label%%_*}","label":"${label}","clients":${clients},"jobs":${jobs},"rate":${rate:-null},
     "failed":true,"failure":"${reason}",
     "tps":0,"qps_per_core":0,"backend_cores_busy":0,"cpu_ms_per_query":0,
     "p50_ms":0,"p95_ms":0,"p99_ms":0,"p999_ms":0,"max_ms":0,
     "duration_s":${DURATION},"warmup_s":${WARMUP},"driver_id":"${DRIVER_ID}"}
JSON
        wait_db_healthy
        return 0
    fi

    # tps excluding connections
    local tps; tps="$(printf '%s\n' "${out}" | awk '/tps =/{for(i=1;i<=NF;i++)if($i=="tps"){print $(i+2);exit}}' | head -1)"
    # percentiles from the per-txn log. pgbench -l format (no --aggregate):
    #   client_id txn_no time script_no time_epoch time_us [schedule_lag] [retries]
    # field 3 (time) = per-transaction latency in MICROSECONDS; fields 5/6
    # (time_epoch, time_us) = wall-clock completion time. We drop the first
    # WARMUP seconds by wall clock, emit field-3 latencies, sort with the
    # shell (mawk has no asort), then percentile. Latencies in ms.
    local pctl
    pctl="$(cat ${logpfx}.* 2>/dev/null | awk -v w="${WARMUP}" '
            NR==1 { w0 = $5 + $6/1e6 }
            { wt = $5 + $6/1e6; if (wt - w0 >= w) print $3 }' \
        | sort -n \
        | awk '
            { a[n++] = $1 }
            END{
              if (n==0) { print "0 0 0 0 0"; exit }
              i50=int(.50*(n-1)); i95=int(.95*(n-1)); i99=int(.99*(n-1)); i999=int(.999*(n-1));
              printf "%.3f %.3f %.3f %.3f %.3f\n", a[i50]/1000, a[i95]/1000, a[i99]/1000, a[i999]/1000, a[n-1]/1000
            }')"
    rm -f ${logpfx}.* 2>/dev/null || true
    read -r p50 p95 p99 p999 pmax <<<"${pctl}"

    local wall djiff cores
    wall="$(awk -v a="$w0" -v b="$w1" 'BEGIN{print b-a}')"
    djiff=$(( j1 - j0 ))
    cores="$(awk -v d="${djiff}" -v w="${wall}" -v c="${CLK}" 'BEGIN{ if(w>0) printf "%.3f", d/c/w; else print 0 }')"
    local qpc; qpc="$(awk -v t="${tps:-0}" -v cr="${cores:-0}" 'BEGIN{ if(cr>0) printf "%.1f", t/cr; else print 0 }')"
    local cpums; cpums="$(awk -v cr="${cores:-0}" -v t="${tps:-0}" 'BEGIN{ if(t>0) printf "%.3f", cr*1000/t; else print 0 }')"

    cat <<JSON
    {"mode":"${label%%_*}","label":"${label}","clients":${clients},"jobs":${jobs},"rate":${rate:-null},
     "tps":${tps:-0},"qps_per_core":${qpc},"backend_cores_busy":${cores},"cpu_ms_per_query":${cpums},
     "p50_ms":${p50},"p95_ms":${p95},"p99_ms":${p99},"p999_ms":${p999},"max_ms":${pmax},
     "duration_s":${DURATION},"warmup_s":${WARMUP},"driver_id":"${DRIVER_ID}"}
JSON
}

{
    echo "{"
    echo "  \"timestamp_utc\": \"${ts}\", \"run_tag\": \"${RUN_TAG}\", \"driver_id\": \"${DRIVER_ID}\","
    echo "  \"topology\": \"${TOPOLOGY:-collapsed-single-host DIAGNOSTIC}\","
    echo "  \"db\": \"${DB_HOST}:${DB_PORT}\", \"pooler\": \"${POOLER_HOST}:${POOLER_PORT}\","
    echo "  \"corpus_rows\": ${CORPUS_ROWS}, \"dim\": ${DIM}, \"metric\": \"${METRIC}\","
    echo "  \"index_kind\": \"${INDEX_KIND}\", \"lists\": ${LISTS}, \"bit_width\": ${BIT_WIDTH},"
    echo "  \"k\": ${K}, \"scan_parallelism\": \"${SCAN_PARALLELISM}\","
    echo "  \"ncpu\": ${NCPU}, \"host\": \"$(uname -n)\","
    echo "  \"cpu_model\": $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //' | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))'),"
    echo "  \"runs\": ["
} > "${OUT}"

first=1
emit() { [ "${first}" = 1 ] && first=0 || echo "," >> "${OUT}"; run_one "$@" >> "${OUT}"; }

log "closed-loop sweep: clients=[${CLIENTS}] duration=${DURATION}s"
for c in ${CLIENTS}; do
    j=$(( c < NCPU ? c : NCPU )); [ "$j" -lt 1 ] && j=1
    log "  closed c=${c} j=${j}"
    emit "closed_c${c}" "${c}" "${j}" ""
done

if [ -n "${OPEN_RATES}" ]; then
    log "open-loop sweep: rates=[${OPEN_RATES}] clients=${OPEN_CLIENTS}"
    for r in ${OPEN_RATES}; do
        j=$(( OPEN_CLIENTS < NCPU ? OPEN_CLIENTS : NCPU )); [ "$j" -lt 1 ] && j=1
        log "  open R=${r} c=${OPEN_CLIENTS}"
        emit "open_r${r}" "${OPEN_CLIENTS}" "${j}" "${r}"
    done
fi

echo "" >> "${OUT}"
echo "  ]" >> "${OUT}"
echo "}" >> "${OUT}"
rm -f "${SQLFILE}"
log "wrote ${OUT}"
echo "${OUT}"
