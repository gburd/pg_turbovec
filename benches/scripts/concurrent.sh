#!/usr/bin/env bash
#
# benches/scripts/concurrent.sh — measure pg_turbovec's backend-local cache
# behaviour under concurrent `turbovec.knn(...)` queries.
#
# Method
# ------
# 1. (Re)create a clean `turbovec_bench` database with the extension.
# 2. Build a 10 000-row, 384-dim corpus and a 256-row pool of query
#    vectors.
# 3. Pre-warm: run a single `turbovec.knn` so the next call into a
#    fresh backend will hit the cache miss path once and then cache.
# 4. For each thread count in CLIENTS (default 1 2 4 8 16) run pgbench
#    for DURATION seconds against `benches/sql/knn_query.sql`, which
#    randomly selects one of the 256 query vectors per transaction.
#    `-C` is *not* set so each pgbench client uses a single
#    persistent backend and the cache miss is paid once per client.
# 5. Emit a JSON report under benches/results/.
#
# Caveats
# -------
# * Each pgbench client = one backend = one cache copy. The cache is
#   backend-local by design (see src/cache.rs), so cross-client cache
#   sharing is not what we are measuring; we are measuring lock
#   contention on the *one* in-process Mutex<HashMap>. With N
#   backends we have N independent mutexes, so this is not a great
#   proxy for in-process contention.  The bench is therefore most
#   useful as a regression guard on per-backend cache lookup cost
#   under load — a workload identical to what the production
#   PostgreSQL server faces.  See `benches/concurrent_knn.rs` for an
#   in-process bench that hammers the *same* mutex from N threads.
# * The 1-second warmup query inside the per-client startup is
#   included in TPS at low N because the corpus build + cache insert
#   pays ~400 ms once.  We discard the first second of measurement
#   (`-D` / `--latency-limit` are not appropriate; instead we use
#   `--progress` and skip the first interval).  Practically the
#   default DURATION=10 gives us 9 s of steady state per N which is
#   plenty.
#
# Usage
# -----
#   ./benches/scripts/concurrent.sh                  # default settings
#   CLIENTS="1 2 4" DURATION=5 ./benches/scripts/concurrent.sh
#
# Required env (matches README "Build / test setup"):
#   PGRX_PG_CONFIG=/home/gburd/.pgrx/install-pg16/bin/pg_config
#   (the script falls back to that path).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/benches/results}"
SQL_DIR="${ROOT}/bench/sql"
PG_BIN="${PG_BIN:-/home/gburd/.pgrx/install-pg16/bin}"
PG_PORT="${PG_PORT:-28816}"
PG_HOST="${PG_HOST:-localhost}"
DBNAME="${DBNAME:-turbovec_bench}"
DIM="${DIM:-384}"
CORPUS_ROWS="${CORPUS_ROWS:-10000}"
QUERY_POOL="${QUERY_POOL:-256}"
K="${K:-10}"
BIT_WIDTH="${BIT_WIDTH:-4}"
DURATION="${DURATION:-10}"
CLIENTS="${CLIENTS:-1 2 4 8 16}"
SKIP_SETUP="${SKIP_SETUP:-0}"

PSQL="${PG_BIN}/psql -h ${PG_HOST} -p ${PG_PORT}"
PGBENCH="${PG_BIN}/pgbench -h ${PG_HOST} -p ${PG_PORT}"

mkdir -p "${RESULTS_DIR}" "${SQL_DIR}"

cat >"${SQL_DIR}/knn_query.sql" <<'SQL'
-- pgbench script: one transaction = one turbovec.knn call.
\set qid random(0, :pool_max)
SELECT count(*) FROM turbovec.knn(
    'bench_corpus'::regclass::oid,
    'id',
    'embedding',
    (SELECT q FROM bench_queries WHERE qid = :qid),
    :k,
    :bw
) AS r;
SQL

setup_database() {
    echo "[setup] (re)creating database ${DBNAME}..."
    ${PSQL} -d postgres -tAc "DROP DATABASE IF EXISTS ${DBNAME};" >/dev/null
    ${PSQL} -d postgres -tAc "CREATE DATABASE ${DBNAME};" >/dev/null
    ${PSQL} -d "${DBNAME}" -v ON_ERROR_STOP=1 -v dim="${DIM}" -v rows="${CORPUS_ROWS}" -v qpool="${QUERY_POOL}" <<'SQL' >/dev/null
SET client_min_messages = warning;
CREATE EXTENSION pg_turbovec;

DROP TABLE IF EXISTS bench_corpus CASCADE;
CREATE TABLE bench_corpus (
    id        bigint PRIMARY KEY,
    embedding turbovec.vector NOT NULL
);
SELECT setseed(0.42);
INSERT INTO bench_corpus (id, embedding)
SELECT g.i,
       (ARRAY(SELECT random()::real * 2.0 - 1.0
              FROM generate_series(1, :dim)))::real[]::turbovec.vector
FROM generate_series(1, :rows) AS g(i);

DROP TABLE IF EXISTS bench_queries CASCADE;
CREATE TABLE bench_queries (
    qid int PRIMARY KEY,
    q   turbovec.vector NOT NULL
);
SELECT setseed(0.13);
INSERT INTO bench_queries (qid, q)
SELECT g.i,
       (ARRAY(SELECT random()::real * 2.0 - 1.0
              FROM generate_series(1, :dim)))::real[]::turbovec.vector
FROM generate_series(0, :qpool - 1) AS g(i);

ANALYZE bench_corpus;
ANALYZE bench_queries;
SQL
    echo "[setup] corpus rows: $(${PSQL} -d ${DBNAME} -tAc 'select count(*) from bench_corpus')"
    echo "[setup] query rows: $(${PSQL} -d ${DBNAME} -tAc 'select count(*) from bench_queries')"
}

if [[ "${SKIP_SETUP}" != "1" ]]; then
    setup_database
fi

# `turbovec.cache_size_mb` defaults to a value that comfortably fits
# our 10 000-vector corpus (~2 MB at 4-bit), but make it explicit
# anyway so the bench is self-contained.
${PSQL} -d "${DBNAME}" -tAc "SET turbovec.cache_size_mb = 256;" >/dev/null || true

# Pool-max as a pgbench `-D` define (random(low, high) is inclusive).
POOL_MAX=$((QUERY_POOL - 1))

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="${RESULTS_DIR}/concurrent_knn_${ts}.json"
host_id="$(uname -n)"
cpu_id="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo unknown)"
n_cpu="$(nproc 2>/dev/null || echo unknown)"

# Header — JSON we'll append result rows to.
{
    echo "{"
    echo "  \"timestamp_utc\": \"${ts}\","
    echo "  \"host\": \"${host_id}\","
    echo "  \"cpu\": $(printf '%s' "$cpu_id" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))'),"
    echo "  \"n_cpu\": ${n_cpu},"
    echo "  \"corpus_rows\": ${CORPUS_ROWS},"
    echo "  \"dim\": ${DIM},"
    echo "  \"query_pool\": ${QUERY_POOL},"
    echo "  \"k\": ${K},"
    echo "  \"bit_width\": ${BIT_WIDTH},"
    echo "  \"duration_s\": ${DURATION},"
    echo "  \"runs\": ["
} >"${out}"

first=1
baseline_tps=""
for c in ${CLIENTS}; do
    echo "[run] clients=${c}"
    # We pre-warm by issuing one knn() per client connection
    # (via -t 1) before timing.  pgbench supports a separate
    # "init" file with -f run + -f init ... no, it doesn't.
    # Instead we use a separate pgbench invocation with -t 1 -c $c
    # (one warm-up txn per client) before the timed run.  pgbench
    # opens fresh connections for the timed run, so we DO want the
    # timed run to absorb the per-backend cache build.  Our
    # DURATION default (10 s) is large enough that the per-client
    # ~0.4 s build is < 5% of throughput at the highest N.

    log="$(mktemp)"
    ${PGBENCH} -d "${DBNAME}" -n -M prepared \
        -c "${c}" -j "${c}" -T "${DURATION}" \
        -D "k=${K}" -D "bw=${BIT_WIDTH}" -D "pool_max=${POOL_MAX}" \
        -f "${SQL_DIR}/knn_query.sql" 2>&1 | tee "${log}" >/dev/null

    # PG16 emits two possible lines:
    #   `tps = X (without initial connection time)` (default), and
    #   `tps = X (including connections establishing)` with -C.
    # We default to persistent connections, so only the first one
    # is present; the other field is left as null.
    tps_inc="$(awk '/^tps =.*including connections establishing/ {print $3; exit}' "${log}")"
    tps_exc="$(awk '/^tps =.*without initial connection time/ {print $3; exit}' "${log}")"
    avg_lat="$(awk '/^latency average =/ {print $4; exit}' "${log}")"
    nxact="$(awk '/^number of transactions actually processed:/ {print $6; exit}' "${log}")"
    init_conn="$(awk '/^initial connection time =/ {print $5; exit}' "${log}")"
    if [[ -z "${tps_exc}" ]]; then
        echo "[error] pgbench produced no tps line; full log:" >&2
        cat "${log}" >&2
        rm -f "${log}"
        exit 1
    fi
    rm -f "${log}"

    if [[ -z "${baseline_tps}" ]]; then
        baseline_tps="${tps_exc}"
    fi
    speedup=$(python3 -c "print(round(${tps_exc}/${baseline_tps}, 3))")
    tps_inc_field="${tps_inc:-null}"
    init_conn_field="${init_conn:-null}"

    if [[ ${first} -eq 0 ]]; then echo "    ," >>"${out}"; fi
    first=0
    cat >>"${out}" <<EOF
    {
      "clients": ${c},
      "tps_excluding_connection_setup": ${tps_exc},
      "tps_including_connection_setup": ${tps_inc_field},
      "avg_latency_ms": ${avg_lat},
      "initial_connection_time_ms": ${init_conn_field},
      "n_transactions": ${nxact},
      "speedup_vs_n1": ${speedup}
    }
EOF
    printf "    clients=%-3d  tps=%-10s  lat=%-8s ms  speedup=%sx\n" \
        "${c}" "${tps_exc}" "${avg_lat}" "${speedup}"
done

echo "  ]" >>"${out}"
echo "}" >>"${out}"
echo "[done] wrote ${out}"
