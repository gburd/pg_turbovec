#!/usr/bin/env bash
# benches/scripts/pooled/common.sh — shared config + helpers for the
# three-tier pooled benchmark harness.
#
# EVERY pooled script sources this. It defines nothing host-specific:
# all tier endpoints come from environment variables so the SAME scripts
# run on the collapsed single-host diagnostic rig AND the real
# metal + pooler-instance + spot-fleet rig (see README.md / provision.md).
#
# Tier endpoints (override per tier; defaults = collapsed single host):
#   DB_HOST / DB_PORT        — PostgreSQL (the system under test)
#   POOLER_HOST / POOLER_PORT— the connection pooler (PgBouncer/pgcat/...)
#   PGUSER / PGDATABASE      — role + db on the DB tier
#   PG_BIN                   — dir with psql/pgbench/pg_config/pg_ctl
#
# Corpus / index (parameterised so one harness covers every scale):
#   CORPUS_ROWS  DIM  METRIC  INDEX_KIND(flat|ivf)  LISTS  BIT_WIDTH
#   QUERY_POOL   K    OVERSAMPLE  SCAN_PARALLELISM(auto|1|N)
#
# Nothing here starts a server or writes data; sourcing is side-effect free.

set -euo pipefail

# ---- tier endpoints -------------------------------------------------------
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-28818}"            # pgrx pg18 default-ish; override freely
POOLER_HOST="${POOLER_HOST:-localhost}"
POOLER_PORT="${POOLER_PORT:-6432}"     # PgBouncer default
PGUSER="${PGUSER:-$(id -un)}"
PGDATABASE="${PGDATABASE:-turbovec_pooled}"
PG_BIN="${PG_BIN:-$HOME/.pgrx/18.6/pgrx-install/bin}"

export PGUSER PGDATABASE

PSQL="${PG_BIN}/psql"
PGBENCH="${PG_BIN}/pgbench"

# Turbovec's opclass + distance operators live in the `turbovec` schema. We
# do NOT put turbovec on search_path, because that would also make unqualified
# user tables (docs, bench_queries) land in the turbovec schema and then be
# invisible to a transaction-mode pooler connection (which drops SET/options).
# Instead: user tables live in `public` (the default), and every turbovec
# object is schema-qualified in SQL (turbovec.vec_cosine_ops,
# OPERATOR(turbovec.<=>), ::turbovec.vector). This resolves identically on a
# direct connection and through a pooler. See distop()/opclass() below.

# Direct to DB tier (admin / setup); through pooler (load).
psql_db()   { "${PSQL}" -h "${DB_HOST}" -p "${DB_PORT}" -X -q -v ON_ERROR_STOP=1 "$@"; }
psql_pool() { "${PSQL}" -h "${POOLER_HOST}" -p "${POOLER_PORT}" -X -q "$@"; }

# ---- corpus / index knobs -------------------------------------------------
CORPUS_ROWS="${CORPUS_ROWS:-500000}"
DIM="${DIM:-768}"
METRIC="${METRIC:-cosine}"             # cosine|l2|ip  -> vec_<metric>_ops
INDEX_KIND="${INDEX_KIND:-flat}"       # flat|ivf
LISTS="${LISTS:-0}"                    # ivf only; 0 => flat
BIT_WIDTH="${BIT_WIDTH:-4}"            # 1|2|4
QUERY_POOL="${QUERY_POOL:-1000}"
K="${K:-10}"
OVERSAMPLE="${OVERSAMPLE:-}"           # turbovec.oversample; empty => default
SCAN_PARALLELISM="${SCAN_PARALLELISM:-auto}"  # auto|1|<N>

# ---- storage sizing (diagnostic: force working set > shared_buffers) ------
SHARED_BUFFERS="${SHARED_BUFFERS:-256MB}"
EFFECTIVE_CACHE_SIZE="${EFFECTIVE_CACHE_SIZE:-1GB}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-200}"

# ---- misc -----------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/results}"
RUN_TAG="${RUN_TAG:-pooled_$(date -u +%Y%m%dT%H%M%SZ)}"

opclass() { echo "turbovec.vec_${METRIC}_ops"; }
# distance operator for each metric (matches docs/MIGRATING_FROM_PGVECTOR.md)
distop() {
    case "${METRIC}" in
        cosine) echo "<=>";;
        l2)     echo "<->";;
        ip)     echo "<#>";;
        *) echo "ERROR unknown METRIC=${METRIC}" >&2; return 1;;
    esac
}

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }

# Emit a key=value line to stderr summarising the active config.
dump_config() {
    log "config: DB=${DB_HOST}:${DB_PORT} POOLER=${POOLER_HOST}:${POOLER_PORT} db=${PGDATABASE}"
    log "config: rows=${CORPUS_ROWS} dim=${DIM} metric=${METRIC} kind=${INDEX_KIND} lists=${LISTS} bit_width=${BIT_WIDTH}"
    log "config: k=${K} oversample=${OVERSAMPLE:-default} scan_parallelism=${SCAN_PARALLELISM} shared_buffers=${SHARED_BUFFERS}"
}
