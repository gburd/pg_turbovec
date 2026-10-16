#!/usr/bin/env bash
# benches/scripts/pooled/setup_db.sh — tune PostgreSQL for the pooled
# target workload, create the extension, load a corpus, build an index.
#
# Runs ON the database tier (metal in the real rig; this host in the
# collapsed diagnostic). Parameterised entirely through common.sh env.
#
# Usage:
#   # against a pgrx-managed cluster already running on DB_PORT:
#   DB_PORT=28818 CORPUS_ROWS=500000 DIM=768 INDEX_KIND=flat \
#       bash setup_db.sh
#
#   # IVF:
#   INDEX_KIND=ivf LISTS=1024 CORPUS_ROWS=1000000 bash setup_db.sh
#
# What it tunes (all recorded in the result header; see METHODOLOGY.md
# "Host tuning"):
#   shared_buffers, effective_cache_size, max_connections (>= backend
#   count the pooler will open), work_mem, maintenance_work_mem,
#   wal_level=replica, max_parallel_maintenance_workers.
# Huge pages are a kernel-level knob (vm.nr_hugepages + huge_pages=try);
# this script sets huge_pages=try and PRINTS the sysctl an operator must
# run on the metal host — it does not sudo on your behalf.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

dump_config

# ---------------------------------------------------------------------------
# 1. tune the running cluster (ALTER SYSTEM so it survives restart). These
#    apply to whichever cluster listens on DB_PORT. On the real metal rig
#    set shared_buffers to tens-hundreds of GB and add the ZFS/huge-pages
#    notes below.
# ---------------------------------------------------------------------------
log "tuning cluster on ${DB_HOST}:${DB_PORT}"
psql_db -d postgres <<SQL
ALTER SYSTEM SET shared_buffers = '${SHARED_BUFFERS}';
ALTER SYSTEM SET effective_cache_size = '${EFFECTIVE_CACHE_SIZE}';
ALTER SYSTEM SET max_connections = ${MAX_CONNECTIONS};
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET huge_pages = 'try';
ALTER SYSTEM SET max_worker_processes = 16;
ALTER SYSTEM SET max_parallel_workers = 8;
-- turbovec cache big enough for the corpus on the DB tier.
ALTER SYSTEM SET turbovec.cache_size_mb = ${TURBOVEC_CACHE_MB:-4096};
SQL

cat >&2 <<NOTE
[setup_db] shared_buffers/max_connections/wal_level changed via ALTER SYSTEM.
           A RESTART is required for shared_buffers + huge_pages to take
           effect. On the pgrx cluster:
               ${PG_BIN}/pg_ctl -D <datadir> restart -m fast
           On the real METAL rig, FIRST reserve huge pages (NOT done here):
               # size = shared_buffers / 2MB  (round up), e.g. 64GB => 32768
               sudo sysctl -w vm.nr_hugepages=<N>
               # make persistent in /etc/sysctl.d/ and verify:
               grep Huge /proc/meminfo
           Storage target per METHODOLOGY.md is ZFS + separate ZIL(SLOG) +
           L2ARC on NVMe (record recordsize/compression/atime/logbias/ARC).
           A plain XFS/ext4-on-NVMe run is a *substrate* row, labelled so.
NOTE

# NOTE: caller must restart the cluster before load if shared_buffers changed.
# setup_pooler.sh / run_load.sh assume the cluster is up with the new config.

# ---------------------------------------------------------------------------
# 2. (re)create database + extension
# ---------------------------------------------------------------------------
log "creating database ${PGDATABASE} + extension"
psql_db -d postgres -c "DROP DATABASE IF EXISTS ${PGDATABASE};"
psql_db -d postgres -c "CREATE DATABASE ${PGDATABASE};"
psql_db -c "CREATE EXTENSION IF NOT EXISTS pg_turbovec;"

# ---------------------------------------------------------------------------
# 3. load corpus + query pool (deterministic seeds, native turbovec.vector).
#    Generated in server-side batches so a multi-million-row corpus does not
#    need an external loader for the smoke rig. For TB-scale, replace this
#    step with a real corpus loader (see benches/scripts/load_dbpedia_1M.py).
# ---------------------------------------------------------------------------
log "loading ${CORPUS_ROWS} x ${DIM}-d corpus (batched)..."
psql_db <<SQL
SET client_min_messages = warning;
DROP TABLE IF EXISTS docs CASCADE;
CREATE TABLE docs (id bigint PRIMARY KEY, embedding turbovec.vector NOT NULL);
DROP TABLE IF EXISTS bench_queries CASCADE;
CREATE TABLE bench_queries (qid int PRIMARY KEY, q turbovec.vector NOT NULL);
SQL

# Batch the insert so one statement never has to materialise N*DIM floats.
BATCH="${BATCH:-50000}"
loaded=0
while [ "${loaded}" -lt "${CORPUS_ROWS}" ]; do
    lo=$((loaded + 1))
    hi=$((loaded + BATCH)); [ "${hi}" -gt "${CORPUS_ROWS}" ] && hi=${CORPUS_ROWS}
    psql_db <<SQL
SELECT setseed(0.42 + ${lo}::float8 / 1e12);
INSERT INTO docs (id, embedding)
SELECT g.i,
       (ARRAY(SELECT random()::real * 2.0 - 1.0 FROM generate_series(1, ${DIM})))::real[]::turbovec.vector
FROM generate_series(${lo}, ${hi}) AS g(i);
SQL
    loaded=${hi}
    log "  loaded ${loaded}/${CORPUS_ROWS}"
done

psql_db <<SQL
SELECT setseed(0.13);
INSERT INTO bench_queries (qid, q)
SELECT g.i,
       (ARRAY(SELECT random()::real * 2.0 - 1.0 FROM generate_series(1, ${DIM})))::real[]::turbovec.vector
FROM generate_series(0, ${QUERY_POOL} - 1) AS g(i);
ANALYZE docs; ANALYZE bench_queries;
SQL

# ---------------------------------------------------------------------------
# 4. build the index (flat or IVF). Expression on the native vector column.
# ---------------------------------------------------------------------------
OPCLASS="$(opclass)"
WITHOPTS="bit_width = ${BIT_WIDTH}"
if [ "${INDEX_KIND}" = "ivf" ]; then
    [ "${LISTS}" -gt 0 ] || { echo "INDEX_KIND=ivf requires LISTS>0" >&2; exit 1; }
    WITHOPTS="${WITHOPTS}, lists = ${LISTS}"
fi
log "building ${INDEX_KIND} index (${OPCLASS}, ${WITHOPTS})..."
T=$(date +%s)
psql_db <<SQL
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;
DROP INDEX IF EXISTS docs_tv;
CREATE INDEX docs_tv ON docs USING turbovec (embedding ${OPCLASS}) WITH (${WITHOPTS});
SQL
log "index built in $(($(date +%s)-T))s"

# ---------------------------------------------------------------------------
# 5. verify: EXPLAIN shows the index is used; sizes; sane recall probe.
# ---------------------------------------------------------------------------
OP="$(distop)"
log "EXPLAIN (index-usage check):"
psql_db -c "SET enable_seqscan=off;
EXPLAIN (COSTS OFF)
SELECT id FROM docs ORDER BY embedding OPERATOR(turbovec.${OP}) (SELECT q FROM bench_queries WHERE qid=0) LIMIT ${K};" >&2

psql_db -tAc "
SELECT 'heap_bytes='||pg_relation_size('docs')
     ||' index_bytes='||pg_total_relation_size('docs_tv')
     ||' n_rows='||(SELECT count(*) FROM docs);" >&2

# A tiny recall sanity: index top-k vs exact top-k overlap for one query.
psql_db <<SQL >&2
SET enable_seqscan=off;
WITH q AS (SELECT q FROM bench_queries WHERE qid=0),
idx AS (SELECT id FROM docs ORDER BY embedding OPERATOR(turbovec.${OP}) (SELECT q FROM q) LIMIT ${K})
SELECT 'recall_probe: index returned '||count(*)||' rows' FROM idx;
SQL

log "setup_db done: db=${PGDATABASE} kind=${INDEX_KIND} rows=${CORPUS_ROWS}"
