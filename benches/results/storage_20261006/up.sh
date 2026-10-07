#!/bin/sh
# Start the Fix C storage-bench cluster (own data dir + port, non-assert PG 16.15).
set -e
B=/work/pg16rel_c/bin; D=/work/fixc/data
[ -d $D ] || $B/initdb -D $D >/dev/null
grep -q 'fixc-conf' $D/postgresql.conf || cat >> $D/postgresql.conf <<C
# fixc-conf
shared_buffers = 8GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_parallel_maintenance_workers = 4
max_parallel_workers_per_gather = 0
max_wal_size = 32GB
checkpoint_timeout = 30min
jit = off
synchronous_commit = off
unix_socket_directories = '/work/fixc'
port = 55433
listen_addresses = ''
shared_preload_libraries = 'pg_turbovec'
turbovec.cache_size_mb = 2048
C
$B/pg_ctl -D $D -l /work/fixc/pg.log -w start >/dev/null
$B/createdb -h /work/fixc -p 55433 bench 2>/dev/null || true
$B/psql -X -h /work/fixc -p 55433 -d bench -qAt -c "CREATE EXTENSION IF NOT EXISTS pg_turbovec" -c "CREATE EXTENSION IF NOT EXISTS pageinspect" -c "SELECT version()" -c "SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'"
