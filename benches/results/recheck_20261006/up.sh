set -e
B=/tmp/pg16rel/bin; D=/tmp/rck/data
[ -d $D ] || $B/initdb -D $D -U gburd >/dev/null
grep -q 'rck-conf' $D/postgresql.conf || cat >> $D/postgresql.conf <<C
# rck-conf
shared_buffers = 3GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_parallel_maintenance_workers = 4
max_parallel_workers_per_gather = 0
jit = off
synchronous_commit = off
unix_socket_directories = '/tmp/rck'
port = 55432
listen_addresses = ''
shared_preload_libraries = 'pg_turbovec'
C
$B/pg_ctl -D $D -l /tmp/rck/pg.log -w start >/dev/null
$B/createdb -h /tmp/rck -p 55432 bench 2>/dev/null || true
$B/psql -h /tmp/rck -p 55432 -d bench -qc "CREATE EXTENSION IF NOT EXISTS pg_turbovec" -c "SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'" -At
