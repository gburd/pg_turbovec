#!/usr/bin/env bash
# Init PG18 cluster on /data, hugepages sized from postgres -C, bench conf, both DBs + extensions.
# Idempotent-ish: skips initdb if the cluster exists.
set -uxo pipefail
export PATH=/opt/pg18/bin:$PATH
PGDATA=/data/pgdata; CONF=~/postgresql.bench.conf; R=~/results
mkdir -p $R
if [ ! -f $PGDATA/PG_VERSION ]; then
  initdb -D $PGDATA -E UTF8 --locale=C
  cat $CONF >> $PGDATA/postgresql.conf
fi
# hugepages: derive from postgres -C (P0 found the formula under-provisions)
NH=$(postgres -D $PGDATA -C shared_memory_size_in_huge_pages)
echo "shared_memory_size_in_huge_pages=$NH"
sudo sysctl -w vm.nr_hugepages=$((NH + 256))
grep -i huge /proc/meminfo | head -3
# start
pg_ctl -D $PGDATA -l ~/pg.log -w start || { tail -40 ~/pg.log; exit 1; }
env | grep -ci GLIBC_TUNABLES
PS="psql -h /tmp -X -v ON_ERROR_STOP=1"
$PS -d postgres -c "SELECT 1" >/dev/null
# databases
for db in db_tv db_pv; do
  $PS -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1 || $PS -d postgres -c "CREATE DATABASE $db"
done
$PS -d db_tv -c "CREATE EXTENSION IF NOT EXISTS pg_turbovec" -c "CREATE EXTENSION IF NOT EXISTS pg_prewarm" -c "CREATE EXTENSION IF NOT EXISTS pg_buffercache" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
$PS -d db_tv -c "ALTER DATABASE db_tv SET search_path = public, turbovec"
$PS -d db_pv -c "CREATE EXTENSION IF NOT EXISTS vector" -c "CREATE EXTENSION IF NOT EXISTS pg_vectorsearch" -c "CREATE EXTENSION IF NOT EXISTS pg_prewarm" -c "CREATE EXTENSION IF NOT EXISTS pg_buffercache" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
{
  echo "== PG"; $PS -d postgres -Atc "SELECT version()"
  echo "== shared_buffers huge_pages"; $PS -d postgres -Atc "SELECT name,setting FROM pg_settings WHERE name IN ('shared_buffers','huge_pages','wal_level','io_method','effective_io_concurrency','jit','autovacuum','max_parallel_maintenance_workers')"
  echo "== extensions db_tv"; $PS -d db_tv -Atc "SELECT extname,extversion FROM pg_extension ORDER BY 1"
  echo "== extensions db_pv"; $PS -d db_pv -Atc "SELECT extname,extversion FROM pg_extension ORDER BY 1"
  echo "== GLIBC env"; env | grep -i GLIBC || echo "(none)"
} > $R/cluster_init.txt 2>&1
cat $R/cluster_init.txt
echo CLUSTER_INIT_DONE
