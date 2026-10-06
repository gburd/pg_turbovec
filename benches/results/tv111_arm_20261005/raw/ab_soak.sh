#!/bin/bash
# Parallel A/B soak: cluster A (port 5432) = v2.10.3 .so, cluster B (5433) = v2.11.0-fix .so.
# Both PG 16.15 non-assert, same config (8GB shared_buffers each to fit RAM).
set -euxo pipefail
. ~/pgenv.sh
B=/mnt/nvme/pg16r
pg_ctl -D /mnt/nvme/pgdata_r -m fast -w stop || true
# Two install trees so each cluster loads its own .so.
for arm in old fix; do
  [ -d /mnt/nvme/pg16_$arm ] || cp -a $B /mnt/nvme/pg16_$arm
  cp /mnt/nvme/so/pg_turbovec_$arm.so /mnt/nvme/pg16_$arm/lib/postgresql/pg_turbovec.so
done
port=5432
for arm in old fix; do
  D=/mnt/nvme/pgsoak_$arm
  [ -d $D ] || /mnt/nvme/pg16_$arm/bin/initdb -D $D -U admin >/dev/null
  cat >> $D/postgresql.conf <<C
shared_buffers = 6GB
maintenance_work_mem = 2GB
max_wal_size = 32GB
checkpoint_timeout = 30min
synchronous_commit = off
unix_socket_directories = '/tmp'
port = $port
shared_preload_libraries = 'pg_turbovec'
C
  /mnt/nvme/pg16_$arm/bin/pg_ctl -D $D -l /mnt/nvme/pgsoak_$arm.log -w start
  /mnt/nvme/pg16_$arm/bin/createdb -h /tmp -p $port bench
  /mnt/nvme/pg16_$arm/bin/psql -h /tmp -p $port -d bench -qc "CREATE EXTENSION pg_turbovec" -c "CREATE EXTENSION pageinspect"
  port=$((port+1))
done
md5sum /mnt/nvme/pg16_*/lib/postgresql/pg_turbovec.so
PGPORT=5432 TAG=old SEED=60000 DURATION=${DURATION:-2400} VAC_EVERY=120 /mnt/nvme/venv/bin/python ~/soak2.py > /mnt/nvme/soak2_old.out 2>&1 &
PGPORT=5433 TAG=fix SEED=60000 DURATION=${DURATION:-2400} VAC_EVERY=120 /mnt/nvme/venv/bin/python ~/soak2.py > /mnt/nvme/soak2_fix.out 2>&1 &
wait
grep -h -E "RESULT|SOAK" /mnt/nvme/soak_old.log /mnt/nvme/soak_fix.log
echo ABSOAK_DONE
