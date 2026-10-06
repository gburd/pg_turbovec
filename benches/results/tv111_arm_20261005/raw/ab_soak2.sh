#!/bin/bash
set -euxo pipefail
. ~/.cargo/env; . ~/pgenv.sh
for arm in old fix; do /mnt/nvme/pg16_$arm/bin/pg_ctl -D /mnt/nvme/pgsoak_$arm -m fast -w stop || true; done
cd /mnt/nvme/pg_turbovec && git fetch -q origin && git checkout -q origin/release/v2.11.0 && git log --oneline -1
cargo pgrx install --release --pg-config /mnt/nvme/pg16_fix/bin/pg_config > /mnt/nvme/buildr_fix2.log 2>&1
md5sum /mnt/nvme/pg16_*/lib/postgresql/pg_turbovec.so
port=5432
for arm in old fix; do
  D=/mnt/nvme/pgsoak_$arm; mv $D $D.prev.$(date +%s)
  /mnt/nvme/pg16_$arm/bin/initdb -D $D -U admin >/dev/null
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
  /mnt/nvme/pg16_$arm/bin/pg_ctl -D $D -l /mnt/nvme/pgsoak2_$arm.log -w start
  /mnt/nvme/pg16_$arm/bin/createdb -h /tmp -p $port bench
  /mnt/nvme/pg16_$arm/bin/psql -h /tmp -p $port -d bench -qc "CREATE EXTENSION pg_turbovec" -c "CREATE EXTENSION pageinspect"
  port=$((port+1))
done
: > /mnt/nvme/soak_old2.log; : > /mnt/nvme/soak_fix2.log
PGPORT=5432 TAG=old2 SEED=60000 DURATION=${DURATION:-2400} VAC_EVERY=120 /mnt/nvme/venv/bin/python ~/soak2.py > /mnt/nvme/soak2b_old.out 2>&1 &
PGPORT=5433 TAG=fix2 SEED=60000 DURATION=${DURATION:-2400} VAC_EVERY=120 /mnt/nvme/venv/bin/python ~/soak2.py > /mnt/nvme/soak2b_fix.out 2>&1 &
wait
grep -h -E "RESULT|SOAK" /mnt/nvme/soak_old2.log /mnt/nvme/soak_fix2.log
echo ABSOAK2_DONE
