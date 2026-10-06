#!/bin/bash
# swap.sh old|new [ENV=VAL ...]  -- install that .so and restart PG with extra env.
set -euo pipefail
. ~/pgenv.sh
ARM=$1; shift
LIBDIR=$(pg_config --pkglibdir)
pg_ctl -D /mnt/nvme/pgdata -m fast -w stop >/dev/null
cp /mnt/nvme/so/pg_turbovec_$ARM.so $LIBDIR/pg_turbovec.so
env "$@" pg_ctl -D /mnt/nvme/pgdata -l /mnt/nvme/pg.log -w start >/dev/null
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
echo "swapped -> $ARM $* md5=$(md5sum $LIBDIR/pg_turbovec.so | cut -c1-8)"
