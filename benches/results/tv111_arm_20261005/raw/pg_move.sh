#!/bin/bash
# Move bench data from the cassert cluster to a non-assert PG16.15 cluster.
set -euxo pipefail
OLDB=$(dirname $(ls -d ~/.pgrx/16.15/pgrx-install/bin/pg_config))
NEWB=/mnt/nvme/pg16r/bin
$OLDB/pg_ctl -D /mnt/nvme/pgdata -l /mnt/nvme/pg.log -w start >/dev/null 2>&1 || true
$OLDB/psql -h /tmp -p 5432 -d bench -qc "\\copy public.docs TO '/mnt/nvme/docs.txt'"
$OLDB/pg_ctl -D /mnt/nvme/pgdata -m fast -w stop
$NEWB/initdb -D /mnt/nvme/pgdata_r -U admin >/dev/null
tail -n 12 /mnt/nvme/pgdata/postgresql.conf >> /mnt/nvme/pgdata_r/postgresql.conf
cat > ~/pgenv.sh <<E
export PGHOST=/tmp PGPORT=5432 PATH=$NEWB:\$PATH PGDATA_DIR=/mnt/nvme/pgdata_r
E
sed -i 's|/mnt/nvme/pgdata |/mnt/nvme/pgdata_r |g; s|/mnt/nvme/pgdata$|/mnt/nvme/pgdata_r|' ~/swap.sh
grep -n pgdata ~/swap.sh
. ~/pgenv.sh
cp /mnt/nvme/so/pg_turbovec_new.so $(pg_config --pkglibdir)/pg_turbovec.so
pg_ctl -D /mnt/nvme/pgdata_r -l /mnt/nvme/pg_r.log -w start
createdb bench
psql -d bench -v ON_ERROR_STOP=1 -qc "CREATE EXTENSION pg_turbovec" -c "CREATE EXTENSION pageinspect" \
  -c "CREATE TABLE public.docs(id bigint primary key, tv turbovec.vector)" \
  -c "\\copy public.docs FROM '/mnt/nvme/docs.txt'" -c "VACUUM ANALYZE public.docs"
psql -d bench -c "SET maintenance_work_mem='8GB'; SET max_parallel_maintenance_workers=8" -c "\\timing on" \
  -c "CREATE INDEX docs_tv ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width=4)"
psql -d bench -Atc "SELECT count(*) FROM docs; SHOW debug_assertions; SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'"
/mnt/nvme/venv/bin/python ~/relhash.py docs_tv
echo MOVE_OK
