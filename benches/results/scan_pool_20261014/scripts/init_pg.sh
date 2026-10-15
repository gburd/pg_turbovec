#!/usr/bin/env bash
# initdb + P0 postgresql.bench.conf; nr_hugepages derived from postgres -C (P0 finding) + 512 spare.
set -euxo pipefail
export PATH=/opt/pg18/bin:$PATH; D=/data/pgdata; R=~/results/q
[ -d $D ] || initdb -D $D > ~/initdb.log
grep -q 'VS_BENCH_PLAN' $D/postgresql.conf || cat ~/postgresql.bench.conf >> $D/postgresql.conf
HP=$(postgres -D $D -C shared_memory_size_in_huge_pages)
sudo sysctl -w vm.nr_hugepages=$((HP + 512))
echo "shared_memory_size_in_huge_pages=$HP nr_hugepages=$((HP + 512))" > $R/hugepages.txt; grep -i huge /proc/meminfo >> $R/hugepages.txt
pg_ctl -D $D -l ~/pg.log -w start
createdb -h /tmp db_tv
psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_tv -c "CREATE EXTENSION pg_turbovec" -c "CREATE EXTENSION pg_prewarm" -c "ALTER DATABASE db_tv SET search_path = public, turbovec"
psql -h /tmp -X -d db_tv -Atc "SELECT extname, extversion FROM pg_extension" | tee -a $R/hugepages.txt
echo INIT_PG_DONE
