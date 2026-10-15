#!/usr/bin/env bash
# swap.sh old|new : install that .so, restart the postmaster (NO env changes: no RAYON_NUM_THREADS, no
# GLIBC_TUNABLES, no taskset), prewarm heap+indexes+queries, record the arm identity.
set -euo pipefail
export PATH=/opt/pg18/bin:$PATH; D=/data/pgdata; P="psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_tv -Atc"
pg_ctl -D $D -m fast -w stop >/dev/null
cp --remove-destination ~/so/$1.so $(pg_config --pkglibdir)/pg_turbovec.so
pg_ctl -D $D -l ~/pg.log -w start >/dev/null
PM=$(head -1 $D/postmaster.pid)
echo "== swap -> $1 $(date -u +%FT%TZ) installed_so_md5=$(md5sum $(pg_config --pkglibdir)/pg_turbovec.so | cut -d' ' -f1) postmaster=$PM"
echo "affinity: $(taskset -pc $PM)"
tr '\0' '\n' < /proc/$PM/environ | grep -E 'RAYON|GLIBC' || echo "postmaster env: no RAYON*/GLIBC*"
psql -h /tmp -X -d db_tv -At -c "SELECT 'turbovec_version='||turbovec.turbovec_version()" -c "SELECT 'search_concurrency_guc_present='||count(*) FROM pg_settings WHERE name='turbovec.search_concurrency'" -c "SELECT 'extversion='||extversion FROM pg_extension WHERE extname='pg_turbovec'"
for r in docs docs_tv_flat docs_ivf docs_ivf_tv queries; do echo "prewarm $r $($P "SELECT pg_prewarm('$r')")"; done
