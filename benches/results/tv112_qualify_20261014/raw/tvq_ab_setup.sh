#!/bin/bash
# turbovec 1.1.2 A/B on Graviton4 (planes host): v2.12.0 (old) vs this release candidate (new).
# Non-cassert PG16 -O2; 1M x 1024-d Cohere; identical index bytes are NOT expected
# (1.1.2 changes no persisted bytes, but we rebuild per arm only if needed -- we don't).
set -euxo pipefail
. ~/.cargo/env
W=/work/ab; mkdir -p $W/so; cd $W
SRC=$(ls -d ~/.pgrx/16.*/ | grep -v -E 'install|unpack' | head -1)
[ -x $W/pg16r/bin/pg_config ] || { cp -r $SRC $W/pgsrc && cd $W/pgsrc && (make distclean >/dev/null 2>&1 || true) && ./configure --prefix=$W/pg16r --without-icu CFLAGS="-O2" >/dev/null && make -j$(nproc) -s >/dev/null 2>&1 && make -s install >/dev/null && make -C contrib/pg_prewarm -s install >/dev/null; }
PGC=$W/pg16r/bin/pg_config; cd $W
[ -d $W/pgtv-old ] || git clone -q --branch v2.12.0 https://github.com/gburd/pg_turbovec.git $W/pgtv-old
D=$(ls -d /work/pgtv-*/pgtv-wt-deps | tail -1)
# v2.12.0 pins pgrx 0.19.1, so the old arm needs the matching cargo-pgrx.
[ -x $W/pgrx191/bin/cargo-pgrx ] || cargo install -q --locked cargo-pgrx --version 0.19.1 --root $W/pgrx191
for arm in old new; do
  src=$W/pgtv-old; CP=$W/pgrx191/bin; [ $arm = new ] && { src=$D; CP=$HOME/.cargo/bin; }
  (cd $src && PATH=$CP:$PATH cargo pgrx install --release --pg-config $PGC > $W/build_$arm.log 2>&1)
  cp $($PGC --pkglibdir)/pg_turbovec.so $W/so/$arm.so; md5sum $W/so/$arm.so
done
export PATH=$W/pg16r/bin:$PATH
[ -d $W/data ] || { initdb -D $W/data -U fedora >/dev/null; cat >> $W/data/postgresql.conf <<EOC
shared_buffers = 12GB
max_parallel_workers_per_gather = 0
jit = off
max_wal_size = 32GB
unix_socket_directories = '/tmp'
listen_addresses = ''
shared_preload_libraries = 'pg_prewarm'
EOC
}
cp $W/so/new.so $($PGC --pkglibdir)/pg_turbovec.so
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456 pg_ctl -D $W/data -l $W/pg.log -w start
createdb bench 2>/dev/null || true
psql -d bench -qc "CREATE EXTENSION IF NOT EXISTS pg_turbovec; CREATE EXTENSION IF NOT EXISTS pg_prewarm; DROP TABLE IF EXISTS public.docs; CREATE TABLE public.docs(id bigint PRIMARY KEY, tv turbovec.vector)"
python3 -m venv $W/venv && $W/venv/bin/pip install -q numpy pyarrow huggingface_hub
sed -e "s#/mnt/nvme#$W#g" ~/load_corpus.py > $W/load_corpus.py
$W/venv/bin/python $W/load_corpus.py 1000000 200 bench
# (as run: this step and the CREATE INDEX below ran from a separate script with
#  ALTER TABLE and VACUUM as separate psql -c calls -- VACUUM cannot run in a
#  multi-statement -c transaction)
psql -d bench -qc "ALTER TABLE public.docs SET (autovacuum_enabled = off)"
psql -d bench -qc "VACUUM ANALYZE public.docs"
psql -d bench -c "CREATE INDEX docs_flat4 ON public.docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4)"
psql -d bench -c "SELECT pg_size_pretty(pg_relation_size('docs_flat4')), pg_size_pretty(pg_total_relation_size('docs'))"
echo SETUP_OK
