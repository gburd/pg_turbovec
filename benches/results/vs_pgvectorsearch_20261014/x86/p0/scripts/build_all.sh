#!/usr/bin/env bash
# Build PG 18.x, pgvector v0.8.x (default flags), pg_vectorsearch (main HEAD), pg_turbovec v2.12.1.
set -euxo pipefail
PGV=${PGV:-18.6}; PGVEC=${PGVEC:-v0.8.7}; TV=${TV:-v2.12.1}
R=~/results/p0; mkdir -p $R ~/src
if [ ! -x /opt/pg18/bin/postgres ]; then
cd ~/src
[ -d postgresql-$PGV ] || { wget -q https://ftp.postgresql.org/pub/source/v$PGV/postgresql-$PGV.tar.bz2; tar xjf postgresql-$PGV.tar.bz2; }
cd postgresql-$PGV
./configure --prefix=/opt/pg18 --with-liburing --with-lz4 --with-zstd --with-openssl --with-icu CFLAGS="-O2 -g -fno-omit-frame-pointer" > ~/pg_configure.log
make -j96 -s world-bin > ~/pg_make.log 2>&1
sudo make -s install-world-bin > /dev/null
sudo chown -R fedora:fedora /opt/pg18
fi
export PATH=/opt/pg18/bin:$PATH
# pgvector — DEFAULT Makefile flags (R17)
cd ~/src; [ -d pgvector ] || git clone -q --branch $PGVEC https://github.com/pgvector/pgvector.git
cd pgvector; [ -f vector.so ] || make -j16 PG_CONFIG=/opt/pg18/bin/pg_config > ~/pgvector_make.log 2>&1; make install PG_CONFIG=/opt/pg18/bin/pg_config >> ~/pgvector_make.log 2>&1
# pg_vectorsearch
cd ~/src; [ -d pg_vectorsearch ] || git clone -q https://github.com/timescale/pg_vectorsearch.git
cd pg_vectorsearch; [ -d build ] && mv build build.old.$$
meson setup build --buildtype=release -Dsimd=full -Dnative=false -Dblas=enabled -Dtools=disabled -Dc_args=-I/usr/include/openblas -Dpg_config=/opt/pg18/bin/pg_config > ~/pgvs_meson.log 2>&1
meson compile -C build >> ~/pgvs_meson.log 2>&1; meson install -C build >> ~/pgvs_meson.log 2>&1
# pg_turbovec
command -v rustup >/dev/null || true
rustup-init -y --default-toolchain 1.98.0 --profile minimal > ~/rustup.log 2>&1 || true
source ~/.cargo/env
rustup toolchain install 1.98.0 --profile minimal >> ~/rustup.log 2>&1
rustup default 1.98.0
cargo install --locked cargo-pgrx --version 0.19.3 > ~/cargo_pgrx_install.log 2>&1
cargo pgrx init --pg18 /opt/pg18/bin/pg_config > ~/pgrx_init.log 2>&1
cd ~/src; [ -d pg_turbovec ] || git clone -q --branch $TV https://github.com/gburd/pg_turbovec.git
cd pg_turbovec
cargo pgrx install --release --pg-config /opt/pg18/bin/pg_config --no-default-features --features pg18 > ~/pgtv_install.log 2>&1
{
  echo "== pg_config"; pg_config --version; pg_config --configure; pg_config --cflags
  echo "== pgvector"; (cd ~/src/pgvector; git describe --tags; git rev-parse HEAD)
  echo "== pg_vectorsearch"; (cd ~/src/pg_vectorsearch; git rev-parse HEAD; git log -1 --format='%ci %s'; cat VERSION)
  echo "== pg_turbovec"; (cd ~/src/pg_turbovec; git describe --tags; git rev-parse HEAD)
  echo "== rustc"; rustc -vV; cargo pgrx --version
  echo "== .so md5"; md5sum $(pg_config --pkglibdir)/*.so | grep -E 'vector|turbovec'
  echo "== meson config"; grep -iE "simd|native|blas" ~/pgvs_meson.log | head -30
} > $R/versions.txt 2>&1
echo BUILD_ALL_DONE
