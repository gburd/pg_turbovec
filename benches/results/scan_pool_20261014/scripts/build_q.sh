#!/usr/bin/env bash
# Build PG 18.6 (P0 flags), rust 1.98.0, cargo-pgrx 0.19.3 (pg18 + pg16 download), and TWO pg_turbovec .so:
# OLD = tag v2.12.1, NEW = branch fix/scan-concurrency-2.12.2 head at clone time. NEW is installed FIRST and
# OLD LAST, so the installed control file is 2.12.1 and CREATE EXTENSION / index builds run on OLD.
set -euxo pipefail
PGV=18.6; R=~/results/q; mkdir -p $R ~/src ~/so
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
rustup-init -y --default-toolchain 1.98.0 --profile minimal > ~/rustup.log 2>&1 || true
source ~/.cargo/env
rustup toolchain install 1.98.0 --profile minimal >> ~/rustup.log 2>&1
rustup default 1.98.0
cargo install --locked cargo-pgrx --version 0.19.3 > ~/cargo_pgrx_install.log 2>&1
cargo pgrx init --pg18 /opt/pg18/bin/pg_config --pg16 download > ~/pgrx_init.log 2>&1
cd ~/src
[ -d tv-old ] || git clone -q --branch v2.12.1 https://github.com/gburd/pg_turbovec.git tv-old
[ -d tv-new ] || git clone -q --branch fix/scan-concurrency-2.12.2 https://github.com/gburd/pg_turbovec.git tv-new
for arm in new old; do
  (cd ~/src/tv-$arm && cargo pgrx install --release --pg-config /opt/pg18/bin/pg_config --no-default-features --features pg18 > ~/build_$arm.log 2>&1)
  cp $(pg_config --pkglibdir)/pg_turbovec.so ~/so/$arm.so
done
{
  echo "== pg_config"; pg_config --version; pg_config --configure
  for arm in old new; do echo "== tv-$arm"; (cd ~/src/tv-$arm; git describe --tags --always; git rev-parse HEAD; git log -1 --format='%ci %s'; grep -n '^turbovec = ' Cargo.toml; grep '^version' Cargo.toml | head -1); done
  echo "== rustc"; rustc -vV; cargo pgrx --version
  echo "== .so"; md5sum ~/so/*.so; sha256sum ~/so/*.so; ls -l ~/so/
  echo "== installed control"; cat $(pg_config --sharedir)/extension/pg_turbovec.control; ls $(pg_config --sharedir)/extension/ | grep turbovec
} > $R/versions.txt 2>&1
cat $R/versions.txt
echo BUILD_Q_DONE
