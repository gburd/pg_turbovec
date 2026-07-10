#!/usr/bin/env bash
# Toolchain + PG17.5-from-source + extensions install for the qdrant/annbench run.
set -euo pipefail
export HOME=/home/ec2-user
cd /mnt/nvme

echo "=== dnf packages ==="
sudo dnf install -y gcc gcc-c++ make cmake git clang clang-devel llvm \
  readline-devel zlib-devel openssl-devel libicu-devel bison flex \
  perl-core openblas openblas-devel python3.11 python3.11-devel python3.11-pip \
  docker tar gzip which pkgconfig >/dev/null 2>&1
echo "dnf done"

echo "=== docker ==="
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

echo "=== rust ==="
if [ ! -d "$HOME/.rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustc --version

echo "=== build PostgreSQL 17.5 from source ==="
cd /mnt/nvme/src
if [ ! -d /mnt/nvme/pgsql/bin ]; then
  curl -sSL https://ftp.postgresql.org/pub/source/v17.5/postgresql-17.5.tar.bz2 -o pg.tar.bz2
  tar xjf pg.tar.bz2
  cd postgresql-17.5
  ./configure --prefix=/mnt/nvme/pgsql --without-icu >/dev/null
  make -j32 >/dev/null 2>&1
  make install >/dev/null 2>&1
  cd contrib && make -j32 >/dev/null 2>&1 && sudo make install >/dev/null 2>&1 || make install >/dev/null 2>&1
fi
export PATH=/mnt/nvme/pgsql/bin:$PATH
pg_config --version

echo "=== initdb + start PG ==="
export PGDATA=/mnt/nvme/pgdata
if [ ! -f "$PGDATA/PG_VERSION" ]; then
  initdb -D "$PGDATA" -U ec2-user >/dev/null
  cat >> "$PGDATA/postgresql.conf" <<EOF
listen_addresses = ''
unix_socket_directories = '/mnt/nvme/pg'
shared_buffers = 32GB
maintenance_work_mem = 32GB
max_parallel_maintenance_workers = 32
max_parallel_workers = 32
max_parallel_workers_per_gather = 16
work_mem = 512MB
effective_cache_size = 200GB
max_wal_size = 32GB
checkpoint_timeout = 60min
fsync = off
synchronous_commit = off
full_page_writes = off
EOF
fi
pg_ctl -D "$PGDATA" -l /mnt/nvme/logs/pg.log -o "-k /mnt/nvme/pg" start || pg_ctl -D "$PGDATA" -l /mnt/nvme/logs/pg.log start
sleep 3
psql -h /mnt/nvme/pg -d postgres -c "SELECT version();"
psql -h /mnt/nvme/pg -d postgres -c "SELECT 1" >/dev/null && createdb -h /mnt/nvme/pg vecbench 2>/dev/null || echo "vecbench exists"

echo "=== pgvector 0.8.0 ==="
cd /mnt/nvme/src
if [ ! -d pgvector ]; then
  git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git
fi
cd pgvector
make clean >/dev/null 2>&1 || true
PG_CONFIG=/mnt/nvme/pgsql/bin/pg_config make -j32 >/dev/null 2>&1
PG_CONFIG=/mnt/nvme/pgsql/bin/pg_config make install >/dev/null 2>&1
echo "pgvector installed"

echo "=== cargo-pgrx 0.17.0 ==="
source "$HOME/.cargo/env"
if ! cargo pgrx --version 2>/dev/null | grep -q "0.17"; then
  cargo install --locked cargo-pgrx --version 0.17.0 >/dev/null 2>&1
fi
cargo pgrx --version
# init pgrx against the source-built PG17
cargo pgrx init --pg17 /mnt/nvme/pgsql/bin/pg_config 2>&1 | tail -3

echo "=== build+install pg_turbovec v1.25.0 ==="
cd /mnt/nvme/src/pg_turbovec
export RUSTFLAGS="-C target-cpu=native -L /usr/lib64"
# use pgrx install against the running PG (only pg17 feature needed since v1.3.0)
cargo pgrx install --release --no-default-features --features pg17 --pg-config /mnt/nvme/pgsql/bin/pg_config 2>&1 | tail -15
echo "turbovec install rc=$?"

echo "=== python venv ==="
python3.11 -m venv /mnt/nvme/venv
/mnt/nvme/venv/bin/pip install --quiet --upgrade pip
/mnt/nvme/venv/bin/pip install --quiet numpy h5py psycopg2-binary qdrant-client
echo "python deps installed"

echo "=== create extensions ==="
psql -h /mnt/nvme/pg -d vecbench -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_turbovec;"
psql -h /mnt/nvme/pg -d vecbench -c "SELECT extname, extversion FROM pg_extension ORDER BY 1;"

echo "ALL_SETUP_DONE"
