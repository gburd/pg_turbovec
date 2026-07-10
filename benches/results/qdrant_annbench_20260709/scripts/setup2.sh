#!/usr/bin/env bash
# Continuation: pgvector install, cargo-pgrx, pg_turbovec, python venv.
export HOME=/home/ec2-user
source "$HOME/.cargo/env"
export PATH=/mnt/nvme/pgsql/bin:$HOME/.cargo/bin:$PATH
set -x

echo "=== pgvector install ==="
cd /mnt/nvme/src/pgvector
PG_CONFIG=/mnt/nvme/pgsql/bin/pg_config make -j32
PG_CONFIG=/mnt/nvme/pgsql/bin/pg_config make install
echo "pgvector install rc=$?"

echo "=== cargo-pgrx 0.17.0 ==="
if ! cargo pgrx --version 2>/dev/null | grep -q "0.17"; then
  cargo install --locked cargo-pgrx --version 0.17.0
fi
cargo pgrx --version
# point pgrx at the source-built PG17 (no managed cluster; use its pg_config)
cargo pgrx init --pg17 /mnt/nvme/pgsql/bin/pg_config 2>&1 | tail -3

echo "=== build+install pg_turbovec v1.25.0 ==="
cd /mnt/nvme/src/pg_turbovec
export RUSTFLAGS="-C target-cpu=native -L /usr/lib64"
cargo pgrx install --release --no-default-features --features pg17 --pg-config /mnt/nvme/pgsql/bin/pg_config 2>&1 | tail -20
echo "turbovec install rc=$?"

echo "=== python venv ==="
if [ ! -d /mnt/nvme/venv ]; then
  python3.11 -m venv /mnt/nvme/venv
fi
/mnt/nvme/venv/bin/pip install --quiet --upgrade pip
/mnt/nvme/venv/bin/pip install --quiet numpy h5py psycopg2-binary qdrant-client
echo "python venv rc=$?"

echo "=== create extensions ==="
psql -h /mnt/nvme/pg -d vecbench -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_turbovec;"
psql -h /mnt/nvme/pg -d vecbench -c "SELECT extname, extversion FROM pg_extension ORDER BY 1;"

echo "SETUP2_DONE"
