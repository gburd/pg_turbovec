#!/bin/bash
# Whole-load IVF tail diagnosis, end to end. Writes ~/diag/*.
set -uxo pipefail
mkdir -p ~/diag
sudo dnf -y -q install gcc git clang clang-devel llvm-devel openblas-devel readline-devel zlib-devel libicu-devel bison flex perl openssl-devel python3-pip perf > /dev/null 2>&1
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain 1.98.0 > /dev/null 2>&1
. ~/.cargo/env
cargo install -q --locked cargo-pgrx --version 0.19.3
cargo pgrx init --pg16 download > ~/diag/pgrx_init.log 2>&1
git clone -q --branch fix/scan-concurrency-2.12.2 https://github.com/gburd/pg_turbovec.git ~/pgtv
(cd ~/pgtv && git log --oneline -1 > ~/diag/pgtv_sha.txt)
PGB=$(ls -d ~/.pgrx/16.*/pgrx-install/bin | head -1); export PATH=$PGB:$PATH
(cd ~/pgtv && cargo pgrx install --release --pg-config $PGB/pg_config > ~/diag/install.log 2>&1)
python3 -m pip install -q --user numpy pyarrow huggingface_hub psycopg[binary]
D=~/pgdata; initdb -D $D -U fedora > /dev/null
cat >> $D/postgresql.conf <<EOC
port = 5432
shared_buffers = 16GB
max_parallel_workers_per_gather = 0
jit = off
max_wal_size = 32GB
checkpoint_timeout = 30min
unix_socket_directories = '/tmp'
listen_addresses = ''
EOC
pg_ctl -D $D -l ~/pg.log -w start
createdb -h /tmp bench
psql -h /tmp -d bench -qc "CREATE EXTENSION pg_turbovec; CREATE TABLE docs(id bigint PRIMARY KEY, tv turbovec.vector)"
mkdir -p ~/corpus; python3 ~/load_corpus.py 1000000 5200 bench > ~/diag/load.log 2>&1
psql -h /tmp -d bench -qc "VACUUM ANALYZE docs"
psql -h /tmp -d bench -c "CREATE INDEX docs_ivf ON docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 1000)"
echo "=== step A: whole vs ooc per query, planes ON (default)"
python3 ~/tail_diag.py 400 ~/diag/A_planes_on.jsonl > ~/diag/A.txt 2>&1; cat ~/diag/A.txt
echo "=== step B: same, postmaster with TURBOVEC_4BIT_PLANES=0"
pg_ctl -D $D -m fast -w stop; TURBOVEC_4BIT_PLANES=0 pg_ctl -D $D -l ~/pg.log -w start
python3 ~/tail_diag.py 400 ~/diag/B_planes_off.jsonl > ~/diag/B.txt 2>&1; cat ~/diag/B.txt
pg_ctl -D $D -m fast -w stop; pg_ctl -D $D -l ~/pg.log -w start
echo "=== step C: knobs on whole-load (planes on): hi_dim_rerank off; scan_parallelism 1"
python3 ~/tail_knobs.py > ~/diag/C.txt 2>&1; cat ~/diag/C.txt
echo "=== step D: perf profile of the 30 slowest whole-load queries"
python3 ~/tail_perf.py > ~/diag/D.txt 2>&1; cat ~/diag/D.txt
echo ALL_DONE
