#!/bin/bash
# Fedora 44 qualification host bootstrap for pg_turbovec / turbovec.
set -euxo pipefail
sudo dnf -y -q install gcc gcc-c++ make git clang clang-devel llvm-devel pkgconf-pkg-config openssl-devel \
  readline-devel zlib-devel libicu-devel bison flex perl openblas-devel libxml2-devel libxslt-devel \
  lz4-devel libzstd-devel xz tar zip python3 python3-pip rsync numactl perf 2>&1 | tail -2
sudo mkdir -p /work && sudo chown $USER /work
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain 1.98.0
. ~/.cargo/env
cargo install -q --locked cargo-pgrx --version 0.19.3
cargo pgrx init --pg16 download > /work/pgrx-init.log 2>&1
echo BOOTSTRAP_DONE
