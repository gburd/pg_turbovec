#!/bin/bash
# turbovec 1.1.1 qualification bootstrap (Debian 13 arm64, c8gd)
set -euxo pipefail
sudo mkfs.ext4 -q -F /dev/nvme0n1 || true
sudo mkdir -p /mnt/nvme && (mountpoint -q /mnt/nvme || sudo mount -o noatime /dev/nvme0n1 /mnt/nvme)
sudo chown admin:admin /mnt/nvme
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl pkg-config libssl-dev libclang-dev clang \
  libreadline-dev zlib1g-dev bison flex libicu-dev python3-venv python3-dev tmux sysstat jq \
  linux-perf >/dev/null
# CPU governor/THP (best effort on a VM)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null || true
cd /mnt/nvme
# Rust (rustup, LLVM 22 — the nix 1.97/LLVM21 rustc cannot compile turbovec 1.1)
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.0 --profile minimal
. ~/.cargo/env
cargo install --locked cargo-pgrx --version 0.19.1 >/mnt/nvme/pgrx-install.log 2>&1
# PG16 from source via pgrx (also creates ~/.pgrx/config.toml)
cargo pgrx init --pg16 download > /mnt/nvme/pgrx-init.log 2>&1
python3 -m venv /mnt/nvme/venv
/mnt/nvme/venv/bin/pip install -q numpy pyarrow huggingface_hub psycopg[binary]
echo BOOTSTRAP_OK
