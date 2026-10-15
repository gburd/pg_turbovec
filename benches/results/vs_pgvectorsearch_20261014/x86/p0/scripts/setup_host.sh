#!/usr/bin/env bash
# VS_BENCH_PLAN § B setup (x86 P0). Run as fedora with sudo.
set -euxo pipefail
R=~/results/p0; mkdir -p $R
# 1. packages
sudo dnf -y install gcc gcc-c++ clang clang-devel llvm make meson ninja-build bison flex readline-devel zlib-devel \
  libzstd-devel lz4-devel openssl-devel liburing-devel libicu-devel perl python3-devel python3-pip openblas-devel \
  mdadm xfsprogs numactl kernel-tools perf sysstat nvme-cli fio git rustup tmux rsync strace util-linux-core \
  pkgconf-pkg-config wget bc jq python3-numpy
# 2. NVMe RAID0 -> XFS /data
if ! mountpoint -q /data; then
  DEVS=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print "/dev/"$1}')
  N=$(echo $DEVS | wc -w)
  sudo mdadm --create /dev/md0 --level=0 --raid-devices=$N --chunk=512K $DEVS --run
  sudo mkfs.xfs -f -K /dev/md0
  sudo mkdir -p /data && sudo mount -o noatime,nodiratime,logbufs=8 /dev/md0 /data
  sudo chown fedora:fedora /data
fi
# 3. CPU
sudo cpupower frequency-set -g performance || true
sudo cpupower idle-set -D 2 || true
# 4. memory sysctls (THP left at AMI default — recorded only, plan R18)
sudo swapoff -a || true
sudo sysctl -w vm.swappiness=1 vm.zone_reclaim_mode=0 kernel.numa_balancing=0 vm.dirty_background_bytes=1073741824 vm.dirty_bytes=4294967296
# shared_buffers 128GB + 2GB -> 2MiB pages
sudo sysctl -w vm.nr_hugepages=66560
{
  echo "== date"; date -u
  echo "== uname -r"; uname -r
  echo "== THP enabled/defrag (AMI default, unchanged)"; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
  echo "== lscpu"; lscpu
  echo "== numactl -H"; numactl -H
  echo "== microcode"; grep -m1 microcode /proc/cpuinfo
  echo "== isa gate"; grep -o -w -E 'avx512vbmi|avx512_vnni|avx512bw|amx_tile' /proc/cpuinfo | sort -u
  echo "== nvme list"; sudo nvme list
  echo "== md0"; cat /proc/mdstat; df -h /data; xfs_info /data
  echo "== cpupower frequency-info"; cpupower frequency-info
  echo "== cpupower idle-info"; cpupower idle-info | head -30
  echo "== hugepages"; grep -i huge /proc/meminfo
  echo "== sysctls"; sysctl vm.swappiness vm.zone_reclaim_mode kernel.numa_balancing vm.dirty_background_bytes vm.dirty_bytes vm.nr_hugepages
  echo "== swap"; swapon --show; free -g
  echo "== gcc/clang"; gcc --version | head -1; clang --version | head -1
} > $R/host.txt 2>&1
echo SETUP_OS_DONE
