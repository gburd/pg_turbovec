#!/bin/bash
# Build PG 16 (same minor pgrx downloaded) WITHOUT cassert, -O2, install to
# /mnt/nvme/pg16r, point pgrx at it, reinstall both pg_turbovec .so from the
# SAME commits, and move the cluster over (initdb fresh; reload corpus from
# the saved base.f32 via binary COPY is not needed — we dump/restore docs).
set -euxo pipefail
. ~/.cargo/env
SRC=$(ls -d ~/.pgrx/16.*/ | grep -v install | head -1)
VER=$(basename $SRC); echo "PG source $VER"
cd /mnt/nvme && cp -r $SRC pgsrc_r && cd pgsrc_r
make distclean >/dev/null 2>&1 || true
./configure --prefix=/mnt/nvme/pg16r --without-icu CFLAGS="-O2" >/dev/null
make -j32 -s >/dev/null 2>&1 && make -s install >/dev/null
make -C contrib/pageinspect -s install >/dev/null
PGC=/mnt/nvme/pg16r/bin/pg_config
$PGC --configure
cd /mnt/nvme/pg_turbovec
for arm in old new; do
  if [ $arm = old ]; then git checkout -q v2.10.3; else git checkout -q origin/deps/turbovec-1.1.1; fi
  cargo pgrx install --release --pg-config $PGC > /mnt/nvme/buildr_$arm.log 2>&1
  cp $($PGC --pkglibdir)/pg_turbovec.so /mnt/nvme/so/pg_turbovec_${arm}.so
  md5sum /mnt/nvme/so/pg_turbovec_${arm}.so
done
echo PGR_OK
