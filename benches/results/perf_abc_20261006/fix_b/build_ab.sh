#!/bin/bash
# Build base (3ff7768) and fix-b release .so against the private non-assert PG16 copy.
set -euo pipefail
export CARGO_BUILD_JOBS=8
trap "cd /work/wt-fix-b && git checkout -q perf/fix-b-qcache" EXIT
export PATH=$HOME/.cargo/bin:$PATH
PGC=/work/pg16rel_b/bin/pg_config
LIB=$($PGC --pkglibdir)
cd /work/wt-fix-b
HEAD=$(git rev-parse perf/fix-b-qcache)
git checkout -q 3ff7768
nice -n 10 cargo pgrx install --release --pg-config $PGC
cp $LIB/pg_turbovec.so /work/fixb/base.so
git checkout -q $HEAD
nice -n 10 cargo pgrx install --release --pg-config $PGC
cp $LIB/pg_turbovec.so /work/fixb/new.so
git checkout -q perf/fix-b-qcache
md5sum /work/fixb/*.so
