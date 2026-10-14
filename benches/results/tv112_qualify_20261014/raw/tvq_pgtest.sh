set -uo pipefail
. ~/.cargo/env
D=/work/pgtv-$(date +%s); mkdir -p $D && tar xzf ~/pgtv-deps-wt.tgz -C $D && cd $D/pgtv-wt-deps
# Fedora's openblas lives in /usr/lib64; bindgen finds clang via clang-devel.
cargo pgrx test pg16 > /work/pgtest.log 2>&1; echo "EXIT $?" >> /work/pgtest.log
grep -E '^test result|FAILED|panicked|EXIT' /work/pgtest.log | head -20
