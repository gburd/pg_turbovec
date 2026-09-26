#!/usr/bin/env bash
set -e
cd ~/pgtv
git checkout -- src/index/scan.rs 2>/dev/null || true
python3 - <<'PY'
p="src/index/scan.rs"; s=open(p).read()
# 1. time the FIRST read_full_consistent (flat install_whole_index path).
old1='    let (meta, codes, scales, ids) = relfile::read_full_consistent(rel, meta);\n    let meta = &meta;'
new1='''    let __t0 = std::time::Instant::now();
    let (meta, codes, scales, ids) = relfile::read_full_consistent(rel, meta);
    let __t_read = __t0.elapsed();
    let meta = &meta;'''
assert s.count(old1)==2, s.count(old1)
s=s.replace(old1,new1,1)  # first (flat) only
# 2. time the flat stored_index block: wrap the FIRST has_prepared_layout if.
old2='    let stored_index: cache::ReadOnlyIndex = if meta.has_prepared_layout() {'
assert s.count(old2)==2, s.count(old2)
s=s.replace(old2,'    let __t1 = std::time::Instant::now();\n'+old2,1)  # first only
# 3. log before the flat scan_install (unique line).
old3='    relfile::unlock_relfile_read(rel);\n    cache::scan_install(key, stored_index, total_bytes, relfile_node, version_as_i64)'
assert s.count(old3)==1, s.count(old3)
new3='''    let __t_prepare = __t1.elapsed();
    if std::env::var_os("TURBOVEC_SCAN_TRACE").is_some() {
        pgrx::log!("TVSCAN cold-open: read_full_consistent={}us prepare_repack={}us n={} bw={} dim={}",
            __t_read.as_micros(), __t_prepare.as_micros(), meta.n_vectors, meta.bit_width, meta.dim);
    }
    relfile::unlock_relfile_read(rel);
    cache::scan_install(key, stored_index, total_bytes, relfile_node, version_as_i64)'''
s=s.replace(old3,new3,1)
open(p,"w").write(s)
print("instrumented flat cold-open path")
PY
. ~/.cargo/env
cargo pgrx install --release --features pg16 --no-default-features -c $(which pg_config) >/dev/null 2>&1
sudo systemctl restart postgresql@16-main; sleep 4
echo "INSTRUMENTED_BUILT"
