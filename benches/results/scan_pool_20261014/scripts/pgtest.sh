#!/usr/bin/env bash
# Step 6: cargo pgrx test pg16 on a FRESH clone of the branch head (debug profile, pgrx-managed PG 16.15).
set -uo pipefail
. ~/.cargo/env
cd ~/src; D=tv-test-$(date +%s); git clone -q --branch fix/scan-concurrency-2.12.2 https://github.com/gburd/pg_turbovec.git $D; cd $D
{ git rev-parse HEAD; git log -1 --format='%ci %s'; grep -n '^turbovec = ' Cargo.toml; nproc; } > ~/results/q/pgtest_head.txt
cargo pgrx test pg16 > ~/results/q/pgtest_pg16.txt 2>&1; echo "EXIT $?" >> ~/results/q/pgtest_pg16.txt
grep -E '^test result|FAILED|panicked|EXIT' ~/results/q/pgtest_pg16.txt | head -20
