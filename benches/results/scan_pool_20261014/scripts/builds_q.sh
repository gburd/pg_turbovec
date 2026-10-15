#!/usr/bin/env bash
# Build the two indexes ONCE, under the OLD (v2.12.1) .so (installed last by build_q.sh). build_parallelism=32.
set -x
export PATH=/opt/pg18/bin:$PATH; B=~/build_index.sh
psql -h /tmp -X -d db_tv -Atc "SELECT turbovec.turbovec_version()" | tee ~/results/q/build_arm.txt
bash $B db_tv tv_flat4 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_tv_flat ON docs USING turbovec (emb vec_cosine_ops)"
bash $B db_tv tv_ivf1000 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_ivf_tv ON docs_ivf USING turbovec (emb vec_cosine_ops) WITH (lists = 1000)"
bash ~/idx_sha.sh after_build
echo BUILDS_DONE
