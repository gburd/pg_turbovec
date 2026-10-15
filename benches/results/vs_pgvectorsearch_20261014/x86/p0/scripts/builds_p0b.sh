#!/usr/bin/env bash
# serial rebuild (try-2 tv builds overlapped the try-1 HNSW build -> timings invalid)
set -x
export PATH=/opt/pg18/bin:$PATH
B=~/build_index.sh
psql -h /tmp -X -d db_tv -c "DROP INDEX IF EXISTS docs_tv_flat" -c "DROP INDEX IF EXISTS docs_tv_ivf"
psql -h /tmp -X -d db_pv -c "DROP INDEX IF EXISTS docs_hnsw"
psql -h /tmp -X -d db_pv -c CHECKPOINT; psql -h /tmp -X -d db_tv -c CHECKPOINT
bash $B db_tv tv_flat4_t32 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_tv_flat ON docs USING turbovec (emb vec_cosine_ops)"
bash $B db_tv tv_ivf1000_t32 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_tv_ivf ON docs USING turbovec (emb vec_cosine_ops) WITH (lists = 1000)"
bash $B db_pv hnsw_m16_ef64_p32 "SET max_parallel_maintenance_workers = 31" "CREATE INDEX docs_hnsw ON docs USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
echo BUILDS_DONE
