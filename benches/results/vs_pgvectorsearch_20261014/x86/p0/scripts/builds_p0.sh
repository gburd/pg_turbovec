#!/usr/bin/env bash
set -x
export PATH=/opt/pg18/bin:$PATH
B=~/build_index.sh
psql -h /tmp -X -d db_tv -c "ALTER DATABASE db_tv SET search_path = public, turbovec"
psql -h /tmp -X -d db_pv -Atc "SELECT * FROM prism_index_settings('docs_prism')" > ~/results/p0/build/prism_index_settings.txt 2>&1
psql -h /tmp -X -d db_pv -c "\d+ docs" >> ~/results/p0/build/prism_index_settings.txt 2>&1
# headline turbovec @1M: flat 4-bit default, 32 threads
bash $B db_tv tv_flat4_t32 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_tv_flat ON docs USING turbovec (emb vec_cosine_ops)"
# P0 Q2: IVF lists = sqrt(1M) = 1000
bash $B db_tv tv_ivf1000_t32 "SET turbovec.build_parallelism = 32" "CREATE INDEX docs_tv_ivf ON docs USING turbovec (emb vec_cosine_ops) WITH (lists = 1000)"
# reference arm: pgvector HNSW m=16 ef_construction=64, 32 procs
bash $B db_pv hnsw_m16_ef64_p32 "SET max_parallel_maintenance_workers = 31" "CREATE INDEX docs_hnsw ON docs USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
echo BUILDS_DONE
