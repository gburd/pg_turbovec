#!/usr/bin/env bash
# P2 build driver. Makes a per-kind table copy of docs_$TAG, builds the index (build #1 + #2 variance),
# records wall/size/WAL/RSS via build_index.sh. √n lists for IVF/prism-equivalent chosen by $N.
# usage: build_ds.sh <tag> <N> <kinds...>   kinds in {tvflat,tvivf,prism,hnsw}
set -uxo pipefail
export PATH=/opt/pg18/bin:$PATH
TAG=$1; N=$2; shift 2; KINDS="$*"
R=~/results/p2; mkdir -p $R; B=~/build_index.sh
PS="psql -h /tmp -X -v ON_ERROR_STOP=1"
LISTS=$(python3 -c "import math;print(round(math.sqrt($N)))")
echo "build_ds TAG=$TAG N=$N lists(sqrt)=$LISTS kinds=$KINDS"
mk() { # db base-table new-table
  $PS -d $1 -c "DROP TABLE IF EXISTS $3 CASCADE" \
    -c "CREATE TABLE $3 (LIKE $2 INCLUDING ALL)" \
    -c "INSERT INTO $3 SELECT * FROM $2" \
    -c "VACUUM (FREEZE, ANALYZE) $3"
}
for kind in $KINDS; do
 case $kind in
  tvflat)
    mk db_tv docs_$TAG dtv_flat_$TAG
    bash $B db_tv b1_tvflat_$TAG "SET turbovec.build_parallelism=32" "CREATE INDEX ixtvflat_$TAG ON dtv_flat_$TAG USING turbovec (emb vec_cosine_ops)"
    bash $B db_tv b2_tvflat_$TAG "SET turbovec.build_parallelism=32" "CREATE INDEX ixtvflat2_$TAG ON dtv_flat_$TAG USING turbovec (emb vec_cosine_ops)" ;;
  tvivf)
    mk db_tv docs_$TAG dtv_ivf_$TAG
    bash $B db_tv b1_tvivf_$TAG "SET turbovec.build_parallelism=32" "CREATE INDEX ixtvivf_$TAG ON dtv_ivf_$TAG USING turbovec (emb vec_cosine_ops) WITH (lists=$LISTS)"
    bash $B db_tv b2_tvivf_$TAG "SET turbovec.build_parallelism=32" "CREATE INDEX ixtvivf2_$TAG ON dtv_ivf_$TAG USING turbovec (emb vec_cosine_ops) WITH (lists=$LISTS)" ;;
  prism)
    mk db_pv docs_$TAG dpv_prism_$TAG
    $PS -d db_pv -c "ALTER TABLE dpv_prism_$TAG SET (parallel_workers=31)"
    bash $B db_pv b1_prism_$TAG "SET max_parallel_maintenance_workers=31" "CREATE INDEX ixprism_$TAG ON dpv_prism_$TAG USING prism (emb vec32_cosine_ops)"
    bash $B db_pv b2_prism_$TAG "SET max_parallel_maintenance_workers=31" "CREATE INDEX ixprism2_$TAG ON dpv_prism_$TAG USING prism (emb vec32_cosine_ops)"
    $PS -d db_pv -Atc "SELECT prism_index_settings('ixprism_$TAG')" > $R/prism_settings_$TAG.txt 2>&1 || true ;;
  hnsw)
    mk db_pv docs_$TAG dpv_hnsw_$TAG
    $PS -d db_pv -c "ALTER TABLE dpv_hnsw_$TAG SET (parallel_workers=31)"
    bash $B db_pv b1_hnsw_$TAG "SET max_parallel_maintenance_workers=31" "CREATE INDEX ixhnsw_$TAG ON dpv_hnsw_$TAG USING hnsw (emb vector_cosine_ops) WITH (m=16, ef_construction=64)" ;;
 esac
done
echo "BUILD_DONE_$TAG"
