#!/usr/bin/env bash
# P2 chain for one host: load CO1024 1M + 10M, build headline indexes.
# 1M: tvflat (turbovec headline <1M) + prism (defaults) + hnsw (reference)
# 10M: tvivf (turbovec headline @scale) + prism (defaults)
set -uxo pipefail
export PATH=/opt/pg18/bin:$PATH
chmod +x ~/load_ds.sh ~/build_ds.sh ~/build_index.sh
bash ~/load_ds.sh /data/co1024_1m 1024 co1m
bash ~/load_ds.sh /data/co1024_10m 1024 co10m
bash ~/build_ds.sh co1m 1000000 tvflat prism hnsw
bash ~/build_ds.sh co10m 10000000 tvivf prism
echo P2_CHAIN_DONE
