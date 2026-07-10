#!/usr/bin/env bash
# Leg 1 continuation: turbovec IN-RAM (both corpora) + Qdrant (both).
# HNSW already done in leg1.log. Sequential, never concurrent.
set -x
export PATH=/mnt/nvme/pgsql/bin:$PATH
PY=/mnt/nvme/venv/bin/python
cd /mnt/nvme/src

echo "### turbovec sift1m IN-RAM (lists 1000,4000)"
$PY tv_leg.py turbovec sift1m 1000,4000
echo "### turbovec gist1m IN-RAM (lists 1000,4000)"
$PY tv_leg.py turbovec gist1m 1000,4000

echo "### qdrant sift1m"
$PY qdrant_leg.py sift1m
echo "### qdrant gist1m"
$PY qdrant_leg.py gist1m

echo "LEG1B_ALL_DONE"
