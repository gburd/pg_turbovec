#!/usr/bin/env bash
# Leg 1 orchestrator: sequential, never concurrent index builds.
set -x
export PATH=/mnt/nvme/pgsql/bin:$PATH
PY=/mnt/nvme/venv/bin/python
cd /mnt/nvme/src

echo "### drop smoke index"
psql -h /mnt/nvme/pg -d vecbench -c "DROP INDEX IF EXISTS sm CASCADE;" || true

echo "### HNSW sift1m"
$PY tv_leg.py hnsw sift1m
echo "### HNSW gist1m"
$PY tv_leg.py hnsw gist1m

echo "### turbovec sift1m (lists 1000,4000)"
$PY tv_leg.py turbovec sift1m 1000,4000
echo "### turbovec gist1m (lists 1000,4000)"
$PY tv_leg.py turbovec gist1m 1000,4000

echo "### qdrant sift1m"
$PY qdrant_leg.py sift1m
echo "### qdrant gist1m"
$PY qdrant_leg.py gist1m

echo "LEG1_ALL_DONE"
