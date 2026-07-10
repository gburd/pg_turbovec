#!/usr/bin/env bash
# Leg 2 orchestrator: sequential 10M builds with budget cap.
set -x
export PATH=/mnt/nvme/pgsql/bin:$PATH
PY=/mnt/nvme/venv/bin/python
cd /mnt/nvme/src

echo "### turbovec 10m (lists=4000, budget 90m)"
$PY leg2_10m.py turbovec

echo "### hnsw 10m (m32/efc256, budget 90m)"
$PY leg2_10m.py hnsw

echo "LEG2_PG_DONE"
