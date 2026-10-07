#!/bin/bash
set -uo pipefail
: > /work/ab/results_warm.jsonl
for r in 1 2 3; do for arm in old new; do bash ~/tvp_arm_warm.sh $arm $r; done; done
: > /work/ab/attr_warm.jsonl
for arm in old new; do
  bash ~/tvp_arm_warm.sh $arm attr >/dev/null 2>&1 || true
done
echo WARM_LATENCY_DONE
