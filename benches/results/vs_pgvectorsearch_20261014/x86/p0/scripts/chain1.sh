#!/usr/bin/env bash
# wait for the running calib (by its heartbeat log), then prism calib re-run, then Q4/Q5/equal-CPU probe
while ! grep -q '^\[heartbeat\] DONE' ~/hb/calib.log; do sleep 10; done
mkdir -p ~/results/p0/calib/prism_failed_try1; mv ~/results/p0/calib/prism_* ~/results/p0/calib/prism_failed_try1/ 2>/dev/null
bash ~/calib_p0.sh prism
bash ~/q45_probe.sh
echo CHAIN1_DONE
