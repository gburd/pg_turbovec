#!/bin/sh
# finding 8: toast_tuple_target = 8160 arm next to EXTENDED and MAIN, same run
cd /work/fixc
echo "loadavg before: $(cat /proc/loadavg)"
/work/venv/bin/python bench.py 1024 5 200 32,256,1024 ext main ttt > warm_ttt_1024.json
echo "loadavg after:  $(cat /proc/loadavg)"
