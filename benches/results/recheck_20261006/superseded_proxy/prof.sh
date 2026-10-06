#!/bin/bash
# perf-profile ONE warm backend running the real ORDER BY ... LIMIT 10 query
# repeatedly at search_k=$1. Groups samples by component.
K=$1; TAG=$2; N=${3:-400}
P="/tmp/pg16rel/bin/psql -h /tmp/rck -p 55432 -d bench"
. /tmp/rck/env.sh
/tmp/rck/venv/bin/python -c "
import numpy as np; Q=np.load('/tmp/rck/q.npy')
print('SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off; SET turbovec.search_k=$K; SET jit=off;')
print('SELECT pg_backend_pid() \\\\g /tmp/rck/pid.txt')
print('SELECT pg_sleep(2);')
for i in range($N): print(\"SELECT id FROM docs ORDER BY tv <=> '[\" + ','.join(f'{x:.6f}' for x in Q[i%200]) + \"]'::turbovec.vector LIMIT 10;\")
" > /tmp/rck/q_$K.sql
( $P -qAt -o /dev/null -f /tmp/rck/q_$K.sql ) &
sleep 1.2; PID=$(tr -dc 0-9 < /tmp/rck/pid.txt); sleep 1.5
perf record -F 2999 -g --call-graph=fp -p $PID -o /tmp/rck/perf_$TAG.data -- sleep 6 2>/dev/null >/dev/null
wait
perf report -i /tmp/rck/perf_$TAG.data --no-children --sort symbol --stdio 2>/dev/null | grep -E '^\s+[0-9.]+%' | head -40 | cut -c1-150 > /tmp/rck/self_$TAG.txt
perf report -i /tmp/rck/perf_$TAG.data --children --sort symbol --stdio 2>/dev/null | grep -E '^\s+[0-9.]+%\s+[0-9.]+%' | head -60 | cut -c1-150 > /tmp/rck/incl_$TAG.txt
echo "pid=$PID samples: $(perf report -i /tmp/rck/perf_$TAG.data --stdio 2>/dev/null | grep -m1 'Samples' )"
