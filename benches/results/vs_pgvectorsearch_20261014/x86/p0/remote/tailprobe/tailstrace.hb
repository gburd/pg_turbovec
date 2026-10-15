[heartbeat] starting: bash /home/fedora/tail_strace.sh
[heartbeat] interval: 60s
[heartbeat] label:    bash
latencies: 156.3 154.9 152.7 147.6 150.4 153.3 151.0 152.6 156.7 157.3 153.9 152.7 156.2 89.0 156.1 151.2 144.5 156.1 146.2 150.7 133.1 160.5 155.2 151.6 154.1 133.4 145.8 149.8 146.0 148.2 152.3 145.6 143.2 160.8 150.4 160.1 158.0 149.9 157.3 152.8 144.1 154.0 152.8 155.5 150.4 159.9 146.1 151.7 159.6 148.5 161.9 156.4 146.3 148.8 150.7 152.0 157.3 153.6 147.5 155.2
--- syscalls taking > 50 ms:
    196 <...
      1 epoll_wait(468,
47339 <... futex resumed>)              = 0 <0.062089>
47427 <... futex resumed>)              = 0 <0.092933>
47428 <... futex resumed>)              = 0 <0.092849>
47429 <... futex resumed>)              = 0 <0.085039>
47339 <... futex resumed>)              = 0 <0.054272>
47339 <... futex resumed>)              = 0 <0.051140>
47339 <... futex resumed>)              = 0 <0.051286>
47339 <... futex resumed>)              = 0 <0.052781>
STRACE_DONE
[heartbeat] DONE rc=0 (clean exit)
