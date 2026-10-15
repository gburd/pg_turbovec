#!/usr/bin/env python3
"""Snapshot for pgbench windows: /proc/stat cpu line + per-postmaster-child utime+stime (thread group) and
num_threads. Prints one JSON line."""
import os, json, time
pm = int(open("/data/pgdata/postmaster.pid").readline())
kids = {}
for d in os.listdir("/proc"):
    if not d.isdigit(): continue
    try: s = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError): continue
    if int(s[1]) == pm: kids[d] = dict(cpu=int(s[11]) + int(s[12]), nth=int(s[17]))
print(json.dumps(dict(t=time.time(), hz=os.sysconf("SC_CLK_TCK"), stat_cpu=open("/proc/stat").readline().split()[1:], kids=kids)))
