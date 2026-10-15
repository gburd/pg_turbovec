#!/usr/bin/env python3
"""Summarise pgb/<label>.* : TPS over the measured window [15,75) s from the 1 s aggregate logs (sum of all
pgbench threads' logs), latency avg/p50-of-1s-buckets, host CPU% from /proc/stat deltas between snap0/snap1,
postmaster-children CPU-cores and CPU-ms per transaction from the same snapshots, max backend threads."""
import json, glob, os, sys, collections
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/results/q/pgb")
labels = sorted({os.path.basename(f)[:-4] for f in glob.glob(f"{D}/p*_c*.out")})
print("| label | clients | TPS (window 15–75 s) | mean lat ms (window) | host CPU busy % | server CPU cores (postmaster children) | server CPU-ms / txn | max threads / backend | report tps (pgbench, whole run) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for L in labels:
    s0 = json.load(open(f"{D}/{L}.snap0")); s1 = json.load(open(f"{D}/{L}.snap1"))
    t0, t1 = s0["t"], s1["t"]; hz = s0["hz"]
    agg = collections.defaultdict(lambda: [0, 0.0])
    for f in glob.glob(f"{D}/{L}.*"):
        if f.endswith((".out", ".snap0", ".snap1", ".mpstat")): continue
        for line in open(f):
            p = line.split(); ts = int(p[0]); agg[ts][0] += int(p[1]); agg[ts][1] += float(p[2])
    w = [ts for ts in agg if t0 <= ts and ts + 1 <= t1]
    n = sum(agg[ts][0] for ts in w); lat = sum(agg[ts][1] for ts in w) / max(n, 1) / 1000
    dur = len(w)
    a = [int(x) for x in s0["stat_cpu"]]; b = [int(x) for x in s1["stat_cpu"]]
    tot = sum(b[:8]) - sum(a[:8]); idle = (b[3] + b[4]) - (a[3] + a[4]); busy = 100 * (1 - idle / tot)
    k0, k1 = s0["kids"], s1["kids"]
    srv = sum(k1[p]["cpu"] - k0.get(p, {"cpu": 0})["cpu"] for p in k1) / hz
    cores = srv / (t1 - t0); txn = n * (t1 - t0) / dur if dur else 0
    mth = max(v["nth"] for v in k1.values())
    rep = [l for l in open(f"{D}/{L}.out") if l.startswith("tps")]
    C = L.split("_c")[-1]
    print(f"| {L} | {C} | {n/dur:.0f} | {lat:.2f} | {busy:.1f} | {cores:.1f} | {srv*1000/txn:.1f} | {mth} | {rep[0].split()[2] if rep else '?'} |")
