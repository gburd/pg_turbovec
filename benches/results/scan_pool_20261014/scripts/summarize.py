#!/usr/bin/env python3
"""Aggregate warm/*.ids.summary.json -> markdown tables. Round 0 is discarded. For each (arm, cell):
median over rounds 1..3 of p50 / p95 / p99 / CPU-ms/q (thread group), plus min..max of p50 across rounds,
max threads, recall (identical across rounds or flagged), and the SUM over rounds of queries > 60 / > 100 ms."""
import json, glob, sys, os, re, statistics as st
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/results/q/warm")
cells = {}
for f in glob.glob(f"{D}/r*_*.ids.summary.json"):
    s = json.load(open(f)); m = re.match(r"r(\d)_(old|new)_(.*)", s["label"])
    rnd, arm, cell = int(m[1]), m[2], m[3]
    cells.setdefault((cell, arm), {})[rnd] = s
def row(cell, arm):
    r = cells.get((cell, arm));
    if not r: return None
    k = [x for x in sorted(r) if x >= 1]
    if not k: return None
    g = lambda f: [r[x][f] for x in k]
    rec = sorted(set(round(v, 4) for v in g("recall")))
    return dict(n=len(k), p50=st.median(g("wall_p50_ms")), p50lo=min(g("wall_p50_ms")), p50hi=max(g("wall_p50_ms")),
                p95=st.median(g("wall_p95_ms")), p99=st.median(g("wall_p99_ms")), cpu=st.median(g("cpu_ms_per_q_threadgroup")),
                thr=max(g("threads_max")), rec="/".join(f"{v:.4f}" for v in rec), gt60=sum(g("slow_gt60")), gt100=sum(g("slow_gt100")),
                cold=st.median(g("warm_first_ms")), ov=sum(g("order_violations")) + sum(g("short_results")),
                pss=st.median([r[x]["smaps_rollup_kb"].get("Pss_Anon", 0) / 1024 for x in k]))
H = "| cell | arm | rounds | R@10 | p50 ms (median; min–max) | p95 ms | p99 ms | CPU-ms/q | threads (max) | queries >60 ms / >100 ms (Σ 3×1000) | Pss_Anon MB | 1st query ms | order/short viol. |"
print(H); print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
order = ["old", "new"]
def key(c):
    m = re.match(r"flat_sk(\d+)_sp(\w+)", c)
    if m: return (0, int(m[1]), {"def": 0, "1": 1, "88": 2, "96": 3}.get(m[2], 9))
    return (1, c, 0)
for cell in sorted({c for c, _ in cells}, key=key):
    for arm in order:
        x = row(cell, arm)
        if x: print(f"| {cell} | {arm.upper()} | {x['n']} | {x['rec']} | {x['p50']:.2f} ({x['p50lo']:.2f}–{x['p50hi']:.2f}) | {x['p95']:.2f} | {x['p99']:.2f} | {x['cpu']:.1f} | {x['thr']} | {x['gt60']} / {x['gt100']} | {x['pss']:.0f} | {x['cold']:.0f} | {x['ov']} |")
