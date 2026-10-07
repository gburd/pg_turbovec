#!/usr/bin/env python3
"""Turn the raw JSON in this directory into the FINDINGS tables. Run locally:
python3 summarize.py > raw_summary.txt"""
import json, statistics as st
def J(p): return json.load(open(p))
print("== warm kNN, EXPLAIN Execution Time median (ms), default glibc malloc [faults/query]")
for d in (384, 768, 1024, 1536):
    s = J(f"raw_warm_default_{d}.json")["summary"]; p = J(f"raw_warm_pinned_{d}.json")["summary"]
    for k in (32, 256, 1024):
        row = []
        for v in ("ext", "extl", "main", "plain"):
            key = f"{v}/k{k}"
            if key in s:
                row.append(f"{v} {s[key]['median_ms']:6.2f} [{max(s[key]['backend_minor_faults_per_query']):.0f}] pinned {p[key]['median_ms']:6.2f}")
        print(f"  d={d:<4} k={k:<4} " + " | ".join(row))
print("\n== MAIN - EXTENDED at search_k=1024 (1024 rechecked candidates), and the k=256->1024 slope")
for cfg in ("default", "pinned", "run1"):
    for d in (384, 768, 1024, 1536):
        f = f"raw_warm_run1_{d}.json" if cfg == "run1" else f"raw_warm_{cfg}_{d}.json"
        s = J(f)["summary"]; m = lambda v, k: s[f"{v}/k{k}"]["median_ms"]
        dk = m("main", 1024) - m("ext", 1024)
        slope = ((m("main", 1024) - m("main", 256)) - (m("ext", 1024) - m("ext", 256))) / 768 * 1e3 if "ext/k256" in s else float("nan")
        tot_e = (m("ext", 1024) - m("ext", 256)) / 768 * 1e3 if "ext/k256" in s else float("nan")
        tot_m = (m("main", 1024) - m("main", 256)) / 768 * 1e3 if "ext/k256" in s else float("nan")
        print(f"  {cfg:7s} d={d:<4} ext {m('ext',1024):6.2f}  main {m('main',1024):6.2f}  delta {dk:+6.2f} ms = {dk/1024*1e3:+5.2f} us/cand"
              f"   slope-check {slope:+5.2f} us/cand (per-cand total ext {tot_e:5.2f}, main {tot_m:5.2f})")
print("\n== defaults (search_k=32, oversample=1, hi_dim_rerank=auto -> 1024 candidates at 1024-d)")
for k, v in J("raw_bench_1024_defaults.json")["summary"].items(): print(f"  {k}: {v['median_ms']:.2f} ms  rounds {v['round_medians_ms']}")
print("\n== cold (PG restart + drop_caches before every run), 1024-d, median of 3")
c = J("raw_cold_1024.json")["runs"]
for k, v in c.items():
    print(f"  {k:10s} exec {st.median(x['exec_ms'] for x in v):8.1f} ms  runs {[round(x['exec_ms']) for x in v]}  shared_read {v[0]['shared_read_blocks']:6d}"
          f"  io_read {st.median(x['io_read_ms'] for x in v):7.1f} ms  | same query warm {st.median(x['warm_exec_ms'] for x in v):6.2f} ms")
print("\n== COPY 100k x 1024-d, median of 3 (s)")
for v, t in J("raw_copytime_1024.json")["median_s"].items(): print(f"  {v:6s} {t:6.2f}")
