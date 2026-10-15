#!/usr/bin/env python3
"""Aggregate P3 frontier: per (dataset,host) build the matched-recall headline table.
Reads <dir>/operating_points_<tag>.json + r{1,2}_<sys>_<knob>.ids.summary.json (round 0 discarded).
Per system per target T: headline knob = smallest knob with calib recall >= T; latency = median over
the 2 counted rounds of per-round p50 (and p95), CPU-ms/q, threads. Paired bootstrap ratio of medians
(tv/pv) over the shared qids at each system's 0.95 point. Writes <dir>/headline_<tag>.md + .json.
usage: p3_agg.py <dir> <tag> <host>"""
import sys, json, glob, os, numpy as np
D, TAG, HOST = sys.argv[1], sys.argv[2], sys.argv[3]
op = json.load(open(f"{D}/operating_points_{TAG}.json"))
def rounds_for(sysn, knob):
    out=[]
    for r in (1,2):
        f=f"{D}/r{r}_{sysn}_{knob}.ids.summary.json"
        if os.path.exists(f): out.append(json.load(open(f)))
    return out
TARGETS=[0.90,0.95,0.98,0.99]
rows=[]
for sysn,label in (("tv","pg_turbovec"),("pv","pg_vectorsearch")):
    hl=op[sysn]["headline"]; recalls={int(k):v for k,v in op[sysn]["recalls"].items()}
    for T in TARGETS:
        knob=hl[str(T)]
        if knob is None:
            rows.append(dict(sys=label,target=T,knob=None,note="not reached (max=%.4f)"%max(recalls.values()))); continue
        rr=rounds_for(sysn,knob)
        if not rr: rows.append(dict(sys=label,target=T,knob=knob,note="no timed rounds")); continue
        p50=float(np.median([x["wall_p50_ms"] for x in rr])); p95=float(np.median([x["wall_p95_ms"] for x in rr]))
        cpu=float(np.median([x["cpu_ms_per_q_tasksum"] for x in rr])); th=int(np.median([x["threads"] for x in rr]))
        rec=float(np.median([x["recall"] for x in rr])); spread=max([x["wall_p50_ms"] for x in rr])-min([x["wall_p50_ms"] for x in rr])
        rows.append(dict(sys=label,target=T,knob=knob,recall=rec,p50=p50,p95=p95,cpu_ms_q=cpu,threads=th,
                         rounds=len(rr),p50_spread_pct=100*spread/p50 if p50 else 0,calib_recall=recalls[knob]))
# paired bootstrap ratio of medians at 0.95
def load_lat(sysn,knob):
    d={}
    for r in (1,2):
        f=f"{D}/r{r}_{sysn}_{knob}.ids.jsonl"
        if os.path.exists(f):
            for line in open(f):
                j=json.loads(line); d.setdefault(j["qid"],[]).append(j["wall_ms"])
    return {q:float(np.median(v)) for q,v in d.items()}
boot={}
for T in (0.95,):
    tk,pk=op["tv"]["headline"][str(T)],op["pv"]["headline"][str(T)]
    if tk and pk:
        lt,lp=load_lat("tv",tk),load_lat("pv",pk); qs=sorted(set(lt)&set(lp))
        if qs:
            at=np.array([lt[q] for q in qs]); ap=np.array([lp[q] for q in qs])
            rng=np.random.default_rng(42); ratios=[]
            for _ in range(10000):
                idx=rng.integers(0,len(qs),len(qs)); ratios.append(np.median(at[idx])/np.median(ap[idx]))
            boot[str(T)]=dict(ratio_median=float(np.median(at)/np.median(ap)),ci=[float(np.percentile(ratios,2.5)),float(np.percentile(ratios,97.5))],n_shared=len(qs))
json.dump(dict(host=HOST,tag=TAG,rows=rows,boot_ratio_tv_over_pv=boot),open(f"{D}/headline_{TAG}.json","w"),indent=1)
# markdown
L=[f"### {HOST} — CO1024 {TAG} matched-recall headline (round0 discarded + ABBA×2)",""]
L.append("| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads | rounds | p50 spread% |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r.get("note"): L.append(f"| {r['sys']} | {r['target']} | {r.get('knob')} | — | — | — | — | — | — | {r['note']} |")
    else: L.append(f"| {r['sys']} | {r['target']} | {r['knob']} | {r['recall']:.4f} | {r['p50']:.2f} | {r['p95']:.2f} | {r['cpu_ms_q']:.1f} | {r['threads']} | {r['rounds']} | {r['p50_spread_pct']:.1f} |")
if boot:
    b=boot.get("0.95")
    if b: L+=["","Ratio of medians (pg_turbovec / pg_vectorsearch) at R@10>=0.95: **{:.2f}x** (95% CI [{:.2f}, {:.2f}], n={} shared qids). <1 = turbovec faster.".format(b['ratio_median'],b['ci'][0],b['ci'][1],b['n_shared'])]
open(f"{D}/headline_{TAG}.md","w").write("\n".join(L)+"\n")
print("\n".join(L))
