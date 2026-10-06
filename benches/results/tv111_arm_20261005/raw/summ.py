import json, sys, statistics, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows: g[(r.get("label") or f"k{r.get('k')}", r["arm"], r["envset"])].append(r)
labels = sorted({k[0] for k in g}, key=lambda s: (s.split("_k")[0], int(s.rsplit("k", 1)[1]) if s.rsplit("k",1)[1].isdigit() else 0))
key = "p50_ms" if "p50_ms" in rows[0] else "kernel_p50_ms"
print(f"{'arm':22} " + " ".join(f"{l:>16}" for l in labels))
for arm, env in (("old", "default"), ("new", "default"), ("new", "planes_off")):
    cells = []
    for l in labels:
        rs = g.get((l, arm, env), [])
        if not rs: cells.append(f"{'-':>16}"); continue
        v = statistics.median(r[key] for r in rs); rc = min(r.get("recall10", 1) for r in rs)
        cells.append(f"{v:9.2f} R{rc:.3f}" if "recall10" in rs[0] else f"{v:16.2f}")
    print(f"{arm+'/'+env:22} " + " ".join(cells))
o = {l: statistics.median(r[key] for r in g[(l, "old", "default")]) for l in labels if (l, "old", "default") in g}
n = {l: statistics.median(r[key] for r in g[(l, "new", "default")]) for l in labels if (l, "new", "default") in g}
print("speedup old/new:  " + "  ".join(f"{l}={o[l]/n[l]:.3f}x" for l in labels if l in o and l in n))
