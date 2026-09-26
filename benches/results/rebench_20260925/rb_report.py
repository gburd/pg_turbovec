#!/usr/bin/env python3
import json, sys
d=json.load(open(sys.argv[1])); rows=d["results"]
print("# IVF/flat vs HNSW at matched recall — END-TO-END Execution Time")
print(f"# {d['meta']['corpus']} | basis={d['meta']['latency_basis']}")
valid=[r for r in rows if not r["contended"] and not r["seq_fallback"] and r["n"]>0]
dropped=[r for r in rows if r not in valid]
print(f"# recorded {len(valid)}/{len(rows)} arms ({len(dropped)} dropped: contended/seqscan/wrong-index)")
groups={"hnsw":"HNSW","flat_bw4":"flat bw4","flat_bw1":"flat bw1","ivf_bw4":"IVF1024 bw4","ivf_bw1":"IVF1024 bw1"}
def fam(r):
    if r["kind"]=="hnsw": return "hnsw"
    return "_".join(r["label"].split("_")[:2])
for tgt in (0.90,0.95,0.98):
    print(f"\n## at R@10 >= {tgt}")
    hnsw_ms=None
    best={}
    for r in valid:
        if r["recall"]>=tgt:
            f=fam(r)
            if f not in best or r["p50_ms"]<best[f]["p50_ms"]: best[f]=r
    hnsw=best.get("hnsw")
    hnsw_ms=hnsw["p50_ms"] if hnsw else None
    for f,name in groups.items():
        r=best.get(f)
        if r is None:
            # Distinguish grid-limited (recall still climbing in probes/search_k
            # when the sweep stopped) from a genuine plateau. Report the max
            # recall this family actually reached so the reader can't mistake a
            # too-short sweep for a capability wall.
            fam_rows=[x for x in valid if fam(x)==f]
            mx=max((x["recall"] for x in fam_rows), default=None)
            if mx is None:
                print(f"  {name:14s}: no valid arm recorded")
            else:
                print(f"  {name:14s}: not cleared by the swept grid "
                      f"(max R@10={mx:.4f} at the sweep's probe/search_k ceiling; "
                      f"widen the grid to confirm it is a limit, not a missing config)")
        else:
            rel=f" ({r['p50_ms']/hnsw_ms:.2f}x HNSW)" if hnsw_ms and f!='hnsw' else ""
            print(f"  {name:14s}: {r['p50_ms']:8.2f} ms  R@10={r['recall']:.4f}  [{r['label']}]{rel}")
