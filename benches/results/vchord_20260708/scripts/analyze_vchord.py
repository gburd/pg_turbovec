#!/usr/bin/env python3
"""Build recall-band Pareto tables from the 4-engine leg3 JSON results."""
import json, glob, os

RES = "/tmp/vchord_results"

def load(engine, corpus):
    p = f"{RES}/leg3_{engine}_{corpus}.json"
    if not os.path.exists(p):
        return []
    rows = [r for r in json.load(open(p)) if "recall" in r and "p50" in r]
    for r in rows:
        r["_engine"] = engine
    return rows

def cfg_str(r):
    e = r["_engine"]
    if e == "hnsw": return f"{r['variant']} ef={r['ef_search']}"
    if e == "diskann": return f"{r['variant']} sl={r.get('search_list_size')} rs={r.get('query_rescore')}"
    if e == "vchord": return f"L{r['lists']} p={r['probes']} eps={r['epsilon']}"
    if "turbovec" in e: return f"L{r['lists']} p={r['probes']} sk={r['search_k']}"
    return "?"

def idx_mb(r):
    return r.get("idx_bytes", 0) // 10**6

def best_at_band(rows, band):
    """lowest p50 among configs reaching >= band recall."""
    ok = [r for r in rows if r["recall"] >= band]
    if not ok:
        # none reach it: return the highest-recall config as a MISS
        m = max(rows, key=lambda r: r["recall"])
        return m, True
    return min(ok, key=lambda r: r["p50"]), False

ENGINES = ["hnsw", "vchord", "diskann", "turbovec_tuned", "turbovec_default"]
LABEL = {"hnsw":"HNSW", "vchord":"VectorChord", "diskann":"DiskANN",
         "turbovec_tuned":"pg_turbovec(tuned)", "turbovec_default":"pg_turbovec(default)"}

for corpus, dim in [("sift1m", 128), ("gist1m", 960)]:
    print(f"\n{'='*78}\n{corpus.upper()} / {dim}-dim\n{'='*78}")
    data = {e: load(e, corpus) for e in ENGINES}
    for band in [0.90, 0.95, 0.99]:
        print(f"\n--- R@10 >= {band} ---")
        print(f"{'engine':<22}{'config':<26}{'recall':>7}{'p50ms':>9}{'qps':>7}{'buildS':>9}{'idxMB':>7}")
        for e in ENGINES:
            if not data[e]:
                continue
            r, miss = best_at_band(data[e], band)
            tag = " MISS" if miss else ""
            print(f"{LABEL[e]:<22}{cfg_str(r):<26}{r['recall']:>7.3f}{r['p50']:>9.2f}{r['qps_1conn']:>7.0f}{r.get('build_s',0):>9.0f}{idx_mb(r):>7}{tag}")
