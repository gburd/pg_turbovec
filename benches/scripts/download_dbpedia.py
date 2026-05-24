#!/usr/bin/env python3
"""Download dbpedia-entities-openai-1M parquet shards."""
import json, os, sys, urllib.request, time

OUT = "/scratch/pg_turbovec-bench/dbpedia"
os.makedirs(OUT, exist_ok=True)

api = "https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M/tree/main/data"
with urllib.request.urlopen(api, timeout=30) as r:
    files = json.load(r)

shards = sorted([f["path"] for f in files if f["path"].endswith(".parquet")])
print(f"{len(shards)} shards", flush=True)

base = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/"
for i, p in enumerate(shards):
    out = os.path.join(OUT, f"train-{i:05d}.parquet")
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f"[{i:02d}] skip {out} ({os.path.getsize(out)})", flush=True)
        continue
    url = base + p
    t0 = time.time()
    print(f"[{i:02d}] GET {url}", flush=True)
    urllib.request.urlretrieve(url, out)
    sz = os.path.getsize(out)
    print(f"[{i:02d}] {sz/1e6:.1f} MB in {time.time()-t0:.1f}s -> {out}", flush=True)
print("DONE", flush=True)
