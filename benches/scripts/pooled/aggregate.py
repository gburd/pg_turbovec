#!/usr/bin/env python3
"""benches/scripts/pooled/aggregate.py — fold per-driver run JSONs into one
QPS/latency series and emit the headline markdown table.

On the real three-tier rig each load-fleet node writes one JSON
(<run_tag>_<driver_id>.json); this folds them: closed-loop runs at the same
client count sum their TPS (aggregate server QPS) and merge their latency
samples' percentiles conservatively (max of the per-driver percentile, since
we don't ship raw per-txn logs between hosts — a documented approximation;
for an exact aggregate percentile, pull the raw -l logs and re-percentile).

On the collapsed single-host diagnostic there is one driver, so folding is a
no-op and the percentiles are exact.

Usage:
    aggregate.py OUT.md run1.json [run2.json ...]

Headline table (per METHODOLOGY.md): QPS, QPS/core, p50/p95/p99/p99.9 at
each sweep point, for the recall operating point stated in the run header.
Single-client p50 is kept but LABELLED a diagnostic row, never the lead.
"""
import json
import sys
from collections import defaultdict


def load(paths):
    headers, runs = [], []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        headers.append(d)
        for r in d.get("runs", []):
            r = dict(r)
            r["_src"] = p
            runs.append(r)
    return headers, runs


def fold(runs):
    """Group by (mode, clients, rate); sum tps + cores, take max of tail
    percentiles across drivers (conservative). QPS/core recomputed from the
    folded totals."""
    groups = defaultdict(list)
    for r in runs:
        key = (r["mode"], r["clients"], r.get("rate"))
        groups[key].append(r)
    folded = []
    for (mode, clients, rate), rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1], k[0][2] or 0)):
        tps = sum(x["tps"] for x in rs)
        cores = sum(x["backend_cores_busy"] for x in rs)
        qpc = tps / cores if cores > 0 else 0.0
        cpums = cores * 1000.0 / tps if tps > 0 else 0.0
        folded.append({
            "mode": mode, "clients": clients, "rate": rate,
            "n_drivers": len(rs), "tps": tps, "qps_per_core": qpc,
            "backend_cores_busy": cores, "cpu_ms_per_query": cpums,
            "p50_ms": max(x["p50_ms"] for x in rs),
            "p95_ms": max(x["p95_ms"] for x in rs),
            "p99_ms": max(x["p99_ms"] for x in rs),
            "p999_ms": max(x["p999_ms"] for x in rs),
            "max_ms": max(x["max_ms"] for x in rs),
        })
    return folded


def md_table(folded, mode):
    rows = [r for r in folded if r["mode"] == mode]
    if not rows:
        return ""
    out = []
    if mode == "closed":
        out.append("| clients | QPS | QPS/core | CPU-ms/q | p50 | p95 | p99 | p99.9 | max |")
        out.append("|--------:|----:|---------:|---------:|----:|----:|----:|------:|----:|")
        for r in rows:
            diag = " _(diag)_" if r["clients"] == 1 else ""
            out.append("| {clients}{d} | {tps:.0f} | {qpc:.0f} | {cpums:.3f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {p999:.2f} | {mx:.2f} |".format(
                clients=r["clients"], d=diag, tps=r["tps"], qpc=r["qps_per_core"],
                cpums=r["cpu_ms_per_query"], p50=r["p50_ms"], p95=r["p95_ms"],
                p99=r["p99_ms"], p999=r["p999_ms"], mx=r["max_ms"]))
    else:  # open
        out.append("| target rate | achieved QPS | p50 | p95 | p99 | p99.9 | max |")
        out.append("|------------:|-------------:|----:|----:|----:|------:|----:|")
        for r in rows:
            out.append("| {rate} | {tps:.0f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {p999:.2f} | {mx:.2f} |".format(
                rate=r["rate"], tps=r["tps"], p50=r["p50_ms"], p95=r["p95_ms"],
                p99=r["p99_ms"], p999=r["p999_ms"], mx=r["max_ms"]))
    return "\n".join(out)


def peak_knee(folded):
    closed = [r for r in folded if r["mode"] == "closed"]
    if not closed:
        return None
    peak = max(closed, key=lambda r: r["tps"])
    return peak


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: aggregate.py OUT.md run1.json [run2.json ...]")
    out_md, paths = sys.argv[1], sys.argv[2:]
    headers, runs = load(paths)
    folded = fold(runs)
    h = headers[0]

    lines = []
    lines.append("# Pooled benchmark — {}".format(h.get("run_tag", "?")))
    lines.append("")
    lines.append("> **Topology: {}**".format(h.get("topology", "?")))
    lines.append("")
    lines.append("- host: `{}` ({} cores) — {}".format(h.get("host"), h.get("ncpu"), h.get("cpu_model")))
    lines.append("- DB `{}` · pooler `{}` · drivers merged: {}".format(h.get("db"), h.get("pooler"), len(paths)))
    lines.append("- corpus: {:,} × {}-d, metric={}, index={} (lists={}, bit_width={}), k={}".format(
        h.get("corpus_rows", 0), h.get("dim"), h.get("metric"), h.get("index_kind"),
        h.get("lists"), h.get("bit_width"), h.get("k")))
    lines.append("- `turbovec.scan_parallelism` = **{}**".format(h.get("scan_parallelism")))
    lines.append("")
    peak = peak_knee(folded)
    if peak:
        lines.append("**Headline:** peak sustained **{:.0f} QPS** ({:.0f} QPS/core, "
                     "{:.3f} CPU-ms/query) at {} clients; "
                     "p99 {:.2f} ms, p99.9 {:.2f} ms at that point.".format(
                         peak["tps"], peak["qps_per_core"], peak["cpu_ms_per_query"],
                         peak["clients"], peak["p99_ms"], peak["p999_ms"]))
        lines.append("")
    lines.append("## Closed-loop (max QPS) sweep")
    lines.append("")
    lines.append("_Single-client row is a diagnostic (per METHODOLOGY.md), not the headline._")
    lines.append("")
    lines.append(md_table(folded, "closed"))
    open_tbl = md_table(folded, "open")
    if open_tbl:
        lines.append("")
        lines.append("## Open-loop (fixed arrival rate — tail latency under queueing)")
        lines.append("")
        lines.append(open_tbl)
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_md)
    # also echo the headline to stdout
    if peak:
        print("PEAK: {:.0f} QPS @ {} clients, {:.0f} QPS/core, p99 {:.2f}ms".format(
            peak["tps"], peak["clients"], peak["qps_per_core"], peak["p99_ms"]))


if __name__ == "__main__":
    main()
