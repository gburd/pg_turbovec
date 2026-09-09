import json

def load(p): return json.load(open(p))

arts = {d: load(f"/scratch/bqdim-20260909/bq_dimsweep_arnold_d{d}_20260909.json")
        for d in (256, 512, 1024)}
pub = load("/scratch/.done-bqfrontier-20260908/bq_frontier_arnold_20260908.json")

print("=" * 92)
print("STORAGE bytes/vector, 250,000 rows")
print("=" * 92)
print(f"{'dim':>6} {'bw1':>9} {'bw2':>9} {'bw4':>9} {'bw2/bw1':>9} {'bw4/bw1':>9}  {'dim/8':>7}")
for d in (256, 512, 1024):
    b = {i['bit_width']: i.get('bytes_per_vector') for i in arts[d]['indexes']}
    print(f"{d:>6} {b[1]:>9.2f} {b[2]:>9.2f} {b[4]:>9.2f} {b[2]/b[1]:>9.3f} {b[4]/b[1]:>9.3f}  {d/8:>7.1f}")
pb = {i['bit_width']: i.get('bytes_per_vector') for i in pub['indexes']}
print(f"{'1024P':>6} {pb[1]:>9.2f} {pb[2]:>9.2f} {pb[4]:>9.2f} {pb[2]/pb[1]:>9.3f} {pb[4]/pb[1]:>9.3f}   <- published 2026-09-08")

print()
print("=" * 92)
print("BUILD seconds")
print("=" * 92)
for d in (256, 512, 1024):
    b = {i['bit_width']: i.get('build_s') for i in arts[d]['indexes']}
    print(f"  dim={d:>4}:  bw1={b[1]:>6}  bw2={b[2]:>6}  bw4={b[4]:>6}")

print()
print("=" * 92)
print("FULL CURVE  R@10 / R@100 / p50(ms) vs exact-rerank window")
print("=" * 92)
for d in (256, 512, 1024):
    gt = arts[d]['meta']['ground_truth']
    print(f"\n--- dim={d}   (GT {gt['rows']} rows in {gt['seconds']}s) ---")
    hdr = f"{'win':>5} |"
    for bw in (1, 2, 4):
        hdr += f" {'bw%d R@10' % bw:>9} {'R@100':>7} {'p50':>7} |"
    print(hdr)
    byw = {}
    for r in arts[d]['configs']:
        if r['rerank_mode'] != 'off':
            continue
        byw.setdefault(r['rerank_window_predicted'], {})[r['bit_width']] = r
    for w in sorted(byw):
        line = f"{w:>5} |"
        for bw in (1, 2, 4):
            r = byw[w].get(bw)
            if r:
                line += f" {r['recall_at_k']:>9.3f} {r['recall_at_100']:>7.3f} {r['p50_ms']:>7.2f} |"
            else:
                line += f" {'-':>9} {'-':>7} {'-':>7} |"
        print(line)
    for bw in (1, 2, 4):
        a = [r for r in arts[d]['configs'] if r['rerank_mode'] == 'auto' and r['bit_width'] == bw]
        if a:
            r = a[0]
            print(f"  auto bw{bw}: window={r['rerank_window_predicted']:>4} "
                  f"R@10={r['recall_at_k']:.3f} R@100={r['recall_at_100']:.3f} p50={r['p50_ms']:.2f}ms")

print()
print("=" * 92)
print("ISO-RECALL: cheapest window clearing each R@10 target, and p50 there")
print("=" * 92)
for tgt in (0.95, 0.99):
    print(f"\n  target R@10 >= {tgt}")
    print(f"  {'dim':>5} | {'bw1 win':>8} {'bw1 p50':>8} | {'bw2 win':>8} {'bw2 p50':>8} | {'bw4 win':>8} {'bw4 p50':>8} | {'win ratio bw1/bw2':>18}")
    for d in (256, 512, 1024):
        cells = {}
        for bw in (1, 2, 4):
            ok = sorted((r for r in arts[d]['configs']
                         if r['rerank_mode'] == 'off' and r['bit_width'] == bw
                         and r['recall_at_k'] >= tgt),
                        key=lambda r: r['rerank_window_predicted'])
            cells[bw] = ok[0] if ok else None
        line = f"  {d:>5} |"
        for bw in (1, 2, 4):
            r = cells[bw]
            line += f" {r['rerank_window_predicted']:>8} {r['p50_ms']:>8.2f} |" if r else f" {'NEVER':>8} {'-':>8} |"
        if cells[1] and cells[2]:
            line += f" {cells[1]['rerank_window_predicted']/cells[2]['rerank_window_predicted']:>18.1f}x"
        else:
            line += f" {'n/a':>18}"
        print(line)

print()
print("=" * 92)
print("HYPOTHESIS TEST: does the 1-bit penalty shrink as dim rises?")
print("=" * 92)
print("  bw1 R@10 at a FIXED window, across dim (higher = 1-bit doing better):")
print(f"  {'win':>5} | {'d256':>8} {'d512':>8} {'d1024':>8} |  trend")
for w in (32, 100, 256, 400, 800, 1024, 2000):
    vals = {}
    for d in (256, 512, 1024):
        m = [r for r in arts[d]['configs']
             if r['rerank_mode'] == 'off' and r['bit_width'] == 1
             and r['rerank_window_predicted'] == w]
        vals[d] = m[0]['recall_at_k'] if m else None
    if all(v is not None for v in vals.values()):
        t = "RISES with dim" if vals[1024] > vals[256] else "falls with dim"
        print(f"  {w:>5} | {vals[256]:>8.3f} {vals[512]:>8.3f} {vals[1024]:>8.3f} |  {t}")
print()
print("  bw1 R@100 at the `auto` default, across dim:")
for d in (256, 512, 1024):
    a = [r for r in arts[d]['configs'] if r['rerank_mode'] == 'auto' and r['bit_width'] == 1]
    if a:
        print(f"    dim={d:>4}: auto window={a[0]['rerank_window_predicted']:>4} "
              f"R@10={a[0]['recall_at_k']:.3f} R@100={a[0]['recall_at_100']:.3f}")

print()
print("=" * 92)
print("HONESTY CHECKS")
print("=" * 92)
for d in (256, 512, 1024):
    cfg = arts[d]['configs']
    bad = [r for r in cfg if not r['plan']['index_scan']]
    cont = [r for r in cfg if ((r.get('latency') or {}).get('contention') or {}).get('contended_flag')]
    busy = [((r.get('latency') or {}).get('contention') or {}).get('cpu_busy_pct') for r in cfg]
    busy = [b for b in busy if b is not None]
    errs = [i.get('build_error') for i in arts[d]['indexes'] if i.get('build_error')]
    m = arts[d]['meta']
    print(f"  dim={d}: rows={len(cfg)}  non-IndexScan={len(bad)}  contended={len(cont)}/{len(cfg)}"
          f"  cpu_busy%={min(busy):.1f}-{max(busy):.1f}  build_errors={errs or 'none'}")
    print(f"          qs={m['query_set']['table']} gt_tbl=bqd_gt_{d} n_q={m['query_set']['n_queries']}"
          f" prov={m['query_set']['provenance']} latency_publishable={m['latency_publishable']}")
    print(f"          start_load={m['start_loadavg'].split()[0]}  simd={m['simd']['kernel_tier']}")

print()
print("  published 2026-09-08 for comparison:")
pc = pub['configs']
pcont = [r for r in pc if ((r.get('latency') or {}).get('contention') or {}).get('contended_flag')]
print(f"    rows={len(pc)} contended={len(pcont)}/{len(pc)} start_load={pub['meta']['start_loadavg'].split()[0]}")

print()
print("=" * 92)
print("CONSISTENCY CHECK: my d1024 vs published d1024 (same corpus/rows/host)")
print("=" * 92)
print(f"  {'win':>5} | {'bw1 mine':>9} {'bw1 pub':>9} {'delta':>7} | {'bw1 p50 mine':>12} {'pub':>8}")
pubw = {}
for r in pub['configs']:
    if r['rerank_mode'] == 'off':
        pubw.setdefault(r['rerank_window_predicted'], {})[r['bit_width']] = r
minew = {}
for r in arts[1024]['configs']:
    if r['rerank_mode'] == 'off':
        minew.setdefault(r['rerank_window_predicted'], {})[r['bit_width']] = r
for w in sorted(set(pubw) & set(minew)):
    a = minew[w].get(1)
    b = pubw[w].get(1)
    if a and b:
        print(f"  {w:>5} | {a['recall_at_k']:>9.3f} {b['recall_at_k']:>9.3f} "
              f"{a['recall_at_k']-b['recall_at_k']:>+7.3f} | {a['p50_ms']:>12.2f} {b['p50_ms']:>8.2f}")
