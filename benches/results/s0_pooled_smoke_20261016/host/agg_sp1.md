# Pooled benchmark — smoke_flat_sp1_20261016T161217Z

> **Topology: collapsed-single-host DIAGNOSTIC**

- host: `ip-172-31-26-189` (8 cores) — Intel(R) Xeon(R) Platinum 8488C
- DB `localhost:28818` · pooler `localhost:6432` · drivers merged: 1
- corpus: 500,000 × 768-d, metric=cosine, index=flat (lists=0, bit_width=4), k=10
- `turbovec.scan_parallelism` = **1**

**Headline:** peak sustained **100 QPS** (13 QPS/core, 79.753 CPU-ms/query) at 16 clients; p99 261.67 ms, p99.9 291.07 ms at that point.

## Closed-loop (max QPS) sweep

_Single-client row is a diagnostic (per METHODOLOGY.md), not the headline._

| clients | QPS | QPS/core | CPU-ms/q | p50 | p95 | p99 | p99.9 | max |
|--------:|----:|---------:|---------:|----:|----:|----:|------:|----:|
| 1 _(diag)_ | 15 | 15 | 64.555 | 64.55 | 67.02 | 67.46 | 67.70 | 68.13 |
| 2 | 29 | 15 | 68.843 | 68.71 | 72.23 | 73.05 | 73.79 | 73.79 |
| 4 | 58 | 14 | 69.005 | 68.85 | 73.20 | 76.57 | 78.02 | 78.07 |
| 8 | 97 | 13 | 77.874 | 80.03 | 96.61 | 108.23 | 117.79 | 136.48 |
| 16 | 100 | 13 | 79.753 | 158.39 | 224.92 | 261.67 | 291.07 | 301.08 |

## Open-loop (fixed arrival rate — tail latency under queueing)

| target rate | achieved QPS | p50 | p95 | p99 | p99.9 | max |
|------------:|-------------:|----:|----:|----:|------:|----:|
| 200 | 101 | 4981.33 | 7641.03 | 8533.94 | 8998.15 | 9130.28 |
| 500 | 100 | 7696.21 | 11648.13 | 12061.84 | 12256.38 | 12348.43 |
| 1000 | 101 | 8765.80 | 13176.73 | 13510.28 | 13615.17 | 13663.93 |

