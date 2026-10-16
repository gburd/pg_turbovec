# Pooled benchmark — smoke_flat_spauto_20261016T161029Z

> **Topology: collapsed-single-host DIAGNOSTIC**

- host: `ip-172-31-26-189` (8 cores) — Intel(R) Xeon(R) Platinum 8488C
- DB `localhost:28818` · pooler `localhost:6432` · drivers merged: 1
- corpus: 500,000 × 768-d, metric=cosine, index=flat (lists=0, bit_width=4), k=10
- `turbovec.scan_parallelism` = **auto**

**Headline:** peak sustained **97 QPS** (12 QPS/core, 81.806 CPU-ms/query) at 16 clients; p99 246.56 ms, p99.9 271.25 ms at that point.

## Closed-loop (max QPS) sweep

_Single-client row is a diagnostic (per METHODOLOGY.md), not the headline._

| clients | QPS | QPS/core | CPU-ms/q | p50 | p95 | p99 | p99.9 | max |
|--------:|----:|---------:|---------:|----:|----:|----:|------:|----:|
| 1 _(diag)_ | 0 | 0 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 29 | 14 | 69.436 | 68.22 | 70.83 | 71.34 | 71.94 | 72.39 |
| 4 | 57 | 14 | 69.853 | 69.42 | 74.85 | 77.05 | 79.13 | 79.34 |
| 8 | 96 | 13 | 78.892 | 80.37 | 97.90 | 105.06 | 112.36 | 120.66 |
| 16 | 97 | 12 | 81.806 | 159.89 | 222.17 | 246.56 | 271.25 | 311.23 |

## Open-loop (fixed arrival rate — tail latency under queueing)

| target rate | achieved QPS | p50 | p95 | p99 | p99.9 | max |
|------------:|-------------:|----:|----:|----:|------:|----:|
| 200 | 102 | 4949.66 | 7319.52 | 8041.95 | 8321.79 | 8344.91 |
| 500 | 101 | 7819.52 | 11835.04 | 12207.00 | 12436.47 | 12469.98 |
| 1000 | 101 | 8681.62 | 13094.09 | 13488.75 | 13576.73 | 13600.13 |

