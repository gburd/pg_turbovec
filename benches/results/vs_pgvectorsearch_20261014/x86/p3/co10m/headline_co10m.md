### i7i.metal-24xl — CO1024 co10m matched-recall headline (round0 discarded + ABBA×2)

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads | rounds | p50 spread% |
|---|---|---|---|---|---|---|---|---|---|
| pg_turbovec | 0.9 | 12 | 0.8903 | 28.24 | 34.23 | 77.5 | 5 | 2 | 0.1 |
| pg_turbovec | 0.95 | 32 | 0.9502 | 63.22 | 85.61 | 177.9 | 5 | 2 | 0.1 |
| pg_turbovec | 0.98 | 96 | 0.9833 | 139.43 | 162.84 | 256.1 | 5 | 2 | 0.8 |
| pg_turbovec | 0.99 | None | — | — | — | — | — | — | not reached (max=0.9835) |
| pg_vectorsearch | 0.9 | 20 | 0.8951 | 1.18 | 1.41 | 0.9 | 3 | 2 | 0.5 |
| pg_vectorsearch | 0.95 | 56 | 0.9533 | 2.17 | 2.72 | 1.9 | 2 | 2 | 0.3 |
| pg_vectorsearch | 0.98 | 160 | 0.9819 | 4.97 | 6.38 | 4.7 | 2 | 2 | 0.1 |
| pg_vectorsearch | 0.99 | 320 | 0.9916 | 8.63 | 11.17 | 8.5 | 2 | 2 | 0.2 |

Ratio of medians (pg_turbovec / pg_vectorsearch) at R@10>=0.95: **29.08x** (95% CI [28.82, 29.32], n=5000 shared qids). <1 = turbovec faster.
