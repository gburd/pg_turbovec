### r8gd.metal-24xl — CO1024 co10m matched-recall headline (round0 discarded + ABBA×2)

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads | rounds | p50 spread% |
|---|---|---|---|---|---|---|---|---|---|
| pg_turbovec | 0.9 | 12 | 0.8892 | 31.10 | 37.09 | 82.7 | 5 | 2 | 0.8 |
| pg_turbovec | 0.95 | 32 | 0.9510 | 61.27 | 70.60 | 177.1 | 5 | 2 | 0.3 |
| pg_turbovec | 0.98 | 96 | 0.9822 | 59.30 | 68.55 | 125.4 | 5 | 2 | 1.0 |
| pg_turbovec | 0.99 | None | — | — | — | — | — | — | not reached (max=0.9830) |
| pg_vectorsearch | 0.9 | 20 | 0.8922 | 1.67 | 2.12 | 1.4 | 2 | 2 | 0.2 |
| pg_vectorsearch | 0.95 | 56 | 0.9531 | 3.50 | 4.60 | 3.2 | 2 | 2 | 0.4 |
| pg_vectorsearch | 0.98 | 160 | 0.9817 | 8.63 | 11.44 | 8.4 | 2 | 2 | 1.8 |
| pg_vectorsearch | 0.99 | 320 | 0.9906 | 15.86 | 20.87 | 15.8 | 2 | 2 | 0.1 |

Ratio of medians (pg_turbovec / pg_vectorsearch) at R@10>=0.95: **17.51x** (95% CI [17.42, 17.61], n=5000 shared qids). <1 = turbovec faster.
