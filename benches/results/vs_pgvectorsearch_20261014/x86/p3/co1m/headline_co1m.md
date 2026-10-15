### i7i.metal-24xl — CO1024 co1m matched-recall headline (round0 discarded + ABBA×2)

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads | rounds | p50 spread% |
|---|---|---|---|---|---|---|---|---|---|
| pg_turbovec | 0.9 | 10 | 0.9416 | 1.89 | 2.03 | 5.3 | 5 | 2 | 1.9 |
| pg_turbovec | 0.95 | 16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 | 2 | 0.2 |
| pg_turbovec | 0.98 | 16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 | 2 | 0.2 |
| pg_turbovec | 0.99 | 16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 | 2 | 0.2 |
| pg_vectorsearch | 0.9 | 14 | 0.9127 | 0.89 | 1.09 | 0.6 | 1 | 2 | 0.6 |
| pg_vectorsearch | 0.95 | 28 | 0.9580 | 1.26 | 1.61 | 1.0 | 1 | 2 | 0.4 |
| pg_vectorsearch | 0.98 | 56 | 0.9818 | 1.79 | 2.36 | 1.5 | 1 | 2 | 0.3 |
| pg_vectorsearch | 0.99 | 112 | 0.9925 | 2.91 | 3.73 | 2.6 | 1 | 2 | 0.2 |

Ratio of medians (pg_turbovec / pg_vectorsearch) at R@10>=0.95: **1.57x** (95% CI [1.56, 1.57], n=5000 shared qids). <1 = turbovec faster.
