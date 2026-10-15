### r8gd.metal-24xl — CO1024 co1m matched-recall headline (round0 discarded + ABBA×2)

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads | rounds | p50 spread% |
|---|---|---|---|---|---|---|---|---|---|
| pg_turbovec | 0.9 | 10 | 0.9421 | 2.48 | 2.59 | 7.4 | 5 | 2 | 0.6 |
| pg_turbovec | 0.95 | 16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 | 2 | 0.4 |
| pg_turbovec | 0.98 | 16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 | 2 | 0.4 |
| pg_turbovec | 0.99 | 16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 | 2 | 0.4 |
| pg_vectorsearch | 0.9 | 14 | 0.9119 | 1.22 | 1.60 | 0.9 | 1 | 2 | 0.1 |
| pg_vectorsearch | 0.95 | 28 | 0.9564 | 1.84 | 2.56 | 1.5 | 1 | 2 | 0.1 |
| pg_vectorsearch | 0.98 | 56 | 0.9804 | 3.06 | 4.36 | 2.8 | 1 | 2 | 0.1 |
| pg_vectorsearch | 0.99 | 112 | 0.9920 | 5.55 | 7.56 | 5.3 | 1 | 2 | 0.5 |

Ratio of medians (pg_turbovec / pg_vectorsearch) at R@10>=0.95: **1.38x** (95% CI [1.38, 1.39], n=5000 shared qids). <1 = turbovec faster.
