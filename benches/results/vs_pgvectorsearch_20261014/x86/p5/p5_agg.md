### i7i.metal-24xl — P5 closed-loop QPS (20s warm + 120s timed, pinned client)

| dataset | system | clients | TPS | QPS/core | avg lat ms | p99 ms (pgbench log) |
|---|---|---|---|---|---|---|
| co10m | pg_vectorsearch | 1 | 506 | 503.1 | 1.98 | 2.8 |
| co10m | pg_vectorsearch | 16 | 7787 | 486.5 | 2.06 | 2.9 |
| co10m | pg_vectorsearch | 96 | 22294 | 255.1 | 4.31 | 9.7 |
| co10m | pg_turbovec (sp=4) | 1 | 14 | 5.3 | 73.82 | 90.1 |
| co10m | pg_turbovec (sp=4) | 16 | 158 | 4.1 | 101.42 | 130.2 |
| co10m | pg_turbovec (sp=4) | 96 | 229 | 2.6 | 419.77 | 702.4 |
| co10m | pg_turbovec (sp=1) | 1 | 12 | 12.4 | 80.14 | 127.4 |
| co10m | pg_turbovec (sp=1) | 16 | 152 | 9.5 | 105.56 | 159.4 |
| co10m | pg_turbovec (sp=1) | 96 | 194 | 2.2 | 494.85 | 823.8 |
| co1m | pg_vectorsearch | 1 | 1030 | 1020.1 | 0.97 | 1.4 |
| co1m | pg_vectorsearch | 16 | 16761 | 1049.9 | 0.95 | 1.4 |
| co1m | pg_vectorsearch | 96 | 49641 | 566.8 | 1.93 | 4.3 |
| co1m | pg_turbovec (sp=4) | 1 | 568 | 190.6 | 1.76 | 2.3 |
| co1m | pg_turbovec (sp=4) | 16 | 2293 | 43.2 | 6.98 | 7.5 |
| co1m | pg_turbovec (sp=4) | 96 | 2066 | 23.5 | 46.46 | 71.2 |
| co1m | pg_turbovec (sp=1) | 1 | 185 | 183.8 | 5.39 | 6.8 |
| co1m | pg_turbovec (sp=1) | 16 | 1158 | 72.4 | 13.82 | 14.1 |
| co1m | pg_turbovec (sp=1) | 96 | 2108 | 24.0 | 45.54 | 83.2 |
