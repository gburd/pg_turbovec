### r8gd.metal-24xl — P5 closed-loop QPS (20s warm + 120s timed, pinned client)

| dataset | system | clients | TPS | QPS/core | avg lat ms | p99 ms (pgbench log) |
|---|---|---|---|---|---|---|
| co10m | pg_vectorsearch | 1 | 301 | 299.5 | 3.32 | 5.0 |
| co10m | pg_vectorsearch | 16 | 4787 | 304.2 | 3.34 | 5.0 |
| co10m | pg_vectorsearch | 96 | 19137 | 220.9 | 5.02 | 12.5 |
| co10m | pg_turbovec (sp=4) | 1 | 16 | 5.5 | 62.04 | 77.3 |
| co10m | pg_turbovec (sp=4) | 16 | 236 | 5.2 | 67.73 | 84.0 |
| co10m | pg_turbovec (sp=4) | 96 | 363 | 4.2 | 264.20 | 432.4 |
| co10m | pg_turbovec (sp=1) | 1 | 22 | 21.6 | 46.17 | 59.2 |
| co10m | pg_turbovec (sp=1) | 16 | 281 | 17.6 | 56.90 | 75.5 |
| co10m | pg_turbovec (sp=1) | 96 | 618 | 7.1 | 155.26 | 313.1 |
| co1m | pg_vectorsearch | 1 | 612 | 608.8 | 1.63 | 2.8 |
| co1m | pg_vectorsearch | 16 | 9808 | 627.1 | 1.63 | 2.8 |
| co1m | pg_vectorsearch | 96 | 39497 | 452.7 | 2.43 | 6.0 |
| co1m | pg_turbovec (sp=4) | 1 | 457 | 140.3 | 2.19 | 2.5 |
| co1m | pg_turbovec (sp=4) | 16 | 3473 | 72.5 | 4.61 | 8.9 |
| co1m | pg_turbovec (sp=4) | 96 | 3423 | 39.7 | 28.05 | 50.4 |
| co1m | pg_turbovec (sp=1) | 1 | 151 | 150.0 | 6.62 | 7.6 |
| co1m | pg_turbovec (sp=1) | 16 | 2133 | 136.0 | 7.50 | 7.5 |
| co1m | pg_turbovec (sp=1) | 96 | 3525 | 40.5 | 27.23 | 62.0 |
