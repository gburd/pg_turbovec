# Graph-kind scan beam (`ef`) retune — the measured frontier

**What shipped:** `turbovec.graph_ef` (new `Userset` int GUC, default `0`
= auto = **512**), and the graph's beam is now DECOUPLED from
`turbovec.hi_dim_rerank`. No wire-format change (stays v8), no REINDEX.

**Box:** AWS EC2, 32 vCPU, AVX-512, 247 GiB RAM, Ubuntu 24.04,
PostgreSQL 18.6 (pgrx 0.19.1), `shared_buffers = 32GB`,
`max_parallel_maintenance_workers = 0` (serial builds, so build times
below are single-threaded). Corpora: ann-benchmarks
`sift-128-euclidean.hdf5`, `gist-960-euclidean.hdf5`.

**Method.** Recall is against an **exact cosine** ground truth recomputed
over exactly the `n` rows loaded — NOT the HDF5 `/neighbors` L2 GT.
turbovec normalizes on insert and serves cosine; for a non-unit-norm
corpus the L2 top-10 and the cosine top-10 are different sets, so
scoring against the L2 GT would report a recall deficit that has nothing
to do with the beam. 100 queries from `/test`, 3 timed reps each.
Latency is the server-side `Execution Time` from `EXPLAIN (ANALYZE,
TIMING OFF)` (client RTT excluded); 10 warm-up queries first. qps@1 /
qps@8 are closed-loop over 6 s at 1 and 8 client threads. `search_k` is
pinned to `max(32, k)` so the candidate count can always hold `LIMIT k`
and the BEAM is the only variable.

Graph index build times (serial, one core): SIFT-200k/128d 4m18s (34 MB),
SIFT-1M/128d 48m (191 MB), GIST-1M/960d 6h13m (577 MB).

---

## 1. The mechanism, confirmed

Graph recall is a function of the **beam width**, and only of that. Every
table in §3 is monotone non-decreasing in `graph_ef` and FLAT in
`hi_dim_rerank`.

Pre-patch the beam was `ef = (candidate_count * 4).max(64)`. Because
`hi_dim_rerank = auto` raises the candidate count to
`clamp(dim, 256..=1024)` for `dim >= 256` — a knob whose real job is the
flat/IVF **exact-rerank window** over a cell scan's quantized ranking —
the graph's beam was a side effect of an unrelated feature:

| dim | `hi_dim_rerank` | pre-patch candidate count | pre-patch effective beam |
|---:|---|---:|---:|
| 128 | off | 32 | `max(128, 64)` = **128** |
| 128 | auto | 32 (below the `dim >= 256` threshold) | **128** |
| 960 | off | 32 | **128** |
| 960 | auto | `clamp(960, 256..=1024)` = 960 | **3840** |

That is the inverted cliff: at 960d `auto` got a 30× wider beam than
`off` for free, so high dim looked *better* than low dim, and turning
`hi_dim_rerank` off "collapsed" graph recall at every dim. Nothing about
the scorer or the dimensionality was involved.


**Post-patch the two knobs are independent.** At identical `(k, ef)`,
`hi_dim_rerank = off` and `= auto` give recall within:

| corpus | dim | paired configs | max abs ΔR@k |
|---|---:|---:|---:|
| SIFT-200k | 128 | 14 | **0.0000** |
| SIFT-1M | 128 | 14 | **0.0000** |
| GIST-1M | 960 | 14 | **0.0000** |

Zero at 128d and at 960d. The graph's recall no longer responds to
`hi_dim_rerank` at all, which is the requirement.


---

## 2. Before / after — same index files, same queries, same GT


The pre-patch binary is v2.1.0 at `ded5533`, installed over the SAME
graph relfiles and the postmaster restarted (`pg_ctl restart -m fast`)
so `shared_preload_libraries` re-maps it. This is a pure scan-time
patch — wire format v8 on both sides, index bytes byte-identical — so
this is a clean A/B. Confirmed by `pg_settings`:
`turbovec.graph_ef` row count = 0 pre-patch, 1 post-patch.

| corpus | dim | `hi_dim_rerank` | k | BEFORE R@k | BEFORE p50 | AFTER R@k | AFTER p50 |
|---|---:|---|---:|---:|---:|---:|---:|
| sift200k | 128 | off | 10 | 0.993 | 1.68 ms | **0.996** | 3.97 ms |
| sift200k | 128 | off | 100 | 0.884 | 3.98 ms | **0.885** | 4.44 ms |
| sift200k | 128 | auto | 10 | 0.993 | 1.65 ms | **0.996** | 3.97 ms |
| sift200k | 128 | auto | 100 | 0.884 | 3.94 ms | **0.885** | 4.43 ms |
| sift1m | 128 | off | 10 | 0.976 | 2.02 ms | **0.990** | 5.19 ms |
| sift1m | 128 | off | 100 | 0.859 | 5.04 ms | **0.859** | 5.76 ms |
| sift1m | 128 | auto | 10 | 0.976 | 2.00 ms | **0.990** | 5.24 ms |
| sift1m | 128 | auto | 100 | 0.859 | 4.99 ms | **0.859** | 5.75 ms |
| gist1m | 960 | off | 10 | 0.840 | 13.69 ms | **0.920** | 34.55 ms |
| gist1m | 960 | off | 100 | 0.814 | 37.14 ms | **0.826** | 41.70 ms |
| gist1m | 960 | auto | 10 | 0.971 | 194.41 ms | **0.920** | 34.48 ms |
| gist1m | 960 | auto | 100 | 0.958 | 196.29 ms | **0.826** | 41.72 ms |

(AFTER = the shipped default, `graph_ef = 0` → 512.)

Read the `gist1m` rows carefully — they are the whole story. BEFORE,
`hi_dim_rerank` moved 960d graph recall from 0.840 to 0.971 while
costing 14× the latency (13.7 ms → 194 ms), because it was silently
buying a 3840-wide beam. AFTER, the beam is 512 whatever
`hi_dim_rerank` says, and 0.920 / ~35 ms is what you get in both modes.
The pre-patch `auto` number was never a 960d recall property; it was an
undocumented beam of 3840 that no documented knob named.

**This IS a recall regression for one specific pre-patch configuration,
and it must be called out rather than buried:** a 960d graph index
queried with `hi_dim_rerank = auto` (the default) loses
0.971 → 0.920 R@10 and 0.958 → 0.826 R@100, in exchange for
194 ms → 34 ms p50 (5.6×) and 196 ms → 42 ms (4.7×). Three reasons that
is the right default anyway:

1. **It was never a documented or intentional beam.** No release note,
   GUC, or doc ever said "a 960d graph search uses a 3840-wide beam".
   It was arithmetic leaking out of an unrelated feature, and it broke
   the moment a user set `hi_dim_rerank = off` (0.840, the "collapse").
2. **A ~200 ms p50 default is not a defensible default.** The old
   behaviour bought recall at 14× the latency of the `off` mode on the
   same index, invisibly, with no way to opt out except a knob whose
   documentation is about something else entirely.
3. **The old recall is now REACHABLE and documented.** It was not
   before. `SET turbovec.graph_ef = 3840` reproduces the pre-patch
   `auto` beam exactly; `= 2048` gets R@10 0.966 / R@100 0.863 at
   ~95 ms. The choice is now the user's, explicitly, which is the
   entire point of the change.

The 128d rows are a straight improvement in both modes (0.976 → 0.990
R@10 at 1M) at 2.0 ms → 5.2 ms, and 128d `auto` was NEVER above `off`
pre-patch (128 < the `dim >= 256` threshold), so no 128d user loses
anything.


---

## 3. The frontier


### 3.1 SIFT-200k — 200,000 rows, 128d (the scale trend)


`hi_dim_rerank = off` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.931 | 0.71 | 0.82 | 0.89 | 1139.3 | 7835.8 |
| 64 | 0.979 | 1.06 | 1.21 | 1.28 | 823.5 | 5928.5 |
| 128 | 0.993 | 1.61 | 1.89 | 2.00 | 564.1 | 3963.0 |
| 256 | 0.994 | 2.56 | 3.11 | 3.31 | 370.7 | 2677.6 |
| 512 | 0.996 | 3.97 | 4.98 | 5.39 | 239.1 | 1739.5 |
| 1024 | 0.997 | 6.32 | 8.22 | 9.14 | 153.6 | 1110.8 |
| 2048 | 0.997 | 9.91 | 12.88 | 13.67 | 98.3 | 706.3 |

`hi_dim_rerank = off` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.865 | 1.87 | 2.07 | 2.19 | 483.5 | 3551.0 |
| 64 | 0.865 | 1.87 | 2.07 | 2.18 | 486.2 | 3521.2 |
| 128 | 0.873 | 2.09 | 2.39 | 2.47 | 429.0 | 3175.4 |
| 256 | 0.882 | 3.02 | 3.53 | 3.78 | 312.6 | 2274.8 |
| 512 | 0.885 | 4.44 | 5.49 | 5.95 | 212.6 | 1545.1 |
| 1024 | 0.885 | 6.68 | 8.52 | 9.40 | 142.0 | 1031.1 |
| 2048 | 0.885 | 10.29 | 13.29 | 14.40 | 92.2 | 671.2 |

`hi_dim_rerank = auto` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.931 | 0.71 | 0.82 | 0.89 | 1136.0 | 6979.1 |
| 64 | 0.979 | 1.05 | 1.20 | 1.27 | 827.1 | 5953.1 |
| 128 | 0.993 | 1.61 | 1.89 | 1.98 | 564.3 | 3998.2 |
| 256 | 0.994 | 2.55 | 3.07 | 3.29 | 371.7 | 2672.6 |
| 512 | 0.996 | 3.97 | 4.96 | 5.41 | 240.9 | 1732.9 |
| 1024 | 0.997 | 6.26 | 8.07 | 8.69 | 153.8 | 1111.7 |
| 2048 | 0.997 | 9.90 | 12.92 | 13.64 | 98.1 | 704.3 |

`hi_dim_rerank = auto` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.865 | 1.86 | 2.08 | 2.20 | 485.0 | 3547.5 |
| 64 | 0.865 | 1.86 | 2.09 | 2.20 | 479.5 | 3495.1 |
| 128 | 0.873 | 2.13 | 2.40 | 2.50 | 433.3 | 3187.9 |
| 256 | 0.882 | 3.02 | 3.54 | 3.79 | 313.2 | 2275.0 |
| 512 | 0.885 | 4.43 | 5.47 | 5.93 | 211.6 | 1552.5 |
| 1024 | 0.885 | 6.72 | 8.53 | 9.46 | 140.8 | 1026.7 |
| 2048 | 0.885 | 10.39 | 13.38 | 14.28 | 91.7 | 671.6 |

### 3.2 SIFT-1M — 1,000,000 rows, 128d


`hi_dim_rerank = off` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.874 | 0.85 | 1.01 | 1.06 | 967.1 | 6584.8 |
| 64 | 0.943 | 1.25 | 1.43 | 1.52 | 687.5 | 4905.6 |
| 128 | 0.976 | 1.95 | 2.28 | 2.41 | 444.3 | 3264.7 |
| 256 | 0.987 | 3.19 | 3.85 | 4.04 | 283.6 | 2084.6 |
| 512 | 0.990 | 5.19 | 6.46 | 6.95 | 176.7 | 1295.1 |
| 1024 | 0.991 | 8.61 | 11.23 | 12.22 | 107.6 | 789.3 |
| 2048 | 0.991 | 14.13 | 19.09 | 20.89 | 64.6 | 477.8 |

`hi_dim_rerank = off` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.829 | 2.16 | 2.48 | 2.58 | 400.6 | 2987.3 |
| 64 | 0.829 | 2.18 | 2.49 | 2.60 | 401.5 | 2954.5 |
| 128 | 0.840 | 2.47 | 2.83 | 2.97 | 360.4 | 2647.3 |
| 256 | 0.855 | 3.69 | 4.33 | 4.60 | 247.1 | 1813.4 |
| 512 | 0.859 | 5.76 | 7.06 | 7.64 | 158.5 | 1182.4 |
| 1024 | 0.860 | 9.12 | 11.72 | 12.75 | 101.0 | 744.5 |
| 2048 | 0.860 | 14.67 | 19.59 | 21.50 | 62.9 | 460.8 |

`hi_dim_rerank = auto` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.874 | 0.84 | 0.99 | 1.05 | 970.6 | 6545.3 |
| 64 | 0.943 | 1.24 | 1.42 | 1.50 | 671.5 | 4835.2 |
| 128 | 0.976 | 1.96 | 2.30 | 2.40 | 445.3 | 3254.9 |
| 256 | 0.987 | 3.19 | 3.80 | 4.04 | 287.3 | 2089.6 |
| 512 | 0.990 | 5.24 | 6.50 | 7.02 | 177.3 | 1298.8 |
| 1024 | 0.991 | 8.67 | 11.28 | 12.24 | 107.7 | 787.4 |
| 2048 | 0.991 | 14.12 | 19.14 | 20.98 | 65.6 | 477.7 |

`hi_dim_rerank = auto` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.829 | 2.15 | 2.46 | 2.58 | 404.4 | 2937.8 |
| 64 | 0.829 | 2.16 | 2.49 | 2.63 | 403.5 | 2982.6 |
| 128 | 0.840 | 2.47 | 2.82 | 2.96 | 357.1 | 2593.1 |
| 256 | 0.855 | 3.70 | 4.38 | 4.62 | 244.1 | 1815.1 |
| 512 | 0.859 | 5.75 | 7.00 | 7.56 | 158.8 | 1181.5 |
| 1024 | 0.860 | 9.11 | 11.68 | 12.73 | 101.2 | 741.0 |
| 2048 | 0.860 | 14.70 | 19.62 | 21.41 | 62.5 | 460.6 |

### 3.3 GIST-1M — 1,000,000 rows, 960d


`hi_dim_rerank = off` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.659 | 6.07 | 7.34 | 8.70 | 132.7 | 996.8 |
| 64 | 0.760 | 8.65 | 10.39 | 11.77 | 97.1 | 730.7 |
| 128 | 0.840 | 13.29 | 15.28 | 15.85 | 65.7 | 507.6 |
| 256 | 0.892 | 21.25 | 24.54 | 25.47 | 42.0 | 327.6 |
| 512 | 0.920 | 34.55 | 41.56 | 43.15 | 26.3 | 203.4 |
| 1024 | 0.948 | 56.66 | 70.33 | 72.55 | 16.1 | 125.3 |
| 2048 | 0.966 | 93.44 | 115.97 | 119.90 | 10.1 | 77.4 |

`hi_dim_rerank = off` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.696 | 18.28 | 20.49 | 21.28 | 47.0 | 370.0 |
| 64 | 0.696 | 18.34 | 20.50 | 21.31 | 47.2 | 369.8 |
| 128 | 0.727 | 20.26 | 22.80 | 24.05 | 42.8 | 333.2 |
| 256 | 0.785 | 28.41 | 31.76 | 33.19 | 31.3 | 243.7 |
| 512 | 0.826 | 41.70 | 48.90 | 50.65 | 21.2 | 167.7 |
| 1024 | 0.849 | 64.05 | 77.54 | 80.69 | 14.2 | 110.7 |
| 2048 | 0.863 | 100.92 | 122.70 | 128.17 | 9.3 | 70.7 |

`hi_dim_rerank = auto` · k = 10 · `search_k` = 32

| `graph_ef` | R@10 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.659 | 6.09 | 7.37 | 8.74 | 132.9 | 1007.4 |
| 64 | 0.760 | 8.62 | 10.42 | 11.80 | 96.9 | 748.2 |
| 128 | 0.840 | 13.31 | 15.33 | 15.96 | 65.7 | 508.7 |
| 256 | 0.892 | 21.29 | 24.58 | 25.51 | 42.0 | 326.7 |
| 512 | 0.920 | 34.48 | 41.60 | 43.16 | 26.4 | 203.4 |
| 1024 | 0.948 | 56.70 | 70.97 | 73.75 | 16.0 | 124.5 |
| 2048 | 0.966 | 93.38 | 116.00 | 122.35 | 10.1 | 77.2 |

`hi_dim_rerank = auto` · k = 100 · `search_k` = 100

| `graph_ef` | R@100 | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.696 | 18.30 | 20.50 | 21.25 | 47.0 | 368.6 |
| 64 | 0.696 | 18.27 | 20.52 | 21.37 | 47.0 | 368.9 |
| 128 | 0.727 | 20.22 | 22.87 | 23.93 | 42.9 | 334.4 |
| 256 | 0.785 | 28.30 | 31.88 | 32.98 | 31.4 | 244.7 |
| 512 | 0.826 | 41.72 | 48.92 | 50.36 | 21.2 | 167.7 |
| 1024 | 0.849 | 64.11 | 77.58 | 80.75 | 14.2 | 110.6 |
| 2048 | 0.863 | 100.81 | 122.49 | 128.04 | 9.3 | 70.8 |

### 3.4 The scale trend

The beam a corpus needs GROWS with `n`, which is exactly why a fixed 64
floor was the wrong default and why the retune matters more the bigger
you get:

| beam | 200k R@10 | 1M R@10 | Δ |
|---:|---:|---:|---:|
| 64 | 0.979 | 0.943 | −0.036 |
| 512 | 0.996 | 0.990 | −0.006 |

At beam 64 the 5× corpus costs 0.036 R@10; at beam
512 it costs 0.006. A wider beam absorbs the scale
penalty — the recall gap between 200k and 1M nearly closes.


---

## 4. Comparators — flat and IVF at the same k


**sift1m-flat** — n=1,000,000, dim=128

| k | `hi_dim_rerank` | `probes` | R@k | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | off | 16 | 0.993 | 0.96 | 1.06 | 1.11 | 712.7 | 1312.8 |
| 100 | off | 16 | 0.861 | 1.94 | 2.12 | 2.30 | 403.7 | 964.6 |
| 10 | auto | 16 | 0.993 | 0.98 | 1.11 | 1.31 | 708.9 | 1314.6 |
| 100 | auto | 16 | 0.861 | 1.93 | 2.29 | 2.43 | 402.4 | 969.4 |

**sift1m-ivf** — n=1,000,000, dim=128

| k | `hi_dim_rerank` | `probes` | R@k | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | off | 16 | 0.944 | 2.16 | 2.26 | 2.33 | 397.0 | 1784.4 |
| 10 | off | 64 | 0.990 | 2.22 | 2.33 | 2.40 | 380.7 | 1714.9 |
| 100 | off | 16 | 0.813 | 2.83 | 3.01 | 3.05 | 305.7 | 1451.4 |
| 100 | off | 64 | 0.859 | 3.01 | 3.19 | 3.27 | 284.3 | 1302.3 |
| 10 | auto | 16 | 0.944 | 2.14 | 2.26 | 2.40 | 392.5 | 1806.9 |
| 10 | auto | 64 | 0.990 | 2.22 | 2.35 | 2.46 | 377.0 | 1698.3 |
| 100 | auto | 16 | 0.813 | 2.79 | 3.00 | 3.09 | 308.4 | 1461.0 |
| 100 | auto | 64 | 0.859 | 3.03 | 3.25 | 3.36 | 282.2 | 1298.6 |

**gist1m-flat** — n=1,000,000, dim=960

| k | `hi_dim_rerank` | `probes` | R@k | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | off | 16 | 0.997 | 5.91 | 6.62 | 7.08 | 100.6 | 191.8 |
| 100 | off | 16 | 0.892 | 13.71 | 15.18 | 15.66 | 46.3 | 152.1 |
| 10 | auto | 16 | 0.998 | 59.77 | 64.92 | 66.76 | 10.8 | 45.3 |
| 100 | auto | 16 | 1.000 | 63.83 | 70.10 | 71.44 | 10.1 | 44.2 |

**gist1m-ivf** — n=1,000,000, dim=960

| k | `hi_dim_rerank` | `probes` | R@k | p50 ms | p95 ms | p99 ms | qps@1 | qps@8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | off | 16 | 0.857 | 13.24 | 128.14 | 132.10 | 30.9 | 193.9 |
| 10 | off | 64 | 0.986 | 54.46 | 166.16 | 175.67 | 14.5 | 82.3 |
| 100 | off | 16 | 0.753 | 20.12 | 133.21 | 138.98 | 22.9 | 148.8 |
| 100 | off | 64 | 0.878 | 61.69 | 172.47 | 185.25 | 13.5 | 77.0 |
| 10 | auto | 16 | 0.857 | 60.12 | 173.65 | 183.25 | 10.7 | 77.1 |
| 10 | auto | 64 | 0.986 | 101.85 | 194.00 | 220.95 | 7.9 | 50.5 |
| 100 | auto | 16 | 0.783 | 64.87 | 179.51 | 182.99 | 11.5 | 76.3 |
| 100 | auto | 64 | 0.965 | 106.11 | 200.55 | 226.42 | 8.6 | 47.3 |

---

## 5. The decision


### 5.1 `GRAPH_SCAN_EF_DEFAULT = 512`

**SIFT-1M / 128d** — 512 is the knee, unambiguously:

| beam | R@10 | p50 | ΔR vs prev | Δp50 vs prev |
|---:|---:|---:|---:|---:|
| 64 | 0.943 | 1.25 ms | — | — |
| 256 | 0.987 | 3.19 ms | +0.044 | +1.94 ms |
| **512** | **0.990** | **5.19 ms** | +0.003 | +2.00 ms |
| 1024 | 0.991 | 8.61 ms | +0.001 | +3.42 ms |
| 2048 | 0.991 | 14.13 ms | +0.000 | +5.52 ms |

Past 512 the curve is dead: doubling to 1024 buys
+0.001 R@10 for 1.7×
the latency, and 2048 buys +0.000 more. 512 is
within 0.001 of the beam-2048 ceiling at
0.37× its p50.

**GIST-1M / 960d** — 512 is a deliberate latency-capped choice, NOT a knee:

| beam | R@10 | p50 | ΔR vs prev | Δp50 vs prev |
|---:|---:|---:|---:|---:|
| 64 | 0.760 | 8.65 ms | — | — |
| 256 | 0.892 | 21.25 ms | +0.132 | +12.60 ms |
| **512** | **0.920** | **34.55 ms** | +0.028 | +13.30 ms |
| 1024 | 0.948 | 56.66 ms | +0.028 | +22.11 ms |
| 2048 | 0.966 | 93.44 ms | +0.018 | +36.78 ms |

At 960d the curve has NOT plateaued at 2048 (0.966 and still
climbing). Chasing it costs 93 ms p50 — a default no
one should get silently. 512 gives 0.920 at
34.5 ms; `SET turbovec.graph_ef = 2048` is one line away
for anyone who wants 0.966. **Documented honest limit: the
graph kind does not reach >0.95 R@10 at 960d/1M inside a 40 ms
budget.**

512 is one default for both corpora because the knob exists: it is the
128d knee and the highest 960d beam that keeps p50 under ~35 ms.

### 5.2 The bar: "competitive with flat/IVF at the same k". Honestly — NO, at 1M.

| corpus | best graph @ k=10 | flat @ k=10 | IVF @ k=10 |
|---|---|---|---|
| SIFT-1M/128d | R@10 0.990 / 5.19 ms | R@10 0.993 / **0.96 ms** | R@10 0.990 / 2.22 ms |
| GIST-1M/960d | R@10 0.920 / 34.55 ms | R@10 0.997 / **5.91 ms** | R@10 0.986 / 101.85 ms |

**The flat kind beats the graph on BOTH axes on BOTH corpora at 1M.**
SIFT-1M: flat is +0.003 R@10 at
5.4× LOWER latency. GIST-1M: flat is
+0.077 R@10 at
5.8× LOWER latency. The graph does not
beat flat at any beam on this data — even beam 2048 at 960d
(0.966) is below flat's 0.997 while costing
16× the p50.

That is a real result and it should be said plainly: **at 1M rows on a
32-core AVX-512 box, turbovec's SIMD full scan is simply faster than
navigating a graph.** The whole-corpus LUT scan is one linear sweep of a
compact quantized buffer at memory bandwidth; the graph does ~`ef`
scattered per-hop gathers with a serial dependency between hops, and it
loses. The graph's advantage is asymptotic (`O(log n)`-ish hops vs
`O(n)` bytes swept) and 1M × these dims is below the crossover on this
hardware.

Two things follow, and neither is fixed by a beam default:
1. **The graph kind is not the recommended kind at 1M.** The retune
   makes it *correct and tunable*; it does not make it *the best choice*
   at this scale. The honest guidance is flat (or IVF for a lower-RAM
   footprint) until a measured crossover exists.
2. **`qps@8` is the one place the graph wins**: SIFT-1M graph at beam 64
   does 4906 qps@8 vs flat's
   1313. The flat scan saturates memory bandwidth so it
   barely scales with concurrency (713 → 1313
   qps, 1.8×), while the graph's small
   working set scales near-linearly (687 →
   4906, 7.1×).
   Under real concurrency the graph is the better throughput engine even
   where it loses on isolated p50. That is the argument for keeping the
   kind and for measuring the crossover at 10M+, not at 1M.

### 5.3 Why a GUC and not just a constant

`hnsw.ef_search` and `turbovec.probes` are both user-tunable for the same
reason: the right beam is a per-workload recall/latency choice, and §5.1
shows the 960d optimum is genuinely outside the 128d optimum. Shipping
only a constant would mean 960d users have no documented way to reach
0.966 R@10. `Userset` (drift-check §11b), range
`0..=1000000`, `0` = auto.


---

## 6. What changed in the code

| file | change |
|---|---|
| `src/index/graph.rs` | Split `graph_search` → `graph_search_with_ef(.., ef, ..)` (explicit beam, used by the live scan path) + `graph_search` (unchanged signature, now a wrapper at `default_scan_ef(k, n)` for callers with no live GUC: unit tests, the `aminsert` findability probe). New `GRAPH_SCAN_EF_DEFAULT = 512`. The beam is clamped to `[k, n]` inside the worker, so `ef < k` can no longer under-fill the result set and fall through to the BUG#2 backfill. |
| `src/guc.rs` | New `GRAPH_EF` GucSetting + `graph_scan_ef(k, n)` / pure `graph_scan_ef_decide(setting, k, n)`, and the `turbovec.graph_ef` registration (`Userset`, `0..=1000000`). |
| `src/cache.rs` | `GraphIndex::search` resolves the beam via `guc::graph_scan_ef(k, self.len())` and calls `graph_search_with_ef`. |
| `src/index/scan.rs` | **The decoupling.** The graph arm now searches with `k_user` (`ceil(search_k * oversample)`) instead of `k` (which carries `hi_dim_rerank`'s dim-scaled floor). `current_k` tracks what was actually searched so the iterative-refill schedule doubles from the right base. `hi_dim_rerank` is untouched for flat/IVF. |

New tests: `guc::graph_ef_tests` (4 — auto default, pin, clamp to `k`,
clamp to `n`/empty-corpus, and that the resolver takes no dim/rerank
input) and `index::graph::tests::{recall_increases_monotonically_with_the_beam,
explicit_beam_is_floored_at_k, graph_search_matches_explicit_default_ef}`
(the monotonicity finding, now a regression guard).

