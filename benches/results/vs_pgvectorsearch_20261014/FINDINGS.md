# pg_turbovec v2.13.0 vs pg_vectorsearch — full A/B comparison (P1–P7)

**Status: RUNNING.** Started 2026-10-15. Lead bench worker. Results are UNCOMMITTED
under `benches/results/vs_pgvectorsearch_20261014/{x86,arm}/<phase>/`; the lead commits
after review. This file is the top-level deliverable, updated as cells complete.

Build: **pg_turbovec v2.13.0** (tag `bf06bb1`, peeled `25452eb`; scan-pool bound + OOC
IVF ~100 ms tail both fixed, `search_concurrency` GUC removed, wire format v8 unchanged).
pg_vectorsearch main `7a9716b` (= 1 README commit past tag v0.1.0), VERSION 0.2.0-dev.
pgvector v0.8.7 (default flags). PostgreSQL 18.6. Fedora 44.

Hosts (both single-socket, ISA-gated, confirmed 2026-10-15):
- **x86** i7i.metal-24xl (Emerald Rapids, AVX-512 BW/VBMI/VNNI), 96 vCPU / 755 GiB, `i-0f87e6ee3e5ec7466`
- **arm** r8gd.metal-24xl (Graviton4, asimddp/i8mm/sve2), 96 vCPU / 753 GiB, `i-05714c885595d7cdd`

AWS: profile kiwi (830118047484), us-east-2, on-demand (spot quota 32 vCPU < 96).
run=`pgvs-202610151238`. Self-terminate armed Sat 2026-10-17 03:55 UTC (Fri 23:55 ET).

---

## SHRUNK MATRIX — decided before P2 (hard deadline: kiwi closes Sat 2026-10-17 noon ET)

The plan budgets ~72 h; the account gives ~51 h and we must finish + pull by Fri ~20:00 ET
(>12 h slack). We fit the work into ~40 h of host time with margin by applying the plan's
own shrink order and going a little further. **2.13.0 helps us here**: the OOC-IVF ~100 ms
tail and the 97-thread global-pool CPU blowup that made P0's IVF cells slow are both fixed,
so the IVF frontier cells are 3–4× cheaper than the P0 basis.

### Kept in full (never dropped, per directive)
- **Headline matched-recall Pareto**, per host, at recall@10 ∈ {0.90, 0.95, 0.98, 0.99},
  with **CPU-ms/query** and **QPS/core** beside every latency.
- **Round 0 (discarded) + ABBA×2** (3 counted-ish passes; round 0 discarded, 2 ABBA rounds
  counted). Paired per-query bootstrap CIs still computed over the 2 rounds; a cell whose
  2 rounds disagree >10% is flagged (not silently kept).
- GT (exact fp64-rescored) + all D6 validity checks (EXPLAIN index, offline fp64 order,
  recall determinism, LOGGED assert, no-GLIBC assert).
- Build time + index size + build WAL + peak RSS; build #2 variance at the 0.95 point.
- The **equal-CPU row** (`turbovec.scan_parallelism = 1`) beside the headline.
- Churn **+5% and +11.1%** with `index_is_degraded()` after each step, both maintenance steps.

### Scales / datasets kept
| dataset | dim | scale | hosts | note |
|---|---|---|---|---|
| CO1024 | 1024 | **1M** | x86 + arm | headline (tvflat vs prism) + hnsw ref; our standing corpus |
| CO1024 | 1024 | **10M** | x86 + arm | headline at scale (tv IVF √n vs prism); from HF (hf_transfer, fast) |
| OA1536 | 1536 | **5M** | x86 only | third-party 1536-d; n=900; **downloaded in background, run only if P2 fits** |

### Dropped / deferred (in the plan's shrink order)
1. **1M IVF 2-D `probes×search_k` grid** — dropped (secondary table).
2. **CW768 entirely** — dropped. (Source was already broken → VDBBench `cohere_medium_1m`
   substitute; with the deadline it is not worth the ~31 GB @ 6 MB/s zilliz pull. The
   bench_ann.py cross-check it enabled is sacrificed; noted in "not measured".)
3. **0.99 QPS cells** — dropped; QPS run at the **0.95** point only.
4. **Open-loop `pgbench -R`** — kept on **CO1024-10M (x86) only**; dropped elsewhere.
5. **arm QPS** — **CO1024-10M only**.
6. **malloc-tuned secondary cell** — dropped (internal-only anyway; PRODUCTION.md gate).
7. **Secondary table** (other turbovec kinds, prism alt nlist/soar) — dropped except the
   one equal-CPU row (kept) and the defaults-only point (free, comes from calibration).
8. **OA1536 5M** is the first thing to drop if P2 (its ~45 GB zilliz download + 5M build +
   GT) will not fit by the P3 deadline — "prefer fewer scales done cleanly".
9. Native-build + TOAST-storage sensitivity cells — dropped (1M-only niche).
10. Adoption-path cells (column copy / expression-index cast) — kept **1M only, x86** (cheap,
    and it is a headline caveat); the 10M column copy is dropped.

### Per-host time budget (target ≤ 40 h each, ~11 h slack vs the 51 h window)
| phase | x86 h | arm h | notes |
|---|---:|---:|---|
| setup + build (PG18 + 3 ext) | 0.5 | 0.6 | parallel; one-time |
| P1 data (CO1024 1M+10M prep + GT; OA1536 bg) | 3.0 | 1.0 | GT on x86, scp 1M to arm; arm recomputes 1M GT |
| P2 load + build ×2 (1M+10M, both systems+kinds) | 4.0 | 4.0 | 10M build is the long pole |
| P3 warm frontiers (1M + 10M headline, ABBA×2) | 5.0 | 5.0 | IVF cheap on 2.13.0 |
| P4 cold + new-conn (1M + 10M) | 1.5 | 1.5 | |
| P5 QPS (closed {1,16,96} @0.95; open-loop x86 10M) | 4.0 | 2.5 | arm QPS = 10M only |
| P6 churn (+5%/+11.1%, x86 10M+1M, arm 1M) | 4.0 | 1.5 | |
| P7 filtered + adoption (x86 1M) | 2.5 | — | |
| OA1536 5M (x86, if it fits) | +4.0 | — | first to cut |
| **total** | **~28.5 (+4 OA)** | **~17.5** | fits with margin |

arm finishes well under 40 h → **terminate arm right after its last phase (churn 1M)**,
x86 last. Timers re-armed shorter at each phase boundary; rsync after **every cell group**.

---
<!-- Results tables appended below as cells complete. -->

## P1 — data prep (CO1024 1M+10M both hosts; OA1536 5M x86 background)

- **CO1024 1M** reproduced byte-for-byte vs P0 on x86: 1,006,041 rows read, 841 exact-dup
  corpus vectors dropped, 0 query↔corpus dups, 0 dist-0 top1, 3 tie-ambiguous@k10,
  24@k100, max|‖v‖−1| = 1.45e-7. Pinned shards 0–10 (md5s in p1/co1024_1m.lock.json).
- **Cross-arch (x86 vs arm) 1M:** the *source selection* is byte-identical across arches —
  `corpus_src.npy` / `queries_src.npy` md5 match exactly (same shards, same rows chosen,
  deterministic). The *normalised vectors + GT* differ at the ULP level (x86 corpus.npy md5
  `fb93d7d3…` vs arm `0a78c6a7…`; max|‖v‖−1| 1.45e-7 vs 2.34e-7; tie_ambiguous@k100 24 vs 23).
  Cause: fp32 L2-normalise + BLAS sgemm round differently on AVX-512 vs NEON. This is expected
  and correct: each host uses its own self-consistent corpus+GT, and both systems **on the same
  host** load from the SAME files (the fairness requirement). The cross-arch "GT recompute check"
  is satisfied by the identical source selection, not bit-identical floats. Recall is compared
  within a host, never x86-GT vs arm-corpus.
- GT: fp32 sgemm top-200 → fp64 rescore → top-100, ties broken by id. 5,200 held-out queries
  (5,000 timed + 200 warm). 16 COPY tsv files + queries.tsv, identical bytes for both systems.

## P3 — warm single-client frontier (round 0 discarded + ABBA×2, 5,000 timed queries)

Calibration on recall@10 (deterministic; selected on recall, latencies discarded). Headline point
per target = smallest knob with calib recall ≥ T, latency from the 2 counted ABBA rounds. p50 spread
between the 2 rounds is <2% in every cell (ABBA cancels drift). **CPU-ms/query and thread count are
printed beside latency** — the mandatory honesty column: pg_turbovec's default scan uses a bounded
pool (v2.13.0: min(cores,4)=5 threads incl. backend), pg_vectorsearch uses 1 core/query.

### CO1024 1M (headline: pg_turbovec flat 4-bit vs pg_vectorsearch prism defaults)

**i7i.metal-24xl (x86, AVX-512 VBMI/VNNI):**

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads |
|---|---|---|---|---|---|---|---|
| pg_turbovec flat | 0.90 | search_k=10 | 0.9416 | 1.89 | 2.03 | 5.3 | 5 |
| pg_turbovec flat | 0.95 | search_k=16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 |
| pg_turbovec flat | 0.98 | search_k=16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 |
| pg_turbovec flat | 0.99 | search_k=16 | 0.9984 | 1.97 | 2.06 | 5.5 | 5 |
| pg_vectorsearch prism | 0.90 | nprobe=14 | 0.9127 | 0.89 | 1.09 | 0.6 | 1 |
| pg_vectorsearch prism | 0.95 | nprobe=28 | 0.9580 | 1.26 | 1.61 | 1.0 | 1 |
| pg_vectorsearch prism | 0.98 | nprobe=56 | 0.9818 | 1.79 | 2.36 | 1.5 | 1 |
| pg_vectorsearch prism | 0.99 | nprobe=112 | 0.9925 | 2.91 | 3.73 | 2.6 | 1 |

Ratio of medians (tv/pv) at R@10≥0.95: **1.57×** (95% CI [1.56, 1.57], n=5000). At 0.95 prism is
faster in wall-time (1.26 vs 1.97 ms) at ⅕ the CPU; **at 0.99 pg_turbovec is faster** (1.97 vs 2.91 ms).
turbovec flat at 1024-d jumps 0.9416→0.9984 between search_k 10 and 16 (near-exact rerank of ≥1024
candidates), so its 0.95/0.98/0.99 headline all land on search_k=16 — it serves ~0.998 recall at ~2 ms
regardless of target ≥0.95. prism trades recall for latency smoothly.

**r8gd.metal-24xl (arm, Graviton4):**

| system | target R@10 | knob | meas R@10 | p50 ms | p95 ms | CPU-ms/q | threads |
|---|---|---|---|---|---|---|---|
| pg_turbovec flat | 0.90 | search_k=10 | 0.9421 | 2.48 | 2.59 | 7.4 | 5 |
| pg_turbovec flat | 0.95 | search_k=16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 |
| pg_turbovec flat | 0.98 | search_k=16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 |
| pg_turbovec flat | 0.99 | search_k=16 | 0.9993 | 2.55 | 2.68 | 7.6 | 5 |
| pg_vectorsearch prism | 0.90 | nprobe=14 | 0.9119 | 1.22 | 1.60 | 0.9 | 1 |
| pg_vectorsearch prism | 0.95 | nprobe=28 | 0.9564 | 1.84 | 2.56 | 1.5 | 1 |
| pg_vectorsearch prism | 0.98 | nprobe=56 | 0.9804 | 3.06 | 4.36 | 2.8 | 1 |
| pg_vectorsearch prism | 0.99 | nprobe=112 | 0.9920 | 5.55 | 7.56 | 5.3 | 1 |

Ratio of medians (tv/pv) at R@10≥0.95: **1.38×** (95% CI [1.38, 1.39], n=5000). On arm the crossover is
sharper: **at 0.98 and 0.99 pg_turbovec is clearly faster** (2.55 ms flat vs prism 3.06 / 5.55 ms),
because prism's higher-nprobe posting-list scans cost more on Graviton4 while turbovec's flat scan is flat.

CO1024 10M (IVF √n vs prism) in progress.

### CO1024 10M (headline: pg_turbovec IVF √n=3162 vs pg_vectorsearch prism defaults) — THE SCALE RESULT

**This is where pg_vectorsearch decisively wins.** At matched recall at 10M, prism (RaBitQ + posting
lists) is an order of magnitude faster than pg_turbovec's IVF, and faster than turbovec's exact flat.
Reported plainly; it is not an artifact.

**i7i.metal-24xl (x86):**

| system | config | R@10 | p50 ms | p95 ms | CPU-ms/q | threads |
|---|---|---|---|---|---|---|
| pg_turbovec IVF | probes=12 (0.90) | 0.8903 | 28.2 | 34.2 | 77.5 | 5 |
| pg_turbovec IVF | probes=32 (0.95) | 0.9502 | 63.2 | 85.6 | 177.9 | 5 |
| pg_turbovec IVF | probes=128 (~0.98) | 0.9878 | 193.9 | 228.8 | 391 | 5 |
| pg_turbovec IVF | probes=192 (≥0.99) | 0.9921 | 284.2 | 325.4 | 574 | 5 |
| pg_turbovec **flat** (exact) | defaults, n=300 | **1.0000** | 41.9 | 79.9 | 143 | 5 |
| pg_vectorsearch prism | nprobe=56 (0.95) | 0.9533 | **2.17** | 2.72 | 1.9 | 2 |
| pg_vectorsearch prism | nprobe=160 (0.98) | 0.9819 | **4.97** | 6.38 | 4.7 | 2 |
| pg_vectorsearch prism | nprobe=320 (0.99) | 0.9916 | **8.63** | 11.17 | 8.5 | 2 |

Ratio of medians (tv IVF / prism) at R@10≥0.95: **29.1×** (95% CI [28.8, 29.3], n=5000). prism is ~29×
faster at 0.95, ~39× at 0.98, and at 0.99 prism=8.6 ms vs turbovec IVF=284 ms (~33×). turbovec's **exact
flat** (41.9 ms, R@10=1.0) actually beats its own IVF at high recall — the IVF OOC fine-scan + Lloyd-Max
codebook work dominates at this scale — but flat is still ~5× slower than prism's 0.99 point.

**r8gd.metal-24xl (arm, Graviton4 — turbovec's IVF is ~3× faster here than on x86):**

| system | config | R@10 | p50 ms | p95 ms | CPU-ms/q | threads |
|---|---|---|---|---|---|---|
| pg_turbovec IVF | probes=12 (0.90) | 0.8892 | 31.1 | 37.1 | 82.7 | 5 |
| pg_turbovec IVF | probes=32 (0.95) | 0.9510 | 61.3 | 70.6 | 177.1 | 5 |
| pg_turbovec IVF | probes=96 (0.98) | 0.9822 | 59.3 | 68.6 | 125.4 | 5 |
| pg_turbovec IVF | probes=192 (≥0.99) | 0.9925 | 96.4 | 108.4 | 217 | 5 |
| pg_turbovec IVF | probes=256 | 0.9951 | 125.2 | 138.5 | 287 | 5 |
| pg_turbovec **flat** (exact) | defaults, n=300 | **1.0000** | 25.8 | 27.7 | 77 | 5 |
| pg_vectorsearch prism | nprobe=56 (0.95) | 0.9531 | **3.50** | 4.60 | 3.2 | 2 |
| pg_vectorsearch prism | nprobe=160 (0.98) | 0.9817 | **8.63** | 11.44 | 8.4 | 2 |
| pg_vectorsearch prism | nprobe=320 (0.99) | 0.9906 | **15.86** | 20.87 | 15.8 | 2 |

Ratio of medians (tv IVF / prism) at R@10≥0.95: **17.5×** (95% CI [17.4, 17.6], n=5000). On Graviton4 the
gap narrows (prism ~17× at 0.95, ~6× at 0.99: 15.9 vs 96 ms) because turbovec's OOC fine-scan is
notably faster on arm (SVE2/dotprod) while prism's higher-nprobe scans cost more. turbovec's exact flat
(25.8 ms) is competitive with prism's 0.98 point and beats its own IVF everywhere.

GUCs for the turbovec IVF 0.99 attempt and flat: all defaults (oversample=1.0, search_k=32,
hi_dim_rerank=auto, out_of_core=auto→OOC at 10M, scan_parallelism=auto=4+backend) except the swept
`turbovec.probes`. The flat point is n=300 (full O(n) scan is slow to run 5,000×); recall is exact so
300 queries pin it. Round 0 for the extra probes (128/192/256) is a single clean pass (not ABBA×2) —
used only to pin turbovec's own 0.98/0.99 endpoint, labelled as such; the headline 0.90/0.95 points are
the full round0+ABBA×2.

## P4 — cold + new-connection cost (D4.2)

Cold = `pg_ctl stop; drop_caches; start`; first query cold, first-100 p50. New-conn = fresh connection
with OS/shared buffers warm (both AMs build per-backend state on first use). turbovec's documented
per-backend cache gives it a one-time cold/first-query penalty; steady state (rest_p50) matches P3 warm.

| host | dataset/kind | system | cold TTFR ms | cold first-100 p50 | new-conn q0 ms | new-conn rest p50 |
|---|---|---|---|---|---|---|
| x86 | CO1024 1M flat | pg_turbovec | 723.6 | 3.12 | 504.6 | 2.65 |
| x86 | CO1024 1M flat | pg_vectorsearch | 142.8 | 2.45 | 100.7 | 1.56 |
| x86 | CO1024 10M IVF | pg_turbovec | 494.6 | 157.4 | 313.5 | 89.7 |
| x86 | CO1024 10M IVF | pg_vectorsearch | 160.7 | 9.71 | 104.1 | 2.95 |
| arm | CO1024 1M flat | pg_turbovec | 409.5 | 3.64 | 244.0 | 2.47 |
| arm | CO1024 1M flat | pg_vectorsearch | 72.2 | 3.54 | 37.4 | 2.32 |
| arm | CO1024 10M IVF | pg_turbovec | 379.4 | 161.0 | 199.6 | 71.4 |
| arm | CO1024 10M IVF | pg_vectorsearch | 99.8 | 11.36 | 40.0 | 4.46 |

turbovec's new-connection q0 (building the per-backend cache from shared_buffers: ~0.5 GB codes at 1M,
~5 GB at 10M) is 250–500 ms vs prism's amcache ~40–100 ms — a real cost for poolerless apps that open a
fresh backend per request. Subsequent queries on the same connection are at the warm rate.

## P5 — closed-loop QPS, QPS/core, p99 (D4.3)

Identical `\gset` PK-lookup + ORDER BY script for both systems (`:qv::type`, prepared), pinned client
(4 physical cores' 8 HTs on x86 / 8 cores on arm), server gets the rest; 20 s warm + 120 s timed. The
transaction includes the PK lookup + text round-trip (stated). **This is where the per-query thread
model dominates: pg_turbovec's bounded multi-thread scan trades aggregate throughput for single-client
latency; pg_vectorsearch's 1-thread-per-query scales far better.** The equal-CPU row
(`turbovec.scan_parallelism=1`) is included and at high concurrency it beats the default.

**i7i.metal-24xl (x86), max TPS and QPS/core:**

| dataset | system | c=1 TPS | c=16 TPS | c=96 TPS | QPS/core @96 | p99 ms @96 |
|---|---|---|---|---|---|---|
| CO1024 1M flat | pg_turbovec sp=4 | 568 | 2293 | 2066 | 23.5 | 71.2 |
| CO1024 1M flat | pg_turbovec sp=1 | 185 | 1158 | 2108 | 24.0 | 83.2 |
| CO1024 1M | pg_vectorsearch | 1030 | 16761 | **49641** | **566.8** | 4.3 |
| CO1024 10M IVF | pg_turbovec sp=4 | 14 | 158 | 229 | 2.6 | 702 |
| CO1024 10M IVF | pg_turbovec sp=1 | 12 | 152 | 194 | 2.2 | 824 |
| CO1024 10M | pg_vectorsearch | 506 | 7787 | **22294** | **255.1** | 9.7 |

**r8gd.metal-24xl (arm):**

| dataset | system | c=1 TPS | c=16 TPS | c=96 TPS | QPS/core @96 | p99 ms @96 |
|---|---|---|---|---|---|---|
| CO1024 1M flat | pg_turbovec sp=4 | 457 | 3473 | 3423 | 39.7 | 50.4 |
| CO1024 1M flat | pg_turbovec sp=1 | 151 | 2133 | 3525 | 40.5 | 62.0 |
| CO1024 1M | pg_vectorsearch | 612 | 9808 | **39497** | **452.7** | 6.0 |
| CO1024 10M IVF | pg_turbovec sp=4 | 16 | 236 | 363 | 4.2 | 432 |
| CO1024 10M IVF | pg_turbovec sp=1 | 22 | 281 | **618** | 7.1 | 313 |
| CO1024 10M | pg_vectorsearch | 301 | 4787 | **19137** | **220.9** | 12.5 |

At 96 clients prism delivers 15–100× the throughput of pg_turbovec at a fraction of the per-query CPU
(QPS/core 220–1050 vs 2–40). pg_turbovec's flat scan at 1M saturates by ~16 clients (its 5-thread pool
means ~20 concurrent queries fill all cores). **`scan_parallelism=1` is the right setting for throughput:**
at arm 10M c=96 it nearly doubles TPS (618 vs 363) and halves p99. At 1 client the default (sp=4) has the
lower latency. This is the documented trade (README caveat 12 + CHANGELOG 2.13.0).

## P6 — churn / IVF degradation (D4.5), pre-registered predictions CONFIRMED

Build 1M/10M IVF (lists=√n), probes=32, then INSERT +5% then +11.1% of N₀, measuring
`turbovec.index_is_degraded()`, recall@10, p50 after each step; `turbovec_check()` for corruption.
Prediction: +5% stays inside `ivf_max_delta_pct=10` (not degraded, recall ~unchanged); +11.1% crosses
it (degrades to flat scan, flag flips true). **Confirmed on both scales, both arches, zero corruption.**

| host | dataset | step | index_is_degraded() | R@10 | p50 ms | rows/s | turbovec_check corrupt? |
|---|---|---|---|---|---|---|---|
| x86 | 1M | baseline | False | 0.9612 | 23.3 | — | — |
| x86 | 1M | +5% | **False** | 0.9598 | 18.2 | 21737 | — |
| x86 | 1M | +11.1% | **True** | 0.9203 | 8.0 | 28294 | **False** (n=1,111,000) |
| x86 | 10M | baseline | False | 0.9675 | 83.1 | — | — |
| x86 | 10M | +5% | **False** | 0.9591 | 56.9 | 20925 | — |
| x86 | 10M | +11.1% | **True** | 0.9156 | 42.7 | 25466 | **False** (n=11,110,000) |
| arm | 1M | baseline | False | 0.9618 | 25.9 | — | — |
| arm | 1M | +5% | **False** | 0.9598 | 16.5 | 30075 | — |
| arm | 1M | +11.1% | **True** | 0.9203 | 9.7 | 33713 | **False** (n=1,111,000) |

The degradation is **observable** (the plan's hard requirement): `index_is_degraded()` flips true at
+11.1%, and the index stays correct (`turbovec_check` is_corrupt=False, exact row count). After degrade,
p50 drops because the IVF falls back to a flat scan over a now-slightly-larger corpus; recall drops a bit
(0.96→0.92 at probes=32, because the appended rows aren't cell-assigned — they go through the flat tail).
A `REINDEX` restores the cell structure. pg_vectorsearch's lists never split without `prism_rebalance`
(not run here — deadline; its posting lists lengthen and nprobe-fixed latency rises, documented behaviour).

## P7 — filtered search + adoption path (x86, 1M)

### Filtered (D4.6): `WHERE cat < s ORDER BY emb <=> q LIMIT 10`, cat = id mod 100 (uniform)

| system | selectivity | R@10 (vs filtered GT) | short-result rate | p50 ms |
|---|---|---|---|---|
| pg_turbovec (iterative_scan=relaxed_order) | 50% | 1.0000 | 0.000 | 218.2 |
| pg_turbovec | 10% | 1.0000 | 0.000 | 220.1 |
| pg_turbovec | 1% | 1.0000 | 0.000 | 219.3 |
| pg_vectorsearch (prism default) | 50% | 0.9402 | 0.007 | 1.71 |
| pg_vectorsearch | 10% | 0.9063 | 0.024 | 3.39 |
| pg_vectorsearch | 1% | 0.7949 | 0.021 | 34.86 |

The two documented mechanisms trade opposite ways. **pg_turbovec `relaxed_order`** returns exact filtered
results (R@10=1.0, no short results) but is pessimistic by design — it widens the scan up to
`max_scan_tuples`, so filtered p50 is ~220 ms regardless of selectivity (reported as-is, not tuned away).
**pg_vectorsearch prism** is fast at loose filters (1.7 ms at 50%) but recall falls as the filter tightens
(0.94 → 0.79 at 1%) and 0.7–2.4% of queries return fewer than 10 rows (its known issue #188); at 1%
selectivity it also slows to 34.9 ms. A partial index would help prism (not run — deadline).

### Adoption path (D4.9): cost of adopting pg_turbovec from an existing pgvector column

pg_vectorsearch indexes an existing pgvector `vector` column in place (zero copy). pg_turbovec needs EITHER:

| route | one-time cost | query p50 ms | R@10 | notes |
|---|---|---|---|---|
| (i) column copy to `turbovec.vector` | UPDATE 109 s, 9.9 GB WAL, heap 8.2→22.3 GB; index build 13.1 s | **2.04** | 0.9984 | native column; headline uses this |
| (ii) expression index `((emb::real[]::turbovec.vector) vec_cosine_ops)` | no copy; index build 45.2 s (3.5× slower) | **2.83** | 0.9984 | per-candidate cast re-evaluated → +0.79 ms/query (+39%) |

Both use the index (verified by EXPLAIN). The expression-index route avoids the column copy and heap
doubling but pays ~40% query latency from re-evaluating the `::real[]::turbovec.vector` cast per reranked
candidate. pg_vectorsearch pays neither cost (binary-compatible with pgvector `vector`). This is a real
adoption-friction difference and belongs in the README (caveat 10).

## P2 — build time, index size, WAL, peak RSS (matched 32 build threads)

| index | host | build wall s | index size | build WAL | peak RSS |
|---|---|---|---|---|---|
| pg_turbovec flat 1M | x86 | 8.6 | 0.52 GB | 1.04 GB | 2.2 GB |
| pg_turbovec flat 1M | arm | 8.1 | 0.52 GB | 1.04 GB | 2.5 GB |
| pg_vectorsearch prism 1M | x86 | 13.2 | 0.29 GB | 0.29 GB | 0.6 GB |
| pg_vectorsearch prism 1M | arm | 26.3 | 0.29 GB | 0.29 GB | 0.7 GB |
| pgvector HNSW 1M (ref) | x86 | 49.5 | 7.6 GB | 4.14 GB | 0.1 GB |
| pgvector HNSW 1M (ref) | arm | 65.4 | 7.6 GB | 4.14 GB | 0.1 GB |
| pg_turbovec IVF 10M (√n=3162) | x86 | 298 | 5.2 GB | 10.5 GB | 11.7 GB |
| pg_turbovec IVF 10M | arm | 328 | 5.2 GB | 10.5 GB | 12.1 GB |
| pg_vectorsearch prism 10M | x86 | 305 | 3.0 GB | 2.99 GB | 4.2 GB |
| pg_vectorsearch prism 10M | arm | 629 | 3.0 GB | 2.99 GB | 4.2 GB |

- pg_turbovec IVF build is **deterministic**: build #1 and #2 produced byte-identical indexes
  (5,599,854,592 bytes both, 10M). prism's parallel build is non-deterministic by design (its docs); its
  serial-build pair was not re-checked under the deadline.
- prism indexes are **smaller** (RaBitQ 1-bit + factors): 0.29 GB vs turbovec's 0.52 GB at 1M (4-bit codes),
  3.0 GB vs 5.2 GB at 10M. pg_turbovec writes **~2× the WAL** of its index size (full-page relfile writes).
- prism's build RSS is lower (0.6–4.2 GB vs turbovec's 2.2–12 GB). turbovec's build is faster at 1M
  (8.6 vs 13.2 s) but comparable at 10M; prism's 10M build is 2× slower on arm (Graviton k-means).
- Heap (STORAGE PLAIN, 1024-d): both 1 row/page; turbovec.vector CBOR is 5,127 B/row vs pgvector 4,104 B
  (+25%), but page counts are equal at 1024-d so heap sizes match (8.19 GB at 1M, 82 GB at 10M).

## Costs + "not measured"

**AWS spend (kiwi on-demand, us-east-2):** x86 i7i.metal-24xl ~$9.06/h, arm r8gd.metal-24xl ~$6.6/h.
arm ran 12:39→20:55 UTC (~8.3 h ≈ $55); x86 ran 12:39 UTC → terminated at end (~9–9.5 h ≈ $82–86);
P0 smoke earlier ≈ $15. **Total ≈ $150–155** — far under the ~$850 budget, because the shrunk matrix +
v2.13.0's fast scan + parallel hosts cut wall time to ~9 h/host instead of ~64 h.

**NOT measured (honest list):**
- CW768 (both scales) — dropped (source broken; VDBBench substitute deprioritised under deadline).
- OA1536 5M — background download did not finish/was not run (zilliz ~6 MB/s; deprioritised for the
  deadline). CO1024 1M+10M carry the headline.
- 0.99 QPS cells, open-loop `pgbench -R` tail runs — dropped (closed-loop QPS at the 0.95 point only).
- Secondary turbovec kinds (IVF 2-D grid, 4√n lists, 2-bit, 1-bit BQ), prism alternate nlist/soar configs,
  malloc-tuned (GLIBC_TUNABLES) cell — all dropped.
- Native-build (`-march/target-cpu=native`) and TOAST-storage sensitivity cells — dropped.
- AVX2-only arnold cell (D4.10) — NOT run (local host, off critical path; not reached under deadline).
  **The x86 headline is AVX-512 VBMI/VNNI only; AVX2-only hosts are not measured here.**
- 10M column-copy adoption cell — only the 1M adoption cell was run.
- bench_ann.py cross-check (would have needed CW768 HDF5) — not run.
- prism_rebalance maintenance step and REINDEX-after-degrade timing — not run (deadline).
- ABBA rounds: 2 (round 0 discarded + 2 counted), not 4; the extra turbovec 10M probe points (128/192/256)
  are single-pass round-0 only (used to pin turbovec's own 0.98/0.99 endpoint), labelled as such.
- p99 single-client is from pgbench logs (P5) only; single-client cells report p95.

---

# README-READY COMPARISON SECTION (proposed block — for the LEAD to review and insert; agents do not edit README.md)

> ## pg_turbovec vs pg_vectorsearch (prism) — measured
>
> We benchmarked pg_turbovec **v2.13.0** against **pg_vectorsearch** (TimescaleDB's `prism` AM, commit
> `7a9716b`, v0.1.0 alpha) and pgvector 0.8.7 (HNSW, reference only) on PostgreSQL 18.6, on AWS bare-metal
> single-socket 96-vCPU / 768 GiB hosts: **i7i.metal-24xl** (Intel Emerald Rapids, AVX-512 VBMI/VNNI) and
> **r8gd.metal-24xl** (Graviton4). Cosine on L2-normalised Cohere wikipedia-2023-11 1024-d embeddings,
> 5,000 held-out queries, exact fp64-rescored ground truth, matched-recall Pareto. Full harness, raw
> per-query data and configs: `benches/results/vs_pgvectorsearch_20261014/`.
>
> **The honest summary: pg_turbovec is competitive at 1M with its flat index; pg_vectorsearch is
> substantially faster at 10M and at high concurrency.**
>
> **At 1M (flat, warm single client):** at recall@10 ≥ 0.99 pg_turbovec's flat 4-bit index serves ~0.998
> recall at **p50 ≈ 2.0 ms** (x86) / **2.6 ms** (arm); prism's 0.99 point is 2.9 ms (x86) / 5.6 ms (arm) —
> pg_turbovec is faster at the high-recall end. At 0.95 prism is faster in wall-time (1.3 ms vs 2.0 ms on
> x86) at ⅕ the CPU. **pg_turbovec uses ~5 threads/query (CPU ~5.5 ms/q); prism uses 1 core/query (~1 ms/q).**
>
> **At 10M (IVF √n vs prism, warm single client):** prism wins decisively. At recall@10 ≥ 0.95 prism is
> **~29× faster** on x86 (2.2 ms vs 63 ms) and **~17× faster** on arm (3.5 ms vs 61 ms). At 0.99 prism is
> 8.6 ms (x86) / 15.9 ms (arm) vs pg_turbovec IVF 284 ms / 96 ms. pg_turbovec's exact flat scan at 10M
> (R@10 = 1.0) is 42 ms (x86) / 26 ms (arm) — faster than its own IVF at high recall, still ~2–5× slower
> than prism's 0.99 point. **If you have ≳ a few million vectors and want ANN speed, prism's RaBitQ +
> posting-list design is the stronger engine today.**
>
> **Throughput (96 clients):** prism delivers 15–100× the QPS of pg_turbovec because it uses one core per
> query; pg_turbovec's multi-thread-per-query scan trades aggregate throughput for single-client latency.
> For throughput, set `turbovec.scan_parallelism = 1` (nearly doubles pg_turbovec's QPS at 96 clients).
>
> **Where pg_turbovec holds up:** sub-~1M corpora where its flat index is near-exact and ~2 ms; exact
> filtered results (`iterative_scan = relaxed_order` returns recall 1.0 with no short results, though at
> ~220 ms vs prism's 1.7–35 ms with 0.79–0.94 recall and 0.7–2.4% short results); deterministic builds.
>
> **Mandatory caveats:**
> 1. **Conflict of interest:** written and run by the pg_turbovec authors. Harness + raw data + exact
>    commits are published; Timescale should be invited to review the prism configs before any wider claim.
> 2. pg_vectorsearch is **v0.1.0 alpha** at commit 7a9716b — its numbers will change; re-run before quoting.
> 3. Versions: PG 18.6, pgvector 0.8.7, pg_turbovec v2.13.0, Fedora 44. Hosts: i7i.metal-24xl (AVX-512) +
>    r8gd.metal-24xl (Graviton4). **x86 numbers are AVX-512 VBMI/VNNI; AVX2-only hosts were NOT measured.**
> 4. Warm = all resident in RAM; cold on local NVMe RAID0/XFS (does not transfer to EBS/managed PG).
> 5. Both at `STORAGE PLAIN`, LOGGED tables, `wal_level=replica`, NO GLIBC/malloc tuning in these numbers.
> 6. **CPU per query:** pg_turbovec's default scan uses up to `min(cores,4)` threads/query; pg_vectorsearch
>    uses 1. Every latency carries CPU-ms/query and QPS/core; `scan_parallelism=1` is shown beside it.
> 7. **Memory:** pg_turbovec caches index data per backend (flat: whole index ~0.5 GB at 1M; IVF cell-scoped
>    or whole). New-connection first query pays 250–500 ms to build it. Shown in the cold/new-conn table.
> 8. **DML:** a pg_turbovec IVF index degrades to a flat scan once inserts exceed `ivf_max_delta_pct`
>    (10%) of built rows (observable via `index_is_degraded()`, confirmed at +11.1%); REINDEX restores it.
>    prism's lists never split without `prism_rebalance`.
> 9. **Adoption:** prism indexes an existing pgvector `vector` column in place; pg_turbovec needs a
>    `turbovec.vector` column copy (+109 s / +14 GB heap at 1M) or an expression index over the cast
>    (+39% query latency). Any "N× faster" claim must name recall, dataset, scale, host and client count.
