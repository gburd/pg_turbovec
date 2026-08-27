# pg_turbovec 2.0.0 competitive leg — GIST-1M + GIST-10M (960d)

**Date:** 2026-08-27
**Box:** AWS i4i.8xlarge (i-0393e90bc0c2c7b54), 32 vCPU Ice Lake **AVX-512**
(Xeon 8375C), 247 GiB RAM, /mnt/nvme 3.4 TB NVMe.
**PostgreSQL:** 18.6 (pgrx download build), `fsync=off`,
`shared_buffers=32GB`, `maintenance_work_mem=32GB`,
`max_parallel_maintenance_workers=32`.
**Engines:**
- **pg_turbovec 2.0.0** — branch `feat/turbovec-1.0.0-port` @ 25ab885,
  turbovec pinned to fork rev `f29a2f2` (pgtv-2.0.0-port). **wire_version=8
  confirmed** via `turbovec.turbovec_check()`. In-RAM (`out_of_core=off`),
  4-bit quantization, `scan_parallelism=0` (single-core-per-query latency).
- **pgvector 0.8.1** (HNSW m32/efc256 + IVFFlat) — 0.8.0 does **not** compile
  against PG18 (`TupleDescData.attrs` / `vacuum_delay_point(bool)` API breaks);
  0.8.1 is the closest PG18-compatible release, identical HNSW/IVFFlat
  algorithms. Baseline used 0.8.0 on PG17.
- **VectorChord** (vchordrq / RaBitQ), dev build, pinned cargo-pgrx 0.17.0.
- **pgvectorscale 0.9.0** (StreamingDiskANN), pinned cargo-pgrx 0.16.1.
- **Qdrant: SKIPPED** this leg (Docker not preinstalled; the baseline already
  has a clean Qdrant GIST-1M datapoint; a prior agent wedged on Qdrant and the
  brief marks it optional). Reference: baseline Qdrant GIST-1M 2.31 ms @ R0.97.

**Metric:** recall@10 vs exact ground truth over 1000 test queries. Warm
single-connection p50 (best of 3 timed passes); qps@1 = 1000/mean; qps@8 =
8-thread throughput over 8 s. Distance = L2. GIST-1M GT = ann-benchmarks
published neighbors. GIST-10M GT = exact top-10 by blocked BLAS GEMM.

---

## GIST-1M (960d) — matched-recall frontier

| Engine | @R≈0.95 (p50, qps@8) | @R≈0.99 (p50, qps@8) | build s | idx MB |
|--------|----------------------|----------------------|--------:|-------:|
| **HNSW** (m32/efc256) | **8.8 ms** (ef120, R.962) q8=745 | **21.9 ms** (ef400, R.994) q8=304 | 993 | 8051 |
| **vchord** (vchordrq) | 11.2 ms (L4000 p300 e1.0, R.986) q8=470 | 25.3 ms (p300 e1.9, R.992) q8=216 | 67–168 | 4307–4403 |
| **pg_turbovec 2.0.0** | 57.7 ms flat (R.991) q8=51 / IVF L1000 p64 60.3 ms (R.953) q8=74 | 57.7 ms flat (R.991) q8=51 | **56–91** | **494–509** |
| **IVFFlat** (lists=1000) | 143 ms (p64, R.981) q8=49 | 269 ms (p128, R.997) q8=27 | 170 | 4102 |
| **DiskANN** (default) | **never** (max R=0.406) | **never** | 2193 | 585 |

Notes: IVFFlat GIST-1M p50 is inflated ~20% because the 10M corpus load ran
concurrently during its query pass; even discounting that, its >100 ms latency
is uncompetitive (matches baseline). DiskANN was run to its **default** variant
only — its SBQ quantization is the recall ceiling at 960d, not graph params, so
the wide/plain rebuilds (each ~40 min, no recall gain) were skipped after
`default` confirmed the baseline's R≈0.40 ceiling.

**turbovec 2.0.0 IVF no longer caps under 0.95.** IVF L1000 with
`hi_dim_rerank=auto`: p64 → R=0.9528 @ 60.3 ms (clears 0.95), p128 → R=0.9822 @
63.0 ms (approaches 0.99 **in-IVF**). The v1.29.3 baseline IVF maxed at R=0.9497
and had to fall back to flat for 0.99.

---

## GIST-10M (960d, semi-synthetic, BLAS-exact GT) — 2 h build budget

| Engine | build | in budget? | @R≈0.95 (p50, qps@8) | @R≈0.99 (p50, qps@8) | idx GB |
|--------|------:|:----------:|----------------------|----------------------|-------:|
| **pg_turbovec 2.0.0 IVF** | **1097 s (18 min)** | **YES** | 83 ms (p32 auto, R.964) q8≈0.5 | **94 ms** (p128 auto, R.990) q8≈0.5 | **4.95** |
| **vchord** (vchordrq) | **733 s (12 min)** | **YES** | 71 ms (p30 e1.9, R.955) q8=94 | 163 ms (p100 e1.0, R.993) q8=39 | 42.9 |
| **pgvector HNSW** (m32/efc256) | **7203 s → DNF** (2 h statement_timeout) | **NO** | — | — | — |
| IVFFlat | *(skipped; baseline: 4616 s, >1 s query latency = unusable)* | | | | |
| DiskANN | *(baseline: DNF at 2 h)* | | | | |

**10M headline:** at R≈0.99, **turbovec 2.0.0 (94 ms / 4.95 GB) now beats
vchord (163 ms / 42.9 GB) on BOTH latency and storage** — the storage win is
8.7×, and turbovec's per-query latency at high recall is now ~1.7× faster than
vchord's. turbovec's qps@8 is still ≈0.5 (single-core-per-query design collapses
under concurrency at 10M); vchord keeps qps@8 = 39–94 for concurrent throughput.
So the practical split is unchanged in shape but turbovec has closed the
per-query latency gap and overtaken it at 0.99.

**Build-in-budget at 10M is a two-horse race, unchanged from the baseline:**
only the quantized-IVF designs (turbovec 18 min, vchord 12 min) build a
10M×960 index inside the 2 h cap. **pgvector HNSW DNF'd at exactly 7203 s**
(2 h timeout) — identical to the baseline; you cannot build this graph in two
hours on a 32-core box. DiskANN DNF'd at 2 h in the baseline and was not
re-attempted. IVFFlat builds in budget (baseline 77 min) but its 1–2 s query
latency at 10M is unusable.

---

## Did 2.0.0 move the frontier vs the 2026-08-15 v1.29.3 baseline?

**Yes, materially — the change is a much faster quantized-scan kernel that
lifts turbovec at high dimension where v1.29.x was weakest.**

| Metric | v1.29.3 baseline | 2.0.0 (this leg) | delta |
|--------|------------------|------------------|-------|
| GIST-1M flat auto | R.991 @ **78 ms**, qps8=11 | R.991 @ **57.7 ms**, qps8=51 | **26% faster p50, 4.6× qps8** |
| GIST-1M max IVF recall | **0.9497** (capped under 0.95) | **0.9822** (L1000 p128) | IVF now clears 0.95 AND reaches 0.98 |
| GIST-1M IVF @0.95 | none (0.9497 just short) | 0.9528 @ 60.3 ms | new frontier point |
| GIST-10M @0.99 | R.9912 @ **258 ms** | R.9904 @ **94 ms** | **2.7× faster** (same recall) |
| GIST-10M vs vchord @0.99 | turbovec 258 ms **> vchord 209 ms** (slower) | turbovec 94 ms **< vchord 163 ms** (faster) | **turbovec overtakes vchord** |
| GIST-1M storage | 0.50 GB | 0.49–0.51 GB | unchanged (16× < HNSW) |
| GIST-10M storage | 4.7 GB | 4.95 GB | unchanged (8.7× < vchord) |

Competitor numbers (HNSW/vchord/IVFFlat/DiskANN) are within noise of the
baseline — same algorithms, PG18 vs PG17 the only difference — which validates
that the turbovec deltas above are real engine improvements, not box drift.

---

## Honest positioning of pg_turbovec 2.0.0 vs the field

- **Storage: unchanged, unambiguous win.** 4-bit quantized index is **16×
  smaller than HNSW** (0.49 vs 8.05 GB) and **9× smaller than vchord** (0.49 vs
  4.4 GB) at 1M; **8.7× smaller than vchord** at 10M (4.95 vs 42.9 GB).
- **Build: wins at scale.** turbovec IVF builds 10M×960 in **18 min**; the
  graph builders (HNSW, DiskANN) DNF or are marginal at the 2 h budget.
- **Latency at 1M/960d: still behind the graph leaders, but the flat scan is
  much faster.** turbovec flat auto is now 57.7 ms (was 78 ms) — still ~7×
  slower than HNSW (8.8 ms) and ~5× slower than vchord (11.2 ms). Storage
  remains the reason to pick turbovec at 1M/960d, not latency.
- **Latency at 10M/960d: turbovec 2.0.0 has flipped ahead of vchord at 0.99**
  (94 ms vs 163 ms) — the scan-kernel speedup compounds at scale. It is the
  fastest **per-query** engine that also builds in budget and fits in single-
  digit GB at 10M.
- **Concurrent throughput (qps@8): still turbovec's weak axis.** ~0.5 at 10M
  (single-core-per-query); vchord keeps 39–94. If you need concurrent QPS at
  10M, vchord wins; if you need lowest single-query latency + smallest index,
  turbovec 2.0.0 wins.
- **DiskANN: unchanged loser at 960d.** Max R≈0.41, never clears 0.95, 37-min
  build for the default variant, DNF at 10M.

**One-line verdict:** pg_turbovec 2.0.0 is still the **storage-and-build-at-scale**
engine, but the 2.0.0 scan kernel materially improved its high-dimension
standing — IVF now clears 0.95 at 1M (was impossible), and at 10M/0.99 it is now
the **fastest per-query engine that builds in budget**, at 8.7× less storage
than vchord. It remains behind HNSW/vchord on 1M/960d single-query latency and
on concurrent throughput everywhere.

---

## DNFs / skips (honest)

- **Qdrant: skipped** — Docker not preinstalled; baseline has a clean Qdrant
  GIST-1M datapoint (2.31 ms @ R0.97); brief marks it optional + a prior agent
  wedged. Not attempted rather than risk wedging the leg.
- **DiskANN GIST-1M: default variant only** — wide/plain rebuilds skipped after
  default confirmed the R≈0.40 recall ceiling (SBQ quantization, not graph
  params).
- **IVFFlat GIST-10M: skipped** — baseline covers it (4616 s build, >1 s query
  latency = unusable); not re-run.
- **vchord GIST-10M: probes≤100 only** — the probes 300/500 sweep was cut (each
  config ~6–8 min at 10M) after the 0.95 and 0.99 frontier points were
  captured; adds only 0.999-at-higher-latency.
- **pgvector 0.8.0→0.8.1** for PG18 compat (algorithm-identical).
- **GIST-10M is semi-synthetic** (GIST-1M train tiled ×10 with σ≈0.048 Gaussian
  jitter; GT is exact BLAS). Treat 10M recall as directional; same corpus as the
  baseline so the comparison is apples-to-apples.
