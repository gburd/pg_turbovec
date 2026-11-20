# pg_turbovec v2.16.2/v2.16.3 competitive metal bench — FINDINGS

Run: 2026-11-20, AWS `breakfast` (847218423625), us-east-2,
`run=pgvs2162-202611200902`. Hosts:
- x86 i7i.metal-24xl (Emerald Rapids / Xeon Platinum 8559C, AVX-512 VNNI), 96 vCPU
- arm r8gd.metal-24xl (Graviton4, sve2/i8mm/dotprod), 96 vCPU

Stack: PG 18.7 (from source, identical `postgresql.bench.conf`: shared_buffers
128 GB, huge pages, io_uring, jit off, autovacuum off for reads). pgvector v0.8.7
(`f37c13f`, default flags). pg_vectorsearch 0.2.0-dev (`ae2a75f`, prism/RaBitQ,
`-Dnative=false -Dsimd=full`). pg_turbovec **v2.16.2** (`d5cf6d1`, cargo-pgrx
0.19.4, runtime dispatch). Dataset CO1024 (Cohere wikipedia-2023-11 en, 1024-d),
10M corpus + 5000 held-out queries, fp64 ground truth, pinned shard list.

## Headline finding: a build-scaling bug in the e-means IVF trainer (now fixed)

v2.16.0 made e-means the default IVF coarse trainer, qualified on SIFT **1M**
(lists <= 1024). At **10M / lists = round(sqrt(10M)) = 3162 / dim = 1024** the
e-means `CREATE INDEX` **stalled for 40+ minutes pinning a single core** (load
average ~1.05 on 96 cores; all 32 build-pool threads idle). `perf`: 100% in
`turbovec::cluster::cluster_emeans_tuned`. The build never completed.

Root cause: e-means' batch nearest-centroid assignment was a hand-written
scalar serial loop, O(sample · lists · dim) on one core, versus Lloyd's assignment
which runs as a parallel BLAS GEMM. At 1M/k<=1024 the gap was tolerable; at
10M/k=3162 it is ~10 trillion scalar FLOPs on one thread. Lloyd
(`turbovec.ivf_use_emeans = off`) built the identical 10M IVF in **~299 s**.

A second defect in the same loop: no `CHECK_FOR_INTERRUPTS`. The stuck build
ignored `pg_cancel_backend`, `pg_terminate_backend`, and `pg_ctl -m fast stop`;
only `-m immediate` (crash-recovery restart) cleared it.

### Fix (shipped in v2.16.3)

Parallelise the e-means batch assignment (and the revival coverage scan) across
the rayon build pool. The assignment is bit-identical to the serial version
(each point's nearest cell is an independent pure function; the f64 EMA
accumulation still runs in batch order), pinned by the existing
`emeans_determinism_independent_of_rayon_pool` test (1-thread == 8-thread,
byte-exact). **Qualified on this run:** the previously-stuck 10M e-means build
now completes in **~740 s** (whole machine busy, load ~15, >1100% backend CPU),
`turbovec_check` clean (`n_vectors=10000000, is_corrupt=f`), scan R@10=0.982 at
probes=48. Because the centroids are byte-identical, recall and the on-disk
format are unchanged — this is a build-time fix (a patch). Fork carry #8
rev `545febc`.

## Competitive frontier: 10M, matched recall@10 (Lloyd IVF vs prism)

With e-means' build fixed separately, the 10M competitive comparison uses the
Lloyd-trained IVF against prism. Warm, single client, prepared statements,
restart+drop_caches+prewarm per round, ABBA (r0 discarded + r1/r2 timed; r1 and
r2 agreed to ~0.1 ms). fp64-GT recall. **Zero order violations on every point,
both systems, both hosts** (correctness clean).

### x86 i7i.metal-24xl (Emerald Rapids, AVX-512 VNNI)

| target R@10 | pg_turbovec Lloyd IVF (probes, p50) | prism (nprobe, p50) | prism faster by |
|---|---|---|---|
| ~0.95 | 32 → **48.4 ms** (R 0.953) | 56 → **1.63 ms** (R 0.957) | ~30x |
| ~0.97 | 48 → 76.6 ms (R 0.968) | 80 → 1.93 ms (R 0.971) | ~40x |
| ~0.98 | 96 → 170.7 ms (R 0.983) | 160 → 3.29 ms (R 0.986) | ~52x |
| ~0.99 | 128 → 235.0 ms (R 0.988) | 320 → 5.46 ms (R 0.993) | ~43x |

### arm r8gd.metal-24xl (Graviton4)

| target R@10 | pg_turbovec Lloyd IVF (probes, p50) | prism (nprobe, p50) | prism faster by |
|---|---|---|---|
| ~0.95 | 32 → **40.1 ms** (R 0.954) | 56 → **1.77 ms** (R 0.957) | ~23x |
| ~0.97 | 48 → 62.4 ms (R 0.969) | 112 → 2.80 ms (R 0.977) | ~22x |
| ~0.98 | 96 → 111.7 ms (R 0.985) | 160 → 3.79 ms (R 0.983) | ~29x |
| ~0.99 | 128 → 144.4 ms (R 0.989) | 320 → 6.37 ms (R 0.992) | ~22x |

Build (10M, lists=3162, Lloyd): index 5.6 GB, WAL 11.3 GB, peak RssAnon ~13 GB,
CREATE INDEX ~299 s (x86).

## Verdict

Unchanged from the v2.13.0 run: **pg_vectorsearch (prism / RaBitQ + posting
lists) wins decisively at 10M** — ~22-52x faster at matched recall@10. Nothing
on pg_turbovec's scan path changed between v2.13.0 and v2.16.2, so the gap is
the same architecture gap: a brute-force IVF cell scan (tens of ms at 10M) vs
prism's RaBitQ-quantised posting-list traversal (single-digit ms). turbovec's
IVF is notably faster on arm (Graviton4 SVE2/dotprod) than x86. Correctness is
clean on both systems.

This run's actionable output was the e-means build-scaling bug, found and fixed
(v2.16.3). The e-means-vs-Lloyd recall A/B at 10M was deferred: e-means could
not build at 10M before the fix, and with the fix landing late in the run the
decisive comparison is the SIFT1M recall A/B already in the record (e-means
wins recall by +0.01 to +0.02). The standing architecture gap vs prism at 10M
is the real competitive signal, and it is a scan-path/algorithm gap, not a
trainer gap.

Raw per-point summaries: `x86/results_x86.tar.gz`, `arm/results_arm.tar.gz`
(`results/vs2162/front10m/*.summary.json`, build summaries, `versions.txt`,
`emeans_qual.log`).
