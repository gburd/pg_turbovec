# pg_turbovec Benchmarks

Canonical, reproducible head-to-head benchmark page for `pg_turbovec` vs
`pgvector`. The goal is a standardized, VectorDBBench-style result on a public
corpus so the numbers are comparable to published ANN benchmarks — not a
bespoke single-host claim.

> **Status (2026-06-15):** First standardized 1M run on a public corpus
> (Cohere wiki, 1024-d, real embeddings). **Correctness, storage, build, and
> recall are measured and valid.** The **latency frontier for pg_turbovec is
> now measured on AVX2 hardware** (`arnold`, i9-12900H) -- see
> [AVX2 latency frontier](#avx2-latency-frontier-arnold-i9-12900h). The original
> bench host (`meh`) is a pre-AVX2 Xeon, so turbovec there runs its scalar
> fallback (~1000x slower than its AVX2/AVX-512 SIMD
> kernels); that section is kept as the correctness/storage/recall evidence.
> See [Caveats](#caveats). The headline result this run establishes:
> **recall@10 = 1.000 on the fixed v1.8.0 build at 1M × 1024-d real
> embeddings — the pre-AVX2 correctness fix works.**

## Methodology

| Item | Value |
|------|-------|
| Corpus | `Cohere/wikipedia-22-12-en-embeddings`, 1,000,000 rows, **1024-d**, cosine, L2-normalized |
| Standard size | VectorDBBench "Medium" (1M). Real embeddings, not synthetic. |
| Held-out queries | 1,000 vectors (ids 1000000–1000999), held **out** of the index |
| Ground truth | Brute-force exact top-10 by cosine over all 1M rows. BLAS matmul, cross-checked against in-DB seqscan (`enable_indexscan=off`) — 10/10 overlap on sampled queries; reconstructed corpus byte-identical to the DB (`max|diff| = 0.0`) |
| Recall metric | recall@10 vs the exact GT, averaged over the held-out queries |
| Latency (pgvector) | server-side `Execution Time` from `EXPLAIN (ANALYZE)` |
| Latency (turbovec) | client wall over a unix socket; the in-engine scan dominates (>40s here), so the sub-ms cast/RTT term is negligible |
| Warm protocol | ≥1 warmup query (untimed), then N timed queries with fresh held-out vectors (never corpus members) |
| Host | `meh`: Intel **Xeon E5-2697 v2** (Ivy Bridge), 24 cores, 125 GiB RAM, NixOS |
| SIMD | `avx`, `sse4_1`, `sse4_2` — **no `avx2`, no `avx512`** |
| PostgreSQL | 17.9 |
| pgvector | 0.8.0 |
| pg_turbovec | **binary v1.8.0** (git `7d01a51`, turbovec fork `d3d468e`) |
| `shared_buffers` | 640 MB · `maintenance_work_mem` 8 GB · 8 maint. workers |

Indexes built: pgvector HNSW `(m=16, ef_construction=64)`; pg_turbovec 4-bit;
pg_turbovec 2-bit. All on the same 1M-row heap (`docs.emb vector(1024)`); the
turbovec indexes use the expression cast `(emb::real[]::turbovec.vector)`.

## Correctness gate (the headline result)

Before any benchmarking, a correctness gate ran on a 10k × 128-d table of
**distinct** random unit vectors: build a turbovec 4-bit index, compare the
index top-10 against the brute-force top-10 for 20 fresh probes.

```
mean recall@10 = 1.0000   all top-10 sets distinct (10 ids each): True   → PASS
```

And on the full 1M × 1024-d corpus, **every pg_turbovec config returned
recall@10 = 1.000** vs exact GT.

This matters because the *previous* run on this exact host (the old turbovec
v0.7.0 / pg_turbovec v1.7.1 build) scored **recall@10 = 0.0** here — the
pre-AVX2 wrong-results bug. **v1.8.0 fixes it.** Confirmed on real 1024-d
embeddings at 1M scale.

## Storage and build

| Index | Build time | Size | vs HNSW |
|-------|-----------:|-----:|--------:|
| pgvector HNSW (m16, efc64) | 15:29 | 7,806 MB | — |
| pg_turbovec 4-bit | 08:20 | 1,026 MB | **7.6× smaller**, 1.9× faster build |
| pg_turbovec 2-bit | 07:27 | 512 MB | **15.2× smaller**, 2.1× faster build |

Heap: 5,332 MB (incl. TOAST) for 1M × 1024-d. pg_turbovec's compact quantized
codes are its clearest structural advantage and are CPU-independent.

## Recall-vs-latency frontier — pgvector HNSW

`ef_search` sweep, 200 timed queries each. **This frontier is valid and
AVX2-independent** (it's pgvector's own SIMD, unaffected by turbovec's kernel
path).

| Config | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (1 conn) |
|--------|----------:|---------:|---------:|---------:|-------------:|
| HNSW ef=40  | 0.849 |  9.4 | 20.3 | 25.0 | 96.7 |
| HNSW ef=100 | 0.926 | 13.1 | 22.4 | 25.1 | 74.2 |
| HNSW ef=200 | 0.957 | 17.3 | 32.4 | 41.3 | 53.9 |
| HNSW ef=400 | 0.979 | 20.1 | 38.2 | 48.2 | 46.6 |

## Recall-vs-latency frontier — pg_turbovec

**Recall is exact (1.000) at every config.** pg_turbovec is a *quantized
full-scan (flat) index*, so it does not trade recall for speed the way a graph
index does — every query scores the whole corpus.

**Latency on `meh` is the scalar-fallback FLOOR, not a representative
competitive number.** Reported for completeness only:

| Config | recall@10 | p50 (ms) | basis |
|--------|----------:|---------:|-------|
| tv 2-bit, search_k=100  | 1.000 | 41,618 | scalar fallback (pre-AVX2) |
| tv 2-bit, search_k=500  | 1.000 | 42,014 | scalar fallback (pre-AVX2) |
| tv 4-bit, search_k=100  | 1.000 | 69,043 | scalar fallback (pre-AVX2) |
| tv 4-bit, search_k=1000 | 1.000 | 55,701 | scalar fallback (pre-AVX2) |

Note the fingerprint: latency is **independent of `search_k`** and **identical
warm vs cold** — the cost is the fixed `O(n_vectors · dim)` full-corpus
blocked-code scan, not I/O or candidate-set size.

### Why turbovec is slow on this host (diagnosis)

`meh` has `avx` but **no `avx2`**. turbovec v0.9.0 correctly dispatches to its
scalar `score_query_into_heap` path (the same path whose *correctness* bug
v1.8.0 fixed). That path is right but ~1000× slower than the AVX2/AVX-512
kernels: the on-disk codes use a FAISS-style perm0-interleaved layout built for
the AVX2 kernel, so the scalar path must `deinterleave_x86_code_byte` **per
byte, per vector** — ≈256M de-interleave evaluations per query for 2-bit (more
for 4-bit) over 1M × 1024-d. `EXPLAIN (ANALYZE)` confirms it is an
`Index Scan using docs_tv_*` (not a seq scan): the time is genuinely inside the
turbovec scan kernel. On an AVX2/AVX-512 host the SIMD kernel runs and these
latencies are expected to fall to the tens-of-ms range.

### Tunable recall frontier (oversampling)

The `meh` build (`7d01a51`) predates the `turbovec.oversample` feature, so this
run swept `search_k` only. Current `main` adds `turbovec.oversample`, which
fetches `ceil(search_k · oversample)` quantized candidates and re-ranks them by
exact distance — turning a fixed-quantization accuracy point into a tunable
recall frontier comparable to HNSW's `ef_search` (and to Qdrant oversampling /
VectorChord rerank). A monotone recall-vs-oversample curve is verified in
`benches/results/oversample_recall_curve_2026_06_15.json`. Re-running this
1M frontier on an AVX2 host with the oversample sweep is the natural next step.

## AVX2 latency frontier (`arnold`, i9-12900H)

> **Status (2026-06-15):** The latency numbers `meh` (pre-AVX2) could not
> produce, measured on `arnold` -- a **12th Gen Intel i9-12900H** with `avx2`
> + `fma` (no `avx512`; Alder Lake fuses it off). Same v1.9.0 binary
> (`e2d49cf`, turbovec fork `d3d468e`), same Cohere-wiki 1M x 1024-d corpus,
> same 1000 held-out queries, **byte-identical ground truth** (parquet shards
> md5-verified against `meh`; in-DB brute-force seqscan top-10 == `gt_top10.npy`
> with overlap@10 = 1.00 on probe queries). pgvector 0.8.0, PG 17.9.

**Correctness gate (AVX2 path):** the 10k x 128-d distinct-vector sanity
check passed with **mean recall@10 = 1.0000** over 20 probes, all top-10 sets
10 distinct ids. This confirms the AVX2 SIMD kernel (not just `meh`'s scalar
fallback) is correct on v1.9.0.

### Isolation method (this is a busy shared box)

`arnold` runs the user's interactive desktop + other agent sessions +
Discord/Firefox. The bench was insulated, not given priority:

- The **dedicated bench postmaster** (port 28815, socket
  `/scratch/pg_turbovec-bench`, separate from the user's clusters) was started
  under `taskset -c 2-5` -- four dedicated P-cores, away from cores 0-1
  (kernel/IRQ-favored) and the E-cores 14-19. **All backends inherit the CPU
  mask.** The Python sweep driver pinned itself to the same cores. Default
  `nice` (negative nice needs privilege; CPU pinning is the lever -- the goal
  is to insulate, not preempt the user).
- **Latency = server-side `Execution Time` from `EXPLAIN (ANALYZE)` for BOTH
  engines** (the fair engine-to-engine number; excludes client RTT).
- **Warm protocol:** 20 (pgvector) / 5 (turbovec) untimed warmup queries to
  warm the per-backend Arc cache + OS page cache, then timed.
- **Contention measured per batch:** `/proc/loadavg`, `/proc/stat` CPU
  busy/iowait/steal delta, and free RAM sampled before+after each timed batch.
  Per-query `>3x`-median outliers flagged; both raw and outlier-filtered
  p50/p95/p99 plus a 5% trimmed mean recorded. A batch is flagged contended if
  the observed 1-min load exceeded 1.5.
- **Query counts:** 400 timed queries for the fast pgvector configs; 40 for the
  turbovec full-scan configs (~2.5-2.9s each, so 40 keeps wall-clock sane while
  the near-zero variance keeps the median stable).

**Observed load during the timed windows stayed at ~0.3-1.05** (well under the
1.5 gate); **`contended_flag` was False on all 14 configs**, CPU steal ~0
(bare metal), turbovec batches had 0 outliers (p95 within ~3% of p50). No
batches were discarded or re-run. Full per-batch metadata is in
`benches/results/latency_frontier_arnold_cohere_1m_v1_9_0_2026_06_15.json`.

### pgvector HNSW (AVX2, 400 timed queries)

| Config | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (1 conn) |
|--------|----------:|---------:|---------:|---------:|-------------:|
| HNSW ef=40  | 0.866 | 2.76 | 5.54 | 7.57 | 341 |
| HNSW ef=100 | 0.938 | 3.32 | 6.82 | 8.44 | 276 |
| HNSW ef=200 | 0.964 | 5.25 | 10.0 | 11.7 | 180 |
| HNSW ef=400 | 0.981 | 8.63 | 16.1 | 21.5 | 109 |

Recall matches the `meh` HNSW run closely (ef400 0.981 vs 0.979); the much
lower latency is just the faster CPU.

### pg_turbovec (AVX2, 40 timed queries)

The AVX2 SIMD kernel runs here -- **~15-25x faster than `meh`'s scalar
fallback** (2-bit/k100: 2.55s here vs 41.6s on `meh`). But pg_turbovec is a
flat quantized full-scan, so even with AVX2 a query over 1M x 1024-d is
**seconds, not tens of ms** -- and recall is **exact (1.000) at every
config**, including 2-bit.

| Config | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) |
|--------|----------:|---------:|---------:|---------:|
| tv 2-bit, search_k=100  | 1.000 | 2552 | 2604 | 2620 |
| tv 2-bit, search_k=200  | 1.000 | 2523 | 2575 | 2585 |
| tv 2-bit, search_k=500  | 1.000 | 2735 | 2759 | 2802 |
| tv 4-bit, search_k=100  | 1.000 | 2775 | 2852 | 2887 |
| tv 4-bit, search_k=200  | 1.000 | 2711 | 2734 | 2768 |
| tv 4-bit, search_k=500  | 1.000 | 2906 | 2934 | 2973 |
| tv 4-bit, search_k=1000 | 1.000 | 2854 | 2884 | 2918 |

Latency is **flat across `search_k`** (the `O(n_vectors · dim)` scan dominates;
`search_k` only sizes the result heap) -- the same fingerprint `meh` showed,
now at the AVX2 floor.

### Oversample frontier (4-bit, search_k=200)

| oversample | recall@10 | p50 (ms) |
|-----------:|----------:|---------:|
| 1 | 1.000 | 2710 |
| 2 | 1.000 | 2661 |
| 4 | 1.000 | 2644 |

**On this corpus the oversample lever has no recall headroom to recover:**
both 2-bit and 4-bit already reach recall@10 = 1.000 at the *smallest*
`search_k` (100). 4-bit at oversample=1 already exceeds HNSW-ef400's recall
(1.000 vs 0.981), so it never needs oversampling here. The oversample
*mechanism* is verified correct on a harder synthetic corpus (where base
recall < 1) by the in-tree `#[pg_test]`
`oversample_recall_monotone_non_decreasing`.

### Headline: recall-vs-p50 at matched recall@10 >= 0.95 (AVX2)

| Engine | Config | recall@10 | p50 (ms) |
|--------|--------|----------:|---------:|
| pgvector HNSW | ef200 | 0.964 | **5.2** |
| pgvector HNSW | ef400 | 0.981 | **8.6** |
| pg_turbovec 2-bit | search_k=100 | 1.000 | **2552** |
| pg_turbovec 4-bit | search_k=100 | 1.000 | **2775** |

At the 1M x 1024-d scale, **HNSW is ~490x faster at the warm p50** (5.2ms vs
2552ms) while turbovec is **exact and 7.6-15.2x smaller on disk**. They sit at
different points on the recall/latency/storage frontier: turbovec is a flat
index (exact recall, tiny storage, `O(n)` latency that grows with the corpus),
HNSW is a graph (approximate recall, large storage, sublinear latency). The
AVX2 result confirms turbovec's SIMD path is correct and ~15-25x faster than
the scalar fallback, but does **not** make a 1M-row flat scan latency-
competitive with a graph index -- it was never meant to be. turbovec's pitch
is exact recall + compact codes, and at smaller corpora (or with a coarse
pre-filter) its per-query `O(n·d)` cost shrinks proportionally.


## IVF recall-vs-probes (host-independent)

> **This is the recall/scan-work trade-off, measured without needing a quiet
> AVX2 host.** Recall@10 is a function of *which cells are probed vs where the
> true neighbours live* — it is independent of SIMD speed — so this curve is
> reproducible on any host that builds the extension. It is the host-independent
> evidence that the `turbovec.probes` dial trades recall for scan-work exactly
> as IVF is designed to. **Absolute warm-p50 latency on AVX2 is a separate
> measurement** (see [AVX2 latency frontier](#avx2-latency-frontier-arnold-i9-12900h)
> for the flat-scan frontier); the IVF warm-p50 sweep on a quiet `arnold`
> window is **TODO** — not run here because `arnold` is currently contended
> with the user's PostgreSQL test suite. The `blocks_skipped_by_mask` fraction
> below is the CPU-independent proxy for that latency win: a query that skips
> F% of the corpus's 32-vector blocks does proportionally less scan work.

The frontier is produced by the `ivf_recall_vs_probes_frontier` `#[pg_test]`
(it both asserts the contract and writes the artefact). Corpus: 16,334
distinct deterministic pseudo-random unit vectors, 64-d, 4-bit, `lists = 128`
(≈√n), 50 held-out queries, brute-force exact top-10 ground truth
(`enable_indexscan = off`). Random unit vectors have **no cluster structure**,
so the curve is deliberately the *hard* case (true neighbours scatter across
cells); a clustered or real-embedding corpus rises faster for the same probes.
The curve **shape** (monotone, hits 1.0 at `probes = lists`, skips a large
block fraction at the low end) is scale-invariant; a larger corpus is the same
curve.

| probes | recall@10 | blocks scanned | blocks skipped |
|-------:|----------:|---------------:|---------------:|
| 1      | 0.078     | ~1.0%          | 99.0%          |
| 2      | 0.124     | ~2.0%          | 98.0%          |
| 4      | 0.200     | ~3.9%          | 96.1%          |
| 8      | 0.340     | ~7.7%          | 92.3%          |
| 16     | 0.528     | ~15.2%         | 84.8%          |
| 32     | 0.722     | ~29.5%         | 70.5%          |
| 128 (= lists) | **1.000** | 100%   | 0.0%           |

Artefact: [`benches/results/ivf_recall_vs_probes_2026-06-16.json`](../benches/results/ivf_recall_vs_probes_2026-06-16.json).

**The headline this delivers:** "at `probes = P`, recall@10 = R while scanning
F% of the corpus" — e.g. at `probes = 32` the scan touches ~30% of the blocks
for recall@10 = 0.72, and at `probes = 1` it touches ~1% of the blocks. The
dial works. The contract test asserts (1) recall@10 is monotone
non-decreasing in probes, (2) `recall(probes = lists) = recall(flat) ≈ 1.0`
(probing every cell *is* the full scan), and (3) the low-probes end skips a
large fraction of blocks.

Soft multi-assignment (`WITH (assign_dups = M)`, IVF-4) raises recall@10 at any
fixed `probes` by storing boundary vectors in their top-M nearest cells, at a
bounded storage cost — see [Migrating from pgvector](MIGRATING_FROM_PGVECTOR.md)
and an internal design note.

## Caveats

- **Single host, pre-AVX2 CPU.** `meh` is an Ivy Bridge Xeon (`avx`, no
  `avx2`). turbovec's SIMD kernels (AVX2/AVX-512) do **not** run here; it takes
  the scalar fallback. **The pg_turbovec latency numbers are a worst-case
  floor, not a ceiling, and are NOT representative.** Storage, build, and
  recall are CPU-independent and valid.
- **Latency frontier for pg_turbovec is now measured on AVX2** (`arnold`,
  i9-12900H) -- see [AVX2 latency frontier](#avx2-latency-frontier-arnold-i9-12900h).
  The `meh` numbers above are the **pre-AVX2 scalar-fallback floor** and remain
  here only as the correctness/storage/recall evidence.
- **`arnold` is a busy, RAM-constrained shared host.** The AVX2 run was
  CPU-pinned (`taskset -c 2-5`) with per-batch contention measurement; observed
  1-min load stayed <= ~1.05 and no batch was flagged contended. Build times on
  `arnold` are NOT comparable to `meh` (2GB vs 8GB `maintenance_work_mem`, so
  HNSW spilled to disk); storage sizes ARE comparable and match `meh`.
- **pg_turbovec is a flat (quantized full-scan) index**, not a graph index. It
  delivers near-exact recall and tiny storage; its latency is `O(n)` per query
  (SIMD-accelerated), versus HNSW's sublinear-but-approximate traversal. The
  two occupy different points on the recall/latency/storage trade-off.
- **Extension catalog vs binary:** `bench_wiki` reports the extension catalog
  at 1.7.1 (first `CREATE EXTENSION`), but the loaded `.so` is **v1.8.0** — the
  scan path lives in the binary, so all results were produced by v1.8.0.
- **No competitor beyond pgvector.** VectorChord / pgvectorscale comparison is
  future work.
- **10M not run** this round (budget); 1M is the priority standardized size.
- **Concurrent (pgbench) QPS** not run; single-connection QPS captured for
  HNSW.

## Reproduction

On a host with the v1.8.0+ binary installed (ideally **AVX2+** for
representative turbovec latency):

```bash
# 1. Build/install pg_turbovec from main (pgrx must match Cargo.toml's pgrx pin)
cargo pgrx install --release --pg-config <pgrx pg_config>

# 2. Schema + load 1M Cohere-wiki rows (1024-d), hold out 1000 queries
psql -d bench_wiki -f setup_schema.sql
python3 load_wiki_1m.py --corpus 1000000 --held 1000      # binary COPY pipe
python3 load_queryset.py                                  # held-out -> query_set

# 3. Exact brute-force ground truth (BLAS; cross-check vs in-DB seqscan)
python3 compute_gt_blas.py

# 4. Build indexes (HNSW m16/efc64, turbovec 4-bit, turbovec 2-bit)
psql -d bench_wiki -f build_indexes.sql

# 5. Sweep frontiers
python3 sweep_1m.py --which pgv --n-timed 200 --out res_pgv.json
python3 sweep_tv_lean.py --bits 2 --ks 100,500,1000 --out res_tv2.json  # drop the other tv index first
python3 sweep_tv_lean.py --bits 4 --ks 100,1000     --out res_tv4.json
```

Artifacts (this run): `benches/results/vectordbbench_cohere_wiki_1m_v1_8_0_20260615.json`.

The bench scripts used on `meh` live in `/scratch/pg_turbovec-bench/`
(`load_wiki_1m.py`, `load_queryset.py`, `compute_gt_blas.py`,
`setup_schema.sql`, `build_indexes.sql`, `sweep_1m.py`, `sweep_tv_lean.py`,
`sanity_check.py`).
