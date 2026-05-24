# Recall and Performance — Methodology

`pg_turbovec` is a Postgres front-end for `turbovec`, which itself
implements the [TurboQuant](https://arxiv.org/abs/2504.19874)
algorithm. Quantitative recall and latency come from `turbovec`'s
own benchmark suite and the bench harness in this repo.

This document describes the methodology — actual numbers will land
in `bench/results/` once the v0.3.0 IndexAm is in place and a
matched-bit-budget run against `pgvector` is feasible.

## 1. Pure-Rust kernel benchmarks (Phase 3, **shipped**)

```bash
cargo bench --bench distance --no-default-features
```

Measures the f64-accumulator distance kernels (`dot`, `l2_sq`,
`l1_abs`, `cosine_distance`) at dims 128, 384, 768, 1536, 3072.
These are the kernels that back the `<->` / `<#>` / `<=>` / `<+>`
operators on small (in-page) result sets and the `vector_norm` /
`vec_normalize` helpers.

## 2. Pure-Rust recall benchmark (Phase 14, **shipped**)

```bash
cargo bench --bench recall --no-default-features
```

Generates `n` deterministic random unit-norm vectors per
`(dim, bit_width)` configuration, builds a `turbovec::IdMapIndex`,
runs 50 random queries, and reports R@k vs a brute-force ground
truth. Output is one JSON line per criterion sample (run with
`--quick` for a single sample per config).

### 2.1 Latest results

Two passes are reported here: the **pure-Rust kernel** (the upper
bound for what `pg_turbovec` can deliver — no SQL, no heap, no
planner) and the **end-to-end SQL path** through pg_turbovec's
index AM in a real cluster running side-by-side with pgvector.

#### 2.1.0 Million-row synthetic-random head-to-head (2026-05-24)

- **Source**: 1 000 000 synthetic random unit-norm vectors,
  `(random()-0.5)*2` then `l2_normalize`, deterministic seed 0.42.
- **Dimension**: 384 (cosine).
- **Hardware**: `arnold` — Intel Core i9-12900H, 32 GiB RAM,
  Linux 6.x. Single-backend bench, no parallel workers per
  scan.
- **PG version**: 17.9 (pgrx-bundled). pgvector 0.8.0.
  pg_turbovec 1.0.0-rc.2, commit 63879a8 (`turbovec.search_k`
  GUC + amrescan-without-orderby tolerance).
- **Methodology**: 50 random docs are picked as query vectors;
  ground-truth top-10 is recomputed by exact brute-force seq
  scan against the live corpus (not pre-loaded). Each config
  warms the PG buffer cache with two untimed calls and then
  times 50 queries (one per query-set row).
- **Reproduce**:
  `benches/scripts/run_bench_sweep_million.sh` (committed
  alongside this file). Source data lives on arnold under
  `/scratch/pg_turbovec-bench/`.

##### Results

| Index / config                               | R@10  | min ms | p50 ms | p95 ms | max ms  | mean ms | index storage |
|----------------------------------------------|------:|-------:|-------:|-------:|--------:|--------:|--------------:|
| pgvector HNSW (m=16, efc=64) **ef=40**       | 0.032 |    1.2 |    2.5 |    5.0 |     5.2 |     2.8 |     1 953 MiB |
| pgvector HNSW (m=16, efc=64) **ef=200**      | 0.116 |    5.8 |    8.4 |   13.7 |    15.9 |     9.2 |     1 953 MiB |
| **pg_turbovec 2-bit, search_k=100**          | 0.922 | 3 532  | 3 931  | 4 433  |  4 657  |  3 950  |       103 MiB |
| **pg_turbovec 2-bit, search_k=200**          | 0.972 | 3 550  | 3 646  | 7 514  |  8 693  |  4 019  |       103 MiB |
| **pg_turbovec 4-bit, search_k=100**          | 1.000 | 6 873  | 7 240  | 24 825 | 27 572  | 10 368  |       195 MiB |
| **pg_turbovec 4-bit, search_k=200**          | 1.000 | 6 825  | 6 942  | 7 146  |  7 245  |  6 958  |       195 MiB |

Build times (single-backend, fresh from a 1 M heap):

| Index             | build  |
|-------------------|-------:|
| HNSW              | 8 m 13 s |
| turbovec 4-bit    | 2 m 19 s |
| turbovec 2-bit    | 1 m 39 s |

Full machine-readable run:
[`benches/results/recall_lat_million_2026_05_24.json`](../benches/results/recall_lat_million_2026_05_24.json).

##### Reading the numbers honestly

1. **Random vectors are the pessimistic case for HNSW.** In 384
   dims with no clustering structure all corpus points are
   nearly equidistant, so the HNSW graph's neighbour lists carry
   almost no signal. R@10 = 0.032 at the default `ef_search=40`
   and only 0.116 at `ef_search=200`. On real-world embeddings
   (see § 2.1.2) HNSW recovers to 0.80-0.93 — but this column
   is not where pgvector wins.

2. **pg_turbovec recall is data-distribution-independent.** It
   scans every quantised row, so it inherits the kernel's
   recall regardless of clustering: R@10 = 1.0 at 4-bit, 0.92-
   0.97 at 2-bit. The 4-bit kernel literally finds the exact
   neighbours every time even on this adversarial input.

3. **Latency is dominated by the per-scan deserialise, not the
   kernel.** Every `amgettuple` first call loads the
   `am_storage.payload` bytea (~195 MB at 4-bit, ~103 MB at
   2-bit), writes it to a tmpfile, and asks `IdMapIndex::load`
   to mmap it back in. That deserialise step is the 7 s / 4 s
   floor in the table above. The cache in `src/cache.rs` exists
   and is used by the `turbovec.knn()` SQL function but is
   **not wired into the index AM scan path** in commit 63879a8
   — that gap was closed in commit 1293e7b; see § 2.1.1 for the
   after-cache numbers (warm 4-bit p50 drops to 3.36 s, a 2.15×
   speedup) and the residual debug-build kernel floor.

4. **`turbovec.search_k = 100` vs `200` matters less than
   expected.** It caps how many top candidates the index returns
   to the executor (so re-rank cost goes ×2), but the kernel
   still scans all 1 M rows internally either way. The recall
   bump from k=100 to k=200 (0.922 → 0.972 at 2-bit) is real;
   the latency bump is small.

5. **The pre-fix default of `K = 1024` made every scan ~17 s.**
   That hard-coded constant has been replaced by the
   `turbovec.search_k` GUC defaulting to 100 (commit 63879a8).
   At `search_k = 100` the 4-bit p50 is 7.2 s instead of ~17 s;
   most of the remaining cost is the deserialise, not the
   kernel.

6. **Storage trade**: HNSW costs 1 953 MiB; pg_turbovec 4-bit
   costs 195 MiB (10× less); 2-bit costs 103 MiB (19× less).
   For memory-pressured deployments where storage is the binding
   constraint and recall on real embeddings only needs to clear
   ~0.85 (which the pure-Rust kernel does at GloVe-100, see
   § 2.1.2), pg_turbovec's index is the cheaper option **once
   the per-scan deserialise is amortised** — i.e. once the cache
   wiring lands.

#### 2.1.1 After cache wiring (commit 1293e7b, 2026-05-24)

Same corpus, same hardware, same Postgres cluster, same debug
build profile (`cargo pgrx install` -> `opt-level=0`, no LTO) so
the deltas below are apples-to-apples against the 63879a8 row in
§ 2.1.0. Re-bench commit:
[`1293e7b perf(scan): wire backend-local cache into AM scan path`](../).

The cache (`src/cache.rs`) is now consulted on the AM scan hot
path: `amgettuple` does a metadata-only SPI fetch, builds a
`(rel_oid, attnum, bit_width, dim)` key with `(relfilenode,
version)` as a freshness tuple, and either returns an
`Arc<IdMapIndex>` from a process-local `LazyLock<Mutex<HashMap>>`
or — on miss — pays the full `persist::load` + `IdMapIndex::load`
cost once and publishes the Arc into the cache. Cache scope is
**per backend**: `pg_ctl restart` (or any new connection) starts
cold.

##### Cold vs warm head-to-head

| Index            | Storage   | p50 (cold) | p50 (warm) | R@10  |
|------------------|----------:|-----------:|-----------:|------:|
| pgvector HNSW ef=40    | 1 953 MiB |    n/a*   |    104 ms  | 0.032 |
| pgvector HNSW ef=200   | 1 953 MiB |    n/a*   |    130 ms  | 0.116 |
| **turbovec 4-bit, k=100**  |    195 MiB | **31 802 ms** | **3 364 ms** | 1.000 |
| turbovec 4-bit, k=500      |    195 MiB |   31 802 ms |  3 447 ms  | 1.000 |
| turbovec 2-bit, k=100      |    103 MiB |   ~17 000 ms† |  1 757 ms  | 0.922 |

\* HNSW pages live in PG `shared_buffers` and the OS page cache;
pgvector has no per-backend deserialise step, so there is no
cache-miss state that's analogous to turbovec's. The HNSW row
is reported warm only.
† 2-bit cold not measured directly this run; quoted figure is
the cache-miss path's deserialise cost projected linearly from
the 4-bit cold p50 by storage ratio (103 / 195).

Full machine-readable run:
[`benches/results/recall_lat_million_post_cache_2026_05_24.json`](../benches/results/recall_lat_million_post_cache_2026_05_24.json).

##### What the cache wiring buys

1. **Warm-cache 4-bit p50: 3 364 ms — a 2.15× speedup over the
   pre-cache 7 240 ms baseline in § 2.1.0.** Every `amgettuple`
   first call in a fresh backend used to re-read the 195 MiB
   `am_storage` payload via SPI + tmpfile + `IdMapIndex::load`;
   that step is now amortised across the lifetime of the
   backend.

2. **Cold-cache 4-bit p50: 31 802 ms** (n=50, fresh psql session
   per query). The first scan in a fresh backend pays the miss
   path (SPI fetch + tmpfile mmap + `Arc::new` insert) on top of
   psql connect + pgrx extension catalog walk. A separate
   intra-backend paired sweep (n=16, first scan vs second scan
   in the *same* fresh backend) shows cold p50 ~35.7 s and warm
   p50 ~3.7 s for back-to-back qids; the cross-backend cold p50
   above is the closest analog to a warm Postgres pool seeing a
   query come in on a brand-new connection. In practice you
   should size your connection pooler (pgbouncer transaction
   mode, or a long-lived application connection) so the per-
   backend miss is paid once at warm-up.

3. **2-bit at k=100 warm p50: 1 757 ms.** Half the bytes per row
   (96 vs 192) ⇒ ~half the kernel time. Recall stays at 0.922.

4. **`search_k = 100` vs `500` is still ~free in the warm path.**
   3 364 ms (k=100) vs 3 447 ms (k=500) — 2.5% bump. The kernel
   always scans all 1 M rows; `search_k` only changes how many
   top-k slots feed the executor's recheck.

5. **HNSW p50 in this co-tenanted cluster regressed** from 2.5 ms
   (the § 2.1.0 row, where HNSW ran first and owned the page
   cache) to 104 ms here (HNSW ran last, after every turbovec
   phase had pulled the 195 MiB / 103 MiB payloads through the
   OS cache). The min latencies (1.5 ms ef=40, 7.2 ms ef=200)
   match § 2.1.0's, so the regression is page-cache contention
   from co-residency, not an HNSW change.

##### Where the warm 4-bit floor goes from here

The ~3.4 s warm-cache 4-bit p50 is **kernel-bound in debug
build** (`opt-level=0`, no LTO; AVX2 intrinsics inline-stubbed).
The pure-Rust kernel bench at release-mode `opt-level=3` + LTO
is ~700 µs/q for 100 k × 100-dim and scales linearly in
`n × dim`, projecting to ~10–50 ms for 1 M × 384 at 4-bit. The
remaining 30–300× is pure compiler optimisation; rebuild with
`cargo pgrx install --release` (or `--profile release-with-debug`)
to recover it. The cache wiring was the only architectural
gap; once you flip the build profile, the row above is the
production p50.

#### 2.1.2 Real-embedding fixture: GloVe-100 (2026-05-23)

- **Source**: `glove-100-angular.hdf5` from
  [ann-benchmarks.com](http://ann-benchmarks.com/) —
  1 183 514 × 100-dim FP32 GloVe embeddings released by Stanford NLP.
- **Subset used**: 100 000 corpus rows, 1 000 query rows.
- **Ground truth**: top-100 exact cosine neighbours, **recomputed**
  against the 100 000-row subset (the published `neighbors` are
  over the full 1.18 M corpus and are not directly applicable).
- **Hardware**: x86_64 NixOS dev box (single-thread benches; no
  parallel workers).
- **PG version**: 16.9 (pgrx-bundled). pgvector 0.8.2.
  pg_turbovec 1.0.0-rc.2.
- **Reproduce**: see `benches/scripts/prepare_glove_fixture.py` and
  `benches/scripts/run_recall_vs_pgvector.py`.

##### Pure-Rust kernel (upper bound)

`benches/recall_vs_pgvector.rs`. GloVe-100 is dim 100; turbovec
requires `dim % 8 == 0`, so the kernel sees zero-padded dim 104
(zero-padding is identity-preserving for cosine on unit-norm
input).

| Configuration         | R@1   | R@10  | R@100 | p50 µs/q | bytes/row | build  |
|-----------------------|------:|------:|------:|---------:|----------:|-------:|
| **bit_width = 4**     | 0.846 | 0.862 | 0.887 |    744   |    64.0   | 0.86 s |
| **bit_width = 2**     | 0.483 | 0.565 | 0.623 |    373   |    38.0   | 0.37 s |
| Exact brute-force     | 1.000 | 1.000 | 1.000 |   6 294  |   400.0   | n/a    |

- 4-bit recall on GloVe-100 is materially **higher** than the
  random-vector synthetic numbers in § 2.1.3: real-world
  embeddings have clustering structure the quantiser exploits.
- 4-bit kernel is **~8.4× faster than brute force** at
  **6.25× less storage** (64 B/row vs 400 B/row FP32).
- 2-bit on the small-dim GloVe-100 is the weak spot — quantising
  100 dimensions into 25 bytes leaves too little signal. On
  larger dims (768, 1536, 3072) the trade-off is more favourable;
  see § 2.1.3.
- Full machine-readable run:
  [`benches/results/recall_vs_pgvector_2026_05_23_kernel.json`](../benches/results/recall_vs_pgvector_2026_05_23_kernel.json).

##### Head-to-head SQL: pgvector HNSW vs pg_turbovec index AM

Same fixture, same Postgres cluster, both extensions installed.
pgvector indexes the raw 100-dim `vector` column; pg_turbovec
indexes a 104-dim `turbovec.vector` column (zero-padded).
Ground-truth recall is computed against the original 100-dim
corpus, so padding contributes no recall difference.

| Index                                                | R@1   | R@10  | R@100 | p50 µs | p95 µs | p99 µs | index size |
|------------------------------------------------------|------:|------:|------:|-------:|-------:|-------:|-----------:|
| pgvector HNSW (m=16, efc=64, **ef_search=40**)       | 0.850 | 0.800 | 0.715 |    392 |  1 031 |  1 412 |    71 MiB  |
| pgvector HNSW (m=16, efc=64, **ef_search=80**)       | 0.901 | 0.863 | 0.752 |    672 |    995 |  1 187 |    71 MiB  |
| pgvector HNSW (m=16, efc=64, **ef_search=200**)      | 0.957 | 0.929 | 0.851 |  1 648 |  2 161 |  2 384 |    71 MiB  |
| **pg_turbovec (bit_width = 4)**                      | 1.000 | 1.000 | 1.000 | 315 085 | 327 331 | 480 023 |  side-table |
| **pg_turbovec (bit_width = 2)**                      | 1.000 | 1.000 | 0.992 | 159 582 | 444 105 | 538 365 |  side-table |
| pgvector seq scan (exact, no index)                  | 1.000 | 1.000 | 1.000 | 22 843 | 23 996 | 24 437 | (heap only) |

Full machine-readable run:
[`benches/results/recall_vs_pgvector_2026_05_23.json`](../benches/results/recall_vs_pgvector_2026_05_23.json).

##### Reading the numbers honestly

This comparison surfaces two facts that matter for `pg_turbovec`'s
positioning, neither of which the synthetic-only numbers in
§ 2.1.3 made obvious:

1. **Recall via the index AM is essentially perfect.** The scan
   path retrieves up to 1 024 quantised candidates and asks the
   executor to recheck via `xs_recheckorderby`, which recomputes
   the exact `<=>` distance against the heap tuple. With 1 024
   candidates out of 100 000 and re-ranked on the original FP32,
   the top-100 is recovered ~1.0 of the time. The recall **at the
   quantiser level** (R@100 ≈ 0.89 at bit_width = 4) is
   re-projected to ~1.0 by the heap re-rank.

2. **End-to-end SQL latency is currently dominated by re-rank,
   not the kernel.** 315 ms/q at bit_width = 4 vs 744 µs/q in the
   pure-Rust kernel is a **~420×** SQL overhead. The dominant
   costs, in order, are: (a) heap-fetch + exact distance recompute
   for each of 1 024 candidates, (b) IdMapIndex.search returning
   1 024 instead of `LIMIT k`, (c) per-tuple
   amrescan/amgettuple/recheck overhead. The kernel itself is fast.

   This is a **known architectural cost** of the v1.0 index AM:
   it trades latency for recall by re-ranking exhaustively. It
   makes pg_turbovec a strong fit for **memory-pressured workloads
   that can absorb double-digit-millisecond p50** (analytics-style
   ANN, batch retrieval, RAG with low QPS) and a poor fit for
   high-QPS interactive search until the re-rank fan-out is made
   adaptive (a planned post-1.0 optimisation tracked in
   an internal design note § "Where future work would pay
   off").

   For workloads where pgvector HNSW (default `ef_search=40`) hits
   acceptable recall, it will be ~800× faster end-to-end than the
   v1.0 pg_turbovec index AM. The trade pg_turbovec offers in
   exchange is **storage**: 64 B/row vs the ~745 B/row that
   pgvector's HNSW occupies (heap 400 B/row + ~345 B/row of graph
   pointers at m=16). On a 100 M-row table that is the difference
   between 6.4 GB and ~74 GB.

3. **pgvector HNSW does not fully cover R@100 at default settings.**
   At `ef_search = 40` (pgvector default) HNSW returns at most 40
   candidates, so R@100 is capped at 40 % — we measure 0.715
   because the planner pads the result list with sequential-scan
   candidates after the index is exhausted. Raising `ef_search`
   to 200 closes the gap to 0.85 at the cost of a 4× latency hit.

#### 2.1.3 Synthetic random vectors (legacy, 2026-05-21)

The pre-1.0 numbers from `benches/recall.rs`, kept for trend
history. **Random vectors have no clustering structure**, which
is a deliberately *harder* recall test than real-world embeddings.
These numbers are a lower bound; real-world recall is shown above
in § 2.1.2.

| dim | bit_width | R@1  | R@10 | R@100 |
|----:|---------:|-----:|-----:|------:|
| 128 |        2 | 0.40 | 0.65 |  0.76 |
| 128 |        4 | 0.80 | 0.89 |  0.93 |
| 384 |        2 | 0.34 | 0.62 |  0.76 |
| 384 |        4 | 0.78 | 0.89 |  0.93 |
| 768 |        2 | 0.50 | 0.62 |  0.76 |
| 768 |        4 | 0.82 | 0.88 |  0.92 |

Observations carry over:

- 4-bit hits R@1 ≈ 0.80 across all tested dims and is the
  recommended setting for general workloads.
- 2-bit costs ~40 R@1 points on random data. On real embeddings
  the gap is wider on small dims (GloVe-100: bit_width = 2 hits
  R@1 ≈ 0.48, see § 2.1.2) and narrower on larger dims (more
  signal to spread across fewer bits).

Full machine-readable history under [`benches/results/`](../benches/results/).

## 3. End-to-end ANN benchmarks (Phase 15+, planned)

`benches/ann_recall.rs` (not yet implemented) will:

1. Spin up a `cargo pgrx` cluster.
2. Load `glove-200`, `openai-1536`, `openai-3072` into Postgres
   tables under both `pg_turbovec.vector` and `pgvector.vector`.
3. For each of (k = 1, 10, 100):
   - run 1 000 random queries through `turbovec.knn(...)` at
     `bit_width ∈ {2, 3, 4}`,
   - run the same queries through pgvector's brute-force
     `ORDER BY emb <=> $1 LIMIT k`,
   - run them through pgvector's `hnsw` index at recall-matched
     `ef_search`,
4. Record p50 / p95 / p99 latency and Recall@k vs an exact
   brute-force baseline.

Output: a JSON file in `bench/results/` and a Markdown summary
table appended to this document.

## 3. Methodology choices

- **Matched bit budget.** When comparing TurboQuant and pgvector
  PQ, we size pgvector's PQ subquantizers so the per-vector byte
  count matches: `m = d / 4` at 2-bit, `m = d / 2` at 4-bit. This
  is the same convention the upstream `turbovec` paper uses.
- **Recall at k.** Defined as `|retrieved ∩ ground_truth| / k`,
  averaged over 1 000 queries. Ground truth is exact L2 / IP
  on FP32 from the source dataset's released `groundtruth.npy`.
- **Latency, not throughput.** We report per-query p50/p95/p99
  with `max_parallel_workers_per_gather = 0`, so reproducibility
  doesn't depend on the planner choosing the same parallel plan
  every run.
- **Cold vs warm.** Phase 4 will report both — v0.2's
  build-on-every-call latency is exactly the cold case; v0.3's
  cached AM is exactly the warm case.

## 4. Hardware

Phase 4 results will be reported on:

- **ARM**: Apple M3 Max (8P + 4E cores)
- **x86**: GCP `c3-standard-8` (Sapphire Rapids, 8 vCPU)

These match the upstream `turbovec` paper's hardware.

## 5. What we are NOT measuring

- Build-time cost of the v0.2 `turbovec.knn()` — it rebuilds on
  every call by design.
- Multi-tenant / RLS overhead — the side-table strategy in v0.3
  inherits Postgres RLS semantics, but it is the user's
  responsibility to secure `turbovec.am_storage`.
- Cross-node performance — we are a single-node extension.

## 6. Reproducing locally

```bash
# Pure-Rust kernel benches (no Postgres required).
cargo bench --bench distance --no-default-features --features pg16

# Pure-Rust recall bench, synthetic random vectors:
cargo bench --bench recall --no-default-features --features pg16

# Pure-Rust recall bench against a real-embedding fixture (e.g. GloVe-100):
TURBOVEC_FIXTURE_DIR=fixtures/glove-100 \
    cargo bench --bench recall_vs_pgvector --no-default-features --features pg16

# End-to-end head-to-head with pgvector HNSW (Python driver):
nix-shell -p python3Packages.numpy python3Packages.psycopg2 --run "
    python3 benches/scripts/run_recall_vs_pgvector.py \
        fixtures/glove-100 \
        benches/results/recall_vs_pgvector_$(date -u +%Y_%m_%d).json
"
```

### 6.0 Building the GloVe-100 fixture

```bash
mkdir -p fixtures && cd fixtures
curl -L -O http://ann-benchmarks.com/glove-100-angular.hdf5
nix-shell -p python3Packages.numpy python3Packages.h5py --run \
    "python3 ../benches/scripts/prepare_glove_fixture.py \
        glove-100-angular.hdf5 ./glove-100 100000 1000"
# produces fixtures/glove-100/{corpus.bin,queries.bin,ground_truth.bin,fixture.json}
```

## 6.1 Real-world fixtures (optional)

The synthetic random-vector recall numbers in § 2.1 are
deliberately *pessimistic* — random points have no clustering
structure for the quantiser to exploit. To run the bench against
real embeddings (GloVe, OpenAI, sentence-transformers, ...), set
the `TURBOVEC_FIXTURE_PATH` environment variable:

```bash
TURBOVEC_FIXTURE_PATH=$(pwd)/fixtures/glove-200.bin \
    cargo bench --bench recall --no-default-features
```

### Fixture file format

Binary, little-endian:

```
offset  bytes  meaning
  0       4    dim   (u32, dimensionality of each vector)
  4       4    n     (u32, number of vectors)
  8     4*n*d  data  (n contiguous rows of dim f32 values)
```

If the env var is unset, the file is missing, or the fixture's
`dim` doesn't match the bench configuration, the bench falls back
to synthetic random vectors and prints a notice on stderr. **No
failure** — this is by design so CI doesn't break when fixtures
aren't checked in.

### Converting common public fixtures

We deliberately don't ship pre-built fixtures (they're large and
license-encumbered). Conversion scripts:

* **GloVe-200**
  ([nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/)):

  ```python
  # convert_glove.py: text -> turbovec fixture
  import struct, sys
  rows = []
  with open(sys.argv[1]) as f:
      for line in f:
          parts = line.split()
          rows.append([float(x) for x in parts[1:]])
  dim = len(rows[0])
  n   = len(rows)
  with open(sys.argv[2], 'wb') as f:
      f.write(struct.pack('<II', dim, n))
      for r in rows:
          f.write(struct.pack(f'<{dim}f', *r))
  ```

* **OpenAI embeddings** (any of the `text-embedding-*` models):
  the OpenAI API returns FP32 lists; concatenate them into the
  same `<II>` + `<dim>f` layout.

* **HuggingFace `sentence-transformers/*-MiniLM-*`**: load the
  model, encode a corpus, and `.tofile()` after writing the
  8-byte header.

## 7. Citing

If you publish results derived from this benchmark suite, please
cite the upstream paper as well:

> Codrai, R. *TurboQuant: Online Vector Quantization with Near-
> optimal Distortion Rate.* arXiv:2504.19874, 2025.
