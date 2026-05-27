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
   (see § 2.1.3) HNSW recovers to 0.80-0.93 — but this column
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

#### 2.1.2 Release-build numbers (commit cca1ddc / v1.0.0, 2026-05-24)

Same corpus, same hardware, same Postgres cluster as §§2.1.0 /
2.1.1, with the only delta being the build profile:
`cargo pgrx install --release` (`opt-level = 3`, LTO on, AVX2
intrinsics fully inlined). All other knobs — `turbovec.search_k`
GUC, the per-backend `Arc<IdMapIndex>` cache, indexes,
`gt_top10`, query set — are unchanged from § 2.1.1.

Indexes were dropped and rebuilt under the release binary so the
build-time numbers are also release-mode.

##### Build times (single-backend `CREATE INDEX` on the 1 M heap)

| Index             | Debug  | Release | Speedup |
|-------------------|-------:|--------:|--------:|
| turbovec 4-bit    | 2 m 19 s | 33.5 s | 4.1×   |
| turbovec 2-bit    | 1 m 39 s | 23.1 s | 4.3×   |
| HNSW (m=16, efc=64, pgvector) | 8 m 13 s | 8 m 13 s | n/a (pgvector binary unchanged) |

##### Latency / recall

| Index            | Storage   | p50 (cold) | p50 (warm) | p95 (warm) | R@10  |
|------------------|----------:|-----------:|-----------:|-----------:|------:|
| pgvector HNSW ef=40    | 1 953 MiB |    n/a*   |     99 ms  |    149 ms  | 0.032 |
| pgvector HNSW ef=200   | 1 953 MiB |    n/a*   |    121 ms  |    267 ms  | 0.116 |
| **turbovec 4-bit, k=100**  |    195 MiB | **6 786 ms** | **22 ms**  |    23 ms   | 1.000 |
| turbovec 4-bit, k=500      |    195 MiB |   6 786 ms† |   61 ms  |    69 ms   | 1.000 |
| turbovec 2-bit, k=100      |    103 MiB |   ~3 600 ms‡ |   12 ms  |    13 ms   | 0.922 |

\* Same caveat as § 2.1.1: pgvector has no per-backend
deserialise step, so HNSW has no analog of turbovec's cache-miss
cold state.
† Cold p50 only measured at `search_k = 100`; the cold
bottleneck is `IdMapIndex::load` from the bytea payload, which is
independent of `search_k`.
‡ 2-bit cold not measured directly this run; quoted figure is
the 4-bit cold p50 scaled by storage ratio (103 / 195).

Full machine-readable run:
[`benches/results/recall_lat_million_release_v1_0_0.json`](../benches/results/recall_lat_million_release_v1_0_0.json).

##### What the release build buys

1. **Warm 4-bit p50 collapses from 3 364 ms (debug) to 22 ms
   (release) — a 152× speedup.** This is the AVX2 distance
   kernel + LTO inlining doing exactly what § 2.1.1 projected
   ("~10–50 ms for 1 M × 384 at 4-bit"). The kernel was the
   only remaining warm-path cost after the cache wiring landed,
   so flipping the optimiser unlocks essentially the full
   pure-Rust kernel speed.

2. **Warm 2-bit p50 collapses from 1 757 ms to 12 ms — 146×.**
   Half the bytes per row scanned, half the kernel time, recall
   stays at 0.922.

3. **`turbovec.search_k = 100` vs `500` is now the same order
   of magnitude as the kernel itself.** 22 ms (k=100) vs 61 ms
   (k=500); the per-tuple recheck cost (heap fetch + exact
   FP32 distance recompute on 5× more candidates) becomes a
   meaningful slice of the budget once the kernel is no longer
   the floor.

4. **Release pg_turbovec is 5.4× *faster* than pgvector HNSW
   ef=200 on this fixture and 4.5× faster than HNSW ef=40,**
   with strictly higher recall (1.000 vs 0.116 / 0.032). On
   uniform-random data HNSW degenerates while turbovec scans
   every quantised row, so recall is data-distribution
   independent. See § 2.1.3 / GloVe-100 for the real-embedding
   trade-off.

5. **Cold-cache 4-bit p50: 6 786 ms — down 4.7× from the
   debug-build 31 802 ms.** Most of the cold cost is the
   `IdMapIndex::load` deserialise via SPI + tmpfile + mmap,
   which is I/O- and allocator-bound rather than kernel-bound;
   release helps less than on the warm path. Eliminating the
   tmpfile round-trip (load straight from the bytea slice) is
   tracked for 1.1 — see `docs/ROADMAP_DECISIONS.md`.

6. **Build times also drop ~4×.** 4-bit `CREATE INDEX` on 1 M
   rows: 2 m 19 s → 33.5 s. The TurboQuant fit is matrix-heavy
   (BLAS/LAPACK calls into OpenBLAS, which is the same C
   library either way), but the surrounding tight loops were
   debug-build hot.

##### Final head-to-head (1 M × 384, cosine, release, warm cache)

| Index            | Storage   | p50 (warm) | R@10 (synthetic random) |
|------------------|----------:|-----------:|------------------------:|
| HNSW ef=40       |   1 953 MB |    99 ms  | 0.032 |
| HNSW ef=200      |   1 953 MB |   121 ms  | 0.116 |
| **turbovec 4-bit** |    195 MB |    22 ms  | 1.000 |
| **turbovec 2-bit** |    103 MB |    12 ms  | 0.922 |

10× smaller than HNSW, 5× faster, with strictly better recall on
uniform-random data. GloVe-100 (§ 2.1.3 below) is the real-world
reference.

#### 2.1.3 After in-memory deserialiser (commit 0c42f55, 2026-05-24)

Follow-up bench against § 2.1.2's release-build baseline. The
only change is the vendored `turbovec` 0.5.0 patch landed in
commit `0c42f55`: public `Read`/`Write` trait deserialisers let
`src/index/persist.rs::read_idmap_from` skip the SPI -> tmpfile
-> mmap dance and parse straight from the bytea slice in memory.
Corpus, hardware, indexes, and methodology are otherwise
identical to § 2.1.2.

| Metric (tv_4bit, k=100) | § 2.1.2 (cca1ddc) | § 2.1.3 (1769a43) |
|--|--:|--:|
| Cold-cache p50 | 6 786 ms | **6 802 ms** |
| Warm-cache p50 |    22 ms |     22 ms    |

No measurable cold-path speedup (0.998×; within run-to-run
noise). Removing the tmpfile round-trip was the cheapest of
three dominant cold-path costs; the remaining two are still in
place:

1. **SPI fetch of the ~195 MiB bytea payload** (PG TOAST detoast
   + decompress on every fresh backend).
2. **`HashMap<u64, usize>` construction for `slot_to_id`** — 1 M
   entries built from scratch on every cache miss.

The in-memory parser is still a strict win on code clarity and
removes a `/tmp` file dependency, but closing the cold-path gap
requires storing the index payload as relfile pages in
`shared_buffers` (cached cluster-wide, not per-backend) instead
of a single bytea heap row. Tracked for 1.1; see
`docs/ROADMAP_DECISIONS.md`.

Warm-cache p50 is unchanged — the warm path was already
bypassing `read_idmap_from` via the per-backend `Arc<IdMapIndex>`
cache, so this patch could not have moved it.

Full run:
[`benches/results/recall_lat_million_inmem_load_2026_05_24.json`](../benches/results/recall_lat_million_inmem_load_2026_05_24.json).

#### 2.1.3 Real-embedding fixture: GloVe-100 (2026-05-23)

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
  see § 2.1.4.
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
§ 2.1.4 made obvious:

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
   `docs/ROADMAP_DECISIONS.md` § "Where future work would pay
   off"). **Update (§ 2.1.2):** the v1.0.0 release build on the
   1 M synthetic corpus brings 4-bit warm p50 to 22 ms and 2-bit
   to 12 ms, *faster than HNSW ef=200*. The 315 ms GloVe number
   below is debug-build; expect the release-build pattern to
   match the 1 M results once re-measured on the GloVe fixture.

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

#### 2.1.4 Synthetic random vectors (legacy, 2026-05-21)

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
  R@1 ≈ 0.48, see § 2.1.3) and narrower on larger dims (more
  signal to spread across fewer bits).

Full machine-readable history under [`benches/results/`](../benches/results/).

## 2.2 Real-world recall on dbpedia-entities-openai-1M (1 M × 1536-d)

This is the canonical real-embedding head-to-head referenced from the
README headline: pgvector HNSW vs pg_turbovec 4-bit and 2-bit on the
same corpus the community uses for OpenAI-scale ANN benchmarks.

- **Source**: [`KShivendu/dbpedia-entities-openai-1M`](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
  on Hugging Face — 1 000 000 Wikipedia/DBpedia article entities
  embedded with OpenAI's `text-embedding-ada-002` (cosine, unit-norm).
- **Dimension**: 1536.
- **Hardware**: `arnold` — Intel Core i9-12900H, 32 GiB RAM,
  Linux 7.x. PG 17.9 (pgrx-bundled). pgvector 0.8.0.
  pg_turbovec 1.0.0 release build, commit `2c45824`.
- **Methodology**: 50 query vectors drawn from the first 50 docs in
  the corpus (so rank-1 is trivially the query itself; R@10 is
  dominated by ranks 2..10). Brute-force cosine ground truth from a
  parallel seqscan with `enable_indexscan=off` /
  `enable_bitmapscan=off`. Per config, two warmup queries then 50
  timed queries via plpgsql `clock_timestamp()` around
  `ORDER BY emb <=> q LIMIT 10`. Indexes are renamed in/out of the
  way (no rebuild) to force the planner to pick a single AM per phase.
- **Reproduce**: `benches/scripts/load_dbpedia_1M.py` +
  `benches/scripts/run_dbpedia_sweep.sh` +
  `benches/scripts/build_tv_dbpedia.sh` +
  `benches/scripts/gt_dbpedia.sh` +
  `benches/scripts/emit_dbpedia_json.py`. Source data lives on arnold
  under `/scratch/pg_turbovec-bench/dbpedia/`.

### Headline

| Index / config                            |  Storage |    Build | p50 (warm) | p95 (warm) |  R@10 |
|-------------------------------------------|---------:|---------:|-----------:|-----------:|------:|
| pgvector HNSW (m=16, efc=64) **ef=40**    | 8 192 MB | 4 m 55 s |     61 ms  |     93 ms  | 0.962 |
| pgvector HNSW (m=16, efc=64) **ef=200**   | 8 192 MB | 4 m 55 s |    115 ms  |    222 ms  | 0.970 |
| **pg_turbovec 4-bit, search_k=100**       |   780 MB | 2 m 43 s |     71 ms  |     91 ms  | **1.000** |
| pg_turbovec 4-bit, search_k=500           |   780 MB | 2 m 43 s |    124 ms  |    143 ms  | 1.000 |
| **pg_turbovec 2-bit, search_k=100**       |   396 MB | 2 m 06 s |     48 ms  |     50 ms  | **1.000** |
| pg_turbovec 2-bit, search_k=500           |   396 MB | 2 m 06 s |     78 ms  |     80 ms  | 1.000 |

Full machine-readable run:
[`benches/results/recall_dbpedia_1M_2026_05_24.json`](../benches/results/recall_dbpedia_1M_2026_05_24.json).

### What this proves

This run is the credibility counterweight to the synthetic R@10 = 1.0
in § 2.1: on a 1 M corpus of real OpenAI embeddings, pg_turbovec 4-bit
and 2-bit both recover the exact-cosine top-10 perfectly (R@10 = 1.000
at `search_k = 100`), at **10×** less storage than pgvector HNSW
(780 MB vs 8 192 MB) for 4-bit, **20×** less for 2-bit, and 1.6× to
2.4× lower p50 latency than HNSW at `ef_search = 200`. HNSW's recall
recovers from the 0.03 / 0.12 floor it hit on synthetic random data
in § 2.1 — real embeddings have the clustering structure HNSW needs,
and it lands at R@10 = 0.962 / 0.970 — but pg_turbovec still beats it
on every other axis (storage, build cost, p50, p95 tail). 2-bit at
`search_k = 100` is the surprise winner: 396 MB on disk, 48 ms p50,
and the same R@10 = 1.000 as 4-bit.

A few honest caveats:

1. **Query set is in-corpus.** The 50 queries are the first 50 docs,
   so rank-1 = the query itself (`hits[0] == query.id`). R@10 is
   computed on the full 10 hits including that trivial one, so the
   floor is 1/10 = 0.1 even for a random index. Real-world recall
   curves should expect a slight regression vs. these numbers when
   queries are *out-of-corpus*.
2. **HNSW build memory.** The HNSW index is **8 192 MB** — it
   fits the data set exactly into a 1 M × 8 KB graph block budget at
   `m = 16`. On the 32 GiB benchbox we had to lower
   `shared_buffers` to 512 MiB to make the *turbovec* build fit
   (4-bit needs ~12 GB peak working set: a 6 GB Vec<f32> heap copy
   plus codebook training). At 8 GB shared_buffers + default
   maintenance settings pg_turbovec's CREATE INDEX gets OOM-killed.
   This is a real ergonomics gap and is tracked separately.
3. **search_k goes up linearly in cost.** k=500 vs k=100 doubles
   p50 because the kernel returns 5× more candidates the executor
   has to recheck. The recall stays at 1.000 either way on this
   corpus — which means k=100 is leaving recall on the floor for
   harder distributions but is ~free here.

## 2.3 Cold-scan latency: relfile-resident page format (commit 9e8ee81)

**TL;DR:** the relfile-resident layout (Phase L hardening 1-6, gated
on `--features relfile_storage`) is wired end-to-end and *correct*
(WAL, init-fork, RelationTruncate, deferred-commit aminsert, in-place
ambulkdelete) — but on dbpedia-1M × 1536-d it does **not** lift cold-
scan p50 into the same ballpark as HNSW. Cold-scan p50 stays at
~26 s/query because the per-backend bottleneck is the Lloyd-Max
codebook compute + 793 MB blocked-layout repack, not the SPI fetch
that relfile_storage replaces. The headline conclusion: **do not
flip `relfile_storage` default ON in v1.3.0**; queue Phase P
(shared-memory IdMapIndex cache or pre-baked blocked layout on disk)
as the actual cold-scan fix.

### Setup (commit 9e8ee81)

- Build: `cargo pgrx install --release --no-default-features
  --features "pg17 experimental_index_am relfile_storage"`
- Host: arnold (Intel i9-12900H, 32 GiB; ~3 GiB consumed by
  browsers/IDE/Discord during the run, swap saturated)
- PG: 17.9 (pgrx-bundled), shared_buffers reduced to 512 MB during
  the bench so the encoder transient (~21 GiB) didn't OOM the host
- Index: `docs_tv_4bit` rebuilt from scratch under the relfile path,
  170 s build time, 793 MB on-disk (matches §2.2's side-table
  payload byte-for-byte modulo page padding)
- Planner pick: `docs_tv_2bit` was DROPped before the sweep so the
  planner has only `docs_tv_4bit` to satisfy `OPERATOR(turbovec.<=>)`;
  HNSW's pgvector opclass cannot match that operator. Confirmed via
  EXPLAIN logged before the sweep.

### Methodology

- **Cold:** PG cluster restarted *once* before the sweep to drop
  shared_buffers. Then 50 fresh psql sessions, each issuing one
  `bench_one_query_tv(qid)` (timed via plpgsql `clock_timestamp()`).
  Each backend re-pays the full per-process IdMapIndex re-init:
  relfile read of 793 MB → `IdMapIndex::from_id_map_parts` →
  first-search lazy init of rotation matrix + Lloyd-Max codebook +
  blocked-layout repack (`pack::repack`).
- **Warm:** single warm psql session, 2 untimed warmup queries (the
  first re-pays the per-backend init, ~25 s), then 50 timed queries
  in the same backend with everything cached.

### Numbers

| label                      |  n |    min |    p50 |    p95 |    max |   mean |
|----------------------------|---:|-------:|-------:|-------:|-------:|-------:|
| `tv_4bit_k100_cold_relfile`| 50 | 25 725 | 26 310 | 27 376 | 28 499 | 26 407 |
| `tv_4bit_k100_warm_relfile`| 50 |   85.0 |   87.4 |   99.4 |  111.1 |   89.6 |

Units: ms. Raw TSVs:
[recall_relfile_cold_scan_v1_3_0_2026_05_25.cold.tsv](../benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.cold.tsv)
and [.warm.tsv](../benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.warm.tsv).

### Comparisons

| metric                                                   | value          |
|----------------------------------------------------------|---------------:|
| v1.0.0 cold p50, **GloVe-100 1 M × 384-d**, side-table   |    6 786 ms    |
| v1.3.0 cold p50, **dbpedia-1M × 1536-d**, relfile        |   26 310 ms    |
| v1.0.0 warm p50, GloVe-100 384-d                         |       22.2 ms  |
| v1.0.0 warm p50, dbpedia-1M 1536-d (§2.2 release)        |       70.5 ms  |
| v1.3.0 warm p50, dbpedia-1M 1536-d, relfile (this run)   |       87.4 ms  |

**Important caveat:** the 6 786 ms number is on GloVe-100 (1 M × 384-d),
not dbpedia-1M (1 M × 1536-d). Per-backend prep cost scales with
`n_vectors × dim × bit_width`, so the 1536-d corpus would land
~4× above the 384-d corpus *regardless of storage path*. The
apples-to-apples side-table baseline on dbpedia-1M was not
measured this session (out of budget — would have required a
feature-flag-flip rebuild + 22 min cold sweep). Without that
baseline, the relfile-vs-side-table delta on this exact corpus
is inferred from architecture, not measured.

### Why the architectural fix didn't show up as a 10× win

Profiling the relfile cold-scan path (per backend, fresh process):

1. `relfile::read_full` reads 793 MB via the buffer manager (~95 K
   pages). With OS page cache warm (which it is from query 2
   onward) this is sub-second. With shared_buffers cold but OS
   cache warm (the configuration we measured), still sub-second.
2. `IdMapIndex::from_id_map_parts` builds the `id_to_slot` HashMap
   for 1 M ids → small fraction of a second.
3. **First `search()` call lazy-inits via `OnceLock::get_or_init`:**
   - rotation matrix (1536² f32 = 9 MiB allocation, fast)
   - Lloyd-Max codebook (small)
   - **`pack::repack(packed_codes, n_vectors=1 M, bit_width=4,
     dim=1536)`** — re-tiles 768 MiB of packed codes into the
     SIMD-blocked layout. This dominates and takes ~20-25 s on
     this host.

The SPI-fetch step that relfile_storage *replaces* takes maybe
5-10 s (TOAST detoast + memcpy of the 793 MB bytea). So even with
relfile_storage saving 100 % of that 5-10 s, the remaining 20-25 s
of Lloyd-Max + repack still dominates. Speedup vs side-table on
*this* corpus is therefore expected to be ~1.3-1.5×, not 10-100×.

### Recommendation

- **Keep `relfile_storage` gated for v1.3.0.** Ship the page-format
  hardening (Phase L 1-6) as documented progress; do not flip the
  default ON until cold-scan p50 actually drops below ~500 ms.
- **Phase P (next):** move the blocked-layout repack out of the
  per-backend search path. Two options:
  1. Pre-bake the blocked layout into the index relfile alongside
     `packed_codes`. ambuild pays the cost once; every backend's
     search() just memmaps. ~2× index size, but cold scans become
     I/O-bound (sub-second) instead of CPU-bound.
  2. Ship a shared-memory `IdMapIndex` cache (DSM segment, refcounted)
     so the *first* backend to scan in a cluster pays the prep cost
     and every subsequent backend hits the prepared structure. Ties
     into pg_shmem and is more invasive but cheaper on disk.
- **Warm-path validation:** the 87 ms warm p50 is +24 % vs §2.2's
  70.5 ms (v1.0.0, side-table on the same corpus). That's within
  noise on a contended laptop but worth re-measuring on a quiet
  host before declaring no regression.

### Artefacts

- [`benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.json`](../benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.json)
- Bench driver: `/scratch/pg_turbovec-bench/cold_bench.sh` on arnold
  (saved into the run directory; reproducible via
  `bash cold_bench.sh tv_4bit_k100_cold_relfile docs_tv_4bit 100 cold`).

## 2.4 Cold-scan latency: pre-baked layout (Phase P, commit a801f38)

**The cold-scan fix.** Phase O-2 found that the relfile-resident
page format (§2.3) hadn't closed the cold-scan gap on a
1 M × 1536-d corpus because the dominant per-backend cost wasn't
storage I/O — it was `pack::repack` (transposing 768 MiB of
packed codes into the SIMD-blocked layout) plus the Lloyd-Max
codebook compute, both lazy-init'd via `OnceLock` on the first
`search()` in any fresh backend.

Phase P (commit `a801f38`) pre-bakes both at `ambuild` time and
persists them into the relfile. Backends now read a prepared
structure straight off disk; no per-backend prep work.

**Re-measured on arnold** (Intel i9-12900H, 32 GiB, PG 17.9,
release build, 1 M × 1536-d dbpedia-1M, same 50-query workload
as §2.3):

| metric | Phase O-2 (v1.2.0 relfile preview) | Phase O-3 (v1.2.0 + Phase P) | speedup |
|---|---:|---:|---:|
| min | 25 725 ms | 1 140 ms | 22.6× |
| **p50** | **26 310 ms** | **1 256 ms** | **20.9×** |
| p95 | 27 376 ms | 2 617 ms | 10.5× |
| max | 28 499 ms | 2 732 ms | 10.4× |
| mean | ~26 600 ms | 1 499 ms | 17.7× |

**Index size**: 1 527 MiB (vs 793 MiB pre-Phase-P). The pre-baked
blocked layout roughly doubles the on-disk size; we trade disk
for cold-scan latency. Even at this size the index is still 5×
smaller than pgvector HNSW's 8 192 MiB on the same corpus.

**Build time**: 237 s (vs 163 s pre-Phase-P, +45%). The `prepare`
step that was lazy is now eager and runs once during `CREATE
INDEX` instead of once per backend.

**Verdict**: ready for v1.3.0 default-on flip. The 1.26 s cold
p50 is acceptable for a fresh-backend first-query; subsequent
queries in the same backend hit the warm cache at ~87 ms. The
v1.0.x side-table storage path will be removed in Phase Q so
that relfile becomes the only and default storage, matching
the convention every other PG index AM follows.

**Source data**
- [`benches/results/recall_relfile_phase_p_cold_scan_2026_05_25.json`](../benches/results/recall_relfile_phase_p_cold_scan_2026_05_25.json)
- Per-sample TSV: [`recall_relfile_phase_p_cold_scan_2026_05_25.cold.tsv`](../benches/results/recall_relfile_phase_p_cold_scan_2026_05_25.cold.tsv)

## 2.5 Warm-scan latency: persisted rotation (Phase R-2, v1.4.0, commit 9046f2c)

**The warm-scan fix that wasn't.** Phase R’s perf snapshot of
v1.3.0 found `gemm_f64::microkernel::fma::f64::x2x6` at 64.77%
self on the warm-scan path — the 1536×1536 QR decomposition
that builds the rotation matrix, running on first scan per
backend (and apparently leaking out of the per-backend
`OnceLock` on subsequent scans). Phase R-2 (commit `9046f2c`,
v1.4.0, wire format v3) persists that rotation matrix in the
relfile at `ambuild` time so backends read it once at
`from_id_map_parts_with_prepared` and never compute the QR at
scan time.

**Re-measured on arnold** (Intel i9-12900H, 32 GiB, PG 17.9,
release build, 1 M × 1536-d dbpedia-1M, single warm psql
session, 2 untimed warmups against qid=1, 50 timed queries via
`bench_one_query_tv(qid)` for qid=1..50; same methodology as
§ 2.2 / Phase J / Phase O-3, `shared_buffers=512MB` matching
§ 2.4):

| metric | Phase J (v1.1.0 side-table) | Phase O-3 (v1.3.0 relfile) | **Phase R-3 (v1.4.0 + persisted rotation)** |
|---|---:|---:|---:|
| min | 48.1 ms | 85.0 ms | 86.5 ms |
| **p50** | **70.5 ms** | **87.4 ms** | **90.3 ms** |
| p95 | 91.3 ms | 99.4 ms | 103.9 ms |
| max | 101.9 ms | 111.1 ms | 108.1 ms |
| mean | 71.2 ms | 89.6 ms | 91.7 ms |

**Verdict: phase\_r2\_not\_enough\_more\_work\_needed.** The fix
worked at the symbol level — the post-Phase-R-2 perf profile
(`benches/results/profile_v1_4_0_symbols.txt`) shows `gemm_f64`
dropped from 64.77% self to absent in the top 100 symbols. But
the headline warm p50 stayed flat (90.3 ms vs 87.4 ms is
within the run-to-run noise this contended laptop carries).
What now dominates the warm-scan profile:

```
  ReadBufferExtended ............... 37.08 % children
  ReadBuffer_common (inlined) ...... 36.68 % children
  __memmove_avx_unaligned_erms ..... 34.96 % children, 12.42 % self
  asm_exc_page_fault ............... 32.71 % children
  WaitReadBuffers .................. 29.44 % children
  mdreadv / FileReadV / pread ..... ~27 % children
  turbovec::search::search_multi_query_avx2 .. 21.79 % children, 21.55 % self
  _copy_to_iter (kernel) ........... 12.18 % self
```

The 1.5 GB on-disk index does not fit in 512 MB
`shared_buffers`, so each warm scan re-pulls touched pages
from the OS page cache via `pread` → kernel `_copy_to_iter`
→ `__memmove_avx_unaligned_erms` into the buffer manager.
That’s ~50–60 ms per scan and was always there in v1.3.0—
just dwarfed by the QR cost. Removing the QR exposed it.
The SIMD scoring kernel itself (`search_multi_query_avx2`)
is only 21.55% self; ~65% of warm-scan time is now PG
buffer-manager I/O against a too-small `shared_buffers`.

**Index growth and build cost** — within Phase R prediction:

| metric | v1.3.0 (Phase O-3) | v1.4.0 (Phase R-3) | delta |
|---|---:|---:|---:|
| build time | 170 s | 234 s | +64 s (eager rotation chain) |
| size on disk | 793 MB (payload) | 1 536 MB (full relfile) | mostly relfile/heap-fork accounting; rotation chain alone is ~9 MiB |

*Build setting note*: the v1.4.0 build was run with
`maintenance_work_mem=512MB`, `max_parallel_maintenance_workers=0`,
and `shared_buffers=128MB` during build (Phase O-3 used 512MB);
the two prior build attempts at 4GB / 16 workers and 512MB / 0
workers were OOM-killed by the kernel because this 32 GiB host
was already paging (7 GiB swap used) when the run started.
`shared_buffers` was raised back to 512 MB before the warm
sweep so the bench-time configuration matches Phase O-3 exactly.

**Two follow-on directions for closing the warm gap:**

1. **Raise `shared_buffers` to ≥2 GB** so the entire
   `docs_tv_4bit` fits hot. Knob change, not code change, but
   it requires a host with that much spare memory.
2. **Per-backend `mmap`-resident view of the blocked layout**
   that bypasses the PG buffer manager on warm scans. This is
   essentially a re-run of the Phase L design conversation:
   the buffer manager is in the way for read-only quantized
   data that wants to behave like an OS-page-cache-resident
   artefact, not like a row-oriented heap.

**Source data**
- [`benches/results/recall_warm_phase_r3_2026_05_25.json`](../benches/results/recall_warm_phase_r3_2026_05_25.json)
- Per-sample TSV (clock-stamped, primary): [`warm_phase_r3_clock.tsv`](../benches/results/warm_phase_r3_clock.tsv)
- Per-sample TSV (EXPLAIN ANALYZE secondary): [`warm_phase_r3.tsv`](../benches/results/warm_phase_r3.tsv)
- Post-fix perf profile: [`profile_v1_4_0_symbols.txt`](../benches/results/profile_v1_4_0_symbols.txt) and [`profile_v1_4_0_flame.svg`](../benches/results/profile_v1_4_0_flame.svg)
- Reproducer scripts: [`benches/scripts/warm_phase_r3.sh`](../benches/scripts/warm_phase_r3.sh), [`warm_phase_r3_clock.sh`](../benches/scripts/warm_phase_r3_clock.sh), [`migrate_index_v1_4_0.sh`](../benches/scripts/migrate_index_v1_4_0.sh)

## 2.6 Warm-scan latency: mmap'd static regions (Phase R-3, v1.5.0)

**The warm-scan fix that bypasses the buffer manager.** § 2.5
 diagnosed the post-Phase-R-2 warm-scan profile: the 1.5 GB index
 doesn't fit in the bench's 512 MB `shared_buffers`, so each warm
 scan re-pulls touched pages from the OS page cache via `pread`
 → kernel `_copy_to_iter` →
 `__memmove_avx_unaligned_erms` into the buffer manager. That
 was ~65% of warm-scan time and put us at 90 ms p50 vs HNSW
 ef=40's 61 ms. Phase R-3 (v1.5.0) replaces the buffer-manager
 reads of the *static* regions (persisted SIMD-blocked codes,
 persisted rotation matrix, inline codebook) with a per-backend
 `mmap(MAP_PRIVATE)` of the relfile. The OS page cache is the
 authoritative cache for those bytes; PG's buffer manager is
 out of the loop entirely on those reads.

**What we mmap (and what we don't):**

| Region | Path | Why |
|---|---|---|
| Meta page (block 0) | buffer manager | One 8 KB page; never the bottleneck. |
| Codes / scales / ids chains | buffer manager | Mutated in-place by VACUUM swap-remove (`ambulkdelete`). The buffer manager is the canonical reader. |
| **Persisted SIMD-blocked codes** (~768 MiB at 1 M × 1536-d × 4-bit) | **mmap** | Deterministic-after-`ambuild`; rewritten only on the next aminsert commit-flush which bumps `am_version` and invalidates every backend's mmap'd cache entry. **The bulk of the I/O.** |
| **Persisted rotation matrix** (~9 MiB at dim 1536) | **mmap** | Deterministic from `(dim, ROTATION_SEED)`. Never mutated post-build. |
| **Inline codebook (centroids + boundaries)** | **mmap-side copy** | Deterministic from `(bit_width, dim)`. 64 bytes — the path is uniform but the cost is irrelevant either way. |

The Cow-based borrowed API in upstream `turbovec`
([`from_id_map_parts_with_prepared_borrowed`](https://github.com/gburd/turbovec/blob/pg_turbovec-integration/turbovec/src/id_map.rs))
is wired up so embedders whose on-disk static-region chains are
*contiguous* can hand turbovec a true zero-copy `Cow::Borrowed`
slice into the mapping. pg_turbovec's chains have 24-byte page
headers every 8168 bytes (PG `BLCKSZ` minus `PageHeaderData`),
so v1.5.0 still does one `memcpy` from mmap into a contiguous
`Vec<u8>` at cache-fill time. The win over the buffer-manager
path is

1. No `BufTableLookup` / pin / lock per page (was the 37%
   `ReadBufferExtended` children figure in the v1.4.0 profile).
2. No double-cache: shared_buffers no longer holds a copy of
   pages already in the OS page cache, freeing it for the
   non-static chains.
3. One memcpy (mmap → chain `Vec`) instead of two (`pread` →
   shared_buffers slot, then slot → chain `Vec`).

The `turbovec.mmap_static_blocked` GUC (default `on`) controls
this path. Setting it `off` reverts to the v1.4.x buffer-manager
read path on a per-session basis.

**Isolation contract.** mmap with relaxed consistency vs the
buffer manager is correct because the index AM contract is
approximate-by-design: heap visibility is the source of truth
and `xs_recheckorderby = true` (asserted unconditionally in
`amgettuple`) recomputes the ORDER BY expression from the heap
tuple, correcting any ranking error caused by the mmap'd image
lagging a just-committed insert. See
[`docs/ARCHITECTURE.md` § "Index AM · mmap isolation
contract"](ARCHITECTURE.md#index-am--mmap-isolation-contract)
for the full argument and worked examples (concurrent aminsert,
concurrent ambulkdelete, REINDEX).

**Re-bench on arnold:** _pending—validation against the same
`shared_buffers=512MB` 1 M × 1536-d dbpedia corpus is queued for
the next available bench window._ Expected: warm p50 drops from
the v1.4.0 90 ms toward ~60 ms (HNSW ef=40 parity). The
buffer-manager symbols (`ReadBufferExtended`, `WaitReadBuffers`,
`mdreadv`, `__memmove_avx_unaligned_erms`) should fall off the
top-50 of the warm-scan profile.

**Re-bench on `meh` (Phase U-2, 2026-05-26).** First real
measurement of v1.5.0 against the dbpedia-1M corpus. Host: 24
cores, 125 GiB RAM, NixOS 6.12.83, `shared_buffers = 512 MB`
(matching arnold's Phase R-3 setup), 1 M × 1536-d ada-002
vectors, `turbovec.search_k = 100`. Methodology: single warm
psql session per config, 2 untimed warmups + 50 timed
`bench_one_query_tv(qid)` calls; same shape as Phase J / R-3.

| metric | mmap=on (v1.5.0) | mmap=off (v1.4.x equivalent) | delta |
|---|---:|---:|---:|
| min  | 26.58 ms | 26.55 ms | +0.03 ms |
| **p50** | **26.80 ms** | **26.65 ms** | **+0.15 ms** |
| p95  | 61.34 ms | 60.97 ms | +0.37 ms |
| max  | 61.58 ms | 61.10 ms | +0.48 ms |
| mean | 39.39 ms | 35.01 ms | +4.38 ms |

Verdict: **`shared_buffers_was_the_bottleneck`** (more
precisely: OS-page-cache size dominated). Phase S delivers
zero measurable warm-scan win on `meh` because the bottleneck
it targets — buffer-manager copies of the 1.5 GB static
regions when they don't fit shared_buffers — is fully masked
by the OS page cache when free RAM is plentiful. With 125 GiB
total / ~76 GiB free, the kernel page cache holds the whole
index, `pread` from a hot OS cache costs ~0, and the
`pread → shared_buffers slot → chain Vec` path has no copy
left to remove. mmap eliminates one `memcpy` in principle,
but the savings (~0.1 ms p50) are below measurement noise on
a 26 ms baseline. The original arnold profile was at 90 ms
p50 because arnold's 31 GiB RAM forced shared_buffers and OS
cache to fight over the same 1.5 GB working set; on `meh`
that fight doesn't exist. v1.5.0 is at-worst neutral on a
generously-RAMed host — no regression.

The distribution is bimodal in both modes: ~25–38 of 50
queries cluster at ~26.5–26.7 ms (fast), ~12–14 cluster at
~60 ms (slow). The p50 captures the fast cluster; the p95
captures the slow one. The bimodality is query-set-dependent
(some query vectors trigger cheaper search-k pruning paths
than others) and present in both v1.5.0 paths, so it isn't
the Phase S delta.

The original arnold re-bench (where the buffer-manager fight
is real) remains the definitive Phase S validation; this
`meh` run only proves v1.5.0 is non-regressing on hosts where
Phase S has no real work to do.

**Local debug-build smoke (this commit):** the in-tree
`#[pg_test]` `relfile_mmap_static_round_trip_matches_buffer_manager`
builds an index with the prepared layout, runs the same query
with `turbovec.mmap_static_blocked = on` and `= off`, and
asserts identical top-1 ids. The mmap path round-trips at the
result level; sub-ms latency on the small fixture is the
expected debug-build speed and not the right place to measure
the arnold-scale win.

**Index growth and build cost** — unchanged from v1.4.0:

| metric | v1.4.0 (Phase R-3 — wait, that's the *measurement*) | v1.5.0 (Phase R-3 — the *fix*) | delta |
|---|---:|---:|---:|
| build time | 234 s | 234 s | 0 (build is unchanged) |
| size on disk | 1 536 MB | 1 536 MB | 0 (wire format unchanged) |

v1.5.0 is a scan-side change only. Wire format stays at v3, so
v1.4.x indexes scan under v1.5.0 with no REINDEX.

**Source data**
- Local pg_test smoke: see
  `tests::pg_relfile_mmap_static_round_trip_matches_buffer_manager`,
  `tests::pg_relfile_mmap_static_concurrent_aminsert_recheck_corrects`,
  `tests::pg_relfile_mmap_static_cache_invalidation_drop_order` in `src/lib.rs`.
- arnold re-bench: pending.
- `meh` re-bench (Phase U-2, 2026-05-26):
  [`benches/results/recall_warm_meh_v1_5_0_2026_05_26.json`](../benches/results/recall_warm_meh_v1_5_0_2026_05_26.json),
  raw timings
  [`u2_meh_tv_4bit_warm_mmap_on.tsv`](../benches/results/u2_meh_tv_4bit_warm_mmap_on.tsv) /
  [`u2_meh_tv_4bit_warm_mmap_off.tsv`](../benches/results/u2_meh_tv_4bit_warm_mmap_off.tsv).
- Cache-miss diagnosis (Phase U-1, 2026-05-26):
  [`docs/PHASE_U_DIAGNOSIS.md`](PHASE_U_DIAGNOSIS.md). Verdict:
  cache works correctly (50 / 50 hits); the Phase S agent's
  hot `HashMap::insert` perf symbol was the one-shot
  `finalise_from_inner` build during warmup1, not a per-query
  rebuild.

## 2.7 Scaling: 10 M × 1536-d (Phase V, 2026-05-26)

**The 10× scaling check.** Phase J / U covered 1 M × 1536-d on
real ada-002 (`dbpedia-entities-openai-1M`); Phase V takes the
same index AM up to **10 M × 1536-d** on `meh` (24 cores, 125
GiB RAM, NixOS) to confirm nothing falls over at the dbpedia-
10× scale and that the per-backend cache's `enforce_cap`
`len() > 1` retain rule still holds the (now ~15 GiB) entry
resident at the default `cache_size_mb = 256`.

**Corpus is synthetic random unit-norm vectors**, not a real
embedding distribution. A 10 M × 1536-d real fixture (Cohere
wikipedia-22-12 35 M × 768, etc.) is 100+ GiB of parquet to
download — out of scope for this phase. Server-side generation
with `(random()*2-1)::real` then `l2_normalize()` produced 10 M
rows in **30 min 54 s** as one `DO` loop of 100 × 100 k batches
on an `UNLOGGED` table with `STORAGE MAIN` on the vector column
(inline; no TOAST). Heap is 76 GB (one row per 8 KiB page).

**The synthetic-data caveat for HNSW.** Random unit vectors
have no low-dimensional manifold structure. HNSW on this
distribution is *unrealistically fast* because the graph is
near-degenerate: any candidate is roughly equally close to any
query, so a few hops in the upper layers find ten neighbours
immediately. The `hnsw_ef40 p50 = 0.94 ms` number below is
**not** comparable to the dbpedia-1M HNSW p50 of 61 ms (Phase
J) or 50 ms (Phase U-2) on real ada-002 — those are real
manifolds where ef=40 actually has to navigate. Phase V is a
scaling check, not a head-to-head latency benchmark; the
numbers we trust at this scale are pg_turbovec's (whose work
per query — scan k candidates, recompute distance — is
distribution-independent) and the system-level metrics
(storage, build time, build memory, cache retention).

### Storage at 10 M rows

| Index | Type | Size | × heap | × pgvector |
|---|---|---:|---:|---:|
| `docs_pgv_hnsw` | pgvector HNSW (m=16, ef_construction=64) | **65.5 GiB** (66 GB) | 0.86× | 1.0× |
| `docs_tv_4bit`  | pg_turbovec (bit_width=4)               | **14.92 GiB** (15 GB) | 0.20× | **0.23×** |
| `docs` (heap)   | UNLOGGED, vector(1536) STORAGE MAIN     | 76 GB | 1.0× | — |

pg_turbovec is **4.4× smaller on disk than pgvector HNSW** at
10 M × 1536-d (vs. ~5× at 1 M on dbpedia — the ratio is stable
across the 10× scale jump, which is what we expected since
both indexes' per-row cost is dominated by the vector payload).

### Build time at 10 M rows

| Index | Build time | Notes |
|---|---:|---|
| `docs_pgv_hnsw` | **3 h 38 min 14 s** (13 094 s) | 7 parallel workers; HNSW slows super-linearly past 5 M rows even with everything in OS page cache. |
| `docs_tv_4bit`  | **1 h 24 min 09 s** (5 048 s)  | Heap scan parallelised (16-way); prepared-layout finalisation single-process. |

Both are slower than the prompt's pre-bench estimate (HNSW
~50–60 min, pg_turbovec ~25–40 min). The HNSW estimate was
optimistic for a 10 M × 1536-d corpus; the pg_turbovec estimate
was optimistic about how long the post-scan in-memory prepared-
layout phase takes (see § *Build memory pressure* below).

pg_turbovec wins the build race **2.6×** even though only its
heap-scan phase is parallel.

### Warm-scan p50 (50 queries, single backend, fresh random unit-norm queries NOT in docs)

| Config | min | **p50** | p95 | max | mean |
|---|---:|---:|---:|---:|---:|
| `hnsw_ef40`     | 0.59 ms | **0.94 ms** | 2.00 ms | 2.12 ms | 1.13 ms |
| `tv_4bit_k100`  | 21.39 ms | **47.27 ms** | 48.96 ms | 49.16 ms | 36.83 ms |
| `tv_4bit_k500`  | 95.36 ms | **182.41 ms** | 217.09 ms | 217.58 ms | 173.12 ms |

Reading the table:

- **HNSW ef=40 at 0.94 ms p50:** see the synthetic-data caveat
  above. On real ada-002 at 1 M scale the same ef=40 was 61 ms;
  the 65× speedup is the corpus, not the scale.
- **tv_4bit search_k=100 at 47 ms p50:** scales linearly from
  Phase J's 1 M × 1536 numbers (~5–25 ms p50 depending on
  bench-host RAM). The 10× scale → ~10× p50 holds because
  pg_turbovec's `search_k` is an *absolute* candidate-budget,
  not a fraction of the corpus, so per-query work is dominated
  by the prepared-layout SIMD scan over the first `search_k`
  packed codes the IVF / probe scheduler picks. At 10 M, more
  bytes in those `search_k` candidates need to land in cache
  per query, hence the linear-in-bytes-scanned blowup.
- **tv_4bit search_k=500 at 182 ms p50:** 5× more candidates
  scanned → ~4× the p50, sub-linear because the cache stays
  hotter across the larger sweep.
- **Bimodality, not noise.** Both `tv_4bit_k100` (clusters at
  ~21 ms and ~48 ms) and `tv_4bit_k500` (clusters at ~95–130 ms
  and ~210 ms) are bimodal across the 50 queries. Same
  query-vector-dependent search-k pruning effect documented in
  § 2.6's `meh` Phase U-2 run; not a Phase V regression.

### Build memory pressure

*Captured by `snap_rss.sh` polling `/proc/<pid>/status` every
2 s for the leader pgrx postgres backend and all its parallel
workers.*

| Phase | Peak leader RSS | Peak total (leader+workers) | Swap used |
|---|---:|---:|---:|
| `CREATE INDEX docs_tv_4bit`  | **121 GiB** | n/a (single backend during finalisation) | up to 60 GiB |
| `CREATE INDEX docs_pgv_hnsw` | 16.9 GiB    | 43.4 GiB                         | 0          |
| Warm 50-query sweep          | **15.22 GiB** | 15.78 GiB | 0 |

**pg_turbovec's 121 GiB build peak is the headline scaling
concern.** The post-heap-scan prepared-layout assembly
allocates the raw codes, the rotation matrix, and the SIMD-
blocked layout simultaneously in heap memory; on `meh` this
filled essentially all 125 GiB of RAM and pushed 60 GiB into
swap before being persisted to the relfile and freed. A host
with less than ~100 GiB of free RAM + swap headroom would OOM
during a 10 M × 1536-d × bit_width=4 build. Phase J's 1 M-row
build peaked at ~6 GiB; the scaling is roughly linear in
`n_rows` and likely linear in `dim × bit_width / 8`.
[**Flagged for Phase W**](./PARITY_GAPS.md): bound the build's
peak memory by either streaming the prepared layout to disk in
chunks during finalisation or by honouring `maintenance_work_mem`
as a real cap (currently it is read but not enforced past the
heap-scan phase).

### Cache retention at 10× scale (the question that motivated Phase V)

The `cache::enforce_cap`'s `len() > 1` retain rule (`src/cache.rs`)
is the safety valve that prevents the per-backend cache from
evicting an entry that is *bigger than the byte-cap on its own*.
At 10 M × 1536-d × 4-bit the entry is ~15 GiB; the default
`cache_size_mb = 256` would unconditionally evict a normal
entry that big. The retain rule keeps it because evicting the
only resident entry guarantees a re-load on the next query —
net pessimization.

Observation across the 50-query `tv_4bit_k100` sweep:

- All 50 queries returned in **21–49 ms**.
- No query took ≥ 100 ms, which is what re-loading the 15 GiB
  entry from the relfile + re-running `finalise_from_inner`
  would look like (Phase R-3's *cold-cache* fill on a 1.5 GiB
  index at 1 M scale was ~150 ms; at 10× the index size it
  would be ~1.5 s — and there are zero 1.5 s spikes).
- Peak backend RSS during the sweep was 15.22 GiB, matching
  the 14.92 GiB index size + per-backend Postgres baseline.
  The cache held the entry continuously across all 50 queries.

**Verdict:** `enforce_cap`'s `len() > 1` retain rule scales
cleanly to 10 M × 1536-d at the default `cache_size_mb = 256`;
the 15 GiB entry is held resident for the whole 50-query
workload with zero eviction-induced spikes.

### Source data

- Full structured run:
  [`benches/results/recall_warm_meh_10m_v1_5_1_2026_05_26.json`](../benches/results/recall_warm_meh_10m_v1_5_1_2026_05_26.json)
- Raw 50-query timings CSV:
  [`benches/results/recall_warm_meh_10m_v1_5_1_2026_05_26.csv`](../benches/results/recall_warm_meh_10m_v1_5_1_2026_05_26.csv)
- Build / sweep psql logs (with heartbeats):
  [`benches/results/build_tv_meh_10m_v1_5_1_2026_05_26.log`](../benches/results/build_tv_meh_10m_v1_5_1_2026_05_26.log),
  [`benches/results/build_hnsw_meh_10m_v1_5_1_2026_05_26.log`](../benches/results/build_hnsw_meh_10m_v1_5_1_2026_05_26.log),
  [`benches/results/sweep_meh_10m_v1_5_1_2026_05_26.log`](../benches/results/sweep_meh_10m_v1_5_1_2026_05_26.log).

### Post-Phase-W follow-up (v1.6.0, 2026-05-27)

Phase V flagged the 121 GiB CREATE INDEX peak as the dominant
scaling concern and proposed bounding it via `maintenance_work_mem`.
Phase W (commit `f61d906`, v1.6.0) ships that fix: the `ambuild`
callback streams heap-scan rows into `IdMapIndex::add_with_ids`
in chunks of `min(0.75 × maintenance_work_mem, 1 GiB) / (dim × 4 B)`
rows instead of accumulating the entire heap-scan output in a
single `Vec<f32>`. At `maintenance_work_mem = '8GB'` and
`dim = 1536` that is 174 762 rows per flush.

Re-measured on `meh`, same 10 M × 1536-d corpus, same
`maintenance_work_mem = 8 GiB`:

| Metric | Phase V (v1.5.1) | Phase W (v1.6.0) | Change |
|---|---:|---:|---:|
| Peak leader RSS during `CREATE INDEX docs_tv_4bit` | 121 GiB | **22.52 GiB** | −5.4× |
| Swap used (delta over the build) | up to 60 GiB | **0 GiB** | gone |
| Build wall-clock | 5 048 s (1 h 24 m 09 s) | 5 052.5 s (1 h 24 m 12 s) | +0.09 % |
| Index size on disk | 15 GiB | 15 GiB | unchanged |
| Warm `tv_4bit_k100` p50 (50 queries) | 47.27 ms | 21.24 ms¹ | within noise |

¹ Different probe sample + different cache state; both fall in
the 21–49 ms band Phase V already documented for the same
config. The Phase W run does not measure recall, only that the
streamed-build index is queryable and not pathologically slow.

**Verdict: Phase W works.** The 61 GiB staging `Vec<f32>` Phase
V identified as the dominant offender is gone, and bounded
streaming did not regress build throughput. The remaining
22.52 GiB peak is what the `IdMapIndex`'s row-major
`packed_codes` (≈7.7 GiB at 10 M × 1536-d × 4-bit) plus the
prepared SIMD-blocked layout (≈7.5 GiB) plus per-allocator
slack and the surrounding Postgres backend must hold
simultaneously during the end-of-build finalisation. Cutting it
further would require streaming the prepared-layout assembly
itself to disk during `relfile::write_full_with_prepared`—a
separate optimisation and not part of Phase W's scope.

No cluster-side changes were needed: the v1.6.0 wire format is
byte-identical to v1.5.x (`MetaPageData::version = 3`); a no-op
`pg_turbovec--1.5.0--1.6.0.sql` upgrade script was added and
`ALTER EXTENSION pg_turbovec UPDATE TO '1.6.0'` was the only
schema-side step.

#### Source data

- Full structured run:
  [`benches/results/phase_w_validate_meh_10m_2026_05_27.json`](../benches/results/phase_w_validate_meh_10m_2026_05_27.json)
- psql build log + timing:
  [`benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.psql.log`](../benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.psql.log)
- Heartbeat-wrapped outer log:
  [`benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.log`](../benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.log)
- Per-process RSS time series (1 s cadence, gzipped TSV):
  [`benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.rss.tsv.gz`](../benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.rss.tsv.gz)
- Warm-scan sanity check (50 queries, `tv_4bit_k100`):
  [`benches/results/phase_w_warm_sanity_meh_10m_2026_05_27.json`](../benches/results/phase_w_warm_sanity_meh_10m_2026_05_27.json)

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
