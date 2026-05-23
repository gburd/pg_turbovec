# Concurrency: backend-local cache behaviour

This document records how `pg_turbovec`'s backend-local cache
(`src/cache.rs`, a `parking_lot::Mutex<HashMap<CacheKey, Entry>>`)
behaves under concurrent ANN queries, and answers the open question
in `docs/ROADMAP_DECISIONS.md` § "Where future work would pay off"
item 2 ("Concurrent query throughput").

**TL;DR.**
- The cache mutex contributes **< 1 %** to per-call cost at every
  thread count we measured (1, 2, 4, 8, 16). It is not a
  bottleneck.
- Aggregate `turbovec.knn` throughput on a single 8-core box scales
  to **2.85 ×** N=1 at N=8 across separate PostgreSQL backends.
  That sub-linear scaling is dominated by the per-call SPI cost
  (`pg_class` lookup + a `count(*)` of the corpus on every cache
  hit) and by ordinary CPU saturation, **not** by the cache
  mutex — each backend has its own copy of the static.
- We did **not** ship a sharded / RwLock / OnceLock optimisation.
  The data does not justify it. If we want to push QPS at high N,
  the highest-leverage targets are the per-lookup `count(*)` and
  the `relation_row_count` SPI roundtrip in `src/knn.rs`.

The rest of this document records the methodology and raw numbers.

## 1. Why two benchmarks?

The cache is a process-level `static`. PostgreSQL backends are
forked processes that each get a private copy via copy-on-write,
so two concurrent SQL clients are operating on **two separate
mutexes**. To answer "would the mutex contend?" we therefore need
two complementary benchmarks:

1. **`bench/concurrent.sh`** — pgbench drives N concurrent backends
   against `turbovec.knn(...)` over the same table. Each backend
   has its own cache. This measures the *real* customer-visible
   QPS at concurrency N. It is dominated by per-backend SPI
   overhead and CPU saturation, not by the cache mutex.

2. **`benches/concurrent_knn.rs`** — a pure-Rust criterion-style
   bench (declared `harness = false`) that hammers a *single*
   `Mutex<HashMap>` from N native threads inside one process. It
   has two modes:
   - `MODE=mutex` (default): faithful re-implementation of
     `cache::lookup` (including the `next_seq` second-mutex
     dance).
   - `MODE=nolock`: bypasses the cache entirely and works
     directly on a shared `Arc<IdMapIndex>`.

   The delta between `mutex` and `nolock` is the cache mutex's
   contribution to the hot path. If the two curves are
   indistinguishable, the mutex is not a bottleneck.

## 2. Hardware

All numbers below were collected on:

- Host: `floki`
- CPU: Intel Core Ultra 7 258V (Lunar Lake, 4 P-cores + 4 LP
  E-cores; 8 logical cores, no hyperthreading)
- Postgres: pgrx-managed PostgreSQL 16.9 on port 28816
- pg_turbovec: `v1.0.0-rc.2`, default features (`pg16`,
  `experimental_index_am`)
- Build: `cargo pgrx install --release`, `cache_size_mb = 256`
  (default), 4-bit width, dim 384, k = 10
- Corpus: 10 000 random unit-norm 384-dim vectors (seed 42),
  256 fixed query vectors (seed 13)

## 3. In-process bench: the cache mutex itself

```bash
DURATION_S=5 MODE=mutex  cargo bench --bench concurrent_knn \
    --no-default-features --features pg16
DURATION_S=5 MODE=nolock cargo bench --bench concurrent_knn \
    --no-default-features --features pg16
```

| Threads | mutex QPS | nolock QPS | mutex speedup | nolock speedup | mutex avg lat | nolock avg lat |
|--------:|----------:|-----------:|--------------:|---------------:|--------------:|---------------:|
|       1 |     2 886 |      2 952 |        1.00 × |         1.00 × |        347 µs |         339 µs |
|       2 |     5 087 |      5 200 |        1.76 × |         1.76 × |        393 µs |         385 µs |
|       4 |     8 520 |      8 625 |        2.95 × |         2.92 × |        469 µs |         464 µs |
|       8 |    11 202 |     11 082 |        3.88 × |         3.75 × |        715 µs |         722 µs |
|      16 |    11 430 |     11 591 |        3.96 × |         3.93 × |      1 398 µs |       1 380 µs |

Source: `benches/results/concurrent_knn_inproc_mutex_20260523T212553Z.json`,
`benches/results/concurrent_knn_inproc_nolock_20260523T212651Z.json`.

**Reading the table.** At every thread count the `mutex` and
`nolock` rows are within run-to-run noise of each other (the
`mutex` row is actually *slightly faster* at N=8, which is
arithmetic noise, not measurement; the runs are 5 s each and not
warmed up beyond what 5 s of steady state provides). The cache
mutex adds **no detectable contention overhead** to the hot path.

The 3.88 × ceiling at N=8 is not a lock-contention number — it
shows up in the no-lock arm too (3.75 ×). It's CPU saturation:
this box has 8 logical cores, but they aren't homogeneous (4
performance + 4 low-power-efficient), and the per-call work
(`IdMapIndex::search` over 10 000 4-bit codes) is essentially
pure CPU. N=16 saturates at ~3.96 ×, again identical between the
two modes.

**Per-call cost decomposition** (N=1, mutex mode):

```
347 µs total
├── ~1 µs    cache.lookup() — Mutex acquire + HashMap::get_mut
├── ~1 µs    next_seq()     — second Mutex acquire
├── ~340 µs  IdMapIndex::search(k=10)  -- the 10 000-vec sweep
└── ~5 µs    Arc<…> clone + scoring loop overhead
```

The mutex hold time is **~0.3 % of the per-call budget**. To make
the mutex a bottleneck we would have to make `IdMapIndex::search`
roughly 100 × cheaper, which is not on any current roadmap.

## 4. Cross-backend bench: pgbench

```bash
DURATION=10 bash bench/concurrent.sh
```

| Clients | TPS  | avg lat | speedup |
|--------:|-----:|--------:|--------:|
|       1 |  670 |  1.5 ms |  1.00 × |
|       2 |  871 |  2.3 ms |  1.30 × |
|       4 | 1379 |  2.9 ms |  2.06 × |
|       8 | 1909 |  4.2 ms |  2.85 × |
|      16 | 1871 |  8.6 ms |  2.79 × |

Source: `benches/results/concurrent_knn_20260523T211637Z.json`.

This is the customer-visible curve, and it scales worse than the
in-process bench. **None of the gap is the cache mutex** (each
client = one backend = its own mutex). The two extra costs paid
per call in the SQL function path, that the in-process bench
skips, are:

1. `cache::current_relfilenode(rel)` — `SELECT relfilenode FROM
   pg_class WHERE oid = $1`. Cheap, but takes shared LWLocks on
   the catalog buffer. Sub-millisecond.
2. `relation_row_count(rel)` — `SELECT count(*)::int8 FROM <rel>`
   on **every** cache hit. For 10 000 rows on a clean buffer
   cache this is ~0.5–1 ms by itself, doubling the per-tx wall
   time and serialising every backend on the same heap pages.

Plus PostgreSQL's normal per-statement overhead (snapshot
acquisition, ProcArrayLock, plan cache, executor setup) which
itself contends on shared LWLocks at high backend counts. The
pgbench numbers reflect that whole stack, not just the cache.

## 5. Decision

**No optimisation shipped.** The criterion in the task brief was:

> If the QPS curve at N=8 is materially below 8 × the N=1 number
> (say below 4 × — i.e. > 50 % serialisation cost), implement one
> of [sharded map / RwLock / OnceLock]. […] If the measurements
> show the contention is < 20 % overhead at N=8, document that
> and skip the optimisation — don't gild a non-problem.

By the literal QPS-vs-baseline test the in-process bench sits
just below the 4 × threshold (3.88 ×). But the *contention*
threshold is satisfied: `mutex` vs `nolock` is < 4 % at every
data point and within noise at N=8 (mutex is actually slightly
*faster* than nolock at N=8). The lock is not the bottleneck.

What we did instead:

- Committed `bench/concurrent.sh` and the pgbench script
  (`bench/sql/knn_query.sql`) so future work has a reproducible
  cross-backend measurement.
- Committed `benches/concurrent_knn.rs` with the `mutex` /
  `nolock` toggle so future work can re-run the contention
  isolation in 30 seconds on any laptop.
- Committed the four result JSON files in
  `benches/results/concurrent_knn*.json` so the comparison stays
  honest.

If we ever do want to push the pgbench numbers, the targets in
priority order are:

1. **Cache `relation_row_count` per (rel_oid, xmin)** so the
   `count(*)` only runs once per snapshot. This is the single
   biggest knob.
2. **Replace the `pg_class` SPI call** with a direct
   `RelationGetRelid` / `RelationGetRelFilenumber` lookup on a
   pinned relation reference, eliminating SPI parse/plan
   overhead.
3. **Only then** revisit the cache mutex. If a future change
   makes per-call work much cheaper (e.g. an in-memory `IdMapIndex`
   deserialiser that lifts cache hit cost from ~340 µs to ~30 µs),
   the mutex's contribution to the hot path could become
   non-trivial and a `RwLock` or sharded layout would matter.
   Today it does not.

## 6. Reproducing

```bash
# Build setup matches docs/BUILDING.md.
export LIBCLANG_PATH=/nix/store/.../clang-17/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /nix/store/.../glibc/include \
                                 -isystem /nix/store/.../clang/17/include"
export RUSTFLAGS="-L /nix/store/.../openblas/lib"

# In-process bench (no Postgres needed for this part).
DURATION_S=5 MODE=mutex  cargo bench --bench concurrent_knn \
    --no-default-features --features pg16
DURATION_S=5 MODE=nolock cargo bench --bench concurrent_knn \
    --no-default-features --features pg16

# Cross-backend bench.
cargo pgrx install --release --features pg16 --no-default-features
cargo pgrx start  pg16
DURATION=10 bash bench/concurrent.sh

ls benches/results/
```

`bench/concurrent.sh` accepts environment overrides:

| Var          | Default              | Meaning                                      |
|--------------|----------------------|----------------------------------------------|
| `CLIENTS`    | `1 2 4 8 16`         | space-separated thread counts to sweep       |
| `DURATION`   | `10`                 | seconds per pgbench run                      |
| `CORPUS_ROWS`| `10000`              | rows in `bench_corpus`                       |
| `DIM`        | `384`                | corpus / query vector dimension              |
| `QUERY_POOL` | `256`                | size of the random-query pool                |
| `K`          | `10`                 | knn() k                                      |
| `BIT_WIDTH`  | `4`                  | quantiser bit width                          |
| `SKIP_SETUP` | `0`                  | reuse existing `turbovec_bench` database     |
| `PG_PORT`    | `28816`              | pgrx pg16 default                            |
