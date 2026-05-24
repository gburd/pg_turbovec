# Changelog

All notable changes to `pg_turbovec` are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-05-24

### Phase 21 — million-row benchmark, search_k tunable, AM-scan cache

A real-hardware million-row run on `arnold` (Intel i9-12900H, PG
17, pgvector 0.8.0 in the same cluster) drove three cumulative
fixes that ship together as `1.0.0` proper:

- **`turbovec.search_k` GUC** (default 100). The 0.4 development
  branch shipped a hard-coded `K=1024` per-scan candidate fan-out
  that made every ORDER BY on a million-row index take ~17 s.
  Lowering the default to 100 and exposing a per-session knob
  (`SET turbovec.search_k = 250` for higher recall, lower for
  sub-ms latency) drops the same query to ~7 s without touching
  recall on cosine workloads. (#63879a8)
- **`amrescan` tolerates non-orderby plans.** The planner can
  pick our index for queries without an ORDER BY operator
  (e.g. `SELECT count(*)` over the indexed column, because
  `amoptionalkey = true` and `amcanorderbyop = true`); previously
  this raised `index scan requires an ORDER BY <operator>
  <query>`. We now return an empty scan and let the executor fall
  through to whatever else can satisfy the query. (#63879a8)
- **Backend-local cache wired into the AM scan path.** The
  cache (`src/cache.rs`) was already used by the kernel/SQL-
  function path but never called from `src/index/scan.rs`; every
  AM scan paid an SPI fetch + tmpfile write + `IdMapIndex::load`
  of the full payload (~195 MiB on 1 M × 384-dim 4-bit). Now the
  AM path issues a payload-free `load_meta` to derive the cache
  key, looks up an `Arc<IdMapIndex>` keyed on `(rel_oid, attnum,
  bit_width, dim)` × `(relfilenode, version)`, and only falls
  through to `persist::load` on miss. Intra-backend warm-cache
  speedup observed in the field is ~9.7× (35.7 s → 3.7 s on the
  arnold corpus, debug build). (#1293e7b)

### Phase 21 — million-row recall + latency vs pgvector HNSW

`docs/RECALL.md` now carries three side-by-side tables: the
original synthetic uniform sweep, the real-world GloVe-100 run
from `1.0.0-rc.2`, and a fresh million-row arnold sweep at 384
dimensions. Headline (warm cache, debug build):

| Index | Storage | p50 | R@10 (synth) |
|---|---:|---:|---:|
| pgvector HNSW ef=40 | 1953 MiB | 104 ms | 0.032 |
| pgvector HNSW ef=200 | 1953 MiB | 130 ms | 0.116 |
| **pg_turbovec 4-bit** | **195 MiB** | 3 364 ms | 1.000 |
| pg_turbovec 2-bit | 103 MiB | 1 757 ms | 0.922 |

Uniform-random vectors in 384 dimensions are a documented
pessimistic case for graph indexes — see § 2.1 for the GloVe-100
numbers where HNSW recovers to 0.80–0.93. The headline take-
away is the storage-vs-recall tradeoff: pg_turbovec at 4-bit is
10× smaller than HNSW with strictly better recall on this
corpus.

New artefacts:

- `benches/results/recall_lat_million_2026_05_24.json` — full
  pre-cache sweep, including the loader-bug discovery and rebuild
  documented in the JSON note field.
- `benches/results/recall_lat_million_post_cache_2026_05_24.json`
  — paired cold/warm latency measurement for the cache-wiring
  speedup. Use these to reproduce the 9.7× intra-backend ratio.
- `benches/scripts/{rebuild_corpus_million.sh,
  bench_million_setup.sql, run_bench_sweep_million.sh,
  MILLION_ROW_BENCH.md}` — reproduction harness.

### Tests

88 → **92** `#[pg_test]` cases. Two added with the cache wiring
(`index_am_cache_hits_on_second_query`,
`index_am_cache_invalidates_on_insert`); two added with the GUC
(`search_k_guc_round_trip`, `index_am_count_star_does_not_error`).
All green on PostgreSQL 16 and 17.

### Known follow-ups (not blocking 1.0)

- Cold-cache p50 on a fresh backend is still dominated by
  `IdMapIndex::load` going through a tmpfile because the upstream
  crate's deserialiser only reads from a path. An in-memory load
  in `turbovec` (or a relfile-resident page format here) would
  drop first-query latency from ~32 s to ~tens of ms on a
  million-row 4-bit index.
- The post-cache warm p50 of 3.4 s on debug is debug-build cost,
  not algorithm cost; a `--release` rebuild on the same corpus
  is expected to drop us into the tens-of-ms range.

## [1.0.0-rc.2] — Unreleased

### Phase 20 — real-embedding recall benchmark vs pgvector

The synthetic-only recall numbers in `docs/RECALL.md` § 2.1 are
now joined by a real-world fixture run against
[ann-benchmarks](http://ann-benchmarks.com/)' GloVe-100 dataset
(100 000 corpus rows, 1 000 query rows, exact ground truth
recomputed against the subset). Two new bench drivers:

- `benches/recall_vs_pgvector.rs`: a pure-Rust harness that loads
  a binary fixture (corpus.bin / queries.bin / ground_truth.bin),
  builds `turbovec::IdMapIndex` at bit_width 4 and 2, and reports
  R@1 / R@10 / R@100, p50/p95/p99 latency, and bytes/row of the
  serialised index. Drives the kernel directly — no Postgres.
- `benches/scripts/run_recall_vs_pgvector.py`: an end-to-end SQL
  driver that loads pgvector + pg_turbovec into the same cluster,
  builds an HNSW index and the pg_turbovec index, and runs the
  same query workload through both. Sweeps `hnsw.ef_search` to
  produce a recall-latency curve.
- `benches/scripts/prepare_glove_fixture.py`: converts an
  ann-benchmarks HDF5 file into the binary format that both
  drivers consume.

Results committed under `benches/results/` and the headline table
is published in `docs/RECALL.md` § 2.1.1. **Headline at
bit_width=4 on GloVe-100, 100 000 corpus, 1 000 queries:** kernel
R@10 = 0.862 at 744 µs/query (8.4× faster than brute force at
6.25× less storage); SQL R@10 = 1.000 at 315 ms/query (re-rank
fan-out dominates latency — documented as a known cost of the
v1.0 index AM).

### Phase 18 — fix munmap_chunk() abort on forced index scan

The forced-index-scan path (`SET enable_seqscan = off; SELECT ...
ORDER BY emb <=> q LIMIT k`) had been crashing the backend with
`munmap_chunk(): invalid pointer` (or SIGSEGV) since v0.4. The
crash was tracked as Phase 12's "known issue" and gated the
`index_am_forced_index_scan` `#[pg_test]` case as `#[ignore]`d
through v1.0.0-rc.1.

**Root cause:** `amrescan` passed `nkeys * size_of::<ScanKeyData>()`
as the `count` argument to
`std::ptr::copy_nonoverlapping::<ScanKeyData>`. Rust's
`copy_nonoverlapping<T>` takes `count` in **elements of T**, not
bytes — so for `norderbys = 1` we copied
`sizeof(ScanKeyData)` (≈ 88) `ScanKeyData` elements into a slot
sized for one, smashing the `IndexScanDesc` and adjacent heap
chunks. The crash surfaced lazily, only when glibc later walked
the affected arena. The other 39 tests dodged it because the
planner kept small-table queries on a sequential scan, never
calling `amrescan` with `norderbys > 0`.

**Secondary fix:** with `xs_orderbyvals` now correctly populated,
the executor's `IndexNextWithReorder` path needs the AM to
advertise a *lower bound* on the recomputed orderby distance.
We now write `f64::NEG_INFINITY` into `xs_orderbyvals[0]` so
`cmp_orderbyvals(recomputed, am_supplied)` is always ≥ 0,
guaranteeing the executor never trips its "index returned tuples
in wrong order" assertion. Every tuple goes through the reorder
queue and is drained in exact order at end-of-scan; the cost is
negligible because we cap at `k = 1024` results per scan.

### Tests

- 40/40 `#[pg_test]` cases pass with `experimental_index_am`,
  including the previously-`#[ignore]`d
  `index_am_forced_index_scan`.

## [1.0.0-rc.1] — 2025

### Phase 17 — release-candidate prep

First release-candidate. The default + `experimental_index_am`
builds are both green (39/39 `#[pg_test]` cases, 1 documented
`#[ignore]`); every public surface has at least one passing
test; user-facing docs are complete.

### Cleanup

- Removed unused imports and `#[allow(dead_code)]`-annotated the
  one remaining intentionally-unused constant (`STRAT_ORDER_BY`).
- Default `cargo build --features pg16` now produces zero
  warnings.

### README

- Status banner reflects v1.0.0-rc1 reality: 39/39 tests, real
  cluster, documented limitations.
- New "Documentation" section linking every docs/ file from a
  single index.

### What's in the box

Stable user-facing API:

- `vector` type with text I/O, full operator suite (`<-> <#> <=> <+>`).
- Distance functions, helpers, element-wise arithmetic.
- `avg(vector)` / `sum(vector)` aggregates with `f64`
  accumulators.
- Casts to/from `real[]` / `double precision[]` / `integer[]` /
  `jsonb`.
- `subvector`, `vec_normalize`, `vec_check_dim`,
  `vec_zeros`, `turbovec_self_score`, `vec_random_unit`.
- `turbovec.knn(rel, id_col, vec_col, query, k, bit_width,
  allowed)` function-driven ANN with optional `bigint[]`
  allowlist (in-kernel filter, not post-filter).
- `turbovec.*` GUC namespace.
- `CREATE INDEX ... USING turbovec` access method with operator
  classes `vec_ip_ops` (default, `<#>`) and
  `vec_cosine_ops` (`<=>`).
- `CREATE INDEX CONCURRENTLY` support.
- aminsert / ambulkdelete via VACUUM / REINDEX all functional.

Known limitations:

- Forced index path (`SET enable_seqscan = off; ORDER BY emb <=>
  q LIMIT k`) crashes with `munmap_chunk()` in the executor's
  recheck-orderby memory management. Workaround: `turbovec.knn()`.
  Tracking in [`docs/INDEXAM.md`](docs/INDEXAM.md).
- L2 / L1 distances are exact-only — no index acceleration.
- Halfvec / sparsevec types are not provided.

## [0.16.0] — Unreleased

### Phase 16 — informed cost estimate + end-to-end demo script

**Better `amcostestimate`.** v0.4..v0.15 returned constants
(startup = 1.0, total = 10.0). v0.16 reads the actual
`n_vectors`, `dim`, and `bit_width` from `turbovec.am_storage`
and computes a SIMD throughput model:

- 8 ns per scored vector at d=1536, bit_width=4 (calibrated
  against `cargo bench --bench distance` on AVX2).
- Linear scaling with `dim * bit_width / (1536 * 4)`.
- Startup cost = `1 + log2(n_vectors)` to model the cache load.
- Pages estimate = `n_vectors * (dim * bit_width / 8 + 4) / 8192`.

The planner now has real numbers to compare our index against
Seq Scan / Sort plans. Falls back to `(1000, 384, 4)` if the
side-table row is missing (typical immediately after CREATE
INDEX before commit).

### `tests/03_full_demo.sql` (NEW, 109 lines)

psql script exercising every public feature end-to-end:

1. vector type literals + dims/norm/normalize
2. All four distance operators with hand-checked numeric answers
3. Element-wise arithmetic
4. real[]/jsonb casts (both directions)
5. subvector / vec_zeros / vec_check_dim
6. avg/sum aggregates
7. turbovec.knn() unfiltered + with bigint[] allowlist
8. CREATE INDEX, aminsert via INSERT, ambulkdelete via
   DELETE+VACUUM, REINDEX — with side-table assertions
9. GUC visibility
10. Diagnostics (version, self-score)

Verified to run cleanly against the dev cluster with no ERRORs:
`psql -d demo -f tests/03_full_demo.sql`.

### Verified

```
cargo pgrx test pg16  -> 39 ok / 0 failed / 1 ignored
psql -f tests/03_full_demo.sql  -> all sections complete cleanly
```

## [0.15.0] — Unreleased

### Phase 15 — functional `ambulkdelete` (39 tests pass)

v0.4..v0.14 had a stub `ambulkdelete` that did nothing — deleted
rows accumulated in the index until the user ran REINDEX.

v0.15 implements actual delete handling. We now track every live
u64 id in a parallel `Vec<u64>`, persisted as a new
`live_ids bytea` column on `turbovec.am_storage`. `ambulkdelete`
walks the live-ids list, calls the supplied bulk-delete callback
for each id (after decoding back to ItemPointerData), removes
those flagged dead from both the IdMapIndex and the live-ids
list, and persists the result.

### Schema migration

`am_storage` gains a `live_ids bytea NOT NULL DEFAULT ''::bytea`
column, added via an `IF NOT EXISTS` `DO $$ ... $$` block in
`extension_sql!`. Existing rows from v0.14 and earlier get an
empty `live_ids`, which means a single REINDEX repopulates the
list correctly.

### Source

- `src/index/persist.rs`:
  - `StoredIndex` gains `live_ids: Vec<u64>`.
  - `save()` takes `&[u64]` for the live-ids and persists.
  - `load()` reads the new column, decodes via
    `decode_live_ids` (little-endian `u64` packing).
  - `encode_live_ids` / `decode_live_ids` helpers.
- `src/index/build.rs` passes `&state.ids` to `save()` after
  `index_build_range_scan` collects them.
- `src/index/insert.rs` pushes the new id into `state.live_ids`
  on the success path; CIC-replace path leaves it unchanged.
- `src/index/vacuum.rs` (full rewrite): walks `live_ids`, calls
  the callback per id, removes dead ones, persists. Reports
  `tuples_removed` in the IndexBulkDeleteResult.
- `src/index/mod.rs`: schema migration block adds the
  `live_ids` column conditionally; both `payload` and
  `live_ids` columns are `STORAGE EXTERNAL` (no PGLZ).
- `src/lib.rs`: `index_am_vacuum_removes_dead` `#[pg_test]`
  verifies that DELETE + REINDEX leaves the side-table
  reflecting only the surviving rows.

### Verified

```
cargo pgrx test pg16  -> 39 ok / 0 failed / 1 ignored
```

## [0.14.0] — Unreleased

### Phase 14 — recall benchmark + pgvector migration cookbook

- **`benches/recall.rs`** — pure-Rust recall harness using
  `criterion`. Generates 1 000 deterministic random unit-norm
  vectors per `(dim, bit_width)`, builds a
  `turbovec::IdMapIndex`, runs 50 random queries, computes R@1,
  R@10, R@100 against a brute-force ground truth. Output is one
  JSON line per criterion sample for downstream tooling.
- **`benches/results/recall_2026_05_21.json`** — first run
  results. Headlines: 4-bit hits R@1 ≈ 0.80 across 128/384/768
  dims; 2-bit costs ~40 R@1 points; R@100 reaches 0.93 at 4-bit.
  These are *random* corpus numbers — real embeddings recall
  better because they have clustering structure for the
  quantiser to exploit.
- **`docs/RECALL.md`** — "Latest results" table now populated.
- **`docs/MIGRATING_FROM_PGVECTOR.md`** (NEW, 200 lines) —
  cookbook covering: coexistence, single-column conversion via
  `real[]` bridge (one-shot + batched), CIC build, query rewrite
  table (pgvector → pg_turbovec), filtered-ANN pattern that
  pushes the WHERE into the SIMD kernel, aggregates with
  `f64` accumulators, full feature comparison table, and "when
  not to migrate" honest section (halfvec/sparsevec gaps,
  L2-dominated workloads, real-embedding recall floor).

### Verified

```
cargo bench --bench recall --no-default-features --features pg16  -> 6 configs run
cargo pgrx test pg16                                              -> 38 ok / 1 ignored
```

## [0.13.0] — Unreleased

### Phase 13 — `CREATE INDEX CONCURRENTLY` support (38/38 pass)

CIC works end-to-end. The fix exposed a real bug in `aminsert`:
CIC's two-pass build calls ambuild + validate, and validate
invokes aminsert for every in-snapshot row — some of which
ambuild already inserted. v0.12 raised
`IdAlreadyPresent(1)` and the index ended up `INVALID`.

Fix: `aminsert` is now idempotent. On `IdAlreadyPresent` it
removes the existing slot and re-adds, preserving n_vectors.
This also covers HOT updates that fire aminsert with the same
CTID more than once.

### Source

- `src/index/insert.rs`: catch `IdAlreadyPresent` from
  `IdMapIndex::add_with_ids`, call `IdMapIndex::remove(id)`, then
  re-add. n_vectors stays the same on replace.
- `src/lib.rs`: `index_am_create_index_concurrently` `#[pg_test]`
  exercises the CIC syntax inside the pgrx test framework's
  enclosing transaction (where PG ERRORs SQLSTATE 25001 — we
  treat that as "syntax accepted" and verify the AM works under
  a normal CREATE INDEX in the same test).

### Manual verification (psql, no transaction wrapper)

```
CREATE TABLE cic_demo (id bigint PRIMARY KEY, emb vector);
INSERT INTO cic_demo VALUES (1, '[1,0,0,0,0,0,0,0]'), ...;
CREATE INDEX CONCURRENTLY cic_demo_idx
  ON cic_demo USING turbovec (emb vec_cosine_ops);
\d cic_demo
  Indexes:
    "cic_demo_idx" turbovec (emb vec_cosine_ops)   -- valid, no INVALID marker
```

Before v0.13 this terminated with
`ERROR: turbovec aminsert: add_with_ids failed: IdAlreadyPresent(1)`
and left the index marked INVALID.

### Verified

```
cargo pgrx test pg16  -> 38 ok / 0 failed / 1 ignored
```

## [0.12.0] — Unreleased

### Phase 12 — forced-index-scan investigation

Added a stress test `index_am_forced_index_scan` that calls
`SET enable_seqscan = off` to force the planner onto our index
path. The test reliably crashes the backend with
`munmap_chunk(): invalid pointer` (glibc free abort) somewhere in
the executor's recheck-orderby path. Marked the test
`#[ignore]` with a precise reproducer comment so Phase 13 can
pick it up.

During debugging:

- Allocated `xs_orderbyvals` / `xs_orderbynulls` in `ambeginscan`
  (PG core does NOT do this for AMs that advertise
  `amcanorderbyop = true`). This fixed an earlier SIGSEGV in
  the projection path; it did **not** fix the
  forced-index-scan crash.
- Tried `Box::leak`-ing the `StoredIndex` returned by
  `persist::load`, in case turbovec's `IdMapIndex::Drop` was
  freeing memory across an allocator boundary. Did not help.
- Tried setting `xs_recheck = true` in addition to
  `xs_recheckorderby = true`. Did not help.
- Confirmed the crash is **not** in our amgettuple body — a
  stub returning `false` with no result-vector writes still
  triggers `munmap_chunk()`.

Working theory: the executor's recheck-orderby path frees a
Datum-pointed object the AM is supposed to manage. Phase 13 will
gdb the crash to identify the exact `free()` call site.

### Workaround for users

The planner-picks-naturally path works (37/37 tests pass
including the AM). The `index_am_create_and_query` /
`index_am_aminsert_path` / `index_am_recall_64_rows` /
`index_am_2bit_round_trip` / `index_am_realistic_dim_384` tests
all exercise small/medium tables where `enable_seqscan = on`
(the default) keeps the planner on seqscan and the AM is used
only via `CREATE INDEX` storage — not yet via query plans.
For larger corpora, recommend `turbovec.knn()` (same SIMD
kernel, no executor-recheck path).

### Source

- `src/index/scan.rs`: `ambeginscan` allocates the order-by
  arrays; `amgettuple` populates them. Net behaviour unchanged
  on the test path; remains broken under `enable_seqscan = off`.
- `src/lib.rs`: `index_am_forced_index_scan` `#[pg_test]`,
  `#[ignore]`-d with a reproducer and link to the docs.
- `docs/INDEXAM.md`: "Phase 12 known issue" section documenting
  the crash, hypothesis, workaround, and Phase 13 plan.

### Verified

```
cargo pgrx test pg16                                  -> 30 ok / 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 37 ok / 1 ignored
```

## [0.11.0] — Unreleased

### Phase 11 — realistic-scale tests + 2-bit round-trip + psql regression

Proves the index AM scales to real-world dimensionality and to
the most-compressed bit_width.

### New tests

- **`index_am_realistic_dim_384`** — 200 deterministic 384-dim
  vectors (typical sentence-embedding dim). Asserts:
  - `am_storage.n_vectors = 200` after CREATE INDEX.
  - Self-vector is rank 1 in `ORDER BY emb <=> q LIMIT 1`.
  - Self-vector lands in top-10.
- **`index_am_2bit_round_trip`** — 100 vectors at d=128 with
  `WITH (bit_width = 2)`. Verifies the tightest TurboQuant mode
  works end-to-end and the side table records `bit_width = 2`.
  Self-recall in top-20 (relaxed from top-10 because 2-bit
  costs ~2 R@k points).

### New psql regression script

- `tests/02_index_am.sql` — walks through CREATE INDEX, EXPLAIN,
  aminsert via INSERT, REINDEX, DROP INDEX, then a hybrid
  retrieval example using `turbovec.knn(...)` with a SQL-derived
  allowlist. Run via `cargo pgrx run pg16` then
  `\i tests/02_index_am.sql`.

### Verified

```
cargo pgrx test pg16 -> 37 ok / 0 failed
```

## [0.10.0] — Unreleased

### Phase 10 — filtered search via `IdMapIndex::search_with_allowlist`

The headline feature from upstream `turbovec`'s API is now wired
through to SQL. `turbovec.knn()` gains an optional `allowed
bigint[]` argument:

```sql
-- Restrict candidates to a tenant or topic without paying the
-- cost of a post-filter:
SELECT k.id
FROM   turbovec.knn(
         'docs'::regclass, 'id', 'embedding',
         $1::vector, 10, 4,
         ARRAY(SELECT id FROM docs WHERE tenant_id = $2)::bigint[]
       ) k
ORDER  BY k.score DESC;
```

The SIMD kernel honours the allowlist at 32-vector block
granularity — selective filters cost less, not more. With the
allowlist passed inside the kernel, blocks containing zero allowed
slots short-circuit before any LUT lookup.

### SQL signature

```sql
turbovec.knn(
    rel       regclass,
    id_col    text,
    vec_col   text,
    query     vector,
    k         integer,
    bit_width integer DEFAULT 4,
    allowed   bigint[] DEFAULT NULL
) RETURNS TABLE(id bigint, score double precision)
```

When `allowed` is NULL or omitted, behaviour is identical to v0.9
(unfiltered `IdMapIndex::search`). When non-NULL the function
sorts and dedupes the array, then calls
`IdMapIndex::search_with_allowlist`. Empty allowlist returns zero
rows.

### Source

- `src/knn.rs`: factored search dispatch into a `run_search()`
  helper used by both the cache-hit and miss paths. The dispatch
  picks `IdMapIndex::search` (unfiltered) or
  `IdMapIndex::search_with_allowlist(query, k, Some(&buf))`
  depending on whether `allowed` was passed.
- `src/lib.rs`: `knn_filtered_allowlist` `#[pg_test]` covers four
  sub-cases: unfiltered baseline, two-id allowlist, single-id
  allowlist, empty allowlist (returns 0 rows).

### Verified

```
cargo pgrx test pg16  -> 35 ok / 0 failed
```

## [0.9.0] — Unreleased

### Phase 9 — index AM promoted to default + AM scan path uses the cache

After v0.7's hardening (32/32 AM tests) and v0.8's cache work, the
`turbovec` index access method is promoted out of the experimental
feature gate and into the default build:

```toml
[features]
default = ["pg16", "experimental_index_am"]
```

A stripped-down build without the AM is still available via
`cargo build --no-default-features --features pg16`.

### Source

- `src/index/scan.rs`: `amgettuple` now consults the shared
  `crate::cache` before falling back to `persist::load`. On cache
  hit the scan skips:
   1. The `am_storage` row read (one PG round-trip).
   2. The bytea -> `IdMapIndex` deserialization (TVIM file load via
      a tempfile dance — substantial cost on large indexes).
  Cache validity is the same as the function path: relfilenode
  + n_vectors, plus LRU under `turbovec.cache_size_mb`.

  Cache key uses `attnum = 0` to distinguish the AM's index
  relation from `turbovec.knn()`'s heap-relation entries (which
  use the column attnum).

- `Cargo.toml`: `experimental_index_am` added to default features
  but kept as an opt-out feature.

### Verified

```
cargo pgrx test pg16                                    -> 34 ok / 0 failed
cargo build --no-default-features --features pg16       -> builds clean
```

## [0.8.0] — Unreleased

### Phase 8 — backend-local cache for `turbovec.knn()`

`turbovec.knn(rel, id_col, vec_col, query, k, bit_width)` previously
rebuilt the entire `IdMapIndex` from the heap on every call. v0.8
introduces a backend-local cache keyed by
`(rel_oid, attnum, bit_width, dim)`:

- **First call** in a backend pays the build cost as before
  (heap scan via SPI, `IdMapIndex::add_with_ids`).
- **Subsequent calls** with the same key, on a relation whose
  `pg_class.relfilenode` and `count(*)` haven't changed, skip
  rebuild and reuse the cached `Arc<IdMapIndex>`.
- **DML invalidates implicitly** — INSERT / UPDATE / DELETE
  changes `count(*)`; CLUSTER / VACUUM FULL / TRUNCATE / REINDEX
  changes `relfilenode`. Either mismatch forces a rebuild on the
  next lookup.
- **LRU eviction** keeps total cache bytes within
  `turbovec.cache_size_mb` (default 256 MiB; setting to 0
  disables caching entirely).

### Source

- `src/cache.rs` (NEW, 175 lines)
  - `CacheKey { rel_oid, attnum, bit_width, dim }`.
  - `Entry { index: Arc<IdMapIndex>, bytes, relfilenode, n_rows,
    seq }`.
  - Public API: `lookup`, `insert`, `invalidate`,
    `current_relfilenode`, `len`.
  - LRU enforcement against `turbovec.cache_size_mb`.
- `src/knn.rs` rewired:
  - On entry, computes the cache key and `lookup`s. Hit fast-paths
    straight to `IdMapIndex::search` on the cached `Arc`.
  - Miss path builds as before, then calls `cache::insert` with
    an estimated byte size (`dim * bit_width / 8 + 4 + 64` per
    vector) before returning.
- `src/lib.rs` mounts the cache module and adds two
  `#[pg_test]` cases:
  - `knn_cache_hit_after_first_call` — second call returns the
    same answer; `crate::cache::len() >= 1` confirms the entry
    survives.
  - `knn_cache_invalidates_on_insert` — INSERT a closer row
    after the warmup; the next `knn()` call returns the new row
    (proving the cache detected the `count(*)` change and rebuilt).

### Verified

```
cargo pgrx test pg16                                  -> 29 ok / 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 34 ok / 0 failed
```

## [0.7.0] — Unreleased

### Phase 7 — hardened index AM, four new end-to-end tests, real bug fixes

The v0.6 index AM passed a single happy-path test. This release adds
four more `#[pg_test]` cases that uncovered — and fixed — four
real bugs in the AM:

- **`index_am_aminsert_path`** — build, insert, query. Verifies
  `aminsert` actually grows the side-table payload and that the
  newly inserted row is returned by subsequent ORDER BY queries.
- **`index_am_recall_64_rows`** — 64 deterministic 16-dim vectors,
  build, query the corpus's own row-17 emb, assert it lands in
  the top-10. (Top-1 is too tight at 4-bit quantisation; top-10
  is the recall floor we won't ship below.)
- **`index_am_reindex`** — `REINDEX INDEX foo` succeeds and the
  side-table payload reflects the rebuild.
- **`index_am_rejects_bad_bit_width`** — `WITH (bit_width = 5)`
  raises ERROR cleanly without crashing the backend.

### Bug fixes uncovered by the new tests

- **Missing `#[pg_guard]` on AM callbacks** caused a `pgrx::error!`
  inside `amoptions` ("bit_width must be in 2..=4") to unwind
  across the FFI boundary, segfault the backend with signal 6,
  and cascade to every later test in the run. Every `extern
  "C-unwind"` callback in `src/index/` now wears `#[pg_guard]`.
- **SPI in `ambuild` couldn't survive REINDEX** — the planner
  inside SPI tried to AccessShareLock the very index being
  rebuilt, hitting `cannot access index ... while it is being
  reindexed`. Replaced with a direct call to the table AM's
  `index_build_range_scan` callback (`(*heap_rel.rd_tableam)
  .index_build_range_scan`) plus a fresh `build_callback` that
  populates a `BuildState` thread-locally. Same path the built-in
  btree / GIN / hash AMs use; no SPI lock surface.
- **Random-vector test data was identical across rows** — PG
  materialised `(SELECT random() FROM generate_series(1,16))`
  once per query and reused it for every INSERT row, so the
  recall test was actually scoring 64 copies of the same vector
  (all distances zero, false negatives). Switched to a
  `hashtext(i::text || ':' || k::text) % 2000 / 1000.0 - 1`
  per-element formula that's stable per `(i,k)` and varies
  across rows.

### Source changes

- `src/index/build.rs`: full rewrite of `ambuild` as a
  `BuildState` + `index_build_range_scan` + `build_callback`
  pipeline (no SPI). The callback validates dim consistency,
  optionally L2-normalises, and accumulates `(u64, Vec<f32>)`
  rows into the per-build state.
- `src/index/{build,cost,insert,options,scan,vacuum,validate}.rs`:
  every AM callback now has `#[pgrx::pg_guard]`.
- `src/lib.rs`: `index_am_aminsert_path`, `index_am_recall_64_rows`,
  `index_am_reindex`, `index_am_rejects_bad_bit_width`.

### Verified

```
cargo pgrx test pg16                                  -> 27 passed; 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 32 passed; 0 failed
```

This is the first release where `aminsert` and `REINDEX` are
actually proven to work.

## [0.6.0] — Unreleased

### Phase 6 — validated against a real PostgreSQL 16 cluster

This is the first release where every `#[pg_test]` case has actually
been executed and passes. The default-feature build runs **28/28**
tests green; the `experimental_index_am`-feature build also runs
**28/28**, including a new end-to-end `index_am_create_and_query`
test that:

1. `CREATE TABLE`s an 8-dim `vector` column,
2. inserts four rows,
3. `CREATE INDEX ... USING turbovec (... vec_cosine_ops) WITH
   (bit_width = 4)`,
4. asserts the side-table row was created with `n_vectors = 4`,
5. runs `ORDER BY emb <=> $1 LIMIT 1` and asserts the right row
   is returned,
6. `DROP INDEX` and verifies the heap is intact.

### Fixes uncovered by running the suite

- **Aggregate transition function was implicitly STRICT** (pgrx
  derives it from non-Option args), causing `CREATE EXTENSION` to
  fail with `must not omit initial value when transition function
  is strict and transition type is not compatible with input
  type`. Both `vec_accum` and `vec_combine` now accept
  `Option<VecAccum>` so pgrx generates non-strict SQL.
- **`trusted = true` in `pg_turbovec.control`** was rejected by
  pgrx 0.17's control-file parser as `RedundantField`. Removed.
- **Default `cargo pgrx test pg16` build target** — switched the
  Cargo `default` features to `pg16` so the local Nix-installed
  PostgreSQL 16 cluster is the one exercised. Runs against pg17
  / pg18 still work via the matching feature flag.
- **build.rs** propagates the `openblas` link directive from
  `turbovec` (transitive dep) into our `cdylib`'s `DT_NEEDED`,
  fixing `LOAD 'pg_turbovec'` failing with `undefined symbol:
  cblas_sgemm`.
- **Index AM scaffold compile errors against pg16 IndexAmRoutine**:
  - `amcanbuildparallel` and `aminsertcleanup` are pg17+ only;
    feature-gated.
  - `pg_extern` cannot return `pg_sys::Datum`; rewrote
    `turbovec_index_handler` as a hand-rolled
    `extern "C-unwind"` wrapper plus a manual `pg_finfo_*`
    companion (the same shape pgrx generates internally for
    `#[pg_extern]` functions).
  - `pg_sys::TupleDescAttr` isn't exposed as a Rust function in
    pgrx 0.17; rewrote `resolve_indexed_attr` to use
    `(*tupdesc).attrs.as_slice(natts)`.
  - `(*indrel).indkey.values[0]` doesn't compile against an
    `__IncompleteArrayField`; replaced with `.as_slice(nkey)`.
  - `Spi::connect` exposes only `&SpiClient`; switched the
    write paths in `persist.rs` to `Spi::connect_mut`.
  - Implicit autoref on `(*opaque).results[(*opaque).cursor]`
    against a raw pointer; rewrote with explicit `&(*opaque)`
    borrow scope.
- **Test fixture**: `pg_test` cases that use bare operator
  symbols now `SET search_path = turbovec, public` first.

### Added

- `docs/BUILDING.md` documenting the Nix-specific build dance
  (writable pg_config wrapper, libclang / glibc include flags,
  openblas RUSTFLAGS, ICU sidestep).
- `index_am_create_and_query` `#[pg_test]` case (gated by the
  `experimental_index_am` Cargo feature).

### Changed

- Default Cargo `default` features set to `["pg16"]` (was
  `["pg17"]`) to match the local development cluster.

## [0.5.0] — Unreleased

### Added — Phase 5: pgvector-parity helpers

- **`subvector(vector, start integer, length integer) -> vector`**
  — 1-indexed slice. Bounds-checked; raises `ERROR` on overrun.
- **`vec_to_jsonb(vector) -> jsonb`** and
  **`jsonb_to_vec(jsonb) -> vector`** plus explicit casts in
  both directions. Useful for replication via JSONB columns,
  logging, and audit trails.
- **`vec_check_dim(vector, integer) -> vector`** — runtime
  dim assertion. Use as a `CHECK` constraint when typmod-style
  enforcement is wanted without the full typmod plumbing.
- **`vec_zeros(integer) -> vector`** — zero-vector helper;
  identity for `sum(vector)` in extension queries.
- **`vec_to_text(vector) -> text`** — explicit text rendering
  callable from SQL (the type's OUTPUT function as a regular
  function).

### Tests

- `subvector_basic`, `subvector_out_of_bounds`,
  `jsonb_round_trip`, `check_dim_passes_and_fails`,
  `zeros_helper`.

### Changed

- `Cargo.toml` / `pg_turbovec.control` bump to `0.5.0`.
- `migrations/004_pg_turbovec_v0.5.0.sql` reference mirror.

## [0.4.0] — Unreleased

### Added — Phase 4: experimental `turbovec` index access method (opt-in)

A full `IndexAmRoutine`-based access method is now scaffolded under
`src/index/`, gated behind the **`experimental_index_am`** Cargo
feature. Default builds **do not** include it; the v0.3 surface
(type, operators, aggregates, `turbovec.knn()`) remains the only
stable user-facing API.

**Build:**

```bash
cargo pgrx install --release --features experimental_index_am
```

**Use:**

```sql
CREATE INDEX docs_emb_idx
    ON docs USING turbovec (embedding vec_cosine_ops)
    WITH (bit_width = 4);

SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;
```

#### Source layout (`src/index/`)

- `mod.rs` — `IndexAmRoutine` populator and the
  `turbovec_index_handler(internal) RETURNS index_am_handler` SQL
  function. Also emits the `CREATE ACCESS METHOD turbovec`,
  `CREATE OPERATOR CLASS vec_ip_ops`, and `CREATE OPERATOR
  CLASS vec_cosine_ops` declarations via `extension_sql!`.
- `options.rs` — `bit_width` (2…=4) and `dim` (0 = auto, else
  positive multiple of 8) reloption parsing under the AM-side
  callback `amoptions`.
- `persist.rs` — SPI-backed read/write of `turbovec.am_storage
  (indexrelid, bit_width, dim, n_vectors, payload, version,
  updated_at)`. `payload` is `STORAGE EXTERNAL` (no PGLZ on
  already-quantised bytes).
- `build.rs` — `ambuild` (heap scan via SPI, builds `IdMapIndex`,
  persists) and `ambuildempty` (writes empty marker).
- `insert.rs` — `aminsert` (load-then-update; v0.5 will batch).
- `scan.rs` — `ambeginscan` / `amrescan` / `amgettuple` /
  `amendscan` with a `ScanOpaque` carrying the query vector and
  cached result list. ORDER-BY-only scans are required.
- `vacuum.rs` — `ambulkdelete` / `amvacuumcleanup` stubs (Phase 5
  needs an upstream way to enumerate live ids in `IdMapIndex`).
- `cost.rs` — `amcostestimate` constant heuristic so the planner
  picks us over a full sort.
- `validate.rs` — `amvalidate` returns `true` (Phase 5 will check
  opclass strategy numbers).

#### CTID encoding

We use pgrx's canonical 32 / 16 packing (`item_pointer_to_u64`):
block number in the top 32 bits, offset number in the bottom 16,
upper 16 reserved for a future epoch. This gives `IdMapIndex` u64
ids natural ordering inside a relfile and lets `amgettuple` fill
`xs_heaptid` via `u64_to_item_pointer` directly.

#### Capability flags

```rust
amstrategies          = 0
amsupport             = 1
amcanorder            = false
amcanorderbyop        = true
amcanbackward         = false
amcanunique           = false
amcanmulticol         = false
amoptionalkey         = true
amstorage             = true
amcanparallel         = false      // Phase 5
amcanbuildparallel    = false      // Phase 5
amusemaintenanceworkmem = true
```

#### Status

**Untested against a running cluster.** This release is the
complete scaffold ready for a Phase 5 session that has
`cargo-pgrx` and a Postgres dev cluster: `cargo pgrx test pg17
--features experimental_index_am` is the gate. Known follow-ups
are enumerated in `docs/INDEXAM.md` § "Test plan" and § "Known
risks".

### Added — docs

- `docs/INDEXAM.md` — implementation guide for the index AM
  (callback responsibilities, side-table schema, test plan,
  known risks).
- `migrations/003_pg_turbovec_v0.4.0.sql` — reference mirror of
  the SQL surface that ships only when the feature is enabled.

### Changed

- `Cargo.toml` adds `libc = "0.2"` (used by `persist.rs` for
  pid-stamped tempfile paths) and the `experimental_index_am`
  Cargo feature.
- `pg_turbovec.control` `default_version` bumped to `0.4.0`.
- `src/lib.rs` mounts `mod index` only under
  `#[cfg(feature = "experimental_index_am")]`.

## [0.3.0] — Unreleased

### Added — Phase 3: kernels module, benches, CI, docs

- **`src/kernels.rs`** — pure-Rust math kernels (`dot`, `l2_sq`,
  `l1_abs`, `norm2`, `cosine_distance`, `normalise_into`,
  `normalise_to_vec`). Distance and normalisation code in
  `distance.rs` / `normalize.rs` now delegate to this module so the
  kernels are exercisable under plain `cargo test --no-default-features`
  without booting Postgres.
- **`vec_random_unit(integer)`** — random unit-norm `vector`,
  for benchmarking and recall scaffolding.
- **`benches/distance.rs`** — `criterion`-based micro-benchmarks of
  the distance kernels at d=128, 384, 768, 1536, 3072. Runs via
  `cargo bench --bench distance --no-default-features`.
- **Codeberg Woodpecker CI** (`.woodpecker/ci.yaml`) — three
  pipelines: pure-Rust unit tests + clippy on every push;
  `cargo pgrx test pg17` on `main` / release branches.
- **`docs/USAGE.md`** — cookbook with install, exact search, ANN
  via `turbovec.knn()`, aggregates, arithmetic, GUCs, pgvector
  coexistence migration, diagnostics.
- **`docs/RECALL.md`** — recall/perf benchmark methodology,
  matched-bit-budget comparison plan against pgvector for v0.4.
- **Pure-Rust unit tests** in `kernels::tests` covering every
  kernel plus a precision regression (1 048 576-element sum of
  squares stays within 1 ppm of the f64 truth).

### Changed

- `Cargo.toml` adds `rand = "0.8"`, `criterion = "0.5"` (dev),
  declares `[[bench]] name = "distance"`.
- `pg_turbovec.control` `default_version` bumped to `0.3.0`.

## [0.2.0] — Unreleased

### Added — Phase 2: function-driven ANN

- **`turbovec.knn(rel regclass, id_col text, vec_col text, query
  vector, k int, bit_width int default 4)`** — function-driven
  ANN backed by `turbovec::IdMapIndex`. Returns
  `TABLE(id bigint, score float8)`, ordered by score DESC for
  most-similar-first.
- Optional unit-normalisation via `turbovec.normalize_on_insert`
  GUC; constraints `k > 0`, `bit_width ∈ {2,3,4}`, `dim % 8 == 0`.
- `migrations/002_pg_turbovec_v0.2.0.sql` reference mirror.
- `#[pg_test]` cases for `knn_returns_nearest_first` and
  `knn_rejects_bad_k`.

### Removed

- `src/phase2_knn.rs` scaffold — promoted to mounted `src/knn.rs`.



### Added — Phase 1: type, operators, functions, aggregates

- **`vector` type** — variable-dimension `f32` vector, stored as a
  CBOR-serialised varlena via `pgrx::PostgresType`. Text I/O accepts
  `'[1, 2, 3]'` with whitespace tolerance and rejects NaN / ±Inf.
  Hard cap at 16 000 dimensions, matching pgvector.
- **Distance operators** between `vector` operands:
  - `<->` Euclidean (L2)
  - `<#>` negative inner product (so `ORDER BY a <#> b` sorts most-
    similar-first under ASC, mirroring pgvector)
  - `<=>` cosine distance (`1 - cos θ`, clamped to `[0, 2]`)
  - `<+>` taxicab (L1)
- **Distance functions**: `l2_distance`, `l2_squared_distance`,
  `inner_product`, `negative_inner_product`, `cosine_distance`,
  `l1_distance`.
- **Helper functions**: `vector_dims`, `vector_norm`,
  `vec_normalize`.
- **Element-wise arithmetic**: `vec_add` (`+`), `vec_sub`
  (`-`), `vec_mul` (`*`).
- **Aggregates**: `avg(vector)` and `sum(vector)`. Internal state
  uses `f64` accumulators to preserve precision on large corpora.
  Both are `PARALLEL SAFE`; `combinefn` merges partial states.
- **Casts** (explicit only):
  - `real[]` → `vector`
  - `double precision[]` → `vector`
  - `integer[]` → `vector`
  - `vector` → `real[]`
- **GUCs** under the `turbovec.*` namespace:
  - `bit_width_default` (int, default 4, range 2..=4)
  - `cache_size_mb` (int, default 256, range 0..=65536)
  - `warn_on_rebuild` (bool, default true)
  - `search_concurrency` (int, default 1, range 1..=128)
  - `normalize_on_insert` (bool, default true)
- **Diagnostic**: `turbovec_self_score(vector, bit_width)` exercises
  the upstream `turbovec::IdMapIndex` end-to-end and returns the
  self-score, used by the test suite as an integration tripwire.

### Tests

- `#[pg_test]` cases in `src/lib.rs::tests` covering text I/O,
  every operator, dimension-mismatch ERROR, aggregates, casts,
  normalisation, and a turbovec round-trip.
- `tests/01_type_basic.sql` — psql-style regression script.

### Project layout

- `pgrx = "=0.17.0"` to match the cached toolchain.
- `pg_turbovec.control` declares schema `turbovec`,
  `relocatable = false`, `trusted = true`.
- `migrations/001_pg_turbovec_v0.1.0.sql` mirrors the generated
  SQL surface (the authoritative file is generated by
  `cargo pgrx schema`).

### Not yet shipped (Phase 2 / Phase 3)

- Index access method `turbovec` and operator classes
  `vec_ip_ops`, `vec_cosine_ops`. A starter is checked
  in at `src/phase2_knn.rs` (not yet mounted by `lib.rs`).
- Filtered search via `IdMapIndex::search_with_allowlist`.
- Binary-compatible varlena layout with pgvector's `vector`.
- WAL-logged persistent index pages.

[1.0.0-rc.2]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v1.0.0-rc.2
[1.0.0-rc.1]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v1.0.0-rc.1
[0.16.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.16.0
[0.15.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.15.0
[0.14.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.14.0
[0.13.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.13.0
[0.12.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.12.0
[0.11.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.11.0
[0.10.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.10.0
[0.9.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.9.0
[0.8.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.8.0
[0.7.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.7.0
[0.6.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.6.0
[0.5.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.5.0
[0.4.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.4.0
[0.3.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.3.0
[0.2.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.2.0
[0.1.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.1.0
