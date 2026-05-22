# Changelog

All notable changes to `pg_turbovec` are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

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

1. `CREATE TABLE`s an 8-dim `tvector` column,
2. inserts four rows,
3. `CREATE INDEX ... USING turbovec (... tvector_cosine_ops) WITH
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
  type`. Both `tvector_accum` and `tvector_combine` now accept
  `Option<TvectorAccum>` so pgrx generates non-strict SQL.
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

- **`subvector(tvector, start integer, length integer) -> tvector`**
  — 1-indexed slice. Bounds-checked; raises `ERROR` on overrun.
- **`tvector_to_jsonb(tvector) -> jsonb`** and
  **`jsonb_to_tvector(jsonb) -> tvector`** plus explicit casts in
  both directions. Useful for replication via JSONB columns,
  logging, and audit trails.
- **`tvector_check_dim(tvector, integer) -> tvector`** — runtime
  dim assertion. Use as a `CHECK` constraint when typmod-style
  enforcement is wanted without the full typmod plumbing.
- **`tvector_zeros(integer) -> tvector`** — zero-vector helper;
  identity for `sum(tvector)` in extension queries.
- **`tvector_to_text(tvector) -> text`** — explicit text rendering
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
    ON docs USING turbovec (embedding tvector_cosine_ops)
    WITH (bit_width = 4);

SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;
```

#### Source layout (`src/index/`)

- `mod.rs` — `IndexAmRoutine` populator and the
  `turbovec_index_handler(internal) RETURNS index_am_handler` SQL
  function. Also emits the `CREATE ACCESS METHOD turbovec`,
  `CREATE OPERATOR CLASS tvector_ip_ops`, and `CREATE OPERATOR
  CLASS tvector_cosine_ops` declarations via `extension_sql!`.
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
- **`tvector_random_unit(integer)`** — random unit-norm `tvector`,
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
  tvector, k int, bit_width int default 4)`** — function-driven
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

- **`tvector` type** — variable-dimension `f32` vector, stored as a
  CBOR-serialised varlena via `pgrx::PostgresType`. Text I/O accepts
  `'[1, 2, 3]'` with whitespace tolerance and rejects NaN / ±Inf.
  Hard cap at 16 000 dimensions, matching pgvector.
- **Distance operators** between `tvector` operands:
  - `<->` Euclidean (L2)
  - `<#>` negative inner product (so `ORDER BY a <#> b` sorts most-
    similar-first under ASC, mirroring pgvector)
  - `<=>` cosine distance (`1 - cos θ`, clamped to `[0, 2]`)
  - `<+>` taxicab (L1)
- **Distance functions**: `l2_distance`, `l2_squared_distance`,
  `inner_product`, `negative_inner_product`, `cosine_distance`,
  `l1_distance`.
- **Helper functions**: `vector_dims`, `vector_norm`,
  `tvector_normalize`.
- **Element-wise arithmetic**: `tvector_add` (`+`), `tvector_sub`
  (`-`), `tvector_mul` (`*`).
- **Aggregates**: `avg(tvector)` and `sum(tvector)`. Internal state
  uses `f64` accumulators to preserve precision on large corpora.
  Both are `PARALLEL SAFE`; `combinefn` merges partial states.
- **Casts** (explicit only):
  - `real[]` → `tvector`
  - `double precision[]` → `tvector`
  - `integer[]` → `tvector`
  - `tvector` → `real[]`
- **GUCs** under the `turbovec.*` namespace:
  - `bit_width_default` (int, default 4, range 2..=4)
  - `cache_size_mb` (int, default 256, range 0..=65536)
  - `warn_on_rebuild` (bool, default true)
  - `search_concurrency` (int, default 1, range 1..=128)
  - `normalize_on_insert` (bool, default true)
- **Diagnostic**: `turbovec_self_score(tvector, bit_width)` exercises
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
  `tvector_ip_ops`, `tvector_cosine_ops`. A starter is checked
  in at `src/phase2_knn.rs` (not yet mounted by `lib.rs`).
- Filtered search via `IdMapIndex::search_with_allowlist`.
- Binary-compatible varlena layout with pgvector's `vector`.
- WAL-logged persistent index pages.

[0.8.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.8.0
[0.7.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.7.0
[0.6.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.6.0
[0.5.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.5.0
[0.4.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.4.0
[0.3.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.3.0
[0.2.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.2.0
[0.1.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.1.0
