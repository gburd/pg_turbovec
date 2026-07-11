# pg_turbovec — Architecture

> Source of truth for implementing agents. Read this end-to-end before
> writing code. When in doubt, prefer the design here over what
> `pgvector` does; we intentionally diverge in places.

**Status:** v1.3.0 shipped. The `vector` type, distance
operators + functions, aggregates, casts, the `turbovec` index
AM (default-on; the `experimental_index_am` and
`relfile_storage` Cargo features were retired in Phase Q),
halfvec / sparsevec / bitvec types, deferred-commit `aminsert`,
and the relfile-resident page format with persisted SIMD-
blocked layout + Lloyd-Max codebook (Phase P) all live in
`main`. The relfile is the only storage strategy. The roadmap
history below is truncated; the canonical per-version log is
[`CHANGELOG.md`](../CHANGELOG.md).
**Target Postgres:** 13–18 (all six tested in CI; see
[`docs/PG_VERSION_SUPPORT.md`](PG_VERSION_SUPPORT.md)).
**Toolchain:** Rust 1.85+, `pgrx = "=0.17.0"`.
**Upstream:** [`turbovec`](https://crates.io/crates/turbovec) 0.5.x,
locally vendored under `vendor/turbovec/` with a small patch (see
`vendor/turbovec/PATCH_NOTES.md`).
**License:** Apache-2.0 (matches `turbovec`).

---

## 1. Goals and non-goals

### 1.1 Goals

1. **First-class Postgres extension** that exposes TurboQuant
   quantised vector search to SQL — type, operators, functions,
   aggregates, and an index access method named `turbovec`
   (shipped in v0.4 experimental, default-on since v0.9).
2. **Coexistence with `pgvector`**, not replacement. We use:
   - Type name `vector` (not `vector`).
   - Schema `turbovec` for everything we own.
   - Operator symbols `<#>`, `<=>`, `<->`, `<+>` are reused but
     dispatch by operand type (`vector`, not `vector`); pgvector's
     own operators on `vector` are unaffected when both extensions
     are installed in the same database.
3. **Memory wins** matching the upstream paper: a 1536-dim corpus at
   2-bit-per-coordinate occupies ≈ 388 B/vector (4-bit is ≈ 772 B/vector)
   — a ~16× (2-bit) or ~8× (4-bit) reduction over pgvector's float32
   storage. (Corrected 2026-07-06: this previously said "4-bit... ≈388
   B/vector", which was the 2-bit figure mislabeled — 1536 × 4 bits / 8
   + 4B scale = 772B, not 388B. Caught during a PQ-subvector-
   quantization feasibility review; see an internal design note-
   adjacent research notes.)
4. **SIMD-accelerated ANN** for inner-product and cosine queries via
   the upstream `turbovec` kernels (NEON, AVX2, optional AVX-512BW).
5. **Filtered search** that pushes the predicate into the SIMD
   kernel via `IdMapIndex::search_with_allowlist` (shipped in
   v0.10 through `turbovec.knn(..., allowed bigint[])`).
6. **Honest about limits**: the `turbovec` index AM persists
   into the index relation's main fork via PostgreSQL's buffer
   manager, the same mechanism every other PG index AM uses.
   Per-backend `IdMapIndex` cache + `shared_buffers` cache the
   relfile pages cluster-wide. Phase P's pre-baked SIMD-blocked
   layout + Lloyd-Max codebook bring cold-scan p50 to ~1.26 s on
   dbpedia-1M; remaining gap to pgvector HNSW (~100 ms cold) is
   bounded by the per-backend `IdMapIndex` reconstruction cost.

### 1.2 Non-goals

- **Drop-in pgvector replacement.** No `vector` type, no `ivfflat` or
  `hnsw` AM names, no implicit casts. We provide *explicit* casts
  via `array_to_vec` etc.
- **L2 / L1 ANN.** TurboQuant scores inner-product on unit vectors;
  Euclidean and Manhattan are not supported by the index. We expose
  `l2_distance` / `l1_distance` as exact functions only — no
  operator class, no index path, brute-force scan only.
- **Variable per-row dimensionality inside a single index.** A
  `vector` column may store mixed dims, but a `turbovec` index over
  that column locks dim at first build and rejects mismatches.
- **External codebook training.** TurboQuant is data-oblivious; we
  deliberately do not expose any `train()`-shaped API.
- **Replication of TurboQuant internals.** We treat `turbovec` as a
  black box — no reaching into private modules, no forks.

### 1.2.1 Known upstream limitation: TQ+ calibration on tiny tables

`turbovec`'s TQ+ per-coordinate calibration (a lightweight data-
dependent refinement layered on the data-oblivious base algorithm)
fits once, on the FIRST `add_with_ids` batch, and freezes for the
life of the index (upstream issue
[RyanCodrai/turbovec#107](https://github.com/RyanCodrai/turbovec/issues/107),
open as of 2026-07-06). If that first batch has fewer than 1000 rows,
the fit silently falls back to an identity transform — frozen forever,
no error, no warning — and the index never gets TQ+'s accuracy
improvement. pg_turbovec's IVF build path (`WITH (lists = N)`) is
already safe: it primes calibration with a fixed 16,384-row
cell-ordered prefix (`IVF_CALIB_ROWS` in `src/index/build.rs`),
comfortably above the 1000-row threshold regardless of corpus size.
**The flat (non-IVF, default) build path is NOT protected**: its
streaming flush batches at `turbovec.build_parallelism`-independent,
`maintenance_work_mem`-derived `chunk_rows` (typically 8k-100k rows
at realistic dimensions with the default 64MB `maintenance_work_mem`
— see `BuildState::compute_chunk_rows`), so a table with FEWER rows
than that single first-flush batch size builds its whole flat index
in one under-1000-row `add_with_ids` call, silently freezing TQ+ to
identity. This is a real, if narrow, gap — small tables (dev/test
fixtures, small tenant partitions, cold-start corpora) are exposed;
large tables are not (the streaming flush's first chunk is almost
always ≥ 1000 rows once the corpus itself is that big). Tracked as a
follow-up, not fixed here — the fix belongs upstream (see
an internal design note §1) or as a pg_turbovec-side
workaround (e.g. padding tiny first batches to the threshold with
harmless synthetic dummy rows before calibration and correcting the
slot count afterward) once a maintainer decides which side should own
the mitigation.

### 1.3 Why coexist instead of drop-in?

Pgvector users have years of `vector(1536)` columns and tooling.
Pretending to be a drop-in replacement would create silent semantic
drift, and TurboQuant's assumption that vectors are unit-normalised
would make some queries silently wrong. Better to let users opt in
explicitly: `ALTER TABLE … ADD COLUMN embedding turbovec.vector`.

---

## 2. Crate layout

### 2.1 Single pgrx crate, no workspace

`turbovec` is itself a workspace member depended upon via crates.io.
Wrapping it in a Postgres extension is one `cdylib` — there is no
multi-crate problem to solve. A workspace adds friction without
payoff.

### 2.2 Top-level layout

```
pg_turbovec/
├── Cargo.toml                  # pgrx 0.17, pg13..pg18 features
├── pg_turbovec.control         # extension control file
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                     # Apache-2.0
├── docs/                       # this directory
├── migrations/                 # one .sql per version bump
├── sql/                        # generated by `cargo pgrx schema`
├── vendor/
│   └── turbovec/               # locally vendored upstream crate
│                               # with a small patch (see
│                               # vendor/turbovec/PATCH_NOTES.md)
├── src/
│   ├── lib.rs                  # pgrx::pg_module_magic!(), _PG_init
│   ├── vec.rs                  # the `vector` SQL type, text I/O
│   ├── distance.rs             # operators + named distance functions
│   ├── aggregate.rs            # avg(vector), sum(vector); f64 accum
│   ├── cast.rs                 # array / jsonb ↔ vector casts
│   ├── normalize.rs            # vec_normalize, turbovec_self_score
│   ├── guc.rs                  # turbovec.* GUC registration
│   ├── knn.rs                  # function-driven ANN: turbovec.knn(...)
│   ├── kernels.rs              # pure-Rust math kernels (Phase 3)
│   ├── cache.rs                # backend-local Arc<RwLock<IdMapIndex>>
│   │                           # cache shared by knn() + AM scan path
│   ├── extras.rs               # subvector, vec_zeros, vec_check_dim,
│   │                           # vec_to_text (Phase 5 helpers)
│   ├── halfvec.rs              # f16 SQL type
│   ├── halfvec_ops.rs          # halfvec operators / casts / aggregates
│   ├── sparsevec.rs            # sparse (dim, indices, values) SQL type
│   ├── sparsevec_ops.rs        # sparsevec operators + casts
│   ├── bitvec.rs               # packed binary SQL type + Hamming/Jaccard
│   ├── xact.rs                 # PreCommit/AbortCurrentTransaction hooks
│   │                           # for deferred-commit aminsert (Phase K)
│   ├── index/                  # the `turbovec` index access method
│   │   ├── mod.rs              # IndexAmRoutine populator + handler
│   │   ├── options.rs          # bit_width / dim reloption parsing
│   │   ├── build.rs            # ambuild / ambuildempty
│   │   ├── insert.rs           # aminsert (deferred-commit path)
│   │   ├── scan.rs             # ambeginscan / amrescan / amgettuple / amendscan
│   │   ├── vacuum.rs           # ambulkdelete / amvacuumcleanup
│   │   ├── cost.rs             # amcostestimate (informed model)
│   │   ├── validate.rs         # amvalidate (opclass support fns)
│   │   ├── page.rs             # relfile meta-page byte layout (v2: blocked + codebook)
│   │   └── relfile.rs          # bufmgr-backed page I/O + WAL via GenericXLog
│   └── bin/pgrx_embed.rs
├── benches/                    # criterion + recall harnesses
└── tests/                      # psql regression scripts
```

`src/index/` was introduced in v0.4 as the experimental index AM
and promoted to default-on in v0.9. The relfile-resident format
(`page.rs` + `relfile.rs`) was introduced as a Phase L preview
in v1.1.0 and made the only storage strategy in v1.3.0 (Phase Q,
see an internal design note for the
historical record).

---

## 3. The `vector` SQL type

### 3.1 Phase 1 representation (current)

`Vec` is a `#[derive(PostgresType, Serialize, Deserialize)]`
struct with `#[inoutfuncs]`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct Vector {
    pub data: Vec<f32>,
}
```

On disk this is a CBOR-encoded varlena. Pros: trivial to write and
maintain, automatically TOAST-able, parallel-safe. Cons: ~10–15 %
storage overhead vs raw `f32[dim]`, and `FromDatum` pays one CBOR
deserialization per value.

### 3.2 Binary-compatible varlena layout (deferred)

A hand-rolled `[i32 vl_len_, i16 dim, i16 unused, f32[dim]]`
layout byte-compatible with pgvector's `Vector` would let casts
to/from `pgvector.vector` collapse to a single `memcpy` and let
libpq COPY BINARY clients targeting pgvector reuse their wire
encoders. We considered it for Phase 6 / 1.0 and explicitly
**skipped it**; the rationale lives in
`an internal design note § "Binary-compatible varlena layout"`.

Short version: the cross-extension migration is a one-shot
`UPDATE` through the `real[]` bridge, the storage win comes
from 16× quantisation rather than varlena layout, and the
implementation needs a non-trivial chunk of `unsafe` FFI bypassing
pgrx's `PostgresType` derive. The handoff document
an internal design note enumerates the
work if a future session wants to pick it up.

### 3.3 Text I/O

Format: `'[1, 2, 3]'`. Whitespace tolerated everywhere except inside
numeric literals. NaN and ±Inf rejected at parse time, mirroring
pgvector. Empty `'[]'` rejected (a vector must have at least one
dimension).

### 3.4 Binary I/O (deferred)

The planned `tvector_send` / `tvector_recv` mirror of pgvector's
`vector_send` / `vector_recv` is part of the binary-compatible
varlena layout that we skipped — see § 3.2 above and
an internal design note. Today's
`vector` uses pgrx's CBOR-derived varlena and reaches pgvector
through explicit `real[]` casts.

### 3.5 Casts (current)

| From               | To                  | Implicit? |
|--------------------|---------------------|-----------|
| `real[]`           | `vector`           | no        |
| `double precision[]` | `vector`         | no        |
| `integer[]`        | `vector`           | no        |
| `vector`          | `real[]`            | no        |

NULL elements raise. Non-finite elements raise.

---

## 4. Operators and functions

### 4.1 Operators (all `(vector, vector) -> double precision`)

| Op    | Procedure                    | Phase 2 opclass strategy |
|-------|------------------------------|--------------------------|
| `<->` | `l2_distance`                | exact only (no AM)       |
| `<#>` | `negative_inner_product`     | `vec_ip_ops` strat 1 |
| `<=>` | `cosine_distance`            | `vec_cosine_ops` strat 1 |
| `<+>` | `l1_distance`                | exact only (no AM)       |

`<#>` returns `-dot(a, b)`, not `dot(a, b)`, so that `ORDER BY a <#>
b ASC` returns most-similar-first — same convention as pgvector.

### 4.2 Element-wise arithmetic

| Op | Procedure       |
|----|-----------------|
| `+`| `vec_add`   |
| `-`| `vec_sub`   |
| `*`| `vec_mul` (Hadamard) |

### 4.3 Named functions

```sql
l2_distance         (vector, vector) RETURNS double precision
l2_squared_distance (vector, vector) RETURNS double precision
inner_product       (vector, vector) RETURNS double precision
negative_inner_product (vector, vector) RETURNS double precision
cosine_distance     (vector, vector) RETURNS double precision
l1_distance         (vector, vector) RETURNS double precision
vector_dims         (vector)          RETURNS integer
vector_norm         (vector)          RETURNS double precision
vec_add         (vector, vector) RETURNS vector
vec_sub         (vector, vector) RETURNS vector
vec_mul         (vector, vector) RETURNS vector
vec_normalize   (vector)          RETURNS vector
turbovec_self_score (vector, integer) RETURNS double precision
```

All are `IMMUTABLE PARALLEL SAFE`. Distance accumulators are `f64`,
results are `double precision`. Element output is `f32` (matches
pgvector's `float4` storage in a `vector`).

### 4.4 Dimension policy

- Distance / arithmetic / aggregate over two `vector` operands:
  dimensions must match exactly. Mismatch raises `ERROR`.
- `cosine_distance` returns `NaN` if either operand has zero L2 norm
  (matches pgvector). The result is otherwise clamped to `[0, 2]` to
  defend against sqrt-induced overflow drifting `cos θ` outside
  `[-1, 1]`.

---

## 5. Aggregates

```sql
avg(vector) RETURNS vector
sum(vector) RETURNS vector
```

### 5.1 Internal state

```rust
#[derive(Clone, Debug, Default, Serialize, Deserialize, PostgresType)]
pub struct VecAccum {
    pub sum: Vec<f64>,    // element-wise running sum
    pub count: i64,       // number of values accumulated
}
```

`f64` rather than `f32` accumulators: 1 M `f32` values can lose ~3
decimal digits of precision in a naive sum. Pgvector exhibits the
same drift on its own `avg(vector)` — we deliberately do better.

### 5.2 Transition functions

```sql
vec_accum   (VecAccum, vector) RETURNS VecAccum   -- sfunc
vec_combine (VecAccum, VecAccum) RETURNS VecAccum -- combinefunc
vec_avg_finalfn (VecAccum) RETURNS vector            -- finalfunc for avg
vec_sum_finalfn (VecAccum) RETURNS vector            -- finalfunc for sum
```

Both aggregates are `PARALLEL SAFE`. The first call to
`vec_accum` on a new state allocates the running-sum vector to
the input's dim; subsequent calls validate dim and accumulate.
`vec_combine` rejects mismatched-dim partial states.

The aggregates return `NULL` when no rows match (matches `avg(int)`
behaviour).

---

## 6. Index access method `turbovec` (Phase 2 / Phase 3)

### 6.1 Lifecycle

```sql
CREATE INDEX docs_emb_cosine_idx
    ON docs USING turbovec (embedding vec_cosine_ops)
    WITH (bit_width = 4);
```

The handler is registered via:

```sql
CREATE FUNCTION turbovec_index_handler(internal) RETURNS index_am_handler
  LANGUAGE c AS '$libdir/pg_turbovec', 'turbovec_index_handler';

CREATE ACCESS METHOD turbovec TYPE INDEX HANDLER turbovec_index_handler;
```

`turbovec_index_handler` returns a heap-allocated `IndexAmRoutine`
populated from a Rust-side static. We mirror the slot layout of
`btree_handler` rather than `ivfflat`'s — in particular we set
`amcanorderbyop = true` and `amoptionalkey = true` so the planner
picks us up for `ORDER BY emb <=> $1 LIMIT k`.

### 6.2 Reloptions

```rust
struct TurbovecOptions {
    bit_width: u8,    // 2 | 3 | 4 — default from turbovec.bit_width_default
    dim: i32,         // 0 = auto-detect on first build, else fixed > 0
}
```

Validation in `amoptions`:

- `bit_width ∈ {2, 3, 4}` (turbovec asserts `(2..=4).contains`).
- `dim == 0` or (`dim > 0` and `dim % 8 == 0`) — turbovec requires
  dim be a multiple of 8 internally.

### 6.3 Operator classes

```sql
CREATE OPERATOR CLASS vec_ip_ops
  DEFAULT FOR TYPE vector USING turbovec AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 negative_inner_product(vector, vector);

CREATE OPERATOR CLASS vec_cosine_ops
  FOR TYPE vector USING turbovec AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vector, vector);
```

No L2 / L1 opclass: turbovec's kernel does not score those, and
falling back to the heap is faster than rebuilding the index every
query.

### 6.4 Storage strategy

Three options were considered:

| Option | Pros | Cons | Phase |
|--------|------|------|-------|
| Rebuild on every scan | Trivial; correct | Rebuild cost dominates | discarded |
| Side table `turbovec.am_storage`, backend-local LRU | Reuses existing varlena infra; survives crash via WAL of the side table | Whole-index rewrite on each commit; lock contention; large `payload` is TOASTed | shipped in v1.0.x..v1.2.0 (preview), retired in v1.3.0 |
| Relfile-resident pages, WAL-logged | Crash-safe, incremental, vacuumable, integrates with `shared_buffers` | Need a real page format; lots of new code | **shipped in v1.3.0** (Phase L + Phase P) |

v1.3.0 (Phase Q) ships option C — relfile-resident pages with
persisted SIMD-blocked layout + Lloyd-Max codebook. The on-disk
layout is sketched in [`src/index/page.rs`](../src/index/page.rs);
the v0.4..v1.2 side-table layout shown below is preserved here as
historical context only.

#### On-disk per-vector byte formula (Phase Q-0, wire v7)

Through wire v6, the relfile persisted each vector's quantized codes
**twice**: the row-major bit-plane `packed_codes` chain AND the
SIMD-`blocked` chain (`pack::repack(packed_codes, …)`). The blocked
layout is a pure function of the packed codes, so **v1.27.0 (Phase
Q-0) dropped the blocked chain from disk** and recomputes it once per
backend at index-open (per-query latency unchanged; the recomputed
layout is bit-identical to what used to be persisted). The dominant
O(n) term is now stored ONCE:

```text
per-vector on-disk bytes ≈ dim/8 * bit_width   (codes, ONCE — was ×2 pre-v7)
                         + 4                    (per-vector f32 scale)
                         + 8                    (slot→id u64)
```

plus O(1) fixed overhead (meta page, rotation matrix `dim*dim*4`,
inline codebook). At scale the codes term dominates, so dropping the
second copy roughly HALVES the index. Worked examples (100M vectors,
codes term only):

| dim / bits | codes/vec | pre-v7 (×2) | v7 (×1) |
|---|---|---|---|
| 768 / 2-bit | 192 B | 39.6 GB | **19.8 GB** |
| 768 / 4-bit | 384 B | 78 GB | **39.6 GB** (now fits 40 GB) |
| 1536 / 2-bit | 384 B | 78 GB | **39.6 GB** (now fits 40 GB) |

This cleared the storage blocker for the large-index storage target.
See
[`UPGRADING.md`](UPGRADING.md) — v7 is a wire bump, REINDEX required.

```sql
-- Pre-Phase-Q (v1.0.x..v1.2.0); no longer used.
CREATE TABLE turbovec.am_storage (
    indexrelid  oid PRIMARY KEY,
    bit_width   int4 NOT NULL,
    dim         int4 NOT NULL,
    n_vectors   int8 NOT NULL,
    payload     bytea NOT NULL,
    version     int4 NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
```

A backend-local LRU keyed by `(indexrelid, version)` caches
deserialised index instances; total cache size is bounded
by `turbovec.cache_size_mb`. The cache is shared with the
`turbovec.knn(...)` function path.

As of v1.7.3 (cold-scan parity gap #3) a cache entry holds one
of two variants (`cache::Stored`): a `Mutable`
`Arc<RwLock<IdMapIndex>>` (installed by `aminsert` and
`turbovec.knn`) or a `ReadOnly` `Arc<ReadOnlyIndex>` (installed
by the index-AM scan path). The `ReadOnly` variant stores only
the positional `TurboQuantIndex` + the `slot_to_id` `Vec` and
skips the O(n) `id_to_slot` HashMap that `IdMapIndex` builds
eagerly — the scan path's `search(q, k)` only needs slot→id
translation (a `Vec` index), never id→slot. The first
`aminsert` in a backend evicts the read-only entry (via
`am_lookup_for_mutation` returning `None`) and rebuilds a full
`Mutable` `IdMapIndex`, so the HashMap build is deferred to the
first mutation that actually needs it. A read-only / pooled
backend that only ever scans never pays it.

### 6.5 Callback responsibilities (Phase 2 sketch)

| Callback              | Action                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `ambuild`             | Heap-scan the column, normalise (if GUC on), build `IdMapIndex`, persist to the relfile via `relfile::write_full_with_prepared` (codes/scales/ids chains + rotation + inline codebook + meta page; Phase Q-0 no longer persists the blocked chain — recomputed at open) |
| `ambuildempty`        | INSERT empty payload row                                              |
| `aminsert`            | Lazy-load `IdMapIndex` from relfile pages into cache, `add_with_ids`, mark dirty; `PreCommit` xact callback flushes via `relfile::write_full_with_prepared` once per transaction |
| `ambeginscan`         | Resolve opclass strategy, attach query datum slot                      |
| `amrescan`            | Materialise the query `vector`; if `WHERE` predicate yielded a TID set, build u64 allowlist |
| `amgettuple`          | Build/lookup a read-only `ScanHandle` (a `ReadOnlyIndex`, no `id_to_slot` HashMap) on first call, run `search` on it, then drain cached results |
| `amendscan`           | Drop scan state                                                       |
| `ambulkdelete`        | For each dead CTID, `IdMapIndex::remove(ctid_to_u64)`; mark dirty     |
| `amvacuumcleanup`     | No-op — `ambulkdelete` already wrote the new meta page and truncated trailing pages |
| `amcostestimate`      | Cost ≈ `n_vectors * dim * bit_width / 8 / SIMD_WIDTH * limit`         |
| `amoptions`           | Validate reloptions                                                   |
| `amvalidate`          | Verify opclass support functions resolve                              |

### 6.6 CTID encoding

```text
u64 layout: [reserved 16 | block 32 | offset 16]
```

Block number is the 32-bit `BlockNumber`, offset is the 16-bit
`OffsetNumber`. The reserved high 16 bits are zero in v0.2 and
reserved for a future "epoch" bit so `IdMapIndex` ids survive a
relfile rewrite (CLUSTER, VACUUM FULL).

---

## 7. Filtered search

`turbovec::IdMapIndex::search_with_allowlist(queries, k, &[u64])`
evaluates `WHERE` predicates inside the SIMD kernel, short-circuiting
blocks with no allowed slots before any LUT work — selective
filters get cheaper, not more expensive.

Today this is reachable through `turbovec.knn(rel, id_col,
vec_col, query, k, bit_width, allowed bigint[])` (shipped in
v0.10). The corresponding planner-driven bitmap path through the
index AM is still on the roadmap:

1. The planner picks the `turbovec` index for `ORDER BY <#>` /
   `<=>`. If a `WHERE` clause is also present, the planner may issue
   a bitmap scan for it.
2. An `amgetbitmap` path would merge the bitmap into a `Vec<u64>`
   allowlist and invoke `search_with_allowlist`.
3. For `WHERE tenant_id = $1 ORDER BY emb <=> $2 LIMIT k`, the
   bitmap heap scan stage produces a `TIDBitmap`; we would walk it
   to build the allowlist before invoking the kernel.

---

## 8. Concurrency, WAL, crash safety

- `turbovec::IdMapIndex::search` takes `&self` and is thread-safe via
  internal `OnceLock` caches; pgrx + Postgres semantics restrict us
  to one backend per scan, so concurrent search across backends
  works without further locking.
- `IdMapIndex::add_with_ids` and `swap_remove` take `&mut self` and
  invalidate the SIMD-blocked cache. Mutation is routed through a
  per-`indexrelid` `parking_lot::RwLock` in `src/cache.rs`; Phase
  K's deferred-commit `aminsert` mutates under a write guard and
  defers the relfile rewrite to a `PreCommit` xact callback.
- Crash safety derives from WAL on the index relation's main
  fork: every page write goes through `GenericXLog`
  (`GenericXLogStart` / `RegisterBuffer` / `Finish`), and
  `RelationTruncate` after a shrinking `ambulkdelete` emits
  `XLOG_SMGR_TRUNCATE`. An interrupted `INSERT` rolls back,
  leaving the cached `IdMapIndex` to be discarded on the next
  cache invalidation. **A crash during a long ambuild that has
  not yet committed will leave the index empty until rebuilt**
  — acceptable because Postgres expects that of `CREATE INDEX`.
- Unlogged indexes ship with a populated `INIT_FORKNUM`
  (an internal design note item 2)
  so crash recovery copies the init fork over the main fork,
  restoring an empty queryable index.

### 8.1 Index AM · buffer-cache-only reads

**All index data is read through PostgreSQL's buffer manager**
(`ReadBufferExtended` → pin → `page_data` → `UnlockReleaseBuffer`).
pg_turbovec does **not** mmap or `pread` the relfile directly — the
buffer manager is the single source of truth for page access. This
is required for managed/sandboxed Postgres and gives consistent
pinning/locking and clean crash + streaming-replication semantics.
(v1.5.0–v1.18.x had an opt-out `mmap(MAP_PRIVATE)` fast path for the
static regions; it was removed in v1.19.0. See
`docs/BUFFER_CACHE_ONLY_DESIGN.md`.)

**Correctness backstops (unchanged):** the AM returns TIDs and the
executor calls `heap_fetch`, which enforces transaction visibility —
so a cache entry that lags a just-committed write can only return
TIDs the visibility filter then rejects, never wrong rows.
`xs_recheckorderby = true` makes the executor recompute the exact
ORDER BY distance from the heap tuple, correcting any ranking error
from the lossy quantized scan. Cache invalidation keys on
`(relfilenode, am_version)`: REINDEX or any committed mutation bumps
it and the next `cache::lookup` re-reads through the buffer manager.

**Out-of-core (>RAM) serving** is preserved without mmap: the IVF
cell-scoped path (`OocIvfIndex::search_ooc`) gathers only the probed
cells' contiguous code ranges via `relfile::gather_codes_ranges`,
which reads only those cells' pages through the buffer manager — so
the per-backend resident set stays O(probes·cell_size), not O(n). The
cost the mmap path saved (per-page lookup/pin/lock + avoiding a
second copy on a >shared_buffers index) is traded for full
buffer-cache discipline; pg_turbovec's 7–15× compression mitigates
it by making the index small enough to fit `shared_buffers`.

**Locks:** scans hold `AccessShareLock`, aminsert `RowExclusiveLock`,
ambulkdelete `ShareUpdateExclusiveLock` — unchanged.


## 9. GUCs

| GUC                              | Type | Default | Range          | Notes |
|----------------------------------|------|---------|----------------|-------|
| `turbovec.bit_width_default`     | int  | 4       | 2..=4          | applied when `WITH (bit_width)` is omitted |
| `turbovec.cache_size_mb`         | int  | 256     | 0..=65536      | 0 disables caching (forces rebuild every scan); also the size threshold `turbovec.out_of_core=auto` compares codes-bytes against |
| `turbovec.warn_on_rebuild`       | bool | true    | -              | emit `NOTICE` when a per-backend `IdMapIndex` is reconstructed from the relfile pages |
| `turbovec.search_concurrency`    | int  | 1       | 1..=128        | caps rayon fan-out inside a single batched search |
| `turbovec.normalize_on_insert`   | bool | true    | -              | unit-normalise vectors before passing to `add_with_ids` |
| `turbovec.search_k`              | int  | 32      | 1..=100000     | candidate count per scan batch; the reorder-recheck floor lever (Tier-1 #1a) |
| `turbovec.probes`                | int  | 16      | 1..=65536      | IVF: number of coarse cells scanned per query |
| `turbovec.iterative_scan`        | enum | off     | off, relaxed_order | off is the default since v1.20.1 (the old `relaxed_order` default drove a 450× latency regression — see CHANGELOG) |
| `turbovec.max_scan_tuples`       | int  | 20000   | 1..=10000000   | cap on total candidates examined under iterative refill |
| `turbovec.build_parallelism`     | int  | 0       | 0..=128        | 0 = auto; bounds the rayon pool used by IVF build (k-means + assign-sweep) |
| `turbovec.scan_parallelism`      | int  | 0       | 0..=128        | 0 = auto = `min(cores,4)`; parallelizes the out-of-core per-query fine-scan across probed cells |
| `turbovec.oversample`            | float| 1.0     | 1.0..=100.0    | widens the initial candidate set to `ceil(search_k * oversample)` |
| `turbovec.max_probes`            | int  | 64      | 1..=65536      | cap on IVF probe-set widening under iterative refill |
| `turbovec.out_of_core`           | enum | auto    | off, auto, on  | auto goes cell-scoped only when an IVF index's codes exceed half of `turbovec.cache_size_mb` |
| `turbovec.coarse_graph`          | enum | auto    | off, auto, on  | Phase G-1: navigate an in-memory centroid graph for IVF coarse-cell selection instead of a linear scan, once `lists` is large enough to be worth it; only engages on the out-of-core scan path (needs `turbovec.out_of_core` to have selected cell-scoped serving) |
| `turbovec.allowlist`             | str  | `""`    | CSV of bigint ids | Phase C: per-session heap-TID allowlist ANDed into the scan mask |

`turbovec.mmap_static_blocked` was removed in v1.22.0 (it had been a
deprecated no-op since v1.19.0, when the relfile mmap fast path it
controlled was deleted). `SET turbovec.mmap_static_blocked = ...` now
errors like any other unknown GUC. See CHANGELOG.md.

All are `USERSET` — settable per-session.

---

## 10. Roadmap history

The per-version log is canonical in [`CHANGELOG.md`](../CHANGELOG.md);
please consult it for the deliverables of each release. The
short version of the path from 0.1 to 1.1:

- **Phases 1–3** (v0.1–0.3): `vector` type, distance operators,
  aggregates, casts, kernels module, criterion benches.
- **Phases 4–7** (v0.4–0.7): experimental index AM scaffold under
  `src/index/`, side-table persistence, hardening pass with four
  end-to-end `#[pg_test]` cases that uncovered real bugs.
- **Phases 8–11** (v0.8–0.11): backend-local cache for
  `turbovec.knn()`, AM promoted to default-on, filtered search
  via `IdMapIndex::search_with_allowlist`, scale-up tests at
  d=384 and bit_width=2.
- **Phases 12–16** (v0.12–0.16): forced-index-scan
  `munmap_chunk()` triage, CIC support, functional
  `ambulkdelete`, recall benchmark + pgvector migration cookbook,
  informed `amcostestimate`, end-to-end demo script.
- **Phases 17–21** (v1.0.0-rc.1 → v1.0.0): release-candidate
  prep, real-embedding GloVe-100 recall sweep, the binary-compat
  varlena handoff (deferred per an internal design note), forced-index
  crash root-caused and fixed, million-row arnold sweep, the
  `turbovec.search_k` GUC, AM cache wiring.
- **v1.0.1**: pg13/pg14/pg15/pg18 build compatibility (`#[cfg]`
  gates around AM-callback fields that moved across PG releases).
- **v1.1.0** — **Phase J + K + L**: head-to-head benchmark on
  `dbpedia-entities-openai-1M`, deferred-commit `aminsert`
  (~3000× bulk-INSERT speedup) plus latent
  codebook-recompute and `costestimate` fixes, and the
  relfile-resident page-format preview
  (`--features relfile_storage`, default OFF in v1.1).
- **v1.2.0** — **Phase L hardening + Phase P**: all six Phase L
  items closed (WAL, init fork, truncate, deferred-commit,
  migration NOTICE, in-place ambulkdelete walk); Phase P
  pre-bakes the SIMD-blocked layout + Lloyd-Max codebook into
  the relfile so backends opening the index for the first time
  skip the per-backend `pack::repack` and codebook compute.
  Cold-scan p50 on dbpedia-1M dropped from ~26.5 s to 1.26 s.
- **v1.3.0** — **Phase Q**: side-table storage retired. The
  relfile-resident format is the only storage strategy; the
  `experimental_index_am` and `relfile_storage` Cargo features
  are gone. Hard migration boundary: `ambeginscan` raises
  `ERROR` on a v1.0.x..v1.2 index and asks the user to
  `REINDEX`. See an internal design note
  for the per-item record.

---

## 11. Testing strategy

### 11.1 pgrx unit tests

`#[pg_test]` cases boot a temp cluster via `cargo pgrx test pg17`.
They exercise:

- Text round-trip
- All four distance operators with hand-checked numerical answers
- Dim-mismatch raises `ERROR`
- `avg(vector)` over a 3-row table
- Array casts (both directions)
- `vec_normalize` followed by `vector_norm` returns 1.0
- `turbovec_self_score` of a unit vector returns a high score
  (catches upstream regressions)

### 11.2 SQL regression tests

`tests/01_type_basic.sql` is a psql script intended to be run under
`pg_regress` in Phase 2 (with expected output captured under
`tests/expected/`).

### 11.3 Recall benchmark (Phase 2)

`benches/recall.rs` will run pgvector + pg_turbovec side by side on:

- GloVe-200 (low-dim)
- OpenAI ada-002 (1536)
- OpenAI text-embedding-3 (3072)

at matched bit budgets, recording R@1, R@10, R@100, p50/p95/p99
search latency, and on-disk size.

---

## 12. Open questions and risks

1. **Pgrx index AM API stability across pg17/pg18/pg19.** The
   `IndexAmRoutine` struct gains fields each major release. We will
   feature-gate against `pgrx/pg17`, `pgrx/pg18`, etc., and
   compile-test on each.
2. **Alignment quirks on strict-align platforms.** `f32` data after a
   4-byte int16 + 2 reserved bytes is naturally 4-byte aligned, but
   `VARDATA_ANY` returns a possibly-unaligned pointer. We will
   detoast to a 4-byte form before reading floats.
3. **`turbovec::IdMapIndex::swap_remove` cost.** Each `swap_remove`
   invalidates the SIMD-blocked cache. For delete-heavy tables this
   means an `O(n)` rebuild on the next search. Mitigation: batch
   deletes during VACUUM (Phase 3).
4. **Normalisation policy.** TurboQuant assumes unit-norm inputs.
   `turbovec.normalize_on_insert = true` (default) does the right
   thing transparently. Users who turn this off and feed non-
   unit-norm vectors get measurably worse recall — we should emit a
   `NOTICE` on `CREATE INDEX` when the GUC is off.
5. **Multi-tenant isolation.** The relfile pages live in the
   index relation's main fork; standard PG ACLs and
   row-level-security on the *underlying heap* apply. We do not
   encrypt page payloads at rest beyond what the underlying
   tablespace provides.
6. **Cross-version migration.** Dump/restore preserves the
   index relation. The on-disk wire format is versioned in the
   meta page (currently v2, Phase P). Pre-Phase-P (v1) indexes
   are detected at scan time and the user is asked to `REINDEX`
   (the v1.3.0 hard migration boundary).
7. **Concurrency of `parking_lot::Mutex` inside a Postgres backend.**
   Postgres backends are single-process / single-thread. We use the
   mutex defensively because `rayon` from `turbovec` may spawn
   worker threads inside `search`. Phase 2 will measure signal-
   handling latency under long parallel scans.

---

*Last updated: this commit. When the SQL surface or roadmap changes,
update this document in the same patch.*
