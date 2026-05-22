# pg_turbovec

Open-source vector similarity search for PostgreSQL, backed by Google
Research's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm
via the [`turbovec`](https://crates.io/crates/turbovec) Rust crate.

Store your vectors in 2- or 4-bit-quantised form alongside the rest of
your data. Supports:

- exact and approximate nearest-neighbour search
- single-precision (`f32`) vectors with on-disk compression to ~16×
  smaller than `pgvector` at 4-bit
- L2 distance (`<->`), inner product (`<#>`), cosine distance (`<=>`),
  L1 distance (`<+>`)
- in-kernel filtered ANN — push your `WHERE` predicate into the SIMD
  scoring loop so selective filters get *cheaper*, not more expensive
- pgvector-compatible function names (`to_tvector`, `array_to_tvector`,
  `subvector`, `vector_dims`, `vector_norm`, `inner_product`,
  `l2_distance`, `cosine_distance`, `l1_distance`)
- any [language](https://www.postgresql.org/docs/current/external-pl.html)
  with a Postgres client

Plus [ACID](https://en.wikipedia.org/wiki/ACID) compliance, point-in-
time recovery, JOINs, GUCs, parallel-safe aggregates, and all of the
[other great features](https://www.postgresql.org/about/) of Postgres.

[![Rust 1.85+](https://img.shields.io/badge/rust-1.85+-93450a)](https://www.rust-lang.org/)
[![PostgreSQL 16+](https://img.shields.io/badge/postgres-16+-336791)](https://www.postgresql.org/)
[![Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

> **Status:** v1.0.0-rc.2 — release candidate. 65/65 `#[pg_test]`
> cases pass against PostgreSQL 16; `cargo clippy -D warnings` is
> clean. Open work tracked for 1.0.0 final: pgvector-binary-compatible
> varlena layout, real-world recall validation against pgvector, and
> WAL-logged persistent index pages. See [CHANGELOG.md](CHANGELOG.md)
> for the phase-by-phase release notes.

## Why pg_turbovec?

| Feature                          | pg_turbovec       | pgvector + HNSW   |
|----------------------------------|-------------------|-------------------|
| Storage / 1536-dim row (4-bit)   | **≈ 388 B**       | 6 144 B           |
| Build cost                       | Zero training, single pass | Multi-pass HNSW build |
| Filtered search                  | In-kernel SIMD allowlist | Post-filter |
| Index AM lifecycle               | CREATE / CIC / aminsert / ambulkdelete / VACUUM / REINDEX | same |
| Distance ops indexed             | `<#>` `<=>` (turbovec kernel) | `<->` `<#>` `<=>` `<+>` (HNSW + IVF) |
| L2 / L1 distance ANN             | exact only        | indexed via HNSW  |
| Halfvec, sparsevec, bitvec       | ✗                 | ✓                 |
| Query latency at 1 M × 1536, k=10 | not yet measured | published numbers |
| License                          | Apache-2.0        | PostgreSQL        |

If your workload is mostly cosine / inner-product semantic search and
you'd rather pay disk bytes than RAM bytes, `pg_turbovec` is the
right tool. If you need L2 / L1 ANN, halfvec, sparse, or measured
hyperscale latency *today*, pgvector + HNSW is the safer pick.

## Installation

`pg_turbovec` requires PostgreSQL 16+ and a Rust toolchain ≥ 1.85.

```bash
# One-time setup.
cargo install --locked cargo-pgrx --version 0.17.0
cargo pgrx init                # bootstraps a private PostgreSQL cluster

# Build & install into the dev cluster.
git clone https://codeberg.org/gregburd/pg_turbovec
cd pg_turbovec
cargo pgrx install --release   # default features include the index AM

# Or build a stripped-down variant without the experimental index AM:
cargo pgrx install --release --no-default-features --features pg16
```

For a Nix-based build (the dev environment for this project) see
[`docs/BUILDING.md`](docs/BUILDING.md). For migrating from pgvector
see [`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md).

## Getting Started

Enable the extension (do this once per database):

```sql
CREATE EXTENSION pg_turbovec;
SET search_path = public, turbovec;
```

Create a table with a `tvector` column:

```sql
CREATE TABLE items (
    id        bigserial PRIMARY KEY,
    body      text,
    embedding tvector
);
```

Insert vectors:

```sql
INSERT INTO items (body, embedding) VALUES
  ('hello',  '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'),
  ('world',  '[0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7]');

-- Or via array cast:
INSERT INTO items (body, embedding)
VALUES ('greeting',
        ARRAY[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]::real[]::tvector);

-- Or via the pgvector-style function:
INSERT INTO items (body, embedding)
VALUES ('hi',  to_tvector('[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 8, false));
```

Get the nearest neighbours by cosine distance:

```sql
SELECT id, body, embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::tvector AS dist
FROM   items
ORDER  BY embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::tvector
LIMIT  5;
```

Also supports inner product (`<#>`), L2 (`<->`), and L1 (`<+>`).
`<#>` returns the *negative* inner product so `ORDER BY ... ASC`
returns most-similar-first — same convention as pgvector.

## Storing

```sql
-- Variable dimension:
CREATE TABLE items (id bigserial PRIMARY KEY, embedding tvector);

-- Or with a runtime dim assertion via a CHECK constraint
-- (typmod-style enforcement without typmod plumbing):
CREATE TABLE items (
    id        bigserial PRIMARY KEY,
    embedding tvector,
    CHECK (turbovec.tvector_check_dim(embedding, 1536))
);

-- Or add to an existing table:
ALTER TABLE items ADD COLUMN embedding tvector;
```

`tvector` accepts dim 1..16000 (matching pgvector's cap). The TurboQuant
kernel additionally requires dim be a multiple of 8 when used with the
index AM; pad your embeddings if your model emits something awkward
(e.g. 384 = 48 × 8 ✓, 1536 = 192 × 8 ✓).

Insert vectors in bulk via `COPY`:

```sql
COPY items (embedding) FROM STDIN WITH (FORMAT TEXT);
[1,2,3,4,5,6,7,8]
[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
\.
```

Upsert via `ON CONFLICT`:

```sql
INSERT INTO items (id, embedding) VALUES (1, '[1,2,3,4,5,6,7,8]')
    ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
```

## Querying

### Distance operators

| Op    | Meaning                            | Indexed (with `experimental_index_am`)? |
|-------|------------------------------------|------------------------------------------|
| `<->` | Euclidean (L2) distance            | exact only                               |
| `<#>` | negative inner product             | yes — `tvector_ip_ops` (default)         |
| `<=>` | cosine distance (`1 - cos θ`)      | yes — `tvector_cosine_ops`               |
| `<+>` | taxicab (L1) distance              | exact only                               |

Distances are returned as `double precision`. Distance accumulators
are `f64` internally — `pg_turbovec`'s `avg(tvector)` and `sum(tvector)`
preserve more precision than pgvector's `f32` accumulators on
million-row corpora.

### Named functions (pgvector-compatible)

```sql
l2_distance(a tvector, b tvector)            RETURNS double precision
inner_product(a tvector, b tvector)          RETURNS double precision
cosine_distance(a tvector, b tvector)        RETURNS double precision
l1_distance(a tvector, b tvector)            RETURNS double precision
vector_dims(v tvector)                       RETURNS integer
vector_norm(v tvector)                       RETURNS double precision

to_tvector(text)                             RETURNS tvector
to_tvector(text, integer, boolean)           RETURNS tvector  -- with dim check
array_to_tvector(real[])                     RETURNS tvector
array_to_tvector(real[], integer, boolean)   RETURNS tvector  -- with dim check

subvector(v tvector, start integer, length integer) RETURNS tvector
tvector_normalize(tvector)                   RETURNS tvector
tvector_zeros(integer)                       RETURNS tvector
tvector_check_dim(tvector, integer)          RETURNS tvector  -- assertion
```

### Aggregates

```sql
SELECT avg(embedding) FROM items WHERE topic = 'cats';   -- centroid
SELECT sum(embedding) FROM items;
```

Both are `PARALLEL SAFE` and use `f64` accumulators.

### JSONB I/O

```sql
SELECT '[1, 2.5, -3]'::tvector::jsonb;             -- → [1, 2.5, -3]
SELECT '[1, 2.5, -3]'::jsonb::tvector;             -- → tvector
```


## Indexing

`pg_turbovec` ships an index access method named `turbovec`, included
in the default build via the `experimental_index_am` Cargo feature.

```sql
-- Cosine-distance ordering (most common for semantic search):
CREATE INDEX items_emb_idx ON items USING turbovec (embedding tvector_cosine_ops)
    WITH (bit_width = 4);

-- Inner-product ordering:
CREATE INDEX items_emb_ip_idx ON items USING turbovec (embedding tvector_ip_ops)
    WITH (bit_width = 2);   -- 32× compression vs FP32; some recall loss

-- Online (non-blocking) build:
CREATE INDEX CONCURRENTLY items_emb_idx ON items
    USING turbovec (embedding tvector_cosine_ops);
```

Reloptions:

| Option      | Default        | Range  | Notes |
|-------------|----------------|--------|-------|
| `bit_width` | `turbovec.bit_width_default` (4) | 2, 3, 4 | Lower = smaller index, lower recall |

### Index AM lifecycle

The `turbovec` AM supports the full PostgreSQL index lifecycle:

- `CREATE INDEX [CONCURRENTLY]` (parallel build is roadmap)
- `INSERT` → `aminsert` (idempotent on `IdAlreadyPresent` — handles
  CIC's two-pass build and HOT updates)
- `DELETE` + `VACUUM` → `ambulkdelete` (we track every live `u64` id
  in a parallel `Vec` so dead rows are removed correctly)
- `REINDEX [CONCURRENTLY]`
- `DROP INDEX`
- Order-by-op scans (`ORDER BY emb <=> q LIMIT k`) via the executor's
  recheck-orderby path

### Filtered (hybrid) search

`pg_turbovec` pushes `WHERE` predicates into the SIMD scoring loop —
selective filters get cheaper, not more expensive. The kernel
short-circuits 32-vector blocks whose entire allowed-slot mask is
empty before any LUT lookup.

```sql
-- Top-10 nearest, restricted to a tenant or topic:
SELECT k.id, d.body
FROM   turbovec.knn(
         'items'::regclass,
         'id', 'embedding',
         '[…]'::tvector,
         10, 4,
         ARRAY(SELECT id FROM items WHERE tenant_id = $1)::bigint[]
       ) k
JOIN   items d USING (id)
ORDER  BY k.score DESC;
```

The function-driven `turbovec.knn(...)` API is recommended for
production filtered ANN today; the `experimental_index_am` route
is fully supported but the AM scan path is newer.

## Performance

> **Honest caveat.** As of v1.0.0-rc.2 we have **not** run head-to-
> head benchmarks against pgvector's HNSW or pgvectorscale's
> StreamingDiskANN. The numbers below are from the upstream
> [TurboQuant paper](https://arxiv.org/abs/2504.19874) and from our
> own pure-Rust kernel benches; treat them as the upper bound the
> Postgres SQL layer can deliver.

### Recall (synthetic, 1 000 random unit-norm vectors, 50 queries)

| dim | bit_width | R@1  | R@10 | R@100 |
|----:|----------:|-----:|-----:|------:|
| 128 |         2 | 0.40 | 0.65 |  0.76 |
| 128 |         4 | 0.80 | 0.89 |  0.93 |
| 384 |         2 | 0.34 | 0.62 |  0.76 |
| 384 |         4 | 0.78 | 0.89 |  0.93 |
| 768 |         2 | 0.50 | 0.62 |  0.76 |
| 768 |         4 | 0.82 | 0.88 |  0.92 |

Random vectors have no clustering structure for the quantiser to
exploit — real embeddings (GloVe, OpenAI ada-002) recall meaningfully
better. Reproduction:

```bash
cargo bench --bench recall --no-default-features --features pg16
```

Real-world fixtures via the `TURBOVEC_FIXTURE_PATH` env var; format
documented in [`docs/RECALL.md`](docs/RECALL.md) § 6.1.

### Compression (from the TurboQuant paper)

| dim   | FP32 / vec | TurboQuant 4-bit / vec | TurboQuant 2-bit / vec |
|-------|-----------:|-----------------------:|-----------------------:|
|   128 |    512 B   |                  68 B  |                  36 B  |
|   384 |  1 536 B   |                 196 B  |                 100 B  |
|   768 |  3 072 B   |                 388 B  |                 196 B  |
|  1536 |  6 144 B   |                 772 B  |                 388 B  |
|  3072 | 12 288 B   |               1 540 B  |                 772 B  |

A 10 M-row × 1536-dim corpus that needs ~62 GiB of RAM as FP32 fits in
~7.7 GiB at 4-bit and ~3.9 GiB at 2-bit — without any data-dependent
codebook training.

### Search speed (from the TurboQuant paper, x86 AVX-512BW)

100 K vectors, 1 K queries, k=64, single-threaded:

- TurboQuant **matches or beats** FAISS `IndexPQFastScan` at every
  4-bit configuration tested (d=384, 768, 1536, 3072).
- TurboQuant runs **within ±1%** of FAISS at 2-bit single-threaded.
- On ARM (Apple M3 Max), TurboQuant **beats** FAISS by 12–20% at
  every config the paper measured.

We have not yet run pg_turbovec end-to-end against pgvector + HNSW or
pgvectorscale + StreamingDiskANN. That comparison is the next item
on the v1.0.0 roadmap; see [`docs/RECALL.md`](docs/RECALL.md).

### How it works

TurboQuant compresses each vector to 2/3/4 bits per coordinate using:

1. **Normalize** — strip the L2 norm; store as a single `f32` scale.
2. **Random rotation** — multiply by a fixed orthogonal matrix so
   each coordinate independently follows a known Beta distribution.
3. **Lloyd-Max scalar quantisation** — bucket each coordinate into
   2/3/4-bit codes optimal for the known distribution.
4. **Bit-pack** — `dim` coordinates → `dim * bit_width / 8` bytes.
5. **Length-renormalised scoring** — one extra scalar per vector
   removes the inner-product downward bias the quantiser introduces.

No codebook training, no data passes — adding vectors is `O(dim)` per
vector with no rebuild as the corpus grows. Search rotates the query
once and scores directly against the bit-packed codes via SIMD
nibble-LUT kernels (NEON, AVX2, AVX-512BW).


## Configuration

`pg_turbovec` exposes five GUCs under the `turbovec.*` namespace
(USERSET — settable per session):

| GUC                              | Type | Default | Range          |
|----------------------------------|------|---------|----------------|
| `turbovec.bit_width_default`     | int  | `4`     | `2..=4`        |
| `turbovec.cache_size_mb`         | int  | `256`   | `0..=65536`    |
| `turbovec.warn_on_rebuild`       | bool | `true`  | -              |
| `turbovec.search_concurrency`    | int  | `1`     | `1..=128`      |
| `turbovec.normalize_on_insert`   | bool | `true`  | -              |

```sql
-- Compress harder during this session:
SET turbovec.bit_width_default = 2;

-- Disable the backend-local index cache:
SET turbovec.cache_size_mb = 0;
```

## Migrating from pgvector

`pg_turbovec` and `pgvector` coexist cleanly — different schema, type
name, and operator-dispatch table. See
[`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md)
for the full cookbook. TL;DR:

```sql
ALTER TABLE docs ADD COLUMN embedding_tv turbovec.tvector;

UPDATE docs SET embedding_tv = embedding::real[]::turbovec.tvector;

CREATE INDEX CONCURRENTLY docs_emb_tv_idx
    ON docs USING turbovec (embedding_tv tvector_cosine_ops)
    WITH (bit_width = 4);
```

A binary-compatible `tvector` varlena layout (zero-copy cast to/from
pgvector's `vector`) is on the v1.0 roadmap; until then the `real[]`
bridge is the supported interop path.

## Reference

- **Type:** `tvector` (variable dimension, `f32` coordinates, 1..16000)
- **Schema:** `turbovec` (set on the search_path or fully qualify)
- **Operator classes:** `tvector_ip_ops` (default, `<#>`),
  `tvector_cosine_ops` (`<=>`)
- **Index AM:** `turbovec` (build with `WITH (bit_width = 2|3|4)`)
- **Aggregates:** `avg(tvector)`, `sum(tvector)`
- **Full surface listing:** [`docs/USAGE.md`](docs/USAGE.md) and the
  generated `sql/pg_turbovec--<version>.sql` after
  `cargo pgrx schema`.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map,
  type / operator / aggregate signatures, index AM contract, GUC
  semantics, phased roadmap.
- [`docs/USAGE.md`](docs/USAGE.md) — cookbook covering install,
  exact + ANN search, aggregates, arithmetic, tuning.
- [`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md)
  — hands-on migration with query rewrite tables and a feature
  comparison.
- [`docs/INDEXAM.md`](docs/INDEXAM.md) — implementation guide for
  the `turbovec` index access method.
- [`docs/RECALL.md`](docs/RECALL.md) — recall benchmark methodology
  and the latest measured numbers.
- [`docs/BUILDING.md`](docs/BUILDING.md) — Nix-specific build
  recipe (writable `pg_config` wrapper, `BINDGEN_EXTRA_CLANG_ARGS`,
  `RUSTFLAGS` for openblas).
- [`RELEASING.md`](RELEASING.md) — release process, version-bump
  checklist, Codeberg release flow.
- [`CHANGELOG.md`](CHANGELOG.md) — phase-by-phase release notes.
- [`tests/`](tests/) — psql regression scripts you can run yourself
  (`cargo pgrx run pg16`, then `\i tests/03_full_demo.sql`).

## FAQ

**Is `pg_turbovec` a drop-in replacement for `pgvector`?**

No, by design. We coexist: type name `tvector`, schema `turbovec`,
operator dispatch by argument type. Pgvector users have years of
`vector(1536)` columns and tooling — pretending to be a drop-in would
silently change semantics around normalisation and recall. The
[migration cookbook](docs/MIGRATING_FROM_PGVECTOR.md) shows the
explicit `real[]` bridge.

**What about `halfvec`, `sparsevec`, `bit`?**

Not supported. `pg_turbovec` quantises full-precision `f32` input —
half-precision halfvec and sparse-vector representations don't map
cleanly onto the TurboQuant kernel.

**What about L2 / L1 ANN?**

The TurboQuant kernel scores inner-product on unit-normalised vectors.
We expose `l2_distance` / `l1_distance` as exact functions only — there
is no L2 / L1 index path. For workloads dominated by Euclidean ANN,
pgvector + HNSW is the right pick.

**Why two crates: `pg_turbovec` and `turbovec`?**

[`turbovec`](https://crates.io/crates/turbovec) is the upstream
TurboQuant implementation in pure Rust by Ryan Codrai. `pg_turbovec`
is the PostgreSQL extension built with [pgrx](https://github.com/pgcentralfoundation/pgrx)
on top of it. We track upstream releases.

## Contributing

Issues and patches: <https://codeberg.org/gregburd/pg_turbovec>.

```bash
# Run the full test suite (boots a private PG cluster):
cargo pgrx test pg16

# Run pure-Rust kernel + recall benches (no Postgres):
cargo bench --bench distance --no-default-features --features pg16
cargo bench --bench recall   --no-default-features --features pg16

# Lints + format:
cargo clippy --features pg16 --tests -- -D warnings
cargo fmt --all -- --check
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

- **Ryan Codrai** for the [`turbovec`](https://github.com/RyanCodrai/turbovec)
  Rust crate and the SIMD kernels that do all the actual work.
- **Google Research** for [TurboQuant](https://arxiv.org/abs/2504.19874)
  (ICLR 2026) — the algorithm.
- **The pgvector authors** for setting the API conventions
  (`<-> <#> <=> <+>`, `to_vector`, `array_to_vector`, `subvector`,
  `vector_dims`, `vector_norm`) we mirror.
- **The pgrx maintainers** for making PostgreSQL extension
  development in Rust possible.

## License

Apache-2.0 © Greg Burd. See [`LICENSE`](LICENSE).
