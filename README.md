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
- filtered / hybrid ANN — partial index, **in-kernel allowlist**
  (a selective filter gets *cheaper*, not more expensive), or
  iterative scan ([guide](docs/FILTERING.md))
- pgvector-compatible function names (`to_vec`, `array_to_vec`,
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

> **Status:** v1.3.0 - 109/109 `#[pg_test]` cases pass against
> PostgreSQL 13, 14, 15, 16, 17, and 18 with the default build
> flags (the relfile-resident page format and the `turbovec`
> index AM are now default-on; the `experimental_index_am` and
> `relfile_storage` Cargo features were retired in Phase Q).
> See [`docs/ROADMAP_DECISIONS.md`](docs/ROADMAP_DECISIONS.md)
> for what we deliberately skipped, and
> [`docs/PARITY_GAPS.md` § "Performance gaps"](docs/PARITY_GAPS.md)
> for the honest scoreboard of every metric vs pgvector.

## Why pg_turbovec?

**On 1 M × 1536-d real OpenAI `text-embedding-ada-002` embeddings
([`dbpedia-entities-openai-1M`](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M))
pg_turbovec 4-bit matches pgvector HNSW recall at ~10× less storage
and ~1.6× faster p50.**

Head-to-head, warm cache, release build (full sweep:
[`benches/results/recall_dbpedia_1M_2026_05_24.json`](benches/results/recall_dbpedia_1M_2026_05_24.json)):

| Index                          | Storage  | Build  | p50 (warm) | R@10  |
|--------------------------------|---------:|-------:|-----------:|------:|
| pgvector HNSW (ef_search=40)   | 8 192 MB | 4.9 min |    61 ms |  0.962 |
| pgvector HNSW (ef_search=200)  | 8 192 MB | 4.9 min |   115 ms |  0.970 |
| **pg_turbovec 4-bit (k=100)**  |   780 MB | 2.7 min | **71 ms** | **1.000** |
| pg_turbovec 4-bit (k=500)      |   780 MB | 2.7 min |   124 ms |  1.000 |
| **pg_turbovec 2-bit (k=100)**  |   396 MB | 2.1 min | **48 ms** | **1.000** |
| pg_turbovec 2-bit (k=500)      |   396 MB | 2.1 min |    78 ms |  1.000 |

Feature breakdown:

| Feature                          | pg_turbovec       | pgvector + HNSW   |
|----------------------------------|-------------------|-------------------|
| Storage / 1536-dim row (4-bit)   | **≈ 780 B (measured)** | 8 192 B (measured) |
| Build cost (1 M × 1536-d)        | **2.7 min (4-bit) / 2.1 min (2-bit)** | 4.9 min |
| p50 warm @ R@10 ≥ 0.96 (1 M × 1536-d) | **48 ms (2-bit, k=100)** | 61 ms (ef=40) |
| Filtered search                  | In-kernel SIMD allowlist | Post-filter |
| Index AM lifecycle               | CREATE / CIC / aminsert / ambulkdelete / VACUUM / REINDEX | same |
| Distance ops indexed             | `<#>` `<=>` (turbovec kernel) | `<->` `<#>` `<=>` `<+>` (HNSW + IVF) |
| L2 / L1 distance ANN             | exact only        | indexed via HNSW  |
| Halfvec, sparsevec, bitvec       | ✗                 | ✓                 |
| License                          | Apache-2.0        | PostgreSQL        |

If your workload is mostly cosine / inner-product semantic search and
you'd rather pay disk bytes than RAM bytes, `pg_turbovec` is the
right tool. If you need L2 / L1 ANN, halfvec, sparse, or measured
hyperscale latency *today*, pgvector + HNSW is the safer pick.

*Caveat: the dbpedia query set is drawn from inside the corpus, so
rank-1 is trivially the query itself; R@10 is dominated by ranks
2..10. See [`docs/RECALL.md § 2.2`](docs/RECALL.md) for methodology.*

## Why pg_turbovec instead of `binary_quantize() + bit_hamming_ops`?

If you've reached for pgvector's `binary_quantize()` + `bitvec` +
`bit_hamming_ops` HNSW index, the reason is almost always memory
pressure: "I have 100 M × 1536-dim embeddings, FP32 doesn't fit, I'll
trade recall for 32× compression."

**At the same byte budget, pg_turbovec's 2-bit mode wins on recall.**
Measured numbers from
[`benches/results/recall_dbpedia_1M_2026_05_24.json`](benches/results/recall_dbpedia_1M_2026_05_24.json)
on 1 M × 1536-d OpenAI ada-002 embeddings; the 1-bit Hamming line is
the upper bound from the upstream pgvector docs since we don't have a
direct 1 M-row measurement on the same corpus.

| Approach | Bytes / 1536-dim row | R@10 (real OpenAI ada-002 embeddings) |
|---|---:|---:|
| FP32 (raw `vector`) | 6 144 | 1.00 (ground truth) |
| FP16 (`halfvec`) | 3 072 | ≈ 1.00 |
| TurboQuant 4-bit (`turbovec` index, default) | 780 (measured payload / 1 M rows) | **1.000 (search_k = 100)** |
| TurboQuant 2-bit (`turbovec` index) | 396 (measured payload / 1 M rows) | **1.000 (search_k = 100)** |
| 1-bit + Hamming HNSW (pgvector `bit_hamming_ops`) | 192 | ≈ 0.65-0.75 (literature) |

The 4-bit / 2-bit numbers come from the 50-query head-to-head sweep
in [`docs/RECALL.md § 2.2`](docs/RECALL.md); the synthetic random-vector
measurements in [`docs/RECALL.md § 2.1`](docs/RECALL.md) are
deliberately pessimistic because random points have no clustering
structure to exploit - the dbpedia run shows what real embedding
geometry buys you.

**Why does Lloyd-Max scalar quantization beat 1-bit thresholding at
the same byte count?** TurboQuant first rotates the input by a fixed
orthogonal matrix so that, after rotation, each coordinate
independently follows a known Beta distribution that converges to
N(0, 1/d). It then assigns buckets via Lloyd-Max scalar quantization
- provably the *distortion-rate-optimal* scalar code for that
distribution - and packs them at 2, 3, or 4 bits per coordinate.
1-bit thresholding (pgvector's `binary_quantize()`) is the same idea
pinned to `bit_width = 1`: it keeps the sign and throws the magnitude
away. At 2 bits, Lloyd-Max with 4 reconstruction levels lands
materially closer to the Shannon distortion-rate lower bound than a
2-bucket sign threshold can - so pg_turbovec at `bit_width = 2`
occupies essentially the same byte budget as 1-bit Hamming with
strictly higher recall. See the [TurboQuant paper, arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
for the full distortion analysis.

**When is `bit_hamming_ops` still the right tool?** When the bit
vector is *the data*, not a compression of an `f32` vector - i.e.
native binary embeddings (Cohere's binary mode), perceptual / image
fingerprints (pHash, dHash for near-duplicate detection), and
SimHash / MinHash signatures over text shingles. For those
workloads pgvector's HNSW on `bit_hamming_ops` is the right tool
and there is no reason to use pg_turbovec instead.

## Choose your `bit_width`

| Workload | Recommended | Storage / 1536-dim (measured) | R@10 (1 M dbpedia) |
|---|---|---:|---:|
| Want pgvector-equivalent recall, halve storage | `halfvec` (no quantization) | 3 072 B | ≈ 1.0 |
| RAG / semantic search, R@10 ≥ 0.95 acceptable | **`bit_width = 4` (default)** | 780 B | 1.000 |
| Memory pressure dominates, R@10 ≥ 0.85 acceptable | `bit_width = 2` | 396 B | 1.000 |
| Replacing `binary_quantize() + bit_hamming_ops` | `bit_width = 2` (strictly better) | 396 B | 1.000 (vs 0.65-0.75) |

Measured storage and recall come from the head-to-head sweep on
1 M × 1536-d OpenAI ada-002 embeddings; methodology and the synthetic
random-vector numbers (§ 2.1) live in [`docs/RECALL.md`](docs/RECALL.md).
For dimensions other than 1536, multiply storage through by `dim / 1536`.

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

# Or build a stripped-down variant without the index AM:
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

Create a table with a `vector` column:

```sql
CREATE TABLE items (
    id        bigserial PRIMARY KEY,
    body      text,
    embedding vector
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
        ARRAY[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]::real[]::vector);

-- Or via the pgvector-style function:
INSERT INTO items (body, embedding)
VALUES ('hi',  to_vec('[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 8, false));
```

Get the nearest neighbours by cosine distance:

```sql
SELECT id, body, embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector AS dist
FROM   items
ORDER  BY embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector
LIMIT  5;
```

Also supports inner product (`<#>`), L2 (`<->`), and L1 (`<+>`).
`<#>` returns the *negative* inner product so `ORDER BY ... ASC`
returns most-similar-first - same convention as pgvector.

## Storing

```sql
-- Variable dimension:
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector);

-- Or with a runtime dim assertion via a CHECK constraint
-- (typmod-style enforcement without typmod plumbing):
CREATE TABLE items (
    id        bigserial PRIMARY KEY,
    embedding vector,
    CHECK (turbovec.vec_check_dim(embedding, 1536))
);

-- Or add to an existing table:
ALTER TABLE items ADD COLUMN embedding vector;
```

`vector` accepts dim 1..16000 (matching pgvector's cap). The TurboQuant
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

| Op    | Meaning                            | Indexed (turbovec AM)? |
|-------|------------------------------------|------------------------------------------|
| `<->` | Euclidean (L2) distance            | exact only                               |
| `<#>` | negative inner product             | yes - `vec_ip_ops` (default)         |
| `<=>` | cosine distance (`1 - cos θ`)      | yes - `vec_cosine_ops`               |
| `<+>` | taxicab (L1) distance              | exact only                               |

Distances are returned as `double precision`. Distance accumulators
are `f64` internally - `pg_turbovec`'s `avg(vector)` and `sum(vector)`
preserve more precision than pgvector's `f32` accumulators on
million-row corpora.

### Named functions (pgvector-compatible)

```sql
l2_distance(a vector, b vector)            RETURNS double precision
inner_product(a vector, b vector)          RETURNS double precision
cosine_distance(a vector, b vector)        RETURNS double precision
l1_distance(a vector, b vector)            RETURNS double precision
vector_dims(v vector)                       RETURNS integer
vector_norm(v vector)                       RETURNS double precision

to_vec(text)                             RETURNS vector
to_vec(text, integer, boolean)           RETURNS vector  -- with dim check
array_to_vec(real[])                     RETURNS vector
array_to_vec(real[], integer, boolean)   RETURNS vector  -- with dim check

subvector(v vector, start integer, length integer) RETURNS vector
vec_normalize(vector)                   RETURNS vector
vec_zeros(integer)                       RETURNS vector
vec_check_dim(vector, integer)          RETURNS vector  -- assertion
```

### Aggregates

```sql
SELECT avg(embedding) FROM items WHERE topic = 'cats';   -- centroid
SELECT sum(embedding) FROM items;
```

Both are `PARALLEL SAFE` and use `f64` accumulators.

### JSONB I/O

```sql
SELECT '[1, 2.5, -3]'::vector::jsonb;             -- → [1, 2.5, -3]
SELECT '[1, 2.5, -3]'::jsonb::vector;             -- → vector
```


## Indexing

`pg_turbovec` ships an index access method named `turbovec`,
included in the default build (the `experimental_index_am` and
`relfile_storage` Cargo features were retired in v1.3.0; the
relfile-resident storage path is now the only strategy).

```sql
-- Cosine-distance ordering (most common for semantic search):
CREATE INDEX items_emb_idx ON items USING turbovec (embedding vec_cosine_ops)
    WITH (bit_width = 4);

-- Inner-product ordering:
CREATE INDEX items_emb_ip_idx ON items USING turbovec (embedding vec_ip_ops)
    WITH (bit_width = 2);   -- 32× compression vs FP32; some recall loss

-- Online (non-blocking) build:
CREATE INDEX CONCURRENTLY items_emb_idx ON items
    USING turbovec (embedding vec_cosine_ops);
```

Reloptions:

| Option      | Default        | Range  | Notes |
|-------------|----------------|--------|-------|
| `bit_width` | `turbovec.bit_width_default` (4) | 2, 3, 4 | Lower = smaller index, lower recall |

### Index AM lifecycle

The `turbovec` AM supports the full PostgreSQL index lifecycle:

- `CREATE INDEX [CONCURRENTLY]` (parallel build is roadmap)
- `INSERT` → `aminsert` (idempotent on `IdAlreadyPresent` - handles
  CIC's two-pass build and HOT updates)
- `DELETE` + `VACUUM` → `ambulkdelete` (we track every live `u64` id
  in a parallel `Vec` so dead rows are removed correctly)
- `REINDEX [CONCURRENTLY]`
- `DROP INDEX`
- Order-by-op scans (`ORDER BY emb <=> q LIMIT k`) via the executor's
  recheck-orderby path

### Filtered (hybrid) search

Three patterns, one decision matrix — full guide in
[`docs/FILTERING.md`](docs/FILTERING.md): a **partial index**
(`CREATE INDEX ... WHERE tenant_id = X`) for known filter values,
the **in-kernel allowlist** `turbovec.knn(..., allowed)` for
selective per-query id sets, and **iterative scan** for the normal
`ORDER BY ... LIMIT` ergonomics. The allowlist (shown below) pushes
the id set into the SIMD scoring loop — a *selective* filter gets
cheaper, not more expensive (the kernel short-circuits 32-vector
blocks whose allowed-slot mask is empty before any LUT lookup;
measured crossover at ~7–10% selectivity).

```sql
-- Top-10 nearest, restricted to a tenant or topic:
SELECT k.id, d.body
FROM   turbovec.knn(
         'items'::regclass,
         'id', 'embedding',
         '[...]'::vector,
         10, 4,
         ARRAY(SELECT id FROM items WHERE tenant_id = $1)::bigint[]
       ) k
JOIN   items d USING (id)
ORDER  BY k.score DESC;
```

The function-driven `turbovec.knn(...)` API and the
`turbovec` index AM share the same TurboQuant kernel and the
same backend-local cache; pick whichever fits your query
shape (the AM integrates with `ORDER BY ... LIMIT`; `knn(...)`
lets you pass an allowlist for hybrid retrieval).

## Performance

> **Operations note: `shared_buffers`.** As of v1.5.1 (Phase
> R-3), the bulk of a pg_turbovec index — the persisted
> SIMD-blocked codes, the rotation matrix, and the inline
> codebook — is read from disk via per-backend
> `mmap(MAP_PRIVATE)` of the relfile, **bypassing PG's buffer
> manager entirely** for those bytes. The OS page cache is the
> authoritative cache, and `shared_buffers` size no longer
> bounds warm-scan latency on the dominant chains.
>
> You still want `shared_buffers` big enough to hold the meta
> page, the codes/scales/ids chains (which stay on the buffer
> manager because VACUUM mutates them in place), and your other
> relations. A few hundred MiB is fine; 1.5× the index size is
> no longer required.
>
> The fall-back GUC `turbovec.mmap_static_blocked = off` reverts
> to the v1.4.x buffer-manager-only read path on a per-session
> basis. With it off, the v1.4.x advice applies:
> `shared_buffers ≥ 2 × (sum of all turbovec indexes you query
> hot)` to keep the warm-scan profile clean.
>
> Full diagnosis: [`docs/RECALL.md § 2.5`](docs/RECALL.md) (the
> v1.4.0 buffer-manager-bound profile) and
> [`docs/RECALL.md § 2.6`](docs/RECALL.md) (the v1.5.1 mmap
> fix); architecture +
> isolation contract:
> [`docs/ARCHITECTURE.md § 8.1`](docs/ARCHITECTURE.md#81-index-am--mmap-isolation-contract).

> **Performance methodology.** The headline numbers in the
> table at the top of this README come from a real
> head-to-head against pgvector 0.8.0 HNSW on the
> [`dbpedia-entities-openai-1M`](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
> corpus (1 M Wikipedia/DBpedia entities × 1536-d OpenAI
> `text-embedding-ada-002` embeddings) running on a single Intel
> i9-12900H box with 32 GiB RAM and PG 17.9 from the pgrx-managed
> install tree. Full methodology, query set, ground-truth
> generation, and reproduction scripts in
> [`docs/RECALL.md § 2.2`](docs/RECALL.md) and
> [`benches/scripts/`](benches/scripts/). The synthetic-uniform
> tables below are from a pure-Rust kernel bench and are kept for
> historical comparison - they understate real-world recall because
> uniform-random vectors have no clustering structure for
> quantization to exploit.

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
exploit - real embeddings (GloVe, OpenAI ada-002) recall meaningfully
better. Reproduction:

```bash
cargo bench --bench recall --no-default-features --features pg16
```

Real-world fixtures via the `TURBOVEC_FIXTURE_PATH` env var; format
documented in [`docs/RECALL.md`](docs/RECALL.md) § 6.1.

### Compression (from the TurboQuant paper)

| dim   | FP32 / vector | TurboQuant 4-bit / vector | TurboQuant 2-bit / vector |
|-------|-----------:|-----------------------:|-----------------------:|
|   128 |    512 B   |                  68 B  |                  36 B  |
|   384 |  1 536 B   |                 196 B  |                 100 B  |
|   768 |  3 072 B   |                 388 B  |                 196 B  |
|  1536 |  6 144 B   |                 772 B  |                 388 B  |
|  3072 | 12 288 B   |               1 540 B  |                 772 B  |

A 10 M-row × 1536-dim corpus that needs ~62 GiB of RAM as FP32 fits in
~7.7 GiB at 4-bit and ~3.9 GiB at 2-bit - without any data-dependent
codebook training.

### Search speed (from the TurboQuant paper, x86 AVX-512BW)

100 K vectors, 1 K queries, k=64, single-threaded:

- TurboQuant **matches or beats** FAISS `IndexPQFastScan` at every
  4-bit configuration tested (d=384, 768, 1536, 3072).
- TurboQuant runs **within ±1%** of FAISS at 2-bit single-threaded.
- On ARM (Apple M3 Max), TurboQuant **beats** FAISS by 12-20% at
  every config the paper measured.

We have not yet run pg_turbovec end-to-end against pgvector + HNSW or
pgvectorscale + StreamingDiskANN. That comparison is the next item
on the v1.0.0 roadmap; see [`docs/RECALL.md`](docs/RECALL.md).

### How it works

TurboQuant compresses each vector to 2/3/4 bits per coordinate using:

1. **Normalize** - strip the L2 norm; store as a single `f32` scale.
2. **Random rotation** - multiply by a fixed orthogonal matrix so
   each coordinate independently follows a known Beta distribution.
3. **Lloyd-Max scalar quantisation** - bucket each coordinate into
   2/3/4-bit codes optimal for the known distribution.
4. **Bit-pack** - `dim` coordinates → `dim * bit_width / 8` bytes.
5. **Length-renormalised scoring** - one extra scalar per vector
   removes the inner-product downward bias the quantiser introduces.

No codebook training, no data passes - adding vectors is `O(dim)` per
vector with no rebuild as the corpus grows. Search rotates the query
once and scores directly against the bit-packed codes via SIMD
nibble-LUT kernels (NEON, AVX2, AVX-512BW).


## Configuration

`pg_turbovec` exposes five GUCs under the `turbovec.*` namespace
(USERSET - settable per session):

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

`pg_turbovec` and `pgvector` coexist cleanly - different schema, type
name, and operator-dispatch table. See
[`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md)
for the full cookbook. TL;DR:

```sql
ALTER TABLE docs ADD COLUMN embedding_tv turbovec.vector;

UPDATE docs SET embedding_tv = embedding::real[]::turbovec.vector;

CREATE INDEX CONCURRENTLY docs_emb_tv_idx
    ON docs USING turbovec (embedding_tv vec_cosine_ops)
    WITH (bit_width = 4);
```

A binary-compatible `vector` varlena layout (zero-copy cast to/from
pgvector's `vector`) is on the v1.0 roadmap; until then the `real[]`
bridge is the supported interop path.

## Reference

- **Type:** `vector` (variable dimension, `f32` coordinates, 1..16000)
- **Schema:** `turbovec` (set on the search_path or fully qualify)
- **Operator classes:** `vec_ip_ops` (default, `<#>`),
  `vec_cosine_ops` (`<=>`)
- **Index AM:** `turbovec` (build with `WITH (bit_width = 2|3|4)`)
- **Aggregates:** `avg(vector)`, `sum(vector)`
- **Full surface listing:** [`docs/USAGE.md`](docs/USAGE.md) and the
  generated `sql/pg_turbovec--<version>.sql` after
  `cargo pgrx schema`.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - module map,
  type / operator / aggregate signatures, index AM contract, GUC
  semantics, phased roadmap.
- [`docs/USAGE.md`](docs/USAGE.md) - cookbook covering install,
  exact + ANN search, aggregates, arithmetic, tuning.
- [`docs/PRODUCTION.md`](docs/PRODUCTION.md) - **deployment guide**:
  install, GUC tuning, replication, monitoring, troubleshooting.
- [`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md)
  - hands-on migration with query rewrite tables and a feature
  comparison.
- [`docs/FILTERING.md`](docs/FILTERING.md) - **filtering & hybrid
  search guide**: partial index vs in-kernel allowlist `knn()` vs
  iterative scan, a decision matrix, and the measured allowlist
  selectivity crossover.
- [`docs/INDEXAM.md`](docs/INDEXAM.md) - implementation guide for
  the `turbovec` index access method.
- [`docs/RECALL.md`](docs/RECALL.md) - recall benchmark methodology
  and the latest measured numbers.
- [`docs/ROADMAP_DECISIONS.md`](docs/ROADMAP_DECISIONS.md) - what
  we deliberately did *not* ship in 1.0 (binary-compat varlena,
  `bit_hamming_ops` ANN) and the reasoning.
- [`docs/PARITY_GAPS.md`](docs/PARITY_GAPS.md) - feature-by-feature
  comparison against pgvector.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) - the published
  head-to-head benchmark (Cohere-wiki 1M vs pgvector HNSW), incl.
  the AVX2 latency frontier and the honest "flat-scan loses on
  latency, wins on storage + exact recall" finding.
- [`docs/IVF_PLAN.md`](docs/IVF_PLAN.md) - design plan for the IVF
  coarse-quantizer layer that would turn the flat O(n) scan into a
  tunable sublinear ANN structure (the path to a competitive
  latency story at scale).
- [`docs/COMPETITIVE_ANALYSIS.md`](docs/COMPETITIVE_ANALYSIS.md) /
  [`docs/COMPETITIVE_LANDSCAPE_2026-06.md`](docs/COMPETITIVE_LANDSCAPE_2026-06.md)
  - positioning vs pgvector, VectorChord, pgvectorscale, Qdrant.
- [`docs/PHASE19_PROGRESS.md`](docs/PHASE19_PROGRESS.md) - handoff
  notes for the binary-compatible varlena work, if a future session
  picks it up.
- [`docs/BUILDING.md`](docs/BUILDING.md) - Nix-specific build
  recipe (writable `pg_config` wrapper, `BINDGEN_EXTRA_CLANG_ARGS`,
  `RUSTFLAGS` for openblas).
- [`RELEASING.md`](RELEASING.md) - release process, version-bump
  checklist, Codeberg release flow.
- [`CHANGELOG.md`](CHANGELOG.md) - phase-by-phase release notes.
- [`tests/`](tests/) - psql regression scripts you can run yourself
  (`cargo pgrx run pg16`, then `\i tests/03_full_demo.sql`).

## FAQ

**Is `pg_turbovec` a drop-in replacement for `pgvector`?**

No, by design. We coexist: type name `vector`, schema `turbovec`,
operator dispatch by argument type. Pgvector users have years of
`vector(1536)` columns and tooling - pretending to be a drop-in would
silently change semantics around normalisation and recall. The
[migration cookbook](docs/MIGRATING_FROM_PGVECTOR.md) shows the
explicit `real[]` bridge.

**What about `halfvec`, `sparsevec`, `bit`?**

Not supported. `pg_turbovec` quantises full-precision `f32` input -
half-precision halfvec and sparse-vector representations don't map
cleanly onto the TurboQuant kernel.

**What about L2 / L1 ANN?**

The TurboQuant kernel scores inner-product on unit-normalised vectors.
We expose `l2_distance` / `l1_distance` as exact functions only - there
is no L2 / L1 index path. For workloads dominated by Euclidean ANN,
pgvector + HNSW is the right pick.

**What's not in 1.0?**

TL;DR - see [`docs/ROADMAP_DECISIONS.md`](docs/ROADMAP_DECISIONS.md)
for the full cost/benefit reasoning. The two items most likely to be
asked about:

- **Binary-compatible varlena layout for `vector`.** We use a
  CBOR-derived varlena rather than pgvector's
  `[vl_len_, dim, unused, f32[dim]]` byte layout. The cross-extension
  migration via `::real[]::vector` is one-shot and finishes in
  seconds on a million rows; the 16× quantization savings dominate
  the per-row layout overhead, so binary-compat is a nice-to-have,
  not a 1.0 blocker.
- **Indexed Hamming / Jaccard ANN on `bitvec`.** The TurboQuant
  kernel is a scalar quantizer for dense `f32` vectors; it doesn't
  fit Hamming-space ANN. And the workload that motivates
  `bit_hamming_ops` (memory-pressured semantic search) is already
  covered better by `bit_width = 2` - same byte budget, materially
  higher recall.

**Why two crates: `pg_turbovec` and `turbovec`?**

[`turbovec`](https://crates.io/crates/turbovec) is the upstream
TurboQuant implementation in pure Rust by Ryan Codrai. `pg_turbovec`
is the PostgreSQL extension built with [pgrx](https://github.com/pgcentralfoundation/pgrx)
on top of it. We track upstream releases.

## Ecosystem

Vector search on PostgreSQL has three serious open-source options.
The honest comparison:

- **[pgvector](https://github.com/pgvector/pgvector)** - production-
  tested at scale, larger feature surface (HNSW for L2 *and* inner
  product *and* L1, plus `halfvec`, `sparsevec`, `bitvec`), and an
  ecosystem of clients that already speak its types and operators.
  Choose pgvector if you don't have memory pressure and you value
  maturity.
- **[pgvectorscale](https://github.com/timescale/pgvectorscale)** -
  SOTA published latency on 50 M+ row corpora via StreamingDiskANN,
  layered on top of pgvector. Choose pgvectorscale if your corpus is
  in the tens-of-millions-of-rows range and you can run TimescaleDB.
- **pg_turbovec** - smallest on-disk footprint, in-kernel filtered
  ANN (selective `WHERE` clauses make scans *cheaper*, not more
  expensive), zero codebook training. Choose pg_turbovec if memory
  dominates your cost equation.

The three coexist cleanly in the same database - separate schemas,
separate type oids, separate operator dispatch. You can A/B them on
your own data without committing to one.

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
  (ICLR 2026) - the algorithm.
- **The pgvector authors** for setting the API conventions
  (`<-> <#> <=> <+>`, `to_vector`, `array_to_vector`, `subvector`,
  `vector_dims`, `vector_norm`) we mirror.
- **The pgrx maintainers** for making PostgreSQL extension
  development in Rust possible.

## License

Apache-2.0 © Greg Burd. See [`LICENSE`](LICENSE).
