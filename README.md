# pg_turbovec — Vector Index for PostgreSQL (TurboQuant)

> **Status:** v1.0.0-rc.2 — release candidate.
>
> **🚧 RC2 — testing in progress. Not yet recommended for production
> data; please file issues against any rough edge you hit.**
>
> 62/62 `#[pg_test]` cases pass against a real PostgreSQL 16 cluster
> (default build with `experimental_index_am`); 51/51 pass on the
> kernel-only / `--no-default-features` build. Stable surface: type /
> operators / aggregates / casts / `turbovec.knn()` /
> `CREATE INDEX ... USING turbovec` / `CREATE INDEX CONCURRENTLY` /
> aminsert / ambulkdelete via VACUUM / REINDEX / forced-index-scan
> (`SET enable_seqscan = off`).
>
> Open work tracked for 1.0.0 final:
> 1. Binary-compatible varlena layout with pgvector's `vector` (Phase
>    19, sibling-agent worktree).
> 2. WAL-logged persistent index pages (Phase 20).
>
> See [`CHANGELOG.md`](CHANGELOG.md) and [`RELEASING.md`](RELEASING.md).

`pg_turbovec` is a PostgreSQL extension that provides a vector data type
and an approximate-nearest-neighbour index access method, built in Rust
with [pgrx](https://github.com/pgcentralfoundation/pgrx) on top of the
[`turbovec`](https://github.com/RyanCodrai/turbovec) crate.

It mirrors the public SQL surface of [`pgvector`](https://github.com/pgvector/pgvector)
so existing applications and ORMs (SQLAlchemy, ActiveRecord, Diesel,
sqlx, …) work with minimal changes — but uses TurboQuant's data-oblivious
2/4-bit scalar quantizer to fit a 10 M-document corpus in 4 GB instead
of 31 GB and search it faster than FAISS PQ.

To avoid colliding with `pgvector` when both are installed, the type is
named `tvector` and lives in the `turbovec` schema. Casts to and from
`pgvector`'s `vector` type are provided when both extensions are present.

## Why pg_turbovec?

| Feature | pg_turbovec (TurboQuant) | pgvector (HNSW/IVF + FP32) |
|---|---|---|
| Storage per 1536-dim vector at 4-bit | **~ 384 B** | 6 144 B (FP32) |
| Build cost | Zero training, single pass | Multi-pass, codebook training (PQ) |
| Search kernel | Hand-tuned NEON / AVX-512BW | Generic FP32 |
| Filtered search | SIMD allowlist at block level | Post-filter |
| Dimensionality cap | 16 000 (varlena) | 16 000 |
| License | Apache-2.0 | PostgreSQL |

## Features

### v0.5.0 (current default build)

- **`tvector` type** — variable dimension `f32` vectors with text and
  binary I/O (`'[1, 2, 3]'::tvector`, COPY BINARY, libpq binary).
- **Distance operators**:
  - `<->` Euclidean (L2) distance
  - `<#>` negative inner product
  - `<=>` cosine distance
  - `<+>` taxicab (L1) distance
- **Functions**: `l2_distance`, `inner_product`, `cosine_distance`,
  `l1_distance`, `vector_dims`, `vector_norm`, `tvector_normalize`,
  `tvector_random_unit`, `turbovec_self_score`, `subvector`,
  `tvector_check_dim`, `tvector_zeros`, `tvector_to_text`.
- **JSONB I/O**: `tvector_to_jsonb`, `jsonb_to_tvector`, plus casts.
- **Aggregates**: `avg(tvector)`, `sum(tvector)` — `f64` accumulators,
  `PARALLEL SAFE`.
- **Casts**: explicit `real[]` / `double precision[]` / `integer[]`
  / `jsonb` ↔ `tvector`.
- **`turbovec.knn(rel, id_col, vec_col, query, k, bit_width)`** —
  function-driven ANN search backed by `turbovec::IdMapIndex`.
  Returns `TABLE(id bigint, score float8)`, ordered by score DESC.
- **GUCs**: `turbovec.bit_width_default`, `turbovec.cache_size_mb`,
  `turbovec.warn_on_rebuild`, `turbovec.search_concurrency`,
  `turbovec.normalize_on_insert`.

### v0.4.0 — experimental index access method (opt-in)

Build with `--features experimental_index_am`:

```bash
cargo pgrx install --release --features experimental_index_am
```

Then:

```sql
CREATE INDEX docs_emb_idx
    ON docs USING turbovec (embedding tvector_cosine_ops)
    WITH (bit_width = 4);

SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;
```

The scaffold is complete (`IndexAmRoutine` callbacks, side-table
persistence, operator classes for inner product and cosine) but
**untested against a real cluster**. Read `docs/INDEXAM.md` before
enabling on data you care about.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design,
phased roadmap, and risks.

## Installation

`pg_turbovec` requires PostgreSQL 14+ and a Rust toolchain ≥ 1.85.

```bash
# Install cargo-pgrx (one-time setup).
cargo install --locked cargo-pgrx --version 0.17.0
cargo pgrx init

# Build and install the extension into the dev cluster.
git clone https://codeberg.org/gregburd/pg_turbovec
cd pg_turbovec
cargo pgrx install --release

# Then in a connected psql session:
CREATE EXTENSION pg_turbovec;
```

## Quick start

```sql
CREATE EXTENSION pg_turbovec;
SET search_path = public, turbovec;

CREATE TABLE items (
    id     bigserial PRIMARY KEY,
    body   text,
    embedding tvector
);

INSERT INTO items (body, embedding) VALUES
  ('hello',  '[0.1, 0.2, 0.3]'),
  ('world',  '[0.2, 0.1, 0.4]');

-- exact (no index) cosine-distance lookup
SELECT id, body, embedding <=> '[0.1, 0.2, 0.3]' AS dist
FROM items
ORDER BY dist
LIMIT 5;

-- Phase 2: approximate, index-backed nearest neighbour
-- CREATE INDEX ON items USING turbovec (embedding tvector_cosine_ops)
--   WITH (bit_width = 4);
```

## Layout

```
pg_turbovec/
├── Cargo.toml
├── pg_turbovec.control
├── README.md
├── docs/
│   └── ARCHITECTURE.md
├── migrations/                  # versioned SQL declarations
├── sql/                         # generated by `cargo pgrx schema`
├── src/
│   ├── lib.rs
│   ├── tvector.rs               # type + I/O
│   ├── distance.rs              # operators / functions
│   ├── aggregate.rs             # avg, sum
│   ├── guc.rs                   # GUC registration
│   └── bin/pgrx_embed.rs
└── tests/                       # SQL regression tests
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design, module
  map, type/operator/aggregate signatures, index AM contract,
  GUCs, phased roadmap.
- [`docs/USAGE.md`](docs/USAGE.md) — cookbook with install,
  exact + ANN search, aggregates, arithmetic, GUCs.
- [`docs/MIGRATING_FROM_PGVECTOR.md`](docs/MIGRATING_FROM_PGVECTOR.md)
  — hands-on migration from pgvector with query rewrite tables
  and a feature comparison.
- [`docs/INDEXAM.md`](docs/INDEXAM.md) — implementation guide
  for the `turbovec` index access method, callback responsibilities,
  storage strategy, known issues, Phase 13+ plans.
- [`docs/RECALL.md`](docs/RECALL.md) — recall benchmark
  methodology and latest measured numbers.
- [`docs/BUILDING.md`](docs/BUILDING.md) — step-by-step Nix build
  recipe.
- [`CHANGELOG.md`](CHANGELOG.md) — phase-by-phase release notes.
- [`tests/`](tests/) — psql regression scripts you can run
  yourself with `cargo pgrx run pg16` then `\i tests/...`.

## License

Apache-2.0 © Greg Burd. See [`LICENSE`](LICENSE).

## Contributing

Issues and patches: <https://codeberg.org/gregburd/pg_turbovec>.

For the algorithmic background, please read
[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874).
