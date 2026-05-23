# Using pg_turbovec

Cookbook-style examples. For the full reference see the source comments
in `src/distance.rs`, `src/aggregate.rs`, `src/cast.rs`, and `src/knn.rs`,
or read the generated `sql/pg_turbovec--<version>.sql`.

## 1. Install and load

```sql
CREATE EXTENSION pg_turbovec;
SET search_path = public, turbovec;
```

All extension objects live in the `turbovec` schema. Set
`search_path` once per session — or qualify references with
`turbovec.` — and the rest of these examples work as written.

## 2. Define a column

```sql
CREATE TABLE docs (
    id        bigserial PRIMARY KEY,
    body      text,
    embedding turbovec.vector
);
```

`vector` accepts any dimension from 1 to 16 000. **A single index
fixes the dimension at build time**, so keep your column homogeneous —
mixed-dim rows are stored fine, but the ANN function will skip rows
with a mismatched dim.

The TurboQuant kernel additionally requires that **dim be a multiple
of 8**. Pad your embeddings to the next multiple of 8 if your model
emits an awkward dimension.

## 3. Insert and read

```sql
INSERT INTO docs (body, embedding) VALUES
  ('hello',  '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'),
  ('world',  '[0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7]');

-- Casting from a Rust / Python emitter that produces an array literal:
INSERT INTO docs (body, embedding)
VALUES ('greeting',
        ARRAY[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]::real[]::vector);
```

## 4. Exact (brute-force) similarity search

For corpora up to ~10 000 rows, brute-force scan with the cosine
operator beats ANN on latency and is exact:

```sql
SELECT id, body, embedding <=> $1 AS distance
FROM   docs
ORDER  BY embedding <=> $1
LIMIT  10;
```

Operators:

| Op    | Meaning                            |
|-------|------------------------------------|
| `<->` | Euclidean (L2) distance            |
| `<#>` | Negative inner product             |
| `<=>` | Cosine distance                    |
| `<+>` | Taxicab (L1) distance              |

`<#>` is *negated* deliberately: under `ORDER BY ... ASC`, the most-
similar row sorts first — same convention as `pgvector`.

## 5. ANN search via `turbovec.knn()`

For larger corpora the function-driven ANN beats brute force by
1–2 orders of magnitude:

```sql
SELECT k.id, k.score, d.body
FROM   turbovec.knn(
         'docs'::regclass,
         'id', 'embedding',
         '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector,
         10
       ) k
JOIN   docs d ON d.id = k.id
ORDER  BY k.score DESC;
```

Higher `score` means more similar (raw inner product on unit
vectors). The function is `STABLE PARALLEL SAFE` and rebuilds the
in-memory index on every call in v0.2 — caching across calls comes
in v0.3.

### 5.1 Tuning bit width

```sql
SELECT * FROM turbovec.knn(
    'docs'::regclass, 'id', 'embedding',
    '[...]'::vector, 10,
    bit_width => 2);   -- 2-bit, 32x compression vs FP32
```

Choices: `2` (most compressed, slight recall loss), `3`, or `4`
(default; near-FP32 recall).

### 5.2 Disabling implicit normalisation

TurboQuant assumes unit-norm inputs. By default we normalise both
the corpus and the query inside `knn()`. If your upstream emits
already-unit vectors and you want to skip the work:

```sql
SET turbovec.normalize_on_insert = off;
```

If you turn this off and feed non-unit vectors, recall *will* drop.

## 6. Aggregates

```sql
SELECT avg(embedding) FROM docs;     -- element-wise mean (centroid)
SELECT sum(embedding) FROM docs WHERE topic = 'pets';
```

Both aggregates use `f64` accumulators internally and merge in
parallel-safe `combinefunc`s, so they run cleanly under
`max_parallel_workers_per_gather > 0`.

## 7. Element-wise arithmetic

```sql
-- Difference of centroids — the classic "is X more like A or B?"
-- composition.
SELECT k.id
FROM   turbovec.knn(
         'docs'::regclass, 'id', 'embedding',
         (SELECT avg(embedding) FROM docs WHERE topic = 'cats')
           - (SELECT avg(embedding) FROM docs WHERE topic = 'dogs'),
         5
       ) k;
```

Operators on `vector`:

| Op  | Function       | Result                      |
|-----|----------------|-----------------------------|
| `+` | `vec_add`  | element-wise sum            |
| `-` | `vec_sub`  | element-wise difference     |
| `*` | `vec_mul`  | Hadamard (element-wise) product |

## 8. Configuration GUCs

| GUC                              | Type | Default | Effect |
|----------------------------------|------|---------|--------|
| `turbovec.bit_width_default`     | int  | 4       | default `bit_width` for indexes built without an explicit reloption |
| `turbovec.cache_size_mb`         | int  | 256     | per-backend cache cap for materialised indexes (v0.3+) |
| `turbovec.warn_on_rebuild`       | bool | true    | NOTICE on rematerialisation |
| `turbovec.search_concurrency`    | int  | 1       | rayon fan-out cap inside a single batched search |
| `turbovec.normalize_on_insert`   | bool | true    | unit-normalise on ingestion / query |

All five are `USERSET` — settable per-session.

## 9. Coexisting with pgvector

`pg_turbovec` and `pgvector` can coexist in the same database. They
own different types (`turbovec.vector` vs `public.vector`) and
their distance operators dispatch by operand type — no collisions.

To migrate from a `pgvector.vector` column to `turbovec.vector`:

```sql
-- Phase 1: add a parallel column.
ALTER TABLE docs ADD COLUMN embedding_tv turbovec.vector;

UPDATE docs
SET    embedding_tv = embedding::real[]::turbovec.vector;

-- Phase 2: drop the old column at your leisure.
ALTER TABLE docs DROP COLUMN embedding;
ALTER TABLE docs RENAME COLUMN embedding_tv TO embedding;
```

## 10. Diagnostics

```sql
SELECT turbovec.turbovec_version();          -- '0.3.0'
SELECT turbovec.vector_dims(emb) FROM docs LIMIT 1;
SELECT turbovec.vector_norm(emb) FROM docs LIMIT 1;
SELECT turbovec.turbovec_self_score(turbovec.vec_normalize(emb), 4)
  FROM docs LIMIT 1;
```

`turbovec_self_score` round-trips a vector through the upstream
`turbovec::IdMapIndex` and reports the inner-product self-score —
useful for verifying the SIMD kernel is producing sane answers on
your hardware.
