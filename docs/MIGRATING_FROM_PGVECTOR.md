# Migrating from pgvector to pg_turbovec

A pragmatic, copy-paste-friendly guide to running the two
extensions side-by-side and migrating columns and queries.

`pg_turbovec` and `pgvector` use compatible *text* representations
(`'[1, 2, 3]'`) but different *binary* varlena layouts. That means:

- Casting via `text` or `real[]` is always safe.
- Direct binary cast (`vector::tvector`) is **not** supported in
  v0.x. Roadmap item: a binary-compatible varlena layout in a
  future release.

For now use the `real[]` bridge — Postgres optimises the cast
chain reasonably well and the conversion is O(dim).

## 1. Coexistence

Both extensions can be installed in the same database without
collisions: `pgvector`'s `vector` lives in the `public` schema by
default; `pg_turbovec`'s `tvector` lives in `turbovec`. Distance
operator symbols (`<->`, `<#>`, `<=>`, `<+>`) are reused but
dispatched by argument type, so:

```sql
SELECT '[1,2,3]'::vector            <-> '[1,2,3]'::vector;            -- pgvector
SELECT '[1,2,3]'::turbovec.tvector <-> '[1,2,3]'::turbovec.tvector;  -- pg_turbovec
```

both compile and resolve to their respective implementations.

## 2. Convert a single column

Most pgvector tables look like:

```sql
CREATE TABLE docs (
    id        bigint PRIMARY KEY,
    body      text,
    embedding vector(1536)
);
```

Add a parallel `tvector` column:

```sql
ALTER TABLE docs ADD COLUMN embedding_tv turbovec.tvector;

-- One-shot conversion. real[] is the intermediate format.
UPDATE docs SET embedding_tv = embedding::real[]::turbovec.tvector;

-- Or, in batches if the table is large:
DO $$
DECLARE
    batch_size int := 10000;
    last_id    bigint := 0;
    n          int;
BEGIN
    LOOP
        UPDATE docs
        SET    embedding_tv = embedding::real[]::turbovec.tvector
        WHERE  id > last_id
          AND  embedding IS NOT NULL
          AND  embedding_tv IS NULL
        ORDER  BY id
        LIMIT  batch_size;
        GET DIAGNOSTICS n = ROW_COUNT;
        EXIT WHEN n = 0;
        SELECT max(id) INTO last_id FROM docs WHERE embedding_tv IS NOT NULL;
        COMMIT;
    END LOOP;
END$$;
```

## 3. Build a turbovec index

```sql
CREATE INDEX CONCURRENTLY docs_emb_tv_idx
    ON docs USING turbovec (embedding_tv tvector_cosine_ops)
    WITH (bit_width = 4);
```

Drop the old pgvector index when you're satisfied with the new
one:

```sql
DROP INDEX docs_emb_idx;
```

## 4. Migrate queries

| pgvector                                              | pg_turbovec                                              |
|-------------------------------------------------------|----------------------------------------------------------|
| `SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10` | `SELECT id FROM docs ORDER BY embedding_tv <=> $1::tvector LIMIT 10` |
| `SELECT id FROM docs ORDER BY embedding <#> $1 LIMIT 10` | `SELECT id FROM docs ORDER BY embedding_tv <#> $1::tvector LIMIT 10` |
| `SELECT id FROM docs ORDER BY embedding <-> $1 LIMIT 10` *(L2)* | exact only — no AM. Use `ORDER BY l2_distance(embedding_tv, $1)` |
| `embedding <+> $1` *(L1)*                             | exact only — `l1_distance(embedding_tv, $1)`             |

For ANN-only workloads you can also bypass the index entirely
and use `turbovec.knn()`, which is the recommended API for large
corpora until the index AM exits experimental:

```sql
SELECT k.id, d.body
FROM   turbovec.knn(
         'docs'::regclass,
         'id', 'embedding_tv',
         $1::tvector,
         10,
         4                      -- bit_width
       ) k
JOIN   docs d USING (id)
ORDER  BY k.score DESC;
```

### Filtered ANN

`pgvector` does post-filter: it asks the index for `k * 10` rows
and discards those that don't match the WHERE. `pg_turbovec`
pushes the filter into the SIMD kernel:

```sql
SELECT k.id
FROM   turbovec.knn(
         'docs'::regclass,
         'id', 'embedding_tv',
         $1::tvector, 10, 4,
         ARRAY(SELECT id FROM docs WHERE tenant_id = $2)::bigint[]
       ) k
ORDER BY k.score DESC;
```

The kernel short-circuits 32-vector blocks containing zero
allowed slots before any LUT work, so selective filters get
*cheaper*, not more expensive.

## 5. Aggregates

```sql
-- Centroid (centre of mass).
SELECT avg(embedding_tv) FROM docs WHERE topic = 'cats';

-- Sum (e.g. for batch-mean update in IVF).
SELECT sum(embedding_tv) FROM docs;
```

`pg_turbovec`'s aggregates use `f64` accumulators internally —
they preserve precision better than pgvector's `f32` accumulators
on corpora ≥ 1 M rows.

## 6. Coexistence checklist

| Item                     | pgvector | pg_turbovec | Notes                                  |
|--------------------------|---------:|------------:|----------------------------------------|
| Type name                | `vector` | `tvector`   | namespaced under `turbovec`            |
| Default storage          | `extended` | `extended` | both varlena, both TOAST-able          |
| Storage per 1536-dim row | 6 144 B  | ≈ 388 B (4-bit) | `pg_turbovec` is ~16× smaller    |
| Distance ops             | `<-> <#> <=> <+>` | `<-> <#> <=> <+>` | dispatch by operand type        |
| Index AMs                | `ivfflat`, `hnsw` | `turbovec` | one AM, two opclasses (IP, cosine) |
| Filtered ANN             | post-filter | in-kernel allowlist | kernel short-circuits empty blocks |
| Halfvec / sparsevec      | yes      | no          | not on roadmap                         |
| `subvector`              | yes      | yes         | identical SQL signature                |
| JSONB casts              | no       | yes         | `tvector_to_jsonb`, `jsonb_to_tvector` |

## 7. When **not** to migrate

- You depend on pgvector's `halfvec` (16-bit float) or `sparsevec`
  types — pg_turbovec doesn't expose those.
- Your queries are dominated by `<->` (L2). pg_turbovec's index
  doesn't accelerate L2; you'd lose pgvector's HNSW speed-up.
- Recall floor matters more than memory. pgvector + HNSW with
  high `ef_search` reliably hits R@10 ≈ 1.0 on real
  embeddings; pg_turbovec at 4-bit is ~0.88 in our synthetic
  tests (real embeddings recall better — see `docs/RECALL.md`).

For everyone else, the storage savings and in-kernel filtered
search make the swap worthwhile.
