# pgvector parity gap tracker

What pgvector offers (as of 0.8.x) and where pg_turbovec stands.

## Types

| pgvector type | pg_turbovec status |
|---------------|--------------------|
| `vector` (FP32) | ✓ — `turbovec.vector` |
| `halfvec` (FP16) | ✓ — `turbovec.halfvec` |
| `sparsevec` | ✓ — `turbovec.sparsevec` |
| `bit` (binary) | ✓ — `turbovec.bitvec` (named differently to avoid colliding with PG core's built-in `bit`) |

## Distance operators

| Op | pgvector | pg_turbovec |
|----|----------|-------------|
| `<->` L2 | ✓ | ✓ (vector, halfvec, sparsevec; exact only on AM) |
| `<#>` neg-IP | ✓ | ✓ (indexed for vector) |
| `<=>` cosine | ✓ | ✓ (indexed for vector) |
| `<+>` L1 | ✓ | ✓ (vector, halfvec, sparsevec; exact only on AM) |
| `<~>` Hamming (binary) | ✓ | ✓ (bitvec) |
| `<%>` Jaccard (binary) | ✓ | ✓ (bitvec) |

## Aggregates

| Aggregate | pgvector | pg_turbovec |
|-----------|----------|-------------|
| `avg(vector)` | ✓ | ✓ |
| `sum(vector)` | ✓ | ✓ |
| `avg(halfvec)` | ✓ | ✓ |
| `sum(halfvec)` | ✓ | ✓ |
| `sum(sparsevec)` | ✓ | ✓ |

## Functions

| Function | pgvector | pg_turbovec |
|----------|----------|-------------|
| `l2_distance` | ✓ | ✓ |
| `inner_product` | ✓ | ✓ |
| `cosine_distance` | ✓ | ✓ |
| `l1_distance` | ✓ | ✓ |
| `vector_dims(vector)` | ✓ | ✓ |
| `vector_dims(halfvec)` | ✓ | ✓ |
| `vector_dims(sparsevec)` | ✓ | ✓ |
| `vector_norm(vector)` | ✓ | ✓ |
| `vector_norm(halfvec)` | ✓ | ✓ |
| `subvector` | ✓ | ✓ |
| `to_vector(text)` | ✓ | ✓ (also `to_vec`) |
| `to_vector(text, integer, boolean)` | ✓ | ✓ |
| `array_to_vector(real[])` | ✓ | ✓ (cast + `array_to_vec`) |
| `array_to_vector(real[], integer, boolean)` | ✓ | ✓ |
| `vector_to_float4(vector, integer, boolean)` | ✓ | ✓ |
| `binary_quantize(vector)` | ✓ | ✓ |
| `hamming_distance(bitvec, bitvec)` | ✓ | ✓ |
| `jaccard_distance(bitvec, bitvec)` | ✓ | ✓ |
| `l2_normalize(vector)` | ✓ | ✓ (also `vec_normalize`) |

## Index AMs

| AM | pgvector | pg_turbovec |
|----|----------|-------------|
| `ivfflat` | ✓ (Lloyd k-means) | ✗ |
| `hnsw` | ✓ | ✗ |
| `turbovec` | n/a | ✓ — TurboQuant flat IVF-like |

## Operator classes

| Opclass family | pgvector | pg_turbovec |
|----------------|----------|-------------|
| `vector_l2_ops` (ivfflat + hnsw) | ✓ | ✓ — `vec_l2_ops` (uses recheck-orderby; quality matches cosine for unit-norm vectors) |
| `vector_ip_ops` | ✓ | ✓ (`vec_ip_ops`, default) |
| `vector_cosine_ops` | ✓ | ✓ (`vec_cosine_ops`) |
| `vector_l1_ops` (hnsw) | ✓ | ✓ — `vec_l1_ops` (recheck-orderby; candidate-set quality is approximate, recheck makes final order exact) |
| `halfvec_*_ops` | ✓ | ✓ via expression index: `CREATE INDEX ... USING turbovec ((emb::vector) vec_cosine_ops)` |
| `sparsevec_*_ops` | ✓ | ✓ via expression index, same pattern (note: dense-cast cost on each row may dominate for very high-dim sparse) |
| `bit_hamming_ops` | ✓ | ✗ — TurboQuant kernel doesn't fit Hamming-space ANN; use the exact `<~>` operator (no index) |
| `bit_jaccard_ops` | ✓ | ✗ — same |

## Phase plan

- ~~**Phase HV** — add `halfvec` (FP16) type.~~ ✓ done.
- ~~**Phase SV** — add `sparsevec` type.~~ ✓ done.
- ~~**Phase BV** — add `bitvec` type, Hamming + Jaccard.~~ ✓ done.
- ~~**Phase L2** — indexed L2 / L1 ANN.~~ ✓ done via `vec_l2_ops` /
  `vec_l1_ops` and the existing recheck-orderby path.
- **Phase BV-IDX** — binary-vector ANN index. The TurboQuant kernel
  doesn't fit Hamming-space ANN; if we want indexed bitvec we'd
  need a separate kernel (LSH or multi-index hashing). Out of
  scope for the 1.0 line.
- **Phase BC** — binary-compatible varlena layout for `vector` so
  casts to/from `pgvector.vector` are zero-copy. See
  `docs/PHASE19_PROGRESS.md`.