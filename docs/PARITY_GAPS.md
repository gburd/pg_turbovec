# pgvector parity gap tracker

What pgvector offers (as of 0.8.x) and where pg_turbovec stands.

## Types

| pgvector type | pg_turbovec status |
|---------------|--------------------|
| `vector` (FP32) | ✓ — `turbovec.vector` |
| `halfvec` (FP16) | ✗ — Phase HV (planned) |
| `sparsevec` | ✗ — Phase SV (planned) |
| `bit` (binary) | ✗ — Phase BV (planned, uses Postgres core `bit` type for input + a wrapper) |

## Distance operators

| Op | pgvector | pg_turbovec |
|----|----------|-------------|
| `<->` L2 | ✓ | ✓ (exact only on AM) |
| `<#>` neg-IP | ✓ | ✓ (indexed) |
| `<=>` cosine | ✓ | ✓ (indexed) |
| `<+>` L1 | ✓ | ✓ (exact only on AM) |
| `<~>` Hamming (binary) | ✓ | ✗ — Phase BV |
| `<%>` Jaccard (binary) | ✓ | ✗ — Phase BV |

## Aggregates

| Aggregate | pgvector | pg_turbovec |
|-----------|----------|-------------|
| `avg(vector)` | ✓ | ✓ |
| `sum(vector)` | ✓ | ✓ |
| `avg(halfvec)` | ✓ | ✗ |
| `sum(halfvec)` | ✓ | ✗ |
| `avg(bit)` | ✗ | ✗ |
| `sum(sparsevec)` | ✓ | ✗ |

## Functions

| Function | pgvector | pg_turbovec |
|----------|----------|-------------|
| `l2_distance` | ✓ | ✓ |
| `inner_product` | ✓ | ✓ |
| `cosine_distance` | ✓ | ✓ |
| `l1_distance` | ✓ | ✓ |
| `vector_dims` | ✓ | ✓ |
| `vector_norm` | ✓ | ✓ |
| `subvector` | ✓ | ✓ |
| `to_vector(text)` | ✓ | ✓ (`to_vec`) |
| `to_vector(text, integer, boolean)` | ✓ | ✓ |
| `array_to_vector(real[])` | ✓ | ✓ (cast + `array_to_vec`) |
| `array_to_vector(real[], integer, boolean)` | ✓ | ✓ |
| `vector_to_float4(vector, integer, boolean)` | ✓ | ✗ — Phase HV |
| `binary_quantize(vector)` | ✓ | ✗ — Phase BV |
| `hamming_distance(bit, bit)` | ✓ | ✗ — Phase BV |
| `jaccard_distance(bit, bit)` | ✓ | ✗ — Phase BV |
| `l2_normalize(vector)` | ✓ | ✓ (`vec_normalize`) |
| `vector_dims(halfvec)` | ✓ | ✗ |
| `vector_dims(sparsevec)` | ✓ | ✗ |

## Index AMs

| AM | pgvector | pg_turbovec |
|----|----------|-------------|
| `ivfflat` | ✓ (Lloyd k-means) | ✗ |
| `hnsw` | ✓ | ✗ |
| `turbovec` | n/a | ✓ — TurboQuant flat IVF-like |

## Operator classes

| Opclass family | pgvector | pg_turbovec |
|----------------|----------|-------------|
| `vector_l2_ops` (ivfflat + hnsw) | ✓ | ✗ — TurboQuant kernel doesn't index L2 |
| `vector_ip_ops` | ✓ | ✓ (`vec_ip_ops`, default) |
| `vector_cosine_ops` | ✓ | ✓ (`vec_cosine_ops`) |
| `vector_l1_ops` (hnsw) | ✓ | ✗ |
| `halfvec_*_ops` | ✓ | ✗ |
| `sparsevec_*_ops` | ✓ | ✗ |
| `bit_hamming_ops` | ✓ | ✗ |
| `bit_jaccard_ops` | ✓ | ✗ |

## Phase plan

- **Phase HV** — add `halfvec` (FP16) type. Storage: `[i32 vl_len_, i16
  dim, i16 unused, f16[dim]]`. Operators / aggregates / casts.
  Conversion functions to/from `vector`. `halfvec_*_ops` for the
  index AM.
- **Phase SV** — add `sparsevec` type for high-dim sparse data
  (CSR-style: nnz int4, indices int4[nnz], values f32[nnz]).
- **Phase BV** — add `bit`/`bitvec` flavour: binary quantisation,
  Hamming + Jaccard distance, opclasses for the AM.
- **Phase L2** — investigate whether the TurboQuant kernel can
  support indexed L2 / L1 ANN. Likely requires a separate kernel
  variant; may not be feasible without upstream changes.
