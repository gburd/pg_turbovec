# pgvector parity gap tracker

What pgvector offers (as of 0.8.x) and where pg_turbovec stands.

## Performance gaps (the honest scoreboard)

This section enumerates **known performance regressions** of
pg_turbovec vs pgvector. They are correctness-OK in every case;
the trade-off is that the wins (10× less storage, 5× faster warm
scan) come paired with these losses, which we are working to
close.

| Metric (1 M × 384-d cosine, release build, arnold) | pgvector HNSW | pg_turbovec | Status |
|---|---:|---:|---|
| Storage | 1 953 MiB | 195 MiB (4-bit) | ✅ we win 10× |
| Build time | 8 m 13 s | 33 s | ✅ we win 15× |
| Warm scan p50 | 100 ms | 22 ms | ✅ we win 5× |
| Cold scan p50 (after backend restart) | ~100 ms | 6 800 ms (1 M × 384-d) / 26 310 ms (1 M × 1536-d) | ❌ **we lose 68× (384-d) / 130× (1536-d)** — Phase L preview *did not close it*; Phase P (shared-mem IdMap or pre-baked blocked layout) is the actual fix |
| INSERT throughput (per row, into a 1 M-row index) | ~0.5 ms (HNSW O(log n)) | ~200 ms (full re-serialise per row) | ❌ **we lose ~400×** |
| Recall on uniform-random | 0.03 | 1.000 | ✅ (but synthetic; real-world recall varies) |
| Recall on real OpenAI ada-002 (dbpedia-1M) | ~0.95 | TBD (see `docs/RECALL.md` §2.2) | ❌ / ✅ (depends on `search_k`) |

### Cold-cache latency (the relfile-resident page format)

v1.0 stores the serialised index in a side-table
(`turbovec.am_storage`) read via SPI on first access. Every
fresh PostgreSQL backend pays the full SPI fetch + HashMap
construct cost (~6.8 s on 1 M × 384-d), then caches the result
in a per-backend `Arc<IdMapIndex>`. Connection pools that
create-and-destroy backends, or VACUUM workers, hit this every
time.

pgvector's HNSW lives in the index relation's main fork and is
cached in shared_buffers cluster-wide - first scan after a
restart is the same ~100 ms as the warm scan.

**Fix in flight:** ~~Phase L — relfile-resident page format
putting our serialised index into the index relation's main
fork via the buffer manager.~~ Phase L 1-6 landed in v1.2.0 and
the in-flight default-flip work for v1.3.0 (commit 9e8ee81), but
[§2.3 of `docs/RECALL.md`](RECALL.md#23-cold-scan-latency-relfile-resident-page-format-commit-9e8ee81)
shows the relfile path on dbpedia-1M still p50s at 26 310 ms cold.
Reason: per-backend Lloyd-Max codebook + 793 MB blocked-layout
repack dominate, not the SPI fetch that relfile_storage replaces.
**Phase P (queued):** pre-bake the blocked layout into the relfile
or ship a shared-memory `IdMapIndex` cache so backends don't repeat
the prep work. `relfile_storage` stays gated for v1.3.0 — do not
flip the default ON until cold-scan p50 < 500 ms.

### INSERT throughput (the deferred-commit pattern)

v1.0 `aminsert` calls `persist::load(indexrelid)` (full SPI
fetch) → mutates the in-memory `IdMapIndex` →
`persist::save(...)` (full re-serialise). Every inserted row
pays ~2× 195 MiB of TOAST I/O on a 1 M-row index. A bulk
`INSERT ... SELECT` of 1 M rows would take ~55 hours; pgvector
handles the same in minutes via O(log n) graph walks.

**Fix in flight:** Phase K - hold the modified `IdMapIndex` in
the per-backend cache (clone-on-write or RwLock), register a
`XactCallback` to persist exactly once on `XACT_EVENT_COMMIT`,
so N-row bulk inserts do 1 × (load+save) instead of N ×
(load+save).

### Recall tuning

The trade-off knob is `turbovec.search_k` (default 100). On the
384-d synthetic corpus, K=100 gave R@10 = 1.000 because the
uniform distribution makes ~all candidates within rounding of
each other. On real-world embedding distributions (1536-d
ada-002, GloVe-100), recall depends on K:

- Low K (50-100): low latency (10s of ms), recall ~0.85-0.92.
- High K (500-2000): higher latency (50-100s of ms), recall
  approaches 1.0.

Phase M (post-Phase J) will pick a default that hits ~0.95 on
dbpedia-1M without breaking the warm-p50 latency story.

## Types

| pgvector type | pg_turbovec status |
|---------------|--------------------|
| `vector` (FP32) | ✓ - `turbovec.vector` |
| `halfvec` (FP16) | ✓ - `turbovec.halfvec` |
| `sparsevec` | ✓ - `turbovec.sparsevec` |
| `bit` (binary) | ✓ - `turbovec.bitvec` (named differently to avoid colliding with PG core's built-in `bit`) |

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
| `turbovec` | n/a | ✓ - TurboQuant flat IVF-like |

## Operator classes

| Opclass family | pgvector | pg_turbovec |
|----------------|----------|-------------|
| `vector_l2_ops` (ivfflat + hnsw) | ✓ | ✓ - `vec_l2_ops` (uses recheck-orderby; quality matches cosine for unit-norm vectors) |
| `vector_ip_ops` | ✓ | ✓ (`vec_ip_ops`, default) |
| `vector_cosine_ops` | ✓ | ✓ (`vec_cosine_ops`) |
| `vector_l1_ops` (hnsw) | ✓ | ✓ - `vec_l1_ops` (recheck-orderby; candidate-set quality is approximate, recheck makes final order exact) |
| `halfvec_*_ops` | ✓ | ✓ via expression index: `CREATE INDEX ... USING turbovec ((emb::vector) vec_cosine_ops)` |
| `sparsevec_*_ops` | ✓ | ✓ via expression index, same pattern (note: dense-cast cost on each row may dominate for very high-dim sparse) |
| `bit_hamming_ops` | ✓ | ✗ - TurboQuant kernel doesn't fit Hamming-space ANN; use the exact `<~>` operator (no index) |
| `bit_jaccard_ops` | ✓ | ✗ - same |

## Phase plan

- ~~**Phase HV** - add `halfvec` (FP16) type.~~ ✓ done.
- ~~**Phase SV** - add `sparsevec` type.~~ ✓ done.
- ~~**Phase BV** - add `bitvec` type, Hamming + Jaccard.~~ ✓ done.
- ~~**Phase L2** - indexed L2 / L1 ANN.~~ ✓ done via `vec_l2_ops` /
  `vec_l1_ops` and the existing recheck-orderby path.
- **Phase BV-IDX** - binary-vector ANN index. The TurboQuant kernel
  doesn't fit Hamming-space ANN; if we want indexed bitvec we'd
  need a separate kernel (LSH or multi-index hashing). Out of
  scope for the 1.0 line.
- **Phase BC** - binary-compatible varlena layout for `vector` so
  casts to/from `pgvector.vector` are zero-copy. See
  `docs/PHASE19_PROGRESS.md`.