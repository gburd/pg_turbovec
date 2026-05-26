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
| Warm scan p50 (1 M × 384-d, GloVe) | 100 ms | 22 ms (v1.0.0) | ✅ we win 5× |
| **Warm scan p50 (1 M × 1536-d, dbpedia-1M ada-002)** | 61 ms (ef=40) / 115 ms (ef=200) | **v1.5.0 on `meh` (24 c, 125 GiB RAM): 26.8 ms mmap=on, 26.7 ms mmap=off; arnold re-bench (where the buffer-manager bottleneck is real) still queued** — prior v1.4.0 arnold figure was 90 ms | ✅ **we win 2.3× vs HNSW ef=40 on a generously-RAMed host.** Phase R-3 (v1.5.0, Phase R-3 commit on `phase-r3-mmap-static-regions`) replaces the buffer-manager reads of the deterministic static regions (blocked codes + rotation matrix) with `mmap(MAP_PRIVATE)` of the relfile, so warm scans bypass `ReadBufferExtended` / shared_buffers churn for those bytes. Phase U-2 measurement on `meh` (1 M × 1536-d, `shared_buffers = 512 MB`, `search_k = 100`): mmap=on p50 **26.80 ms**, mmap=off p50 **26.65 ms**, delta +0.15 ms (verdict `shared_buffers_was_the_bottleneck` — OS page cache size dominated; the buffer-manager bottleneck Phase S targets is invisible when free RAM ≫ index size). arnold re-bench at the original 31 GiB-RAM constraint remains the definitive Phase S validation; expected drop from 90 ms toward ~60 ms there. v1.1.0 was 70 ms; v1.3.0 (Phase P) was 87 ms; v1.4.0 (Phase R-2) was 90 ms. See `docs/RECALL.md` § 2.5 / 2.6 and `benches/results/recall_warm_meh_v1_5_0_2026_05_26.json`. |
| Cold scan p50 (after backend restart) | ~100 ms | 1 256 ms (1 M × 1536-d, post-Phase-P, commit a801f38) | ⚠️ **21× speedup vs. pre-fix v1.0.x side-table path**; remaining gap to HNSW is acceptable since subsequent queries warm to ~87 ms in the same backend. The relfile-resident format is the only storage strategy as of v1.3.0; the side-table path is gone. |
| INSERT throughput (per row, into a 1 M-row index) | ~0.5 ms (HNSW O(log n)) | **0.13 ms (post-Phase-K, deferred-commit on the relfile path)** | ✅ **we win 4×** — v1.0.x had ~200 ms/row (full re-serialise per row) and we lost 400×; v1.1.0 (Phase K) shipped the deferred-commit pattern that mutates the cached `Arc<RwLock<IdMapIndex>>` per-row and persists once at xact commit, taking 1k-row bulk inserts from ~400 s to ~136 ms. v1.3.0 (Phase Q) extended the same pattern to the relfile path. |
| Recall on uniform-random | 0.03 | 1.000 | ✅ (but synthetic; real-world recall varies) |
| Recall on real OpenAI ada-002 (dbpedia-1M) | ~0.962 (ef_search=40) / ~0.970 (ef_search=200) | **R@10 = 1.000** at default `turbovec.search_k=100` | ✅ **we win** by 0.030–0.038. See `docs/RECALL.md` §2.2 for the full Phase J head-to-head; the 4-bit and 2-bit configurations both hit 1.000 because TurboQuant's rotation + Lloyd-Max coding preserves rank order for real ada-002 embeddings (the rotation-then-reconstruct cycle is near-lossless on the workload). |

### Cold-cache latency — the relfile-resident page format

v1.0.x..v1.1.0 stored the serialised index in a side-table
(`turbovec.am_storage`) read via SPI on first access. Every
fresh PostgreSQL backend paid the full SPI fetch + HashMap
construct cost (~6.8 s on 1 M × 384-d), then cached the result
in a per-backend `Arc<IdMapIndex>`. Connection pools that
create-and-destroy backends, or VACUUM workers, hit this every
time.

pgvector's HNSW lives in the index relation's main fork and is
cached in `shared_buffers` cluster-wide — first scan after a
restart is the same ~100 ms as the warm scan.

**Status: shipped.** The relfile-resident page format (Phase L,
preview in v1.1.0) plus the persisted SIMD-blocked layout +
Lloyd-Max codebook (Phase P, v1.2.0) close the cold-scan gap:
dbpedia-1M cold p50 is **1 256 ms** post-Phase-P, a 21×
speedup over the v1.0.x side-table baseline. Phase Q (v1.3.0)
removed the side-table path entirely; the relfile is the only
storage strategy and the AM matches every other PostgreSQL
index AM (btree, gist, gin, hnsw, ivfflat).

The remaining gap to pgvector HNSW (~1.2 s vs ~100 ms) is
bounded by the cost of reading the codes + scales + ids +
blocked-layout chains off disk into the per-backend
`IdMapIndex` Arc. Cluster-wide caching of `IdMapIndex` parts
(shared-memory `dsm` segment) is the next lever; tracked as a
follow-up but not a v1.3 gating item.

### INSERT throughput — the deferred-commit pattern

v1.0 `aminsert` did a full SPI fetch + full re-serialise per
row, costing ~2× 195 MiB of TOAST I/O per inserted row on a
1 M-row index. A bulk `INSERT ... SELECT` of 1 M rows would
have taken ~55 hours.

**Status: shipped.** Phase K (v1.1.0) introduced the deferred-
commit pattern: `aminsert` mutates the cached
`Arc<RwLock<IdMapIndex>>` in place, marks the entry dirty,
and registers a `PreCommit` xact callback that persists once
at the end of the transaction. Phase N-C (v1.2.0) extended
this to the relfile path. A 1 k-row bulk `INSERT` on a
turbovec-indexed table now finishes well under 5 s on debug
builds (was ~400 s pre-Phase-K).

For large `INSERT ... SELECT` we still pay one full relfile
rewrite at commit time, which is O(n_vectors). Bulk-build at
ROWS-per-COMMIT scale is order-of-magnitude better than the
pre-Phase-K hot loop, but pgvector's HNSW remains O(log n)
per insert. Tracked as future work; the user-facing
recommendation is to load via `CREATE INDEX` after the bulk
`INSERT` rather than the other way around.

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