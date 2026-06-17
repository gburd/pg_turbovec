# pgvector parity gap tracker

What pgvector offers (as of 0.8.x) and where pg_turbovec stands.

## Performance gaps (the honest scoreboard)

This section enumerates **known performance regressions** of
pg_turbovec vs pgvector. They are correctness-OK in every case;
the trade-off is that the wins (10× less storage, exact recall)
come paired with these losses.

> **2026-06-15 correction — read this first.** An isolated, AVX2,
> contention-controlled benchmark on `arnold` (Cohere-wiki 1M ×
> 1024-d; see `docs/BENCHMARKS.md`) overturned the earlier "we win
> warm p50" claim. pg_turbovec is a **flat quantized full-scan**
> index: `O(n·dim)` per query. At 1M rows its warm p50 is
> **~2.5 s** (AVX2) vs pgvector HNSW's **~5 ms** — HNSW is ~490×
> faster because it's a sublinear graph traversal. The old
> "26.8 ms on meh / we win 2.3×" numbers were produced by the
> **pre-AVX2 scalar-fallback bug** (fixed in v1.7.3) that returned
> fast-but-WRONG results, so they never represented correct
> behaviour. **pg_turbovec's real wins are storage (10–15×),
> exact recall (1.000 vs HNSW's ~0.96), and build memory — NOT
> query latency at scale.** It is the right choice when storage
> and exactness matter more than raw QPS, or at corpus sizes /
> with pre-filters where an `O(n)` scan is acceptable. It is the
> wrong choice for low-latency ANN over millions of rows — use a
> graph index (pgvector HNSW) there. Positioning corrected to
> "best storage efficiency + exact recall for PG vector search
> where an O(n) scan fits the latency budget," NOT "beat HNSW on
> latency."

| Metric (1 M × 384-d cosine, release build, arnold) | pgvector HNSW | pg_turbovec | Status |
|---|---:|---:|---|
| Storage | 1 953 MiB | 195 MiB (4-bit) | ✅ we win 10× |
| Build time | 8 m 13 s | 33 s | ✅ we win 15× (at 384-d; 1.9–2.1× at 1024-d) |
| Warm scan p50 (1 M × 384-d, GloVe) | 100 ms | 22 ms (v1.0.0) | ✅ we win 5× |
| **Warm scan p50 (1 M × 1024-d, Cohere-wiki, AVX2 `arnold`)** | ~5.2 ms (ef=200, R@10 0.96) | **~2552 ms (2-bit/4-bit, R@10 1.000)** | ❌ **we LOSE ~490×.** This is the corrected, contention-controlled AVX2 number (`docs/BENCHMARKS.md`, 2026-06-15). pg_turbovec is a flat `O(n·dim)` quantized scan; HNSW is a sublinear graph. The earlier "26.8 ms on `meh` / we win 2.3×" figure was the **pre-AVX2 scalar-fallback bug** (fast-but-WRONG, fixed v1.7.3) and is retracted. AVX2 makes the correct scan ~15–25× faster than meh's scalar fallback (2.55 s vs 41.6 s), but a 1M-row flat scan is seconds, not ms, by design. The latency knob is corpus size / pre-filter selectivity, not search_k (latency is flat across search_k). Use a graph index for low-latency ANN over millions of rows; use pg_turbovec for exact recall + 10–15× storage where an O(n) scan fits the budget. |
| Cold scan p50 (after backend restart) | ~100 ms | 1 256 ms (1 M × 1536-d, post-Phase-P, commit a801f38); v1.7.3 defers the per-backend `id_to_slot` HashMap build off the read-only scan path (parity gap #3) | ⚠️ **21× speedup vs. pre-fix v1.0.x side-table path**; remaining gap to HNSW is acceptable since subsequent queries warm to ~87 ms in the same backend. v1.7.3 cuts the dominant residual cache-fill term: the read-only scan path now materialises a `ReadOnlyIndex` (positional `TurboQuantIndex` + `slot_to_id` Vec) instead of a full `IdMapIndex`, skipping the O(n) `id_to_slot` HashMap build (~50 ms debug / 200 k rows, scales with n; the dominant cache-fill phase once Phase P pre-baked the blocked layout). The HashMap is deferred to the first mutation, which still needs it. The relfile-resident format is the only storage strategy as of v1.3.0; the side-table path is gone. |
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
blocked-layout chains off disk into the per-backend index
plus, until v1.7.3, the O(n) `id_to_slot` HashMap build.

**v1.7.3 (parity gap #3): lazy `id_to_slot` on the read path.**
Profiling the cache-fill (200 k × 256-d, debug) showed the
dominant residual term was the `id_to_slot:
HashMap<u64,usize>` that `IdMapIndex::from_id_map_parts*`
eagerly materialises in `finalise_from_inner` — ~50 ms at
200 k rows, scaling linearly with `n`, dwarfing the
`read_full` (~16-22 ms) and `read_blocked`+`read_rotation`
(~12-18 ms) data copies. The scan path never reads
`id_to_slot`: `search(q, k)` with `allowlist = None` only ever
indexes `slot_to_id[slot]` (a `Vec`). So the AM scan path now
installs a `cache::ReadOnlyIndex` (the inner positional
`TurboQuantIndex` + the `slot_to_id` `Vec`, no HashMap), and
the HashMap build is deferred to the first `aminsert` /
`remove`, which rebuild a full `IdMapIndex` via `am_install`.
A read-only / pooled-connection backend that only ever scans
never pays the HashMap build. With the fix the read-only
constructor drops from ~50 ms to ~0 ms in the profiled debug
build. Wire format unchanged (scan-side only).

**Deferred follow-ups (not in v1.7.3):**

1. **Read-path mmap of codes / scales / ids.** Today only the
   *static* regions (blocked codes + rotation) are mmap'd; the
   codes/scales/ids chains still go through `read_full` (the
   buffer manager) because `ambulkdelete` swap-removes them in
   place. On a *read-only* cold scan they could be mmap'd RO
   too (same MVCC backstop as the static regions: heap
   visibility + `xs_recheckorderby`). Codes are the bulk of
   the index (≈ 768 MiB at 1 M × 1536-d × 4-bit), so this
   removes the largest remaining data copy on the real cold
   path. The `ReadOnlyIndex::from_prepared_parts_borrowed`
   constructor already accepts `Cow::Borrowed`, so the wiring
   is additive once the relfile path resolution + per-page
   header-gap handling is extended to those chains.
2. **Zero-copy mmap (wire-format change).** Each chain page
   carries a 24-byte PG `PageHeaderData` prefix, so the chain
   bytes are not contiguous in the mmap and must be copied off
   once at cache-fill. A header-gap-free on-disk layout would
   let the SIMD kernel read straight from the mmap with no
   copy at all. That is a `MetaPageData::version` 3 → 4 wire
   bump and belongs in a v1.8 / v2.0 minor, not a scan-side
   patch.
3. **Cross-backend shared cache.** Cluster-wide caching of the
   index parts in a PG DSA/DSM segment keyed by relfilenode so
   the *second* backend onward maps an already-built structure
   instead of rebuilding. Biggest win for pooled workloads but
   the most invasive (DSA allocator, REINDEX invalidation,
   concurrency); XL effort, tracked as a follow-up.

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

Two knobs together form the recall-vs-latency frontier:

1. `turbovec.search_k` (default 100) — how many candidates the kernel
   returns.
2. `turbovec.oversample` (default 1.0, v1.8.x+) — the candidate-set
   widener. The scan fetches `ceil(search_k * oversample)` candidates
   ranked by the lossy quantized distance, and the always-on reorder
   queue (`xs_recheckorderby`) re-ranks them by exact full-precision
   distance, trimming to the true top-k under the LIMIT. This recovers
   true neighbours the quantized ranking placed just outside
   `search_k`, turning quantization from a fixed accuracy point into a
   tunable frontier (Qdrant `oversampling` / VectorChord rerank).

Measured (4-bit, 3000×64, `search_k=10`, 8 query seeds,
`benches/results/oversample_recall_curve_2026_06_15.json`):

| oversample | recall@10 | p50 (ms) |
|-----------:|----------:|---------:|
| 1.0        | 0.8125    | 3.81     |
| 1.5        | 0.9625    | 3.86     |
| 2.0        | 0.9875    | 3.94     |
| 4.0        | 1.0000    | 4.06     |
| 8.0        | 1.0000    | 4.70     |

Recall climbs monotonically to 1.0 as `oversample` grows; latency
rises roughly linearly with the candidate count. There is no separate
`turbovec.rescore` GUC: oversampling plus the reorder queue together
are the rescore mechanism (the reorder queue already re-ranks every
returned tuple by exact distance, so an AM-side rescore would be
redundant). `oversample` composes with iterative scan — it sets the
initial `k`, iterative refill grows it from there.

On the 384-d synthetic corpus, K=100 gave R@10 = 1.000 because the
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

## Arithmetic & concatenation operators

Element-wise add/subtract, the Hadamard (element-wise) product, and
concatenation. pgvector errors on a non-finite result coordinate
(`value out of range: overflow`); pg_turbovec matches this — `+`/`-`/`*`
require equal dimensions and raise on a non-finite result, and `||`
errors if the combined dimension exceeds `MAX_DIM` (16 000). pgvector
does not define arithmetic for `sparsevec`, so neither do we.

| Op | pgvector | pg_turbovec |
|----|----------|-------------|
| `+` element-wise (vector) | ✓ | ✓ |
| `-` element-wise (vector) | ✓ | ✓ |
| `*` Hadamard (vector) | ✓ | ✓ |
| `\|\|` concat (vector) | ✓ | ✓ |
| `+` element-wise (halfvec) | ✓ | ✓ |
| `-` element-wise (halfvec) | ✓ | ✓ |
| `*` Hadamard (halfvec) | ✓ | ✓ |
| `\|\|` concat (halfvec) | ✓ | ✓ |
| arithmetic (sparsevec) | ✗ (not offered) | ✗ (parity: not offered) |

## Index scan features

| Feature | pgvector 0.8.2 | pg_turbovec status |
|---------|----------------|--------------------|
| ANN index scan | ✓ (HNSW, IVFFlat) | ✓ (`turbovec` AM) |
| **Iterative / streaming scan** | ✓ `hnsw.iterative_scan`, `ivfflat.iterative_scan`, `max_scan_tuples`, `scan_mem_multiplier`, `max_probes` | ✓ (v1.8.0) — `turbovec.iterative_scan` (`off` \| `relaxed_order`, default `relaxed_order`). When a selective `WHERE filter ORDER BY emb <=> q LIMIT k` drains the current candidate batch, `amgettuple` re-runs the turbovec search with a doubled `k` and feeds the new (deduplicated) candidates, capped by `turbovec.max_scan_tuples` (default 20000, matches pgvector). Ordering across refill batches is restored by the existing `xs_recheckorderby` reorder queue. pgvector's `strict_order` is future work (our reorder queue already delivers exact ordering on top of `relaxed_order`). |
| Bitmap index scan (`amgetbitmap`) | ✓ | ✗ (not applicable to ANN ordering) |
| **Metadata filtering** | post-filter + iterative + partial idx | three patterns — partial index (native PG pushdown), in-kernel allowlist `turbovec.knn(..., allowed)` (flat-only true pushdown; selective filters get cheaper), iterative scan + recheck. Remaining gap: no true in-traversal pushdown on the `ORDER BY` AM path (the index stores only vector codes + TID, no payload columns). Full guide + measured crossover: [`docs/FILTERING.md`](FILTERING.md). |
| Parallel index build | ✓ (maintenance workers) | ✓ (v1.8.0) — `turbovec.build_parallelism` drives a rayon pool over the quantize/pack stage; relfiles are byte-identical to a serial build. |
| Quantization tuning | manual re-rank CTE | `turbovec.search_k` (candidate count) **plus `turbovec.oversample`** (v1.8.x+): fetch `ceil(search_k * oversample)` quantized candidates, the always-on reorder queue re-ranks by exact distance — oversampling + reorder queue are the rescore mechanism, matching Qdrant oversampling / VectorChord rerank. Recall@10 climbs to 1.0 as oversample grows (see § Recall tuning). |
| `CREATE INDEX CONCURRENTLY` | ✓ | ✓ (standard AM path) |
| Build progress (`pg_stat_progress_create_index`) | ✓ phased | partial (no custom phase labels) |

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
| `vector_concat(vector, vector)` | ✓ | ✓ (also `\|\|` operator) |
| `halfvec_concat(halfvec, halfvec)` | ✓ | ✓ (also `\|\|` operator) |
| `max_sim` / `max_sim_cosine` (ColBERT MaxSim) | ✗ | ✓ — SQL re-rank over `vector[]`; see [`HYBRID_SEARCH.md`](HYBRID_SEARCH.md) |
| `rrf_score` (reciprocal rank fusion) | ✗ | ✓ — `1/(k+rank)` hybrid-fusion helper; see [`HYBRID_SEARCH.md`](HYBRID_SEARCH.md) |

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
- ~~**Phase D (breadth)** - multivector / hybrid SQL surface.~~ ✓ done
  (v1.13.x). `turbovec.max_sim` / `max_sim_cosine` (ColBERT MaxSim
  re-rank over `vector[]`), `turbovec.rrf_score` (reciprocal rank
  fusion), and the named-vector schema pattern. See
  [`HYBRID_SEARCH.md`](HYBRID_SEARCH.md). **Remaining gap:**
  index-native late interaction (per-token index + MaxSim traversal)
  is a documented future phase — MaxSim is a SQL re-rank primitive,
  not an index-accelerated scan.
- **Phase BV-IDX** - binary-vector ANN index. The TurboQuant kernel
  doesn't fit Hamming-space ANN; if we want indexed bitvec we'd
  need a separate kernel (LSH or multi-index hashing). Out of
  scope for the 1.0 line.
- **Phase BC** - binary-compatible varlena layout for `vector` so
  casts to/from `pgvector.vector` are zero-copy. See
  `docs/PHASE19_PROGRESS.md`.