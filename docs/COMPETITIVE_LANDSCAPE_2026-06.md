# Competitive Landscape Report — pg_turbovec v1.8.0

_Date: 2026-06-15. Read-only research; no code changed._

Scope: the full PostgreSQL-vector + standalone-vector-DB field, with a
deep dive on the two most-direct PG-extension competitors (VectorChord,
pgvectorscale), a capability checklist, and an honest verdict to feed a
roadmap decision.

**Sourcing note.** Competitor numbers below are quoted from each
project's own README / docs (fetched 2026-06-15) — these are *vendor
claims*, not independently reproduced. Where a number could not be
verified from a primary source, it is marked **[unverified]**.
pg_turbovec's own numbers come from `benches/results/` in this repo
(single-host runs; see caveats in §4).

---

## pg_turbovec's own numbers (baseline for comparison)

From `benches/results/recall_dbpedia_1M_2026_05_24.json` — corpus
`dbpedia-entities-openai-1M` (1 M × 1536-d, cosine, normalized, OpenAI
ada-002), host `arnold` (i9-12900H, 32 GiB), PG 17.9, pgvector 0.8.0.
Note: this run is tagged `pg_turbovec_version: 1.0.0`; the headline
recall/storage/build numbers predate v1.8.0 but the wire format and
quantizer are unchanged, so they still hold for v1.8.0.

| Config | index/payload size | build s | p50 ms | p95 ms | R@10 |
|---|---|---|---|---|---|
| pgvector HNSW (m16, ef_c64), ef=40 | 8192 MB | 295 | 61.1 | 93.4 | 0.962 |
| pgvector HNSW, ef=200 | 8192 MB | 295 | 114.8 | 222.3 | 0.970 |
| tv 4-bit, search_k=100 | 780 MB | 163 | 70.5 | 91.3 | **1.000** |
| tv 4-bit, search_k=500 | 780 MB | 163 | 123.9 | 143.2 | **1.000** |
| tv 2-bit, search_k=100 | 396 MB | 126 | 48.3 | 49.7 | **1.000** |
| tv 2-bit, search_k=500 | 396 MB | 126 | 78.0 | 79.6 | **1.000** |

Derived wins vs pgvector HNSW on this corpus:
- **Storage:** 396 MB (2-bit) vs 8192 MB = **~20.7×** smaller;
  780 MB (4-bit) = **~10.5×** smaller. (The "~10×" headline is the
  4-bit figure.)
- **Build:** 126 s (2-bit) vs 295 s = **~2.3×**; the "~15×" headline
  is *not* reproduced by this file — it likely refers to a different
  host/config. **Flagged: the "~15× faster build" claim is not
  supported by the dbpedia-1M JSON; reconcile before publishing.**
- **Recall:** R@10 = 1.000 vs 0.962 (ef=40) / 0.970 (ef=200). Genuine
  win, but note this is 50 queries — small sample.
- **Warm p50:** 48.3 ms (2-bit, k=100) vs 61.1 ms — ~1.3× on this
  host. The "2-3× warm" headline is a RAM-constrained-host claim
  (`meh`/`arnold`), not the dbpedia numbers above.

The "26.8 ms warm p50 on meh" headline is not in the dbpedia file; it
comes from a separate `meh` run (`recall_warm_meh_*`). Cite the right
file per claim.

---

## Task 1 — The full competitive set

### Bucket A — PostgreSQL-embedded vector solutions (direct competitors)

These share pg_turbovec's deployment model ("stay inside your Postgres").

#### pgvector — **v0.8.2** (latest tag, confirmed via GitHub tags API)
The baseline. C extension. HNSW + IVFFlat. Types: vector (≤2000-d
indexable), halfvec (≤4000), bit (≤64000), sparsevec (≤1000). Has
`hnsw.iterative_scan` (relaxed/strict, since 0.8.0), `max_scan_tuples`,
`scan_mem_multiplier`, `binary_quantize`, partial indexes. No built-in
rescore knob (manual re-rank CTE). It is the compatibility target;
pg_turbovec, VectorChord, and pgvectorscale all consume its `vector`
type.

#### VectorChord (vchord) — **v1.1.1** (TensorChord, 2026-02-28)
**THE most-direct competitor.** Rust/pgrx PG extension, pgvector-type
compatible, quantization-focused — same three axes as us. Successor to
pgvecto.rs. Index AM `vchordrq` (IVF + RaBitQ) plus a newer graph index
`vchordg`. Quantization is **RaBitQ** (Gao & Long, SIGMOD 2024) with
"autonomous reranking" built in. Since v1.1.0 it also exposes native
quantized column types `rabitq8` (8 bit/dim, "<1% recall loss") and
`rabitq4` (4 bit/dim) with their own operator classes — so quantization
is both an index property and a storable type. Dual-licensed AGPLv3 /
Elastic v2.

Headline vendor claims (from README, **[unverified]** by us):
- Store **400,000 vectors per $1** → "6× more than Pinecone's
  storage-optimized, 26× more than pgvector/pgvecto.rs for the same
  price."
- **100 M × 768-d** vectors on one AWS i4i.xlarge ($247/mo);
  **1 B × 96-d** on i7ie.6xlarge ($2246/mo).
- **Index 100 M vectors in ~20 min** (hierarchical k-means + optimized
  disk ops).
- **1 B-vector indexes built on a 128 GB machine** via dimensionality
  reduction + sampling (out-of-core build).

Feature surface that matters (from docs nav): multi-vector retrieval
(MaxSim/ColBERT), graph index, similarity/range filter, **prefilter**
(true pushdown), **prefetch**, **rerank-in-table**, partitioning,
external/precomputed index build, monitoring, prewarm. This is a *much*
wider surface than pg_turbovec.

#### pgvecto.rs — **v0.4.0** (TensorChord, 2024-11-21) — **DEPRECATED**
README explicitly says: "We have a new implementation VectorChord with
better stability and performance. Users are encouraged to migrate."
Treat as legacy; VectorChord is its live successor. Historically notable
for the **VBASE** method (vector TopK + filter + join without
under-return), ≤65535 dims, runtime SIMD dispatch, fp16/int8/binary
types, separate index storage from PG. Not a forward threat on its own.

#### pgvectorscale (vectorscale) — **v0.9.0** (Timescale, 2025-11-04)
Second most-direct competitor. Rust/pgrx, complements pgvector (installs
it via CASCADE). Three innovations (from README):
- **StreamingDiskANN** index (DiskANN-inspired, Microsoft research) —
  on-disk graph traversal, the core bet for **datasets larger than
  RAM**. This is the axis pg_turbovec explicitly does *not* play
  (we're RAM-resident).
- **Statistical Binary Quantization (SBQ)** — Timescale's improvement
  on standard binary quantization.
- **Label-based filtered vector search** (Microsoft Filtered DiskANN) —
  vector similarity + label filter, integrated.

Headline claim (README): on **50 M Cohere 768-d embeddings**,
pgvector+pgvectorscale gets **28× lower p95 latency** and **16× higher
throughput** vs **Pinecone storage-optimized (s1)** at **99% recall**,
at **75% less cost** self-hosted on EC2. **Important caveat: the
headline comparison is vs Pinecone s1, not vs pgvector HNSW.** The
"vs pgvector HNSW" speedups are in their blog, not the README headline —
**[unverified]** here.

#### Lantern — **v0.5.0** (LanternData, 2024-11-15)
PG extension. Index type `lantern_hnsw`, built on usearch (single-header
state-of-the-art HNSW). Also does in-DB embedding generation and
external index build. PG 11–16. Pure-HNSW play; no quantization story
comparable to ours. Release cadence has slowed (last release Nov 2024).
A real competitor on "HNSW in Postgres" but **not** on storage/quant —
it does not compete with us where we win.

**Bucket A summary:** the field that targets *our exact niche*
(quantization for storage + speed inside PG) is **VectorChord
(RaBitQ)** and **pgvectorscale (SBQ + DiskANN)**. Lantern is HNSW-only;
pgvecto.rs is deprecated; pgvector is the shared baseline.

### Bucket B — standalone vector databases (indirect competitors)

Different deployment model. A PG-extension user picks us to *stay in
Postgres*; these win the user who is willing to run a separate service.
What each claims to tempt that user with:

| DB | Pitch / what it claims to win on |
|---|---|
| **Qdrant** | Filterable HNSW (filter integrated into traversal — no under-return), quantization menu (scalar int8 / PQ / binary) with explicit `rescore` + `oversampling`, native Raft sharding/replication. Wins on filtered-search UX + horizontal scale. |
| **Milvus / Zilliz** | Massive scale (billions), index menu (HNSW, IVF, DiskANN, SCANN, GPU indexes), distributed-native, GPU acceleration. Wins on raw scale + index variety + GPU. Zilliz = managed Milvus. |
| **Weaviate** | Hybrid (BM25 + vector) fusion built in, modules for in-DB embedding, GraphQL API, multi-tenancy. Wins on hybrid search + DX. |
| **Pinecone** | Fully managed, serverless, zero-ops, pod/serverless scaling. Wins on "no infra." It's the cost/throughput yardstick pgvectorscale and VectorChord both benchmark *against*. |
| **Vespa** | Big-data search + ranking engine, tensor/ColBERT, real-time, very large scale. Wins on complex ranking + scale. |
| **Chroma** | Dev-friendly, embedded/local, RAG-prototyping ergonomics. Wins on "easiest to start." Not a scale play. |
| **LanceDB / Lance** | Columnar on-disk format (Lance), embedded, multimodal, versioning, zero-copy. Wins on storage format + embedded analytics. |
| **FAISS** | Library, not a DB. The quantization baseline everyone cites (IVF, PQ, OPQ, HNSW, ScaNN-style). No persistence/transactions/SQL. The academic yardstick for quant quality. |

The temptation for a PG user: scale beyond one box (Milvus/Qdrant/
Vespa), zero-ops (Pinecone), or hybrid fusion (Weaviate). The
counter-pitch for any PG extension is "you already have Postgres —
transactions, backups, joins, SQL, your existing ops." That counter is
strongest for small-to-mid scale (≤ ~100 M) where one box suffices —
which is exactly pg_turbovec's RAM-resident sweet spot.

---

## Task 2 — Deep dive: VectorChord & pgvectorscale vs pg_turbovec

### VectorChord (RaBitQ) vs pg_turbovec (TurboQuant)

| Axis | VectorChord v1.1.1 | pg_turbovec v1.8.0 |
|---|---|---|
| Quantizer | RaBitQ (SIGMOD'24) + extended RaBitQ (SIGMOD'25); autonomous reranking; rabitq4/rabitq8 types | TurboQuant 2/3/4-bit; single quantizer |
| Index AM | `vchordrq` (IVF+RaBitQ), `vchordg` (graph) | single `turbovec` AM |
| Storage claim | "26× more vectors than pgvector for the same $"; rabitq8 ≈ ~1/4 of vector size | 4-bit ~10.5×, 2-bit ~20.7× smaller than HNSW (measured, dbpedia-1M) |
| Build claim | 100 M in ~20 min (hierarchical k-means) **[unverified]** | 1 M in 126 s (2-bit) measured; no published >1 M real-corpus build |
| Scale claim | 100 M on one i4i.xlarge; 1 B on 128 GB box (out-of-core build) | 1 M real + 10 M synthetic (`meh`); RAM-resident |
| Rescore/oversample | yes (autonomous rerank + rerank-in-table) | **no** (search_k only) |
| Filtered pushdown | yes (`prefilter`) | **no** (post-filter + iterative refill) |
| Multivector / ColBERT | yes (MaxSim operators) | **no** |
| Iterative/refill scan | N/A (uses prefilter + VBASE-style) | **yes (new in 1.8.0)** |
| Types | vector/halfvec/sparse/binary + rabitq4/8 | vector/halfvec/sparsevec/bitvec |
| Published large-scale bench | yes (100 M, 1 B blog posts) | no (1 M real) |

**Honest read:** VectorChord wins on breadth (graph index, multivector,
prefilter, rerank, out-of-core build, 100 M/1 B published runs) and has
a mature, heavily-marketed benchmark story. RaBitQ has a *theoretical
error bound* (its selling point) and published recall/QPS at scale.
pg_turbovec's measured R@10=1.000 on dbpedia-1M is a genuinely strong
result, but it is **50 queries on one host** vs VectorChord's published
multi-scale curves. Where we plausibly win: raw storage at equal recall
(2-bit @ 396 MB is aggressive), and the in-PG simplicity of a single AM.
Where we lose today: no rescore knob, no pushdown, no multivector, no
published scale, no out-of-core. **VectorChord is the benchmark to beat
and the feature template to study.**

### pgvectorscale (StreamingDiskANN + SBQ) vs pg_turbovec

| Axis | pgvectorscale v0.9.0 | pg_turbovec v1.8.0 |
|---|---|---|
| Index | StreamingDiskANN (on-disk graph, >RAM) | RAM-resident mmap reads |
| Quantizer | SBQ (statistical binary quantization) | TurboQuant 2/3/4-bit |
| Filtered search | label-based, integrated (Filtered DiskANN) | post-filter + iterative refill |
| Headline | **28× lower p95, 16× higher throughput vs Pinecone s1 @ 99% recall**, 50 M Cohere 768-d, 75% less cost | R@10=1.000, dbpedia-1M, single host |
| Bench target | vs Pinecone s1 (README); vs pgvector HNSW (blog) | vs pgvector HNSW |
| Recall point | 99% (their measured corpus) | 100% @ R@10 on 50 queries |

**Honest read:** pgvectorscale's whole bet is **>RAM datasets** via
on-disk DiskANN — an axis pg_turbovec deliberately doesn't play. Their
headline is measured at 50 M / 99% recall and the comparison is **vs
Pinecone s1**, not pgvector HNSW (a common misreading — verify before
quoting "28×" as a pgvector delta). SBQ is *binary* (1 bit/dim
effectively) with statistical correction; TurboQuant 2-4 bit keeps more
bits, so at equal recall we likely store more than SBQ-binary but
recover recall without a graph. The two are not directly comparable
without a head-to-head on the same corpus — **which neither side has
published.** pg_turbovec loses on: out-of-core, published scale,
integrated filtering. We plausibly win on: build simplicity and recall
at small-to-mid scale in RAM.

### How our numbers stack up — bottom line

- **We have one real-corpus run (dbpedia-1M, 50 queries) + a 10 M
  synthetic run.** Both competitors publish 50 M–1 B curves. On
  *evidence volume*, we lose badly — this is the single biggest
  credibility gap.
- We just fixed a **pre-AVX2 wrong-results bug** (v1.7.3, see
  `docs/PRODUCTION.md`) and just shipped iterative scan + parallel build
  (v1.8.0). We are *newly* at scan-correctness parity, not battle-tested.
- Our storage and recall numbers are strong and defensible *as far as
  they go*; the problem is reach, not the numbers.

---

## Task 3 — Capability checklist

✓ = pg_turbovec v1.8.0 has it · ✗ = missing · ~ = partial

| Capability | pg_turbovec | Who has it (competitor) |
|---|---|---|
| Quantization rescore / oversampling | ✗ | VectorChord (rerank), Qdrant (rescore+oversampling) |
| Filtered search, true pushdown | ✗ (~ post-filter + iterative refill) | VectorChord (prefilter), pgvectorscale (label filter), Qdrant (filterable HNSW) |
| DiskANN on-disk graph (>RAM traversal) | ✗ (RAM-resident) | pgvectorscale (core bet), Milvus DiskANN |
| Streaming / out-of-core index build | ✗ | VectorChord (1 B on 128 GB), pgvectorscale |
| Multiple quantization modes (menu) | ✗ (TurboQuant 2/3/4-bit only) | VectorChord (rabitq4/8), Qdrant (scalar/PQ/binary), Milvus |
| Sparse vectors stored | ✓ (sparsevec type) | pgvector, VectorChord |
| Hybrid dense+sparse w/ fusion (RRF) | ✗ | Weaviate, VectorChord (hybrid-search docs), Qdrant |
| Multivector / ColBERT late-interaction | ✗ | VectorChord (MaxSim), Vespa |
| Index types beyond one | ✗ (single AM) | pgvector (HNSW+IVF), VectorChord (rq+graph), Milvus (many) |
| Iterative / refilling scan | ✓ (new 1.8.0) | pgvector (iterative_scan) |
| Parallel index build | ✓ (new 1.8.0, rayon) | pgvector, VectorChord |
| Published large-scale bench (>10 M) | ✗ (1 M real + 10 M synthetic) | VectorChord (100 M/1 B), pgvectorscale (50 M) |
| ANN-Benchmarks / VectorDBBench presence | ✗ | see below |

**On the public leaderboards.** ANN-Benchmarks (the academic standard:
SIFT1M, GIST1M, glove-100, deep-image-96, fashion-mnist, and the
big-ann-benchmarks track for deep1B / laion / SSNPP at 1 B) and
VectorDBBench (Zilliz's DB-level harness: Cohere, OpenAI, LAION corpora
at 1 M / 10 M / 50 M+) are where credibility is minted. pgvector,
Qdrant, Milvus, Weaviate, Pinecone, and pgvectorscale all appear in
VectorDBBench and/or ANN-Benchmarks results. **pg_turbovec is on
neither.** This is the most actionable single gap for external
credibility — a public VectorDBBench run on a standard corpus would let
us be compared on the same axes everyone else is measured on.
(I could not verify the *current* exact leaderboard standings from a
primary source in this session — treat the "who appears" list as
well-established but **[unverified as of today]**.)

---

## Task 4 — Honest verdict

### Where pg_turbovec credibly wins today
- **Storage efficiency at high recall, in RAM, small-to-mid scale.**
  2-bit @ 396 MB / 4-bit @ 780 MB with R@10=1.000 on dbpedia-1M is a
  real, defensible result against pgvector HNSW's 8 GB / 0.962.
- **Simplicity:** one AM, no graph to tune, ALTER-EXTENSION upgrades,
  wire-format stable since v1.4.0. Lower operational surface than
  DiskANN/IVF tuning.
- **Cold-scan and warm-scan latency on RAM-constrained hosts** (the
  `meh`/`arnold` story), now that the pre-AVX2 bug is fixed and
  iterative scan lands.

### Where it's at parity
- Scalar/type surface vs pgvector (types, ops, aggregates, arithmetic).
- Iterative/refilling scan (now matches pgvector's `iterative_scan`).
- Parallel build (now matches pgvector; still behind VectorChord's
  hierarchical-k-means scale build).

### Where it loses
- **No published large-scale benchmark.** Competitors have 50 M–1 B
  curves; we have 1 M real, 50 queries. Biggest credibility gap.
- **No rescore/oversampling knob.** Both VectorChord and Qdrant turn
  quantization into a tunable recall lever; we expose only `search_k`.
- **No true filtered pushdown.** Post-filter + iterative refill is the
  PG-idiomatic stopgap, not integrated filtering.
- **No out-of-core / DiskANN.** pgvectorscale and VectorChord serve
  >RAM datasets; we cannot.
- **Single quantizer, single index type, no multivector/hybrid.**

### The single most important thing to add next
**A published VectorDBBench (or ANN-Benchmarks) run on a standard
corpus — ideally at 1 M *and* 10 M, OpenAI/Cohere — alongside
pgvector, VectorChord, and pgvectorscale on the same host.** Rationale:
our *features* are now at scan-correctness parity (1.8.0), but our
*evidence* is one host and 50 queries. Every competitor's credibility
comes from public, reproducible, multi-scale curves on the standard
harnesses. Until we have one, "R@10=1.000" reads as a cherry-picked
micro-benchmark to anyone evaluating a swap. This is also cheaper than
the next features and de-risks the roadmap (it tells us where we
*actually* stand vs RaBitQ/SBQ).

A close second is **rescore + oversampling** (effort M, already #5 in
`COMPETITIVE_ANALYSIS.md`): it's the cheapest feature that turns our
storage win into a tunable recall lever and matches the competitor UX —
and it's a prerequisite for an honest VectorDBBench run (you tune the
recall/QPS tradeoff curve, which we currently can't).

### Is "beat everyone on every axis" realistic?
**No — and chasing it would be the over-engineering trap.** Out-of-core
DiskANN, multivector fusion, an index menu, and billion-scale builds are
each multi-month efforts that re-fight battles VectorChord and Milvus
already won. The defensible, narrow positioning is:

> **"Best storage efficiency and recall for in-RAM vector search inside
> Postgres, at small-to-mid scale (≤ ~100 M), with the lowest operational
> surface."**

That sentence is true today, is backed by real numbers, and concedes the
axes we can't win (>RAM, billion-scale, hybrid fusion) to the projects
that own them. Win the in-RAM-in-Postgres user decisively; don't try to
out-DiskANN Timescale or out-feature TensorChord.

---

## Appendix — versions & sources (fetched 2026-06-15)

- pgvector **0.8.2** — GitHub tags API.
- VectorChord (vchord) **1.1.1** — GitHub releases API + README.
- pgvecto.rs **0.4.0** (DEPRECATED, → VectorChord) — README.
- pgvectorscale (vectorscale) **0.9.0** — GitHub releases API + README.
- Lantern **0.5.0** — GitHub releases API + README.
- pg_turbovec numbers — `benches/results/recall_dbpedia_1M_2026_05_24.json`
  (this repo) and `CHANGELOG.md` [1.8.0].
- All competitor performance/cost/scale numbers are **vendor claims**
  from project READMEs/docs, not independently reproduced. Items marked
  **[unverified]** could not be confirmed from a primary source in this
  session (JS-rendered blog/benchmark pages did not yield text via curl).
