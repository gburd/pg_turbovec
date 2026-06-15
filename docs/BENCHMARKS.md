# pg_turbovec Benchmarks

Canonical, reproducible head-to-head benchmark page for `pg_turbovec` vs
`pgvector`. The goal is a standardized, VectorDBBench-style result on a public
corpus so the numbers are comparable to published ANN benchmarks — not a
bespoke single-host claim.

> **Status (2026-06-15):** First standardized 1M run on a public corpus
> (Cohere wiki, 1024-d, real embeddings). **Correctness, storage, build, and
> recall are measured and valid.** The **latency frontier for pg_turbovec is
> DEFERRED to an AVX2+ host** — the bench host (`meh`) is a pre-AVX2 Xeon, so
> turbovec runs its scalar fallback (~1000x slower than its AVX2/AVX-512 SIMD
> kernels). See [Caveats](#caveats). The headline result this run establishes:
> **recall@10 = 1.000 on the fixed v1.8.0 build at 1M × 1024-d real
> embeddings — the pre-AVX2 correctness fix works.**

## Methodology

| Item | Value |
|------|-------|
| Corpus | `Cohere/wikipedia-22-12-en-embeddings`, 1,000,000 rows, **1024-d**, cosine, L2-normalized |
| Standard size | VectorDBBench "Medium" (1M). Real embeddings, not synthetic. |
| Held-out queries | 1,000 vectors (ids 1000000–1000999), held **out** of the index |
| Ground truth | Brute-force exact top-10 by cosine over all 1M rows. BLAS matmul, cross-checked against in-DB seqscan (`enable_indexscan=off`) — 10/10 overlap on sampled queries; reconstructed corpus byte-identical to the DB (`max|diff| = 0.0`) |
| Recall metric | recall@10 vs the exact GT, averaged over the held-out queries |
| Latency (pgvector) | server-side `Execution Time` from `EXPLAIN (ANALYZE)` |
| Latency (turbovec) | client wall over a unix socket; the in-engine scan dominates (>40s here), so the sub-ms cast/RTT term is negligible |
| Warm protocol | ≥1 warmup query (untimed), then N timed queries with fresh held-out vectors (never corpus members) |
| Host | `meh`: Intel **Xeon E5-2697 v2** (Ivy Bridge), 24 cores, 125 GiB RAM, NixOS |
| SIMD | `avx`, `sse4_1`, `sse4_2` — **no `avx2`, no `avx512`** |
| PostgreSQL | 17.9 |
| pgvector | 0.8.0 |
| pg_turbovec | **binary v1.8.0** (git `7d01a51`, turbovec fork `d3d468e`) |
| `shared_buffers` | 640 MB · `maintenance_work_mem` 8 GB · 8 maint. workers |

Indexes built: pgvector HNSW `(m=16, ef_construction=64)`; pg_turbovec 4-bit;
pg_turbovec 2-bit. All on the same 1M-row heap (`docs.emb vector(1024)`); the
turbovec indexes use the expression cast `(emb::real[]::turbovec.vector)`.

## Correctness gate (the headline result)

Before any benchmarking, a correctness gate ran on a 10k × 128-d table of
**distinct** random unit vectors: build a turbovec 4-bit index, compare the
index top-10 against the brute-force top-10 for 20 fresh probes.

```
mean recall@10 = 1.0000   all top-10 sets distinct (10 ids each): True   → PASS
```

And on the full 1M × 1024-d corpus, **every pg_turbovec config returned
recall@10 = 1.000** vs exact GT.

This matters because the *previous* run on this exact host (the old turbovec
v0.7.0 / pg_turbovec v1.7.1 build) scored **recall@10 = 0.0** here — the
pre-AVX2 wrong-results bug. **v1.8.0 fixes it.** Confirmed on real 1024-d
embeddings at 1M scale.

## Storage and build

| Index | Build time | Size | vs HNSW |
|-------|-----------:|-----:|--------:|
| pgvector HNSW (m16, efc64) | 15:29 | 7,806 MB | — |
| pg_turbovec 4-bit | 08:20 | 1,026 MB | **7.6× smaller**, 1.9× faster build |
| pg_turbovec 2-bit | 07:27 | 512 MB | **15.2× smaller**, 2.1× faster build |

Heap: 5,332 MB (incl. TOAST) for 1M × 1024-d. pg_turbovec's compact quantized
codes are its clearest structural advantage and are CPU-independent.

## Recall-vs-latency frontier — pgvector HNSW

`ef_search` sweep, 200 timed queries each. **This frontier is valid and
AVX2-independent** (it's pgvector's own SIMD, unaffected by turbovec's kernel
path).

| Config | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (1 conn) |
|--------|----------:|---------:|---------:|---------:|-------------:|
| HNSW ef=40  | 0.849 |  9.4 | 20.3 | 25.0 | 96.7 |
| HNSW ef=100 | 0.926 | 13.1 | 22.4 | 25.1 | 74.2 |
| HNSW ef=200 | 0.957 | 17.3 | 32.4 | 41.3 | 53.9 |
| HNSW ef=400 | 0.979 | 20.1 | 38.2 | 48.2 | 46.6 |

## Recall-vs-latency frontier — pg_turbovec

**Recall is exact (1.000) at every config.** pg_turbovec is a *quantized
full-scan (flat) index*, so it does not trade recall for speed the way a graph
index does — every query scores the whole corpus.

**Latency on `meh` is the scalar-fallback FLOOR, not a representative
competitive number.** Reported for completeness only:

| Config | recall@10 | p50 (ms) | basis |
|--------|----------:|---------:|-------|
| tv 2-bit, search_k=100  | 1.000 | 41,618 | scalar fallback (pre-AVX2) |
| tv 2-bit, search_k=500  | 1.000 | 42,014 | scalar fallback (pre-AVX2) |
| tv 4-bit, search_k=100  | 1.000 | 69,043 | scalar fallback (pre-AVX2) |
| tv 4-bit, search_k=1000 | 1.000 | 55,701 | scalar fallback (pre-AVX2) |

Note the fingerprint: latency is **independent of `search_k`** and **identical
warm vs cold** — the cost is the fixed `O(n_vectors · dim)` full-corpus
blocked-code scan, not I/O or candidate-set size.

### Why turbovec is slow on this host (diagnosis)

`meh` has `avx` but **no `avx2`**. turbovec v0.9.0 correctly dispatches to its
scalar `score_query_into_heap` path (the same path whose *correctness* bug
v1.8.0 fixed). That path is right but ~1000× slower than the AVX2/AVX-512
kernels: the on-disk codes use a FAISS-style perm0-interleaved layout built for
the AVX2 kernel, so the scalar path must `deinterleave_x86_code_byte` **per
byte, per vector** — ≈256M de-interleave evaluations per query for 2-bit (more
for 4-bit) over 1M × 1024-d. `EXPLAIN (ANALYZE)` confirms it is an
`Index Scan using docs_tv_*` (not a seq scan): the time is genuinely inside the
turbovec scan kernel. On an AVX2/AVX-512 host the SIMD kernel runs and these
latencies are expected to fall to the tens-of-ms range.

### Tunable recall frontier (oversampling)

The `meh` build (`7d01a51`) predates the `turbovec.oversample` feature, so this
run swept `search_k` only. Current `main` adds `turbovec.oversample`, which
fetches `ceil(search_k · oversample)` quantized candidates and re-ranks them by
exact distance — turning a fixed-quantization accuracy point into a tunable
recall frontier comparable to HNSW's `ef_search` (and to Qdrant oversampling /
VectorChord rerank). A monotone recall-vs-oversample curve is verified in
`benches/results/oversample_recall_curve_2026_06_15.json`. Re-running this
1M frontier on an AVX2 host with the oversample sweep is the natural next step.

## Caveats

- **Single host, pre-AVX2 CPU.** `meh` is an Ivy Bridge Xeon (`avx`, no
  `avx2`). turbovec's SIMD kernels (AVX2/AVX-512) do **not** run here; it takes
  the scalar fallback. **The pg_turbovec latency numbers are a worst-case
  floor, not a ceiling, and are NOT representative.** Storage, build, and
  recall are CPU-independent and valid.
- **Latency frontier for pg_turbovec is DEFERRED** to an AVX2+ host. **TODO:**
  re-run this exact 1M frontier (with the `oversample` sweep) on an AVX2 /
  AVX-512 machine to publish competitive turbovec latencies.
- **pg_turbovec is a flat (quantized full-scan) index**, not a graph index. It
  delivers near-exact recall and tiny storage; its latency is `O(n)` per query
  (SIMD-accelerated), versus HNSW's sublinear-but-approximate traversal. The
  two occupy different points on the recall/latency/storage trade-off.
- **Extension catalog vs binary:** `bench_wiki` reports the extension catalog
  at 1.7.1 (first `CREATE EXTENSION`), but the loaded `.so` is **v1.8.0** — the
  scan path lives in the binary, so all results were produced by v1.8.0.
- **No competitor beyond pgvector.** VectorChord / pgvectorscale comparison is
  future work.
- **10M not run** this round (budget); 1M is the priority standardized size.
- **Concurrent (pgbench) QPS** not run; single-connection QPS captured for
  HNSW.

## Reproduction

On a host with the v1.8.0+ binary installed (ideally **AVX2+** for
representative turbovec latency):

```bash
# 1. Build/install pg_turbovec from main (pgrx must match Cargo.toml's pgrx pin)
cargo pgrx install --release --pg-config <pgrx pg_config>

# 2. Schema + load 1M Cohere-wiki rows (1024-d), hold out 1000 queries
psql -d bench_wiki -f setup_schema.sql
python3 load_wiki_1m.py --corpus 1000000 --held 1000      # binary COPY pipe
python3 load_queryset.py                                  # held-out -> query_set

# 3. Exact brute-force ground truth (BLAS; cross-check vs in-DB seqscan)
python3 compute_gt_blas.py

# 4. Build indexes (HNSW m16/efc64, turbovec 4-bit, turbovec 2-bit)
psql -d bench_wiki -f build_indexes.sql

# 5. Sweep frontiers
python3 sweep_1m.py --which pgv --n-timed 200 --out res_pgv.json
python3 sweep_tv_lean.py --bits 2 --ks 100,500,1000 --out res_tv2.json  # drop the other tv index first
python3 sweep_tv_lean.py --bits 4 --ks 100,1000     --out res_tv4.json
```

Artifacts (this run): `benches/results/vectordbbench_cohere_wiki_1m_v1_8_0_20260615.json`.

The bench scripts used on `meh` live in `/scratch/pg_turbovec-bench/`
(`load_wiki_1m.py`, `load_queryset.py`, `compute_gt_blas.py`,
`setup_schema.sql`, `build_indexes.sql`, `sweep_1m.py`, `sweep_tv_lean.py`,
`sanity_check.py`).
