# Competitive Analysis: pg_turbovec vs pgvector & Qdrant

_Last updated: 2026-06-15. Source-of-truth read directly from
`src/`, pgvector README @ master (v0.8.2), and Qdrant's documented
feature set._

This document drives roadmap decisions. It identifies where
`pg_turbovec` already wins, where it has parity, and the one true
gap that bites users today.

---

## Bottom line

> **2026-06-16 — reflects v1.10.1 (IVF shipped).** An earlier draft
> said pg_turbovec "loses to HNSW on latency by ~490×" — true for the
> *flat* scan, but v1.10.0 shipped the **IVF coarse-quantizer layer**
> which closes most of that gap. Updated comparison below.

`pg_turbovec` (v1.10.1) wins decisively on **storage** (4-bit: ~7.6×
smaller, 2-bit: ~15.2× smaller than pgvector HNSW, measured on
Cohere-wiki 1M×1024-d), **build memory** (Phase W/W-2), and **exact
or tunable recall** (flat: R@10 = 1.000; IVF: tunable via `probes`).
It has near-complete feature parity: types, all 4 distance ops,
aggregates, `||` concat + halfvec arithmetic, **iterative scans**,
**parallel build**, **oversampling**, and now an **IVF index** with
`probes`/`max_probes`/`assign_dups` tuning.

**Latency — the v1.10/v1.11 story.** pg_turbovec offers TWO index modes:
- **Flat** (`lists = 0`, default): exact recall (1.000), tiny
  storage, but `O(n·dim)` per query — ~2.5 s at 1M×1024-d (AVX2),
  ~490× slower than HNSW. Right for small corpora, exact-recall
  needs, or pre-filtered subsets.
- **IVF** (`WITH (lists = N)`): sublinear — scans only the `probes`
  nearest cells. **Measured head-to-head at 500k×1024-d (AVX2,
  isolated, v1.11.0):** at recall@10 ≥ 0.95, **IVF warm p50 = 18.5 ms
  (probes=64) vs HNSW 7.9 ms (ef=200)** — HNSW wins the 0.95 point by
  ~2.3×, but IVF is the **same order of magnitude** (tens of ms, not
  the 490× flat gap) and **wins the ≥ 0.99 recall tail**: IVF hits
  0.99 at 25.3 ms while this HNSW config never exceeds 0.983. IVF
  crushes ivfflat at every matched recall (18.5 ms vs ~80–117 ms).
  The recall/latency dial is `probes` (the `ivfflat.probes` /
  `hnsw.ef_search` analogue), all while keeping the 7–15× storage win.

**Honest positioning (v1.11.0):** "compact (7–15× smaller) PG vector
index with a tunable exact↔fast dial — flat for exactness, IVF for
sublinear latency that lands in HNSW's neighbourhood and wins the
high-recall tail." The remaining gaps vs the leaders are
**metadata-filter PUSHDOWN** (we post-filter + iteratively widen
probes, not true in-traversal filtering like Qdrant/VectorChord),
**out-of-core / >RAM** (pgvectorscale's DiskANN bet) — which now also
**blocks the IVF build above ~500k–600k on a 31 GiB host** (the build
accumulates the full corpus + a permuted copy in RAM; Phase B-4), and a
**published 1M+ IVF latency frontier** (currently capped at 500k by
that build ceiling).

---

## Gap table

Severity from a pgvector user's perspective evaluating a swap.
Effort: S (<1wk), M (~2wk), L (~1mo), XL (multi-month).

| Feature | pgvector 0.8.2 | Qdrant | pg_turbovec v1.10.1 | Severity | Effort |
|---|---|---|---|---|---|
| Index types | HNSW + IVFFlat | filterable HNSW | flat + **IVF** (`turbovec` AM) | none (ours wins storage) | ✅ done |
| Index tunability | m, ef_construction, ef_search, lists, probes | m, ef_construct, ef | `bit_width`, `lists`, `assign_dups`, `search_k`, **`probes`**, `max_probes`, `oversample` | none | ✅ done |
| **Iterative/streaming scan** | `iterative_scan`, `max_scan_tuples` | filter-aware traversal | **`iterative_scan` + `max_scan_tuples` + IVF probe-widening** | none | ✅ done (v1.8.0) |
| **Metadata filtering** | post-filter + iterative + partial idx | **integrated (killer feature)** | post-filter + iterative probe-widening (NOT true in-traversal pushdown) | minor | XL (true pushdown) |
| Quantization quality | halfvec, bit, binary_quantize | scalar/PQ/binary | TurboQuant 2/3/4-bit (best storage/recall) | none (we win) | — |
| Quantization tuning | manual re-rank CTE | rescore + oversampling | **`oversample` + `probes` + `assign_dups`** | none | ✅ done (v1.9.0/v1.10.0) |
| Vector arithmetic | `+ - *` and `\|\|` concat | N/A | **`+ - *` + `\|\|` (vector & halfvec)** | none | ✅ done (v1.8.0) |
| Aggregates | avg/sum (vector, halfvec) | N/A | avg/sum (vector, halfvec), sum (sparsevec) | none | — |
| Subvector / helpers | subvector, l2_normalize, etc. | N/A | all present (+ jsonb extras) | none | — |
| Hamming/Jaccard indexed | `<~>` `<%>` on hnsw/ivfflat | N/A | exact only, not indexable | minor | L |
| Max dims (indexable) | vec 2k / half 4k / bit 64k / sparse 1k | — | `vector` opclass; others via `::vector` cast | minor | M |
| **Parallel index build** | yes | yes | **yes** (`build_parallelism`, rayon) | none | ✅ done (v1.8.0) |
| Multivector / named / hybrid | app-side RRF | native fusion | none | minor (scope) | XL |
| Replication / HA | WAL → replication + PITR | native Raft | inherits PG WAL | none | — |
| Observability | `pg_stat_progress_create_index`, EXPLAIN BUFFERS | dashboards | works w/ PG tooling; `blocks_skipped_by_mask` proxy; no build phases | minor | M |
| **Query latency (recall@10 ≥ 0.95, AVX2)** | HNSW ~8 ms (R 0.97, 500k) | in-mem ms | flat ~2.5 s/1M (R 1.0); **IVF 18.5 ms @ probes=64 (R 0.96, 500k); wins ≥0.99 tail @ 25 ms** | flat loses / IVF competitive (~2.3× behind HNSW @ 0.95, ahead @ 0.99) | ✅ IVF measured (v1.11.0, 500k) |
| **Storage (500k / 1M×1024-d)** | 3902 MB / 7806 MB | larger | **518 MB IVF / 1026 MB flat-4bit / 512 MB 2-bit** | ✅ we win 7.5–15× | — |
| Cold-scan latency | ~100 ms | in-mem | lazy `id_to_slot` cut the dominant term (v1.8.0); flat first-scan still O(n) | minor | L |
| Out-of-core (>RAM) | no (HNSW in-RAM) | spill | **no** — also caps IVF BUILD at ~500k–600k on 31 GiB (Phase B-4) | major vs pgvectorscale | XL |
| Large-scale published bench | ann-benchmarks, VectorDBBench | VectorDBBench | Cohere-wiki 1M (storage/recall/flat-latency); **IVF-vs-HNSW frontier measured at 500k** (1M+ blocked on B-4 build) | major | M (B-4 unblocks 1M+) |

---

## Prioritized roadmap

### Done since this analysis was first drafted (v1.8.0 – v1.10.1)

Every "must-have for parity" item from the original draft has
shipped:

1. **Iterative / refilling index scan** — ✅ v1.8.0.
   `turbovec.iterative_scan` (off | relaxed_order) +
   `turbovec.max_scan_tuples`; refill re-enters the search with a
   growing `k` (flat) or widens `probes` (IVF). Fixes the
   selective-`WHERE` under-return.
2. **Parallel index build** — ✅ v1.8.0. `turbovec.build_parallelism`,
   rayon over the encode+repack phases; byte-identical relfiles.
3. **Cold-scan latency** — ✅ v1.8.0. Lazy `id_to_slot` removed the
   dominant per-backend cache-fill term (the O(n) HashMap build).
4. **`||` concat + halfvec arithmetic** — ✅ v1.8.0.
5. **Oversampling** — ✅ v1.9.0. `turbovec.oversample`; recall
   0.81→1.0 as oversample 1→4 (the reorder queue is the rescore).
6. **IVF coarse-quantizer layer** — ✅ v1.10.0/v1.10.1. The big one:
   `WITH (lists = N)` + `turbovec.probes`/`max_probes` +
   `assign_dups` (soft assignment). Turns the flat O(n) scan into a
   sublinear cell scan; measured ~5× AVX2 warm-p50 win at probes=16.

### Remaining roadmap (the real current gaps)

1. **Published large-scale IVF latency frontier** (effort M) — the
   #1 credibility item. We have storage/recall/flat-latency at 1M
   (Cohere-wiki), a host-independent IVF recall-vs-probes curve, and
   a *small* (200k) in-process AVX2 IVF warm-p50. Missing: an
   isolated 1M+ × 1024-d IVF latency sweep on a quiet AVX2 host
   (arnold), head-to-head vs pgvector HNSW + ivfflat — ideally a
   VectorDBBench entry. This is measurement, not code.
2. **True metadata-filter pushdown** (effort XL) — we post-filter +
   iteratively widen probes; Qdrant/VectorChord filter *inside* the
   index traversal. Our approach is correct (no under-return) but
   scans more candidates than a true filtered index under a very
   selective predicate. The PG-idiomatic partial-index + iterative
   combo covers most real cases; full pushdown is a large build.
3. **Out-of-core / >RAM** (effort XL) — pgvectorscale's DiskANN bet.
   pg_turbovec is RAM-resident; a corpus whose IVF index exceeds RAM
   has no streaming-traversal story yet. The 7–15× storage win
   raises the in-RAM ceiling substantially, but it is still a
   ceiling.
4. **IVF build speed at scale** (effort M) — the GEMM fix made it
   feasible, but a 200k×256-d / lists=448 build still took ~2.7 min
   (k-means dominates). Worth a faster k-means (mini-batch, fewer
   Lloyd iters, or sampling) for 1M+ builds.
5. **Multivector / named-vectors / hybrid fusion** (effort XL,
   scope) — express via columns + app-side RRF unless a user asks.
6. **Indexed bitvec ANN** (effort L) — TurboQuant doesn't fit
   Hamming space; keep exact `<~>`/`<%>` only.

#### Legacy detail (kept for reference)

5. **Rescore + oversampling knobs** (effort M) — ✅ **SHIPPED**
   (v1.8.x, scan-side only, additive GUC). `turbovec.oversample`
   (float, default 1.0, range 1.0..100.0): the scan fetches
   `ceil(search_k * oversample)` quantized candidates, and the
   always-on reorder queue (`xs_recheckorderby`) re-ranks them by
   exact full-precision distance, trimming to the true top-k under
   the LIMIT. There is no separate `turbovec.rescore` GUC —
   oversampling plus the reorder queue together ARE the rescore
   mechanism (the reorder queue already re-ranks every returned
   tuple by exact distance, so an AM-side rescore is redundant;
   measured: oversample alone drives recall@10 to 1.0). Composes
   with iterative scan (sets the initial k; iterative refill grows
   it). Measured curve (4-bit, 3000×64, search_k=10): recall@10
   0.81→0.96→0.99→1.0 at oversample 1.0/1.5/2.0/4.0, p50 ~ linear
   (3.8→4.7 ms). Turns the quantization advantage into a tunable
   recall frontier (matching Qdrant `oversampling` / VectorChord
   rerank) and beats pgvector's manual-CTE ergonomics. See
   `docs/PARITY_GAPS.md` § Recall tuning.

6. **Adopt turbovec ≥ 0.9.0 TQ+ calibration** (effort M). Per-
   coordinate shift/scale that improves recall. New persisted state
   → wire-format `MetaPageData::version` 3→4 → minor release with a
   REINDEX migration per `AGENTS.md`. Worth it where TQ currently
   trails HNSW recall. **Note: the v0.9.0 upgrade also fixes the
   pre-AVX2 wrong-results bug (see `docs/PRODUCTION.md`); that part
   ships first in v1.7.3 with identity TQ+ and no wire change.**

### Won't-do (standalone-DB concerns)

7. Own sharding/replication/consensus — Postgres owns this.
8. Multivector / named-vectors / server-side RRF — schema/query
   layer; express via multiple columns + app-side RRF unless a real
   user asks.
9. Indexed bitvec ANN (hamming/jaccard opclass) — TurboQuant doesn't
   fit Hamming space; needs a separate LSH kernel. Keep exact only.

---

## Qdrant takeaways

Qdrant is a standalone vector DB, not a PG extension. Its headline
differentiator is **payload filtering integrated into HNSW
traversal** (filterable HNSW) — it avoids the under-return problem
entirely rather than post-filtering. Its quantization is a menu
(scalar int8 / product / binary) with `rescore` + `oversampling`
knobs. It owns sharding/replication/Raft, which a PG extension
should not reimplement.

The lessons worth importing: (a) the filtering story matters most
to users — our iterative-scan fix (#1) is the PG-idiomatic answer;
(b) rescore/oversampling as explicit knobs (#5) is a clean UX win.
