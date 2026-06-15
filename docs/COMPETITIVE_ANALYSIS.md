# Competitive Analysis: pg_turbovec vs pgvector & Qdrant

_Last updated: 2026-06-15. Source-of-truth read directly from
`src/`, pgvector README @ master (v0.8.2), and Qdrant's documented
feature set._

This document drives roadmap decisions. It identifies where
`pg_turbovec` already wins, where it has parity, and the one true
gap that bites users today.

---

## Bottom line

`pg_turbovec` wins decisively on **storage** (4-bit: ~10× smaller
than pgvector HNSW), **build speed** (~15× faster), and **warm
recall** (R@10 = 1.000 on dbpedia-1M vs HNSW ef=40's 0.962). It has
near-complete scalar parity: types, distance operators, aggregates,
most arithmetic.

The one correctness/usability gap that bites users is **fixed-K
single-batch scans with no iterative refill** — a problem pgvector
already solved with `hnsw.iterative_scan`. Under a selective
`WHERE filter ORDER BY emb <=> q LIMIT k`, the executor post-filters
the fixed `search_k` candidates and silently under-returns. **This
is the #1 roadmap priority.**

---

## Gap table

Severity from a pgvector user's perspective evaluating a swap.
Effort: S (<1wk), M (~2wk), L (~1mo), XL (multi-month).

| Feature | pgvector 0.8.2 | Qdrant | pg_turbovec v1.7.2 | Severity | Effort |
|---|---|---|---|---|---|
| Index types | HNSW + IVFFlat | filterable HNSW | single `turbovec` AM | none (ours wins storage) | — |
| Index tunability | m, ef_construction, ef_search, lists, probes | m, ef_construct, ef, full_scan_threshold | `bit_width` + `search_k` | minor | S–M |
| **Iterative/streaming scan** | `iterative_scan`, `max_scan_tuples`, `scan_mem_multiplier` | filter-aware traversal | **none** — one fixed-K batch | **BLOCKER** | **L** |
| **Metadata filtering** | post-filter + iterative + partial idx | **integrated (killer feature)** | post-filter only; under-returns | **MAJOR** | XL (pushdown) / L (iterative) |
| Quantization quality | halfvec, bit, binary_quantize | scalar/PQ/binary | TurboQuant 2/3/4-bit (best storage/recall) | none (we win) | — |
| Quantization tuning | manual re-rank CTE | rescore + oversampling | `search_k` only | major | M |
| Vector arithmetic | `+ - *` and `\|\|` concat | N/A | `+ - *` vector-only; no `\|\|` | minor | S |
| Aggregates | avg/sum (vector, halfvec) | N/A | avg/sum (vector, halfvec), sum (sparsevec) | none | — |
| Subvector / helpers | subvector, l2_normalize, etc. | N/A | all present (+ jsonb extras) | none | — |
| Hamming/Jaccard indexed | `<~>` `<%>` on hnsw/ivfflat | N/A | exact only, not indexable | minor | L |
| Max dims (indexable) | vec 2k / half 4k / bit 64k / sparse 1k | — | `vector` opclass; others via `::vector` cast | minor | M |
| Parallel index build | yes | yes | **no** (single-threaded) | major | L |
| Multivector / named / hybrid | app-side RRF | native fusion | none | minor (scope) | XL |
| Replication / HA | WAL → replication + PITR | native Raft | inherits PG WAL | none | — |
| Observability | `pg_stat_progress_create_index`, EXPLAIN BUFFERS | dashboards | works w/ PG tooling; no build phases | minor | M |
| Cold-scan latency | ~100 ms | in-mem | ~1,256 ms first scan/backend | major | L |

---

## Prioritized roadmap

### Must-have for parity

1. **Iterative / refilling index scan** — `#1`, fixes a correctness
   gap. `amgettuple` runs one `search_k` batch and returns `false`
   when drained. With a selective `WHERE`, post-filtering those
   ~100 candidates under-returns — exactly what pgvector's
   `iterative_scan` fixes. Minimum lazy fix (effort L): re-enter
   `arc.search()` with a growing K (`search_k`, 2×, 4×…) when the
   cursor drains and the executor still wants tuples, capped by a
   new `turbovec.max_scan_tuples` GUC. The existing
   `xs_recheckorderby = true` reorder-queue already guarantees
   ordering. This also closes most of the metadata-filtering gap
   without true pushdown.

2. **Parallel index build** (effort L). `build.rs` is
   single-threaded; pgvector uses maintenance workers. The build
   win shrinks against a parallel HNSW build on big boxes.

3. **Cold-scan latency** (effort L, on roadmap). 1,256 ms
   first-scan-per-backend vs ~100 ms. A shared DSM segment for the
   `IdMapIndex` parts (vs per-backend Arc rebuild) is the lever;
   pooled-connection workloads churn backends and pay it every time.

4. **Cheap arithmetic parity** (effort S). Add `||` concat for
   `vector`, and `+ - * ||` for `halfvec`/`sparsevec`. Trivial.

### Differentiators (make pg_turbovec strictly better)

5. **Rescore + oversampling knobs** (effort M). Answers "is
   `search_k` enough?" — no. Expose `turbovec.oversample`: fetch
   `k * oversample` quantized candidates, rescore against the
   full/heap vectors, keep top-k. Pairs with the existing recheck
   path. Turns the quantization advantage into a tunable recall
   lever and beats pgvector's manual-CTE ergonomics.

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
