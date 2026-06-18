# Phase F — Index-native late interaction (per-token ColBERT MaxSim)

_Status: **F-1 SHIPPED (v1.16.0); F-2 SHIPPED (v1.17.0); recall win
CONFIRMED across two corpora (SciFact + NFCorpus).** Design
pressure-tested against the
v1.15.1 codebase. This is the last acknowledged feature gap vs
Qdrant/VectorChord; it is deliberately phased so the cheap, reuse-
heavy first cut (F-1) proves the value before the expensive persistent
index (F-2) is funded._

> **F-2 gate result (2026-06-17, `floki`, BEIR/SciFact 5,183 docs, AVX2).**
> `turbovec.colbert_search` (F-1) beat the Phase-D pooled+rerank
> baseline at **every** config, vs real qrels: **+0.06 nDCG@10 /
> +0.09 Recall@10** at the value operating point, rising to **+0.10
> nDCG / +0.14 Recall at small candidate budgets** (`candidate_n=128`,
> where the pooled baseline collapses because the relevant doc's mean
> vector falls outside top-N — F-1 still finds it via its best single
> token: the predicted gap, confirmed on real embeddings). The plan's
> single biggest risk — "2–4 bit quantization destroys the token
> signal" — **did not materialise**: 2/3/4-bit differ by ≤0.002 nDCG
> (2-bit is enough). F-1 reaches the exact brute-force MaxSim ceiling
> (nDCG 0.619 / R 0.744). F-1 latency is 1.5–2.1× the baseline only
> because it **rebuilds the backend-cached token index every call** —
> the exact cost F-2's persistent index removes. **Verdict: GO
> (qualified)** — fund F-2, but first confirm the delta on a second,
> entity-heavier / out-of-domain corpus (NFCorpus / FiQA / LoTTE),
> since the baseline closes most of the gap at large `candidate_n`, so
> F-2's value proposition is specifically "high recall at low
> candidate budget / low latency." Data:
> `benches/results/colbert_f1_gate_floki_scifact_20260617.json`;
> harness: `benches/scripts/colbert/`.
> One bug to fix in F-2 work: `colbert_search` leaks ~28 MB of
> backend RSS per call (the harness works around it by reconnecting
> every 40 queries; noted in the harness README)._

> **F-2 CONFIRMATION (2026-06-18, `floki`, BEIR/NFCorpus 3,633 docs,
> AVX2, persistent `vec_colbert_ops` index).** The SciFact gain
> **REPLICATED** on a second, out-of-domain (medical/nutrition),
> entity-heavier corpus, exercising the shipped F-2 persistent index:
> **+0.044 nDCG@10 / +0.037 Recall@10** at the value point
> (`candidate_n=256`), **+0.065 nDCG at low budget** (`candidate_n=128`,
> where the pooled baseline collapses to 0.220 while `colbert_search`
> holds at 0.285) — same sign, same mechanism, same low-candidate-budget
> shape at *every* config. Absolute deltas are ~30% smaller than
> SciFact (NFCorpus's dense graded qrels compress Recall@10), but the
> win holds exactly where F-2's value lives. Quantization signal
> intact (2-bit ≈ 4-bit, ≤0.0001 nDCG; 2-bit index = 43 MB).
> **The persistent index built cleanly** (561k token slots, 42s/43 MB
> at 2-bit, no OOM) and **served from disk — the F-1 ~28 MB/call leak
> is GONE** (backend RSS plateaus flat at ~360 MB; ~1.4 KB/call warm).
> **Verdict: the qualified GO is upgraded to an established
> cross-domain recall win.** Data:
> `benches/results/colbert_f2_confirm_floki_nfcorpus_20260618.json`._

---

## 0. The honest framing (read first)

pg_turbovec already ships the **stage-2** half of ColBERT: Phase D's
`turbovec.max_sim(query vector[], doc vector[])` computes
`MaxSim(Q,D) = Σ_{q∈Q} max_{d∈D} sim(q,d)` correctly, and
`docs/HYBRID_SEARCH.md` documents the two-stage retrieve-then-rerank
pattern (ANN-retrieve on a pooled vector, MaxSim-rerank from the heap
`vector[]`).

**The one thing Phase D cannot do is stage-1 recall.** Phase D's
candidate recall is capped by the *pooled* (mean) document vector: a
doc is only retrieved if its mean token vector is near the query. The
queries ColBERT was built for — rare entities, specific terms, long
docs where one passage matters and the mean washes it out — are
exactly the ones where a relevant doc's pooled vector falls outside
top-N and Phase D never scores it. Index-native late interaction
retrieves by **best-single-token** proximity (search *all* doc tokens),
so that doc is found via its one matching token.

So Phase F's entire new value is concentrated in **stage-1 recall**.
The literature delta is ~5–15 nDCG@10 points on out-of-domain /
entity-heavy BEIR/LoTTE sets — real, but **workload-dependent**, and
almost entirely a recall effect. For most semantic-similarity queries
the pooled vector is a good summary and Phase D's ceiling ≈ index-
native's. **We ship the cheap first cut, measure the delta on a real
ColBERT corpus, and only fund the expensive persistent index if the
measured delta justifies a 32–512× larger index.**

---

## 1. What the code is today (grounding)

- **AM stores one quantized vector per heap TID.** `build_callback`
  (`src/index/build.rs`) decodes one `Vector` per heap tuple, keyed by
  `item_pointer_to_u64(tid)`. Relfile = parallel codes/scales/ids
  chains keyed by slot, plus blocked/rotation/IVF chains
  (`src/index/page.rs`, `MetaPageData::version = 4`).
- **Scan emits one TID per single-vector distance.** `amrescan` pulls
  `orderbys[0]` (one query vector); `amgettuple` runs one
  `arc.search(query, k)` / IVF `search_masked`; `xs_recheckorderby`
  re-ranks by exact distance. (The `amrescan` scan-key path is the
  Phase-17 `munmap_chunk` crash site — **do not rewrite it.**)
- **The turbovec kernel already batches queries:** `search(queries, k)`
  with `nq = queries.len()/dim`, results row-major per query.
- **`IdMapIndex` enforces unique ids** — but the IVF build already
  sidesteps this (synthetic slot-ids `0..n_slots`, real external ids,
  with duplicates, persisted separately). **A token→doc index is the
  same shape:** many token slots, one repeated doc-id each.
- **`turbovec.knn`** (`src/knn.rs`) is a `#[pg_extern]` SET-returning
  function that builds/loads an index in the **backend cache only**
  (never writes a relfile), bypassing the operator/planner/amrescan
  entirely. This is the F-1 template.

---

## 2. The five hard problems and the chosen answer

| # | Problem | Chosen answer (simplest that keeps invariants) |
|---|---|---|
| Q1 | AM stores 1 vec/tuple; ColBERT needs N tokens/doc | Per-token slots; ids chain holds the repeated doc-id (the IVF soft-assign / synthetic-slot-id trick). 32–512× more slots — intrinsic to ColBERT. |
| Q2 | Two-stage retrieval | **Token-index stage 1 + heap-reread stage 2 via Phase D `max_sim`.** Don't build an index-resident token gather (it breaks OOC cell-contiguity, Q5). |
| Q3 | No MaxSim operator; can't ride `ORDER BY <=>` | **`turbovec.colbert_search(...)` SET-returning function**, `turbovec.knn` model. No new type, no amrescan rewrite, no planner work. |
| Q4 | Wire format | **F-1 needs none** (backend-cache index, version stays 4). F-2's persistent token index is an **additive v5 separate index kind** (new opclass), v4 single-vector indexes byte-identical, `is_legacy_v4()` per contract. |
| Q5 | Determinism / OOC / VACUUM | Tokens cell-contiguous (reuse IVF); tombstone on VACUUM (a deleted doc's TID kills all its token slots via the existing `ivf_tombstone_dead`); **heap-reread stage 2 sidesteps OOC cell-contiguity entirely.** |

---

## 3. Phasing

### F-1 — minimal first cut (no wire change, reuses everything)

A `#[pg_extern(stable, parallel_safe)]` SET-returning function:

```sql
turbovec.colbert_search(
    rel          regclass,
    id_col       text,          -- bigint doc key
    token_col    text,          -- a vector[] column (per-doc token arrays)
    query        vector[],      -- the query's token vectors
    k            integer,       -- final top-k docs
    per_token_k  integer DEFAULT 64,    -- stage-1 hits per query token
    candidate_n  integer DEFAULT 256,   -- max candidate docs into stage 2
    bit_width    integer DEFAULT 4
) RETURNS TABLE(id bigint, score double precision)
```

Algorithm:
1. **Build/load a backend-cached flat token index** (one slot per
   token across all docs, `id = doc-id`), exactly the `turbovec.knn`
   cache model. SPI-unnest the `token_col` arrays. **No relfile;
   `MetaPageData::version` stays 4.** Cache-key includes `token_col`
   so it doesn't collide with a single-vector knn cache entry.
2. **Stage 1:** one batched `idx.search(query_tokens_flat, per_token_k)`
   → |Q|×per_token_k (slot→doc-id) hits; union doc-ids into a
   candidate set capped at `candidate_n`.
3. **Stage 2:** for each candidate doc, call the existing Phase D
   `max_sim` against the heap `token_col` array; return top-k
   `(doc_id, score)` ordered by score DESC.

F-1 delivers the verified gap (stage-1 recall over all tokens) while
reusing the knn cache, the batched kernel, and Phase D's rerank
verbatim. **Honest:** F-1 *is* "a small delta over Phase D" — an
index-accelerated stage 1 bolted in front of the Phase D stage 2. That
is the point: it's cheap to ship and it's exactly enough to measure
whether the recall gap is worth F-2.

**F-1 tests:** colbert_search returns only real doc-ids; recall vs a
brute-force MaxSim over the whole corpus on a small synthetic set;
stage-1 recall strictly ≥ pooled-vector recall on an entity-style
query crafted so one token matches but the mean doesn't (the gap
proof); empty query / empty candidate behaviour; determinism (same
inputs → same ranking).

**F-1 benchmark (the gate for F-2):** on a real ColBERT corpus
(LoTTE or a BEIR slice), measure recall@k and latency of
`colbert_search` vs Phase-D pooled+rerank, sweeping
`(bit_width, per_token_k, candidate_n)`. Two questions: (a) what
operating point reaches ColBERT-grade recall, and (b) is F-1 at that
point actually faster / higher-recall than Phase D pooled+rerank?
**Do not start F-2 until this delta is measured and positive.**

### F-2 — persistent token index AM (only if F-1's delta justifies it)

- Additive **v5** wire format; **`vec_colbert_ops`** opclass over a
  `vector[]` column (or `WITH (multivector = true)` reloption). v4
  single-vector indexes byte-identical; `is_legacy_v4()` per the
  migration contract; minor release with no REINDEX for v4 indexes.
- `build_callback` unnests the `vector[]` into token slots (fixed
  array order = on-disk order → determinism).
- IVF **cell-contiguous token layout** → stage-1 token search is the
  existing OOC IVF path (reuse Phase B-1/B-2 cell-scoped serving;
  n_tokens ≫ n_docs makes OOC matter *more*).
- VACUUM: reuse `ivf_tombstone_dead` — a deleted doc's TID marks all
  its token slots dead naturally (they decode to the same dead TID).
  **Tombstone, never swap-remove** (swap-remove breaks cell
  contiguity).
- Stage 2 **still heap-reread** via Phase D (keeps OOC cell-
  contiguity intact; the heap holds each doc's `vector[]` contiguously).

### Deferred (F-3+)

- Index-resident token gather + dequantized in-index MaxSim — only if
  heap-reread stage 2 is measured as the bottleneck (it breaks OOC
  cell-contiguity, so it's a real cost).
- PLAID-style centroid-interaction pruning.
- A native multivector type + `ORDER BY` operator (touches the
  forbidden amrescan path; highest risk; only if operator ergonomics
  are demanded).

---

## 4. The single biggest risk

**Stage-1 candidate quality under heavy quantization.** ColBERT
tokens are 128-d L2-normalised; at 2–4 bit the per-token quantization
error may be large relative to the tight inter-token distances that
separate a relevant doc's token from a near-miss. If quantized token
ranking is noisy you need a large `per_token_k`, which inflates the
candidate set and erodes F-1's advantage over Phase D. **Unknown until
measured** (§ F-1 benchmark). If the answer is "you need near-exact
tokens," index-native's advantage shrinks toward Phase D — which is
precisely why F-1 is structured to measure this cheaply before F-2 is
funded.

---

## 5. Why this is the right plan

- It isolates the genuinely-new capability (stage-1 recall) and
  reuses the most-tested code (knn cache, batched kernel, Phase D
  rerank) for everything else.
- It avoids, in F-1, every hard invariant: no wire change, no amrescan
  rewrite, no new OOC/VACUUM machinery.
- It refuses to build a 32–512× persistent index on faith — F-1's
  benchmark is the explicit gate for F-2.
- It is honest that the 90% (pooled-summary) case is already covered
  by Phase D, and concentrates effort on the 10% (entity/rare-term)
  case where late interaction actually wins.
