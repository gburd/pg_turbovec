# Item 3 — Sparse ANN opclass (`sparsevec`): design + feasibility memo

**Status: read-only analysis. No code written, no files edited, no EC2.**
**Bottom line: CONFIRM DEMAND BEFORE BUILDING. If forced to build, ship Sparse FLAT (KIND_SPARSE) first, defer WAND.**

---

## 0. Recommendation up front

**Do not build this yet.** There is no evidence any user has asked for
sparse ANN (§1). The gap is a *source-review parity item* against zvec, and
AGENTS.md is explicit that "competitor comparisons are source reviews until
measured" and that a rival having sparse FLAT/HNSW is not the same as a user
needing it. YAGNI applies: skip until asked.

**If/when demand appears**, the smallest first step that delivers value and is
measurable is **Sparse FLAT behind a new `KIND_SPARSE = 4` byte** (§2a, §4),
reusing the existing relfile chains and the sparse dot kernel that already
exists (§3). WAND (§2b) is a second, much larger engine — scope it only if
FLAT's O(n) wall is measured to bite on a real SPLADE corpus (§5).

---

## 1. Demand check (YAGNI) — NO evidence of user demand

Grepped `docs/`, `.agent/notes/`, `CHANGELOG.md`, and `git log --all`:

- **No user/customer/field report mentions sparse ANN, SPLADE, or an inverted
  posting index.** The only field report in `.agent/notes/FIELD_REPORT_SCALES_2026-09-05.md`
  is an IVF *scale-corruption* issue — unrelated.
- Every "sparse" hit is either the existing `sparsevec` *type* work
  (`git log`: commits `298439e`, `a39f2da`, `1cd8b80`, …) or the parity doc
  itself.
- `docs/PARITY_GAPS.md:345–352` is the sole origin: it is a **zvec source
  review** (`Z/src/core/interface/index.cc:1279–1323`), annotated per
  AGENTS.md as un-benchmarked and not to be promoted into a claim.
- `docs/HYBRID_SEARCH.md:279` already documents the **current** answer: keep a
  SPLADE vector in a `sparsevec` column and rank with `<#>` (negative inner
  product) — an *exact sequential* scan, no index. That works today; it is
  just O(n).

**Conclusion:** this is a speculative, competitor-derived gap. The honest
recommendation is: **state the gap in the roadmap, build nothing until a real
workload needs indexed sparse retrieval faster than a seqscan `<#>`.**

---

## 2. Two candidate engines

Distance for learned-sparse is **inner product** (SPLADE ranks by dot). Both
engines target `sparsevec <#> query` (`negative_inner_product`, strategy 1),
mirroring the `vec_ip_ops` opclass at `src/index/mod.rs:203`.

### 2a. Sparse FLAT — small, reuses everything

**What it is:** persist all sparse vectors, scan them all per query, keep top-k
by sparse dot. Exact-ish (exact if no approximation; "ish" only if we later add
value quantization). This is the direct analogue of the flat `KIND_SINGLE`
path, minus TurboQuant (which does not apply to sparse — AGENTS.md, PARITY_GAPS
Z2).

**Reuse of existing machinery in `src/index/`:**

- **KIND byte is the vehicle.** `src/index/page.rs:224–244` defines
  `KIND_SINGLE=0, KIND_COLBERT=1, KIND_GRAPH=2, KIND_BQ=3`. Add
  `KIND_SPARSE=4`. The decode path (`page.rs:940–947`) reads the kind byte at
  offset 6, which is zero/reserved on pre-v5 pages — so **an existing index of
  any other kind still decodes byte-identically. Additive, no wire bump, no
  REINDEX** — exactly the `KIND_BQ` pattern (AGENTS.md: "prefer a new `kind`
  over a wire bump").
- **Chains.** `sparsevec` is `(dim: i32, indices: Vec<i32>, values: Vec<f32>)`
  (`src/sparsevec.rs:26–33`). Unlike dense codes, sparse rows are
  variable-length, so we need **two chains + an offsets array**: a `postings`
  chain (concatenated `(index,value)` pairs, or two parallel `indices`/`values`
  chains) plus a per-row length/offset table, alongside the existing `ids`
  chain. `relfile.rs` already writes fixed-stride chains
  (`write_chain_at`, `read_chain`); the new part is the variable-stride offset
  bookkeeping. That is real but contained.
- **Meta-LAST + running-sum discipline.** AGENTS.md's recurring corruption bug
  is "a chain-offset running sum that omits a chain" (bitten 4×). Adding a
  sparse postings chain + offsets chain means **every running sum in `page.rs`
  and `relfile.rs` must include them.** Non-negotiable per the HARD MANDATE.
- **Scan/insert dispatch** mirrors `KIND_BQ` (`scan.rs:1021`, `insert.rs:390`):
  a `kind == KIND_SPARSE` arm that loads the postings chain and runs the sparse
  top-k. Insert appends a row to the chains (like flat).

**Can it live behind a new KIND byte, additive, REINDEX-free? YES** — this is
the whole point of the KIND design and the reason FLAT is the right first step.

### 2b. Inverted / BlockMax-WAND — the real SOTA, but a SECOND full engine

**What it is:** per-term posting lists (`term -> [(doc, weight)]`), sorted, with
per-block max-weight metadata; query evaluates only the query's terms and uses
the block maxima to skip documents that cannot enter the top-k (WAND / BMW).
This is what dedicated sparse engines use; it is sublinear where FLAT is O(n).

**Honest scope vs the existing IVF layer:** `src/index/ivf.rs` is **2966
lines** — coarse k-means training, cell directory, permutation, soft
assignment, centroid graph, rotation. A posting index is a *comparably large*
new subsystem with a **different persist shape**:

- **New persist surface:** posting lists are variable-length and grow
  per-insert at arbitrary positions (not append-only like FLAT/BQ). This is a
  genuinely new on-disk layout — term dictionary + per-term posting chains +
  per-block skip metadata. None of the existing chain code assumes
  insert-in-the-middle; it assumes whole-chain rewrite or append.
- **VACUUM:** deletes must tombstone entries *inside* posting lists; block
  maxima must be recomputed. The existing tombstone-bitmap approach
  (BQ path) does not directly transfer to per-term lists.
- **Crash safety:** every mid-list mutation is a new torn-write window. AGENTS.md's
  corruption history (5 root causes, `FIELD_REPORT` still-open scales-tear
  investigation) shows how expensive each new write window is. The HARD MANDATE
  requires, for *every* persist-path change: a reproduction test that fails
  before / passes after **AND** a sustained-insert no-recorruption run (the
  v1.28.4 lesson). WAND multiplies these windows.
- **Insert degradation must be observable** (AGENTS.md): if inserts can't
  maintain block maxima cheaply and fall back, that must stamp a
  machine-readable flag + emit a WARNING, same as `ivf_degraded`.

**Verdict:** WAND is IVF-sized *at minimum*, with a harder persist/VACUUM/crash
story than IVF (which is append-and-flat-scan; IVF-1 never even shipped a
cell-restricted scan — `docs/UPGRADING.md:118`). Do not start here.

---

## 3. Kernel decision — the sparse dot already exists, reuse it

`src/sparsevec_ops.rs` already computes exactly what a scan needs:

- `sparse_walk` (`sparsevec_ops.rs:15`) — two-pointer merge over sorted-unique
  index sets, **O(nnz_a + nnz_b)**.
- `sparsevec_inner_product` (`:47`) and `sparsevec_negative_inner_product`
  (`:55`) — the ranking function for strategy-1 `<#>`.

A FLAT scan reuses this directly: for each stored row, decode `(indices, values)`
from the postings chain into a `Sparsevec` and call the existing walk against
the query. **No new kernel, no TurboQuant, no SIMD dispatch decision** — the
sparse dot is inherently gather-bound, not SIMD-friendly, so the scalar
two-pointer walk is the right kernel on every host (this also means, unlike
dense, **meh/rv are usable for latency here** since there is no AVX2 fast path
to miss — though recall/QPS still want a realistic corpus).

For WAND the kernel is different (max-score accumulation with block skipping),
but that only matters if §2b is ever built.

---

## 4. Effort estimates

| Engine | Effort | Why |
|---|---|---|
| **Sparse FLAT** (`KIND_SPARSE=4`, opclass over `sparsevec`, postings+offsets chains, scan/insert dispatch, VACUUM append/tombstone) | **M** | Kernel is free (§3); KIND path is a proven additive pattern (§2a); the only genuinely new work is variable-stride chain offsets + adding the new chains to every running sum + the mandated fail-before/pass-after corruption tests. |
| **Inverted / BlockMax-WAND** | **XL** | New subsystem the size of `ivf.rs` (2966 LOC) with a *harder* persist/VACUUM/crash model (mid-list mutation, per-block maxima, tombstones), each new torn-write window carrying the full HARD-MANDATE test burden. |

**Staged recommendation:**

1. **Stage 0 (do now):** nothing but a roadmap line. Document that seqscan
   `sparsevec <#>` (already in `docs/HYBRID_SEARCH.md:279`) is the supported
   path, and that indexed sparse waits on demand.
2. **Stage 1 (smallest valuable step, only if demanded):** Sparse FLAT behind
   `KIND_SPARSE=4`, IP-only opclass `sparse_ip_ops` over `sparsevec`. Delivers
   a real index (avoids the densify-to-`vector` blowup that PARITY_GAPS Z2
   calls out) and is directly **measurable against two baselines**: (a)
   seqscan `<#>` on the `sparsevec` column, (b) densify-then-turbovec. FLAT
   won't beat WAND asymptotically but it removes densification and is honest
   and corruption-safe.
3. **Stage 2 (only if FLAT's O(n) is measured to bite):** WAND. Gate it on a
   measurement, not on parity aspiration.

---

## 5. What to measure to decide FLAT-vs-WAND, and on what corpus

**Decision metric:** at the target corpus size and query nnz, is FLAT's
per-query scan (O(n · avg_nnz) two-pointer walks) inside the latency budget? If
yes, FLAT is sufficient and WAND is unjustified. WAND only earns its complexity
where n is large enough that the full scan misses budget *and* query terms are
few enough that block-skipping prunes most docs.

**Measure:**
- FLAT p50/p95 latency and QPS vs corpus size (100k → 1M → 10M rows), holding
  query nnz fixed (SPLADE queries are typically ~20–100 non-zero terms).
- Recall is **1.0 for exact FLAT** — so recall is *not* the axis here; latency
  and storage bytes/vec are. (Contrast dense, where recall is the whole game.)
- Storage: bytes/vec for the postings+offsets chains vs the densify-to-`vector`
  baseline (this is where FLAT's win over densification shows).
- For a WAND go/no-go: measure the *fraction of docs a query's terms even
  touch* on the real corpus — that upper-bounds WAND's possible speedup over
  FLAT. If most docs share the common query terms, WAND barely helps.

**Corpus — a REAL SPLADE corpus is required, and none is referenced in-repo.**
No SPLADE/learned-sparse corpus exists anywhere in `benches/` or the docs.
`docs/BQ_RECALL_BENCH.md:447–451` establishes the hard rule: a synthetic
corpus must pass a **resolvability probe** (nn1→nn100 spread ≈ 37–268% on real
data; a synthetic 1M attempt was *discarded* at 6.6–10.4% spread as
statistically unrankable). For sparse:

- **Preferred:** a public SPLADE corpus (e.g. SPLADE++ over MS MARCO passages,
  ~8.8M docs, ~30k vocab) with held-out queries and known relevance judgments —
  because for sparse the meaningful quality metric is **MRR/nDCG against
  qrels**, not self-recall.
- **Since recall of exact FLAT is trivially 1.0**, the resolvability rule
  matters less for FLAT correctness and more for making any *WAND* recall
  comparison meaningful. The smallest realistic rankable synthetic that would
  satisfy `BQ_RECALL_BENCH`'s probe: **~100k docs, vocab ~30k, per-doc nnz
  ~150 drawn from a Zipfian term distribution** (so common terms create the
  long posting lists WAND must skip and the resolvability spread mimics real
  SPLADE) — but this is only for a WAND-vs-FLAT *latency/skip-efficiency*
  study, never for a headline quality claim. A quality claim needs real qrels.

**Host note:** unlike dense turbovec, the sparse dot has no AVX2/AVX-512 fast
path, so `meh` and `rv` are valid for FLAT *latency* here (§3). QPS/skip studies
for WAND still prefer `arnold` for realistic single-thread throughput.

---

## Appendix — file:line index

- Opclass registration (where a new `sparse_ip_ops` would go): `src/index/mod.rs:203–260`.
- KIND constants: `src/index/page.rs:224` (`KIND_SINGLE`), `:226` (`KIND_COLBERT`), `:228` (`KIND_GRAPH`), `:244` (`KIND_BQ`); additive decode `:940–947`.
- Sparse dot kernel: `src/sparsevec_ops.rs:15` (`sparse_walk`), `:47` (`inner_product`), `:55` (`negative_inner_product`).
- Sparse type layout: `src/sparsevec.rs:26–33`.
- Chain write/read primitives: `src/index/relfile.rs:480` (`write_chain_at`), `:646` (`read_chain`); BQ additive relfile write `:725`.
- IVF layer size reference: `src/index/ivf.rs` (2966 lines).
- KIND_BQ scan/insert dispatch precedent: `src/index/scan.rs:1021`, `src/index/insert.rs:390`.
- Parity gap origin: `docs/PARITY_GAPS.md:345–352`, Phase Z2 `:558–561`.
- Current supported sparse path (seqscan): `docs/HYBRID_SEARCH.md:279`.
- Resolvability rule for synthetic corpora: `docs/BQ_RECALL_BENCH.md:447–451`.
