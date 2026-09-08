# 1-bit binary quantization (`WITH (bit_width = 1)`) — design + status

**Status: SHIPPED in v2.6.0 — `bit_width = 1` builds, scans, inserts and
vacuums.** The foundation (reloption + rerank default + pure-Rust sign-BQ
core) landed earlier; v2.6.0 wired the encode path, the Hamming scan
kernel, `aminsert`, VACUUM and `turbovec_check`.

Two corrections to what this doc originally specified, both recorded
below in place:

- **No wire-version bump.** §4 said "bump `VERSION` 7 → 8". That was
  written when v7 was current; by the time the encode path landed the
  format was already at v8, and the right discriminator turned out to be
  a **new `kind` byte (`KIND_BQ = 3`)**, not a version bump. Existing
  indexes keep `kind = SINGLE/COLBERT/GRAPH` and decode byte-identically
  — **no REINDEX**. The `bq_mean_*` fields sit at page offset 316, which
  was reserved-and-zero on every prior version.
- **IVF + 1-bit is not composed yet.** §5's closing line said IVF
  "composes"; it is currently rejected with a clear ERROR
  (`bit_width = 1` with `lists > 0`), because the cell-contiguous layout
  and per-cell Hamming path are not wired. Flat BQ works.

Prior offline study: `.agent/notes/BQ_HNSW_FEASIBILITY.md` (measured
recall + storage; findings respected here, not re-derived).

---

## 1. Is 1-bit TurboQuant-at-1-bit, or sign-BQ? — sign-BQ. (kernel evidence)

**Decisive:** the pinned `turbovec` crate (rev
`befc4cbf73ef40e440232ae597888c71fe1ba50c`) hard-rejects
`bit_width < 2` in BOTH constructors:

- `TurboQuantIndex::new(dim, bit_width)` —
  `turbovec/src/lib.rs:250`: `if !(2..=4).contains(&bit_width) { return
  Err(ConstructError::BitWidthOutOfRange(bit_width)) }`.
- `TurboQuantIndex::new_lazy(bit_width)` — `lib.rs:286`: same check.

`IdMapIndex::new` (the wrapper every pg_turbovec build/scan path calls)
delegates to these, so **turbovec cannot build a 1-bit index at all.**
1-bit is therefore a *distinct scheme*: **sign binary quantization
(sign-BQ)** — the DiskANN/pgvector/Qdrant coarse code:

| | TurboQuant (2/3/4-bit) | sign-BQ (1-bit, this feature) |
|---|---|---|
| code | rotated + Lloyd-Max codes | per-coord sign bit (after centering) |
| distance | rotated dot via LUT | Hamming (`popcount(XOR)`) coarse, then exact heap rerank |
| per-vec storage | `dim/8 * bit_width` + 4 B scale | `dim/8`, **no scale** |
| rotation / codebook | yes | no |
| metric | cosine / IP | angular ≈ cosine (matches the AM's opclasses) |

Even a *hypothetical* TurboQuant-at-1-bit would degenerate to the sign
bit but keep the f32 LUT scorer + 4 B scale — giving up BQ's whole
point (integer popcount speed, half the storage). The scheme that wins
storage/latency is sign-BQ + Hamming, exactly what the feasibility study
measured.

---

## 2. What landed (this branch)

1. **Reloption `WITH (bit_width = 1)`** (`src/index/options.rs`): the
   `bit_width` range is now `1..=4` (was `2..=4`). `bit_width = 1` with
   `graph = true` is rejected (the sign-BQ scan kernel is flat/IVF, not
   Vamana, yet). The GUC *default* stays `2..=4` — BQ is opt-in per the
   study ("never a default"; unusable on non-zero-centered data).

2. **Rerank default auto-widens for 1-bit**
   (`src/guc.rs::hi_dim_rerank_candidate_count`): gained a `bit_width`
   param. A 1-bit index is treated as `effective_dim >=
   HI_DIM_RERANK_MIN_DIM (256)` at ANY dim, so `hi_dim_rerank = auto`
   engages the wider exact-heap-rerank floor for BQ regardless of dim
   (BQ is lossy even at low dim; the study needed a rerank window of a
   few hundred). Reuses the EXISTING `xs_recheckorderby` / `search_k` /
   `oversample` machinery — no new rerank mechanism. A user override
   past the floor still wins.

3. **Pure-Rust sign-BQ core** (`src/index/onebit.rs`, fully unit-tested,
   no pgrx cluster needed):
   - `pack_signs` / `unpack_signs`: MSB-first sign packing, SAME layout
     as `bitvec.rs` / Postgres `bit` (so the SQL Hamming/popcount kernel
     and the future index scorer share one convention).
   - `corpus_mean` + `center`: **the footgun fix — mean-centering.** The
     naive sign-at-zero rule sets every bit to 1 on dense-positive data
     (GIST: R@10 = 0.0). Subtracting the per-dim corpus mean before the
     sign splits each dimension. On zero-centered text embeddings the
     mean is ~0 (centering is a near-no-op).
   - `is_degenerate`: detects the pathological all-same-sign-after-
     centering case (every code identical, Hamming uniformly 0) so the
     build can ERROR instead of shipping an all-ones landmine.
   - `codes_stride(dim) == dim/8` — exactly **half** the 2-bit stride
     `dim/8 * 2` (unit-asserted).

4. **Footgun-safe half-state** (`src/index/build.rs`): until the encode
   path is wired, a `bit_width = 1` build raises a clear ERROR
   ("not yet implemented ... see docs/ONEBIT_BQ.md") at the single
   `ambuild` choke point — NOT a panic (turbovec's `IdMapIndex::new`
   would `expect()`-panic), NOT a silent success. A 1-bit index can
   never come into existence, so `aminsert`/scan paths are unreachable
   for it. `#[pg_test] pg_index_am_onebit_errors_clearly_not_panic`
   asserts this.

---

## 3. Storage (confirmed)

Per-vector on-disk codes for a flat/IVF index:

| bits | codes/vec | + scale | note |
|---|---|---|---|
| 1 (sign-BQ) | `dim/8` | **none** | half of 2-bit |
| 2 | `dim/8 * 2` | 4 B | |
| 4 | `dim/8 * 4` | 4 B | |

1536d: 1-bit = 192 B, 2-bit = 384 B + 4. The `codes_stride` unit test
asserts `2bit == 2 * 1bit`. sign-BQ also drops the per-vector scale
(4 B), the persisted rotation matrix (`dim*dim*4` O(1)), the Lloyd-Max
codebook, and the blocked chain — so the on-disk win is slightly more
than exactly 2× at the O(1) terms. The mean vector (`dim * 4` bytes,
O(1)) is the only NEW header.

---

## 4. Wire-format impact — **YES, a bump is required** (flagged)

A 1-bit index needs a wire bump (`VERSION` 7 -> 8) because a sign-BQ
relfile is NOT byte-decodable by the current reader:

- **no scales chain, no codebook, no rotation chain** — the reader must
  know not to expect them (the meta-page chain-offset fields would be
  ambiguous otherwise).
- **a NEW mean-vector chain** (`dim` f32) the current meta page has no
  field for.
- the codes-chain stride is `dim/8` (bit_width = 1), which the existing
  `codes_stride(1, dim)` math already produces — that part is fine.

So the bump is real and NOT additive-decodable the way v4->v5->v6 were.
Consequences the integration must handle:
- bump `page::VERSION` 7 -> 8 **and** `EXPECTED_WIRE_FORMAT_VERSION` in
  `lib.rs` (the `wire_format_version_is_stable` test).
- existing v7 indexes decode byte-identical (a v8 binary reads v7 as
  before) — so **no REINDEX for existing 2/4-bit indexes**; only a
  1-bit index is new-build-only.
- add `is_legacy_v7()` if a future bump needs it (the current
  `is_legacy_v6` gate stays as-is).
- migration matrix row in `docs/UPGRADING.md`; a `migrations/NNN_*.sql`
  file (empty is fine — additive).
- **sequencing:** this is a MINOR bump. If it ships in the same release
  as another wire change, co-design the single bump; otherwise it
  rebases onto whatever `VERSION` is current.

**This branch does NOT bump the wire format** (VERSION untouched, patch-
safe) precisely because the encode path that WOULD change the wire is
not landed. The bump lands with the encode path, not before.

---

## 5. What shipped (v2.6.0) — was "remaining work"

Not landed here because it (a) is a real new scan kernel + wire path
that can't be validated end-to-end in the shared-cluster sandbox, and
(b) crosses build/scan/relfile/page/cache. The spec:

1. **Encode branch** (`build.rs`, gated `bit_width == 1`): compute
   `corpus_mean` over the (normalised) corpus, `center` each vector,
   `is_degenerate` check (ERROR with a `bit_width >= 2` hint if it
   trips), `pack_signs` into the codes chain. Bypass `IdMapIndex::new`
   entirely — no scales/rotation/codebook/blocked. This is the parallel
   analog of the existing flat/IVF encode, minus the turbovec call.
2. **Meta-page v8** (`page.rs`): a `bq: bool` (or reuse `kind`), a
   mean-vector chain (first/count/bytes), scales/rotation/codebook
   counts = 0. `relfile.rs` write/read for the BQ shape.
3. **Scan kernel** (`cache.rs`): a `ScanHandle::Bq(Arc<BqIndex>)` variant
   holding packed codes + slot_to_id + the mean. `search(query, k)` =
   center the query by the persisted mean, `pack_signs`, then
   top-k by Hamming (`popcount(q XOR code[i])`, ascending). Start
   SCALAR (`bitvec.rs::hamming_distance` is the correct reference);
   SIMD `popcount` is a follow-up (the v1.7.3-class scalar-fallback
   correctness lesson applies — test scalar first). The AM's
   `xs_recheckorderby` already reranks the top-k exactly against the
   heap — compose, don't reinvent.
4. **`#[pg_test]`** (the study's ask): build a `bit_width = 1` index over
   zero-centered synthetic data, assert R@10 >= 0.9 WITH the rerank on a
   favorable set, assert storage is ~half the `bit_width = 2` index over
   the same data (via `pg_relation_size`), and assert the all-positive
   footgun case works-via-centering-or-errors-clearly (never silent
   garbage). The pure-Rust `onebit` tests already cover center/pack/
   degenerate correctness; the pg_test covers end-to-end recall+storage.
5. Wire bump + migration + `UPGRADING.md` row (§4).

IVF `WITH (lists = N, bit_width = 1)` composes (cell-contiguous sign
codes + Hamming per-cell); the graph kind is explicitly excluded for now
(rejected in `options.rs`).

---

## 6. As-built notes (v2.6.0)

What differed from the spec above, and the bugs found wiring it:

1. **`KIND_BQ = 3`, not a version bump** (see the Status note). A BQ
   relfile has: a codes chain of `dim/8` packed sign bits per vector, an
   ids chain, and a corpus-mean chain (`dim` f32). No scales, no
   codebook, no rotation/TQ+, no blocked chain. `MetaPageData::plan_bq`
   is a **separate** constructor rather than a flag on
   `plan_with_blocked`, so a bug in the BQ layout cannot change the
   layout of any existing index.

2. **Three instances of the v1.24.0 corruption class, found and fixed.**
   `write_tombstones_and_meta`, the tombstone placement inside the
   rewrite path, and `MetaPageData::total_blocks()` each summed chain
   page counts *without* `bq_mean_count`. On a BQ index that would have
   placed the tombstone chain **on top of the mean vector** and
   under-sized the relation — the identical shape of the v1.24.0 graph
   bug (which omitted `graph_count`). Found by auditing every
   chain-offset sum in the tree, not just the path being added.

3. **Degeneracy must be checked on the CENTERED corpus.** The first
   implementation checked the raw corpus, which rejects exactly the
   dense-positive corpora this feature exists to handle (they *are*
   degenerate raw — every sign bit is 1 — and index fine after
   centering). The existing unit test
   `all_positive_is_degenerate_raw_but_centering_fixes_it` says so in its
   name. CI caught it. The guard still fires for a corpus collapsed
   *after* centering (constant / near-constant), where Hamming is
   uniformly 0 and results would be arbitrary.

4. **The mean is NOT recomputed on `aminsert`.** Recomputing it would
   invalidate every sign code already packed against the old mean, so a
   single insert would silently degrade the whole index's ranking. The
   build-time mean is treated as fixed; drift is a REINDEX concern, which
   matches the build-then-serve model the reloption's guidance already
   sets out.

5. **VACUUM is tombstone-only**, sharing the graph kind's path
   (`graph_tombstone_dead` was already kind-agnostic slot arithmetic).
   Compacting would require rewriting the whole packed codes chain and
   renumbering every slot.

6. **`turbovec_check` skips the scales validation for BQ only.** BQ has
   no scales chain, so the v2.2.2 check that closed the scan-fatal blind
   spot would otherwise report every BQ index corrupt. It reports
   `kind = 'bq'`.

7. **The Hamming kernel is deliberately scalar.** `u8::count_ones`
   lowers to `POPCNT`; a hand-vectorised version must be proven
   bit-identical against `onebit::topk_hamming` first. This is the
   v1.7.3 lesson (a mis-specialised kernel returned *wrong* ANN results
   on pre-AVX2 CPUs) applied pre-emptively. Ties break toward the lower
   slot, and since Hamming over `dim` bits has only `dim + 1` distinct
   values, ties are the common case — which is why the AM's exact rerank
   does the fine ranking and `hi_dim_rerank` treats a 1-bit index as
   high-dim at any `dim`.

### Still open

- **IVF + 1-bit** (`lists > 0` with `bit_width = 1`) — rejected with a
  clear ERROR; the cell-contiguous sign layout and per-cell Hamming scan
  are unwired.
- **Graph + 1-bit** — rejected in `options.rs` (and the graph kind is
  deprecated as of v2.5.0, so this will not be pursued).
- **SIMD popcount** — see note 7.
- **Recall at scale.** The `#[pg_test]`s prove correctness, storage and
  end-to-end scan behaviour on synthetic corpora. The published
  recall/latency frontier for BQ still needs a real-corpus run on an
  AVX2 host (per `AGENTS.md`, latency numbers may only come from
  `arnold`). **The harness for that run is built and validated but has
  NOT been run — no numbers exist yet.** See
  [`docs/BQ_RECALL_BENCH.md`](BQ_RECALL_BENCH.md) for the runbook, the
  host rules, the re-rank-window controls, and the predictions recorded
  in advance; the driver is `benches/scripts/bq/bq_frontier.py`.
