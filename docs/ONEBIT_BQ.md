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
   correctness lesson applies — test scalar first). [Resolved: §7 — the
   follow-up landed as a CPU-independent wide-word kernel; AVX2 was
   measured and declined.] The AM's
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

7. **The Hamming kernel is wide-word, and deliberately NOT SIMD.**
   `hamming` folds 8 bytes at a time through `u64::count_ones` (one
   `POPCNT` per 8 bytes instead of per byte) with a byte-wise tail. That
   tail loop IS the original scalar kernel and is the only path below
   `dim = 64`, so it stays live and covered. There is **no**
   `is_x86_feature_detected!`, no `target_feature`, no `unsafe`, and no
   runtime dispatch anywhere in the module: every machine executes the
   same instruction sequence over the same word decomposition, so this
   cannot become a second v1.7.3 (where a mis-specialised kernel returned
   *wrong* ANN results on pre-AVX2 CPUs). An AVX2 variant WAS written,
   proven bit-identical and benchmarked; it was declined — the numbers are
   in §7 below. Ties break toward the lower slot, and since Hamming over
   `dim` bits has only `dim + 1` distinct values, ties are the common case
   — which is why the AM's exact rerank does the fine ranking and
   `hi_dim_rerank` treats a 1-bit index as high-dim at any `dim`.

### Still open

- **IVF + 1-bit** (`lists > 0` with `bit_width = 1`) — rejected with a
  clear ERROR; the cell-contiguous sign layout and per-cell Hamming scan
  are unwired.
- **Graph + 1-bit** — rejected in `options.rs` (and the graph kind is
  deprecated as of v2.5.0, so this will not be pursued).
- **Recall at scale.** The `#[pg_test]`s prove correctness, storage and
  end-to-end scan behaviour on synthetic corpora. The published
  recall/latency frontier for BQ still needs a real-corpus run on an
  AVX2 host (per `AGENTS.md`, latency numbers may only come from
  `arnold`). **The harness for that run is built and validated but has
  NOT been run — no numbers exist yet.** See
  [`docs/BQ_RECALL_BENCH.md`](BQ_RECALL_BENCH.md) for the runbook, the
  host rules, the re-rank-window controls, and the predictions recorded
  in advance; the driver is `benches/scripts/bq/bq_frontier.py`.

---

## 7. Hamming kernel: what was measured, and why AVX2 was declined

"SIMD popcount" was listed as open under §6 note 7. It was investigated.
**Outcome: shipped the safe wide-word path (measured 4.4–4.8× at
embedding dims), declined the AVX2 intrinsics path (only ~1.8× further,
and *slower* than wide-word below `dim = 512`).**

### 7.1 What was tried

Four kernels, all producing `popcount(a XOR b)`:

| kernel | how | `unsafe` | CPU-dependent |
|---|---|---|---|
| `bytewise` (was shipped) | `u8::count_ones` per byte | no | no |
| **`u64` (now shipped)** | `u64::count_ones` per 8 bytes + byte tail | no | **no** |
| `u64x4` | as above, 4 independent accumulators, 32 B/iter | no | no |
| `avx2` | `pshufb` nibble-LUT + `vpsadbw`, 32 B/iter | yes | **yes** |

### 7.2 Latency — the shipped change vs what it replaced

The **exact in-tree `topk_hamming`** against the verbatim pre-change
byte-wise kernel. Median of 5–7 reps; agreement asserted on every timed
fixture. Host: Intel Core Ultra 7 258V (Lunar Lake, AVX2, **no AVX-512**),
random packed codes, `k` = the two ends of the BQ rerank window.

| n | dim | stride | k | old (byte-wise) | new (`u64`) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 100k | 128 | 16 B | 10 | 1.15 ms | 0.52 ms | **2.23×** |
| 100k | 128 | 16 B | 800 | 1.53 ms | 0.90 ms | **1.71×** |
| 100k | 768 | 96 B | 10 | 6.13 ms | 1.28 ms | **4.81×** |
| 100k | 768 | 96 B | 800 | 6.57 ms | 1.48 ms | **4.46×** |
| 100k | 1536 | 192 B | 10 | 11.94 ms | 2.67 ms | **4.48×** |
| 100k | 1536 | 192 B | 800 | 12.06 ms | 3.21 ms | **3.76×** |
| 1M | 768 | 96 B | 10 | 59.04 ms | 13.66 ms | **4.32×** |
| 1M | 768 | 96 B | 800 | 61.01 ms | 13.88 ms | **4.39×** |
| 4M | 768 | 96 B | 10 | 237.9 ms | 54.0 ms | **4.40×** |
| 4M | 768 | 96 B | 800 | 249.7 ms | 57.4 ms | **4.35×** |

The speedup is stable from 9 MiB (L3-resident) to 366 MiB (firmly DRAM),
so the scan is **not** memory-bandwidth-bound at these sizes: the
byte-wise kernel ran at ~1.4 GiB/s, the wide-word one at ~6.2 GiB/s and
AVX2 at ~11 GiB/s, all far below this host's DRAM bandwidth.

The win is smallest at low `dim` (`dim = 128` is 16 B/vector, only two
words) and largest once several words fit per row, which is where BQ is
actually used — BQ exists for 768/1024/1536-d embeddings.

### 7.3 Why AVX2 was declined

AVX2 vs the shipped wide-word kernel, same host, `n` = 200k, `k` = 10,
full `topk` including the heap, median of 9 reps:

| dim | stride | byte-wise | `u64` (shipped) | AVX2 | `u64` vs byte | AVX2 vs byte | **AVX2 vs `u64`** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4 B | 1.17 ms | 1.27 ms | 1.73 ms | 0.92× | 0.68× | **0.72×** |
| 64 | 8 B | 1.44 ms | 0.80 ms | 1.38 ms | 1.81× | 1.05× | **0.58×** |
| 128 | 16 B | 2.75 ms | 1.27 ms | 1.82 ms | 2.16× | 1.51× | **0.70×** |
| 256 | 32 B | 5.08 ms | 1.17 ms | 1.24 ms | 4.33× | 4.08× | **0.94×** |
| 384 | 48 B | 7.51 ms | 1.93 ms | 2.10 ms | 3.90× | 3.58× | **0.92×** |
| 512 | 64 B | 9.93 ms | 2.42 ms | 1.69 ms | 4.10× | 5.88× | **1.43×** |
| 768 | 96 B | 14.99 ms | 3.30 ms | 1.88 ms | 4.54× | 7.96× | **1.75×** |
| 1024 | 128 B | 22.35 ms | 4.52 ms | 2.50 ms | 4.94× | 8.95× | **1.81×** |
| 1536 | 192 B | 31.47 ms | 6.79 ms | 3.78 ms | 4.63× | 8.32× | **1.80×** |
| 3072 | 384 B | 64.74 ms | 13.63 ms | 6.73 ms | 4.75× | 9.63× | **1.81×** |

Read the last column: **AVX2 is a net LOSS below `dim = 512`** (the
32-byte main loop never runs at `dim <= 256`, so short rows pay the
dispatch and setup for nothing) and worth at most ~1.8× above it. In log
terms the wide-word change captures **~73% of the total available
reduction** (byte-wise → AVX2) at `dim = 768`, and 69–80% across
`dim` 512–3072 — for zero `unsafe` and zero CPU-dependent behaviour.

Against that ~1.8× on long rows, the AVX2 path costs:

- a second `unsafe` block on the scan hot path;
- a `dim`-dependent dispatch threshold, i.e. a second axis of
  CPU/shape-dependent divergence in a project that already needs a CI
  `layout` matrix axis because turbovec's two scoring layouts diverged
  enough to flip near-ties;
- a path CI **cannot** exercise both sides of. GitHub runners are AVX2,
  `is_x86_feature_detected!` is a runtime check, and `-C
  target-feature=-avx2` would not force the fallback at runtime — the
  same blind spot documented in `docs/CI.md` that let the v1.7.3 bug
  ship.

The shipped kernel has none of those properties: one code path, every
CPU, every arch.

Also measured and rejected: `u64x4` (four independent accumulators) is
**slower** than the plain single-accumulator `u64` loop below `dim =
1024` (distance-only, n=100k: 2.02× vs 2.66× over byte-wise at
`dim = 768`; it only pulls ahead at `dim = 1536`, 3.46× vs 2.89×) — LLVM
already extracts the ILP, and the manual unroll mostly adds a tail. A
deferred-SAD AVX2 variant (reduce `vpsadbw` once every 31 iterations
instead of every iteration) was also written and proven bit-identical;
it matched plain AVX2 within noise, because at these strides the
reduction is not the bottleneck.

### 7.4 The bit-identity proof

The change is a word-size refactor, but "obviously a pure refactor" is
exactly what was believed about the kernel that shipped wrong ANN
results in v1.7.3. So it is proven, in-tree, as plain `#[test]`s (no
cluster — they run under both CI matrix lanes and under a bare `cargo
test --lib`):

- `hamming_agrees_with_bitwise_reference_across_dims` — **5720 random
  code pairs across 143 dims** (every `dim` in `1..=130`, so every
  `dim % 8` and `dim % 64` residue and every sub-word length, plus 255,
  256, 257, 383, 384, 511, 512, 768, 960, 1000, 1024, 1536, 3072)
  against a bit-by-bit MSB-first reference that shares no code and no
  word decomposition with the kernel. Also asserts symmetry and
  zero self-distance on every fixture, and cross-checks the old
  byte-wise kernel against the same reference.
- `hamming_extremes_agree_at_every_word_boundary` — all-zero vs all-ones
  is exactly `dim` at 15 dims straddling byte and word boundaries
  (7/8/9, 63/64/65, 71/72, 127/128/129), the case random fixtures never
  draw.
- `hamming_counts_sign_disagreements_on_real_vectors` — ties the kernel
  to the *semantics*: pack real f32 vectors, assert the packed-code
  Hamming equals a direct count of sign disagreements on the f32s.
- `topk_hamming_agrees_with_bitwise_brute_force_including_ties` — **825
  top-k cases** (11 dims × 5 corpus sizes × 3 query kinds × 5 `k`)
  asserting the full `(distance, slot)` sequence, tie order included,
  equals a brute-force sort scored by the independent reference. Query
  kinds include the all-zero code, which maximally saturates ties.
- `topk_fixtures_really_are_tie_dense` — guards the above against
  passing vacuously by asserting the fixtures genuinely collide
  (≤ `dim + 1` distinct distances over 333 rows).
- `topk_tie_break_prefers_the_lower_slot` — the tie-break contract in
  isolation, on an all-identical corpus where every slot ties.

**Mutation-tested** (each mutation applied to a copy of the module and
the suite re-run): dropping the byte tail, `from_be_bytes` on one
operand only (endianness divergence), `AND` for `XOR`, silently skipping
the last word, and `count_zeros` for `count_ones` each fail 3–6 tests.
Flipping the top-k tie-break direction, or removing the tie clause while
reversing the visit order, each fail exactly the three top-k tests. The
AVX2 and `u64x4` candidates were held to the same bar in the scratch
harness and also passed — they were declined on cost/benefit, **not**
because they disagreed.

One substantive finding en route: under the current ascending visit
order the `d == worst && slot < worst_slot` half of the heap condition
**never fires** (0 firings in 4.2M evaluations over tie-saturated
corpora) — a tie is already resolved by arriving later. It is kept, and
now documented, because it makes the tie-break a property of the
*comparison* rather than of the loop order: `topk_tie_break_prefers_the_
lower_slot` still passes with the loop reversed, and fails if the clause
or the order is broken alone. A future chunked or parallel BQ scan can
therefore reorder safely.

### 7.5 What was NOT verified

- **No `#[pg_test]` / `cargo pgrx test` run.** Sibling agents held the
  shared pgrx cluster (one fixed port, one data dir — see `AGENTS.md`).
  Verified instead: `cargo check --no-default-features --features "pg18
  pg_test"` clean in-tree, `cargo fmt --check` clean, and the module
  extracted verbatim into a dependency-free scratch crate where all 19
  `onebit` tests pass. The kernel is pure and has no Postgres surface,
  and `BqIndex::search` is unchanged, so the `#[pg_test]` risk is the
  usual full-suite regression check, not kernel correctness.
- **Single host, single microarchitecture.** Numbers are from one Lunar
  Lake laptop CPU (AVX2, no AVX-512). Not re-measured on `arnold`
  (the project's AVX2 latency host), `meh` (pre-AVX2), or `rv`
  (riscv64). The wide-word kernel has no CPU-feature dispatch, so
  *correctness* is architecture-independent by construction; the
  *ratios* are not, and per `AGENTS.md` a published latency claim must
  come from `arnold`. Cross-checked one way: forcing `-C
  target-feature=-popcnt` (so `count_ones` lowers to SWAR rather than
  `POPCNT`, a proxy for the scalar-path hosts) still gives 2.40× / 4.62× /
  4.69× at dim 128/768/1536, against 2.69× / 4.51× / 4.22× with `POPCNT`
  enabled — the win comes from doing 1/8 as many count operations, not
  from the `POPCNT` instruction.
- **No end-to-end query latency.** These are kernel microbenchmarks. A
  real BQ `ORDER BY` also pays planning, buffer reads and the exact
  rerank, so the end-to-end improvement will be smaller than 4.4× by
  whatever fraction of query time the Hamming scan represents. That
  fraction was not measured.
- **Big-endian.** Argued from the algebra (popcount of XOR is invariant
  under any shared byte permutation) and asserted by the bitwise
  reference tests, but no big-endian machine was available to run them
  on.
