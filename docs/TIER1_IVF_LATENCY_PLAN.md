# Tier 1 — IVF latency: close the gap to pgvector HNSW (scan-path only)

_Status: PLAN. Pressure-tested against the v1.17.1 code + the A-2
frontier data. **The original premise (wasted whole-index block
iteration) was REFUTED by the data** — see § 0. The real lever is the
reorder-recheck floor. No wire change, no graph index (that's Tier 2)._

---

## 0. The premise we started with, and why it was wrong

**Original theory:** turbovec's SIMD kernel does `for b in 0..n_blocks
{ if !block_has_allowed(mask) { continue } }` — iterating ALL blocks of
the whole index and skipping masked ones, so an IVF scan touching
~probes/lists of cells wastes 85–99% of iterations + cache-walks the
whole codes buffer. Fix: scan only the probed cell ranges.

**What the data actually shows (pressure-test, 2026-06):**
1. The 500k×1024×4-bit bench codes buffer is **exactly 256 MB** =
   `turbovec.cache_size_mb` default, which is **> `0.5 ×
   cache_size_mb`** (`AUTO_OOC_FRACTION`, `src/guc.rs`). So
   `out_of_core=auto` (the default) already routes it to the **OOC
   compact-gather path** (`OocIvfIndex::search_ooc`, `src/cache.rs`),
   which gathers ONLY the probed cells and scans a compact sub-index —
   it never iterates the whole index. The wasteful whole-index masked
   walk (`search_masked`, `src/cache.rs:213`) is only hit by
   *sub-threshold* in-RAM indexes.
2. Latency is **probes-independent**: probes=1 (R@10=0.42) = 19.9 ms;
   probes=64 (R@10=0.96) = 18.5 ms. Flat across a 64× change in
   probed cells. **The fine-scan is NOT the bottleneck** (the v1.4.0
   warm-scan profile: ~37% ReadBuffer/pread/page-fault, ~35% memmove,
   only ~22% in the SIMD kernel).

**The actual floor** is in `amgettuple` (`src/index/scan.rs` ~636):
every candidate is advertised with `xs_orderbyvals =
f64::NEG_INFINITY` and `xs_recheckorderby = true`, which forces ALL
`search_k`=200 candidates onto the executor's reorder queue → ~200
random **heap-tuple fetches + 200 exact 1024-d fp32 distance
recomputes per query**, just to emit LIMIT 10. That cost is paid
identically at probes=1 and probes=64 — it IS the probes-independent
~17 ms floor.

**Lesson:** the IVF block-iteration optimization (a turbovec fork API)
would optimize a path the benchmark never runs. Don't build it first.

---

## 1. The corrected lever: cut the reorder-recheck floor

`amgettuple` advertises a worthless order key (`NEG_INFINITY`) for
every candidate, so the executor cannot trust ANY of our ordering and
re-rechecks all `search_k` of them against the heap. Two sub-levers:

- **(1a) Lower `search_k`.** At R@10=0.96 / LIMIT 10 we almost
  certainly don't need 200 candidates through recheck. Sweep
  `search_k ∈ {25, 50, 100, 200}` at fixed probes; find the minimum
  that holds R@10 ≥ 0.96. Fewer candidates → proportionally fewer heap
  fetches + exact recomputes. **Pure GUC, no code.**
- **(1b) Advertise the REAL quantized distance** instead of
  `NEG_INFINITY`, so the executor only re-queues tuples that are
  genuinely out of order, cutting heap fetches from ~`search_k` toward
  ~`LIMIT`. **REJECTED (investigated 2026-06): a no-op.** Under
  `xs_recheckorderby = true`, PostgreSQL's `IndexNextWithReorder`
  UNCONDITIONALLY fetches the heap tuple and recomputes the exact
  distance for EVERY candidate `amgettuple` returns, *before* it reads
  our advertised value (the advertised value only governs the
  wrong-order ERROR + final drain ordering). So a tighter lower bound
  is legal but reduces zero heap fetches / recomputes. Behaviour is
  identical across PG 13-18. The ONLY lever that cuts the recheck
  floor is returning fewer candidates (= 1a). The AM-side
  "recompute-exactly-and-set-recheck=false" alternative duplicates the
  heap I/O and is an MVCC-correctness minefield — also rejected.
  Documented in `src/index/scan.rs` so it isn't re-asked.

Expected: the recheck is ~200 random heap fetches + 200 exact
recomputes; the profile attributes ~37% to ReadBuffer. Cutting it to
~10–20 candidates plausibly removes **5–10 ms** of the floor →
pg_turbovec **~10–13 ms at R@0.96**, matching HNSW ef40–ef100
(10–33 ms) and within ~1.0–1.3× of ef200's best.

---

## 2. Ranked Tier-1 backlog (payoff/effort)

| # | Item | Effort | Payoff | Wire | Fork | Det-safe |
|---|---|---|---|---|---|---|
| 1 | **Reduce reorder-recheck cost** (1a lower search_k; 1b advertise real quantized distance so only out-of-order tuples re-queue) | S (1a) / M (1b) | **High** — attacks the ~17 ms floor directly | N | N | Y (recheck still corrects ranking) |
| 2 | **`assign_dups=2` Pareto sweep** — hit 0.96 at fewer probed vecs → smaller scan + cache-resident working set (the soft-assign machinery already exists) | S | Med (helps once #1 lifts the floor; shrinks 23 MB working set toward L3) | **Y (opt-in REINDEX)** | N | Y |
| 3 | **SIMD `coarse_probe` + `rotate_query`** (today scalar: ~2.5 M FLOPs/query fixed floor) | S | Low-Med | N | N | Y (must bit-match scalar tie-break) |
| 4 | **Compact-gather for the whole-load path** (replace the `search_masked` whole-index masked walk for sub-threshold in-RAM indexes) | M | Low-Med (zero effect on the 500k bench; helps small/in-RAM indexes) | N | N | Y |
| 5 | **`search_with_cell_ranges` turbovec fork API** (scan only probed contiguous block ranges in place) | M-L | Low (whole-load path only; mostly superseded by #4) | N | **Y** | Y |
| 6 | `madvise`/prefetch on gather; early-termination | M | Low (downstream of #1) | N | N | Y |

## 2a. Resolution (worked 2026-06-18)

| # | Status | Why |
|---|---|---|
| **1a** lower `search_k` default 100→32 | ✅ **DONE** | recall@10 plateaus by ~25 (frontier); `recall_floor_{2,3,4}bit` pass at 32. ~3× less recheck, zero recall loss. Latency confirmation deferred to a quiet AVX2 host. |
| **1b** advertise real/tighter distance | ❌ **REJECTED (no-op)** | `IndexNextWithReorder` rechecks (heap fetch + exact recompute) EVERY returned candidate unconditionally under `xs_recheckorderby`; the advertised bound only governs the wrong-order ERROR + drain order. Identical PG 13–18. Documented in scan.rs. |
| **2** `assign_dups`×probes Pareto | ✅ **DONE** | `assign_dups_probes_pareto` test: min-probes-for-target is monotone non-increasing in assign_dups — higher dups reaches matched recall at ≤ probes (fewer cells scanned). Opt-in (REINDEX), already-existing build machinery. |
| **3** SIMD `coarse_probe`/`rotate_query` | ⚠️ **ASSESSED, DEFERRED** | `coarse_probe` is a fixed-floor term, NOT the dominant cost (the recheck floor is). A SIMD horizontal-sum changes reduction order vs scalar → could flip the `(dist, cell_id)` argmin near ties → cross-ISA recall drift, breaking the determinism + recall-stability invariant. Poor risk/reward; revisit only if a profile shows `sq_dist` hot AND with a proven bit-identical SIMD `sq_dist`. |
| **4** compact-gather for whole-load path | ⏸️ **NOT WARRANTED BY DATA** | Zero effect on the measured (OOC-gathered) bench; helps only sub-threshold in-RAM indexes. Build when a profiled workload shows the whole-load masked path is hot. |
| **5** `search_with_cell_ranges` fork API | ⏸️ **NOT WARRANTED** | Whole-load path only; mostly superseded by #4; needs a turbovec fork change. Defer until #4 is shown insufficient. |
| **6** prefetch / early-term | ⏸️ **DOWNSTREAM OF #1** | Low payoff while the recheck floor dominates. |

**Net Tier-1 deliverable:** the two evidence-backed wins (#1a lower
`search_k`, #2 the `assign_dups` Pareto frontier) ship; #1b and #3 are
rejected/deferred with source-level + determinism reasons rather than
built speculatively; #4–6 are documented as not-warranted-by-the-data
(building them would optimize paths the profile shows aren't hot). The
discipline: don't write speculative optimization code for cold paths.

---

## 3. The first step: PROFILE, don't build

The plan's original primary lever was refuted by data, so the cheapest
de-risking move is to **measure where v1.17.1's 18 ms actually goes**
before writing optimization code. One afternoon, no code:

1. Re-run `benches/scripts/profile_warm_scan.sh` against the current
   v1.17.1 binary at probes=64 / search_k=200. Confirm the split:
   ReadBuffer/recheck-heap-fetch vs `coarse_probe` vs the compact
   `search`/`repack` vs gather memmove.
2. **In the same session, sweep `search_k ∈ {25,50,100,200}` at fixed
   probes=64; record R@10 + p50.** If p50 drops materially as search_k
   falls while R@10 holds ≥ 0.96, **item #1a is the win and #1b is
   worth the careful work.** If it doesn't move, the floor is heap I/O
   we can't avoid via candidate count and we reassess.
3. Read `turbovec::search::BLOCKS_SKIPPED_BY_MASK` to confirm whether
   the whole-load masked path is even reached at 500k (it should not
   be — confirming #4/#5 are not the bench's bottleneck).

This must run on an **AVX2 host** (arnold/floki) for meaningful
latency — `meh` is pre-AVX2 and gives meaningless numbers (the
documented trap).

---

## 4. Honest ceiling

At 500k/R@0.96, HNSW ef200 = 8–17 ms; pg_turbovec = 18.5 ms — already
within 1.1–2.3×. The gap is **not** the vector scan (~22%, OOC-bounded)
— it's the per-query reorder-recheck floor that our `NEG_INFINITY`
strategy maximizes. **Tier-1 (item #1) can realistically MATCH HNSW at
the 0.95–0.97 operating point up to ~1M rows** (~10–13 ms projected).
Above a few million rows the IVF `O(probes·cell·dim)` term reasserts
and the asymptotic gap to HNSW's `O(log N·ef)` reopens — that's the
Tier-2 (quantized graph) regime, deliberately out of scope. Tier-1's
honest claim is "HNSW-class latency at the common operating point and
scale, at 7–15× less storage, exact recall available, out-of-core" —
NOT "beats HNSW asymptotically."

_Note: Cargo.lock pins turbovec rev `d3d468e` (the prompt's `d72c29c`
was a stale checkout); the `for b in 0..n_blocks` kernels are at
`search.rs:188/422/1600` in `d3d468e`._
