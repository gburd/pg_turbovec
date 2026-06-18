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
  ~`LIMIT`. **The catch (must respect):** the advertised distance must
  never be *below* the exact value, or the `elog(ERROR, "index
  returned tuples in wrong order")` guard (`scan.rs:636`) fires. The
  quantized distance is a lossy *approximation* — it can be above or
  below exact. So (1b) needs either a proven lower-bound on the
  quantized distance, or a safety margin, or it stays as a candidate
  only if we can guarantee the monotonicity the executor requires.
  **This is the substance of the lever and the one place it needs real
  care, not a GUC flip.**

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
