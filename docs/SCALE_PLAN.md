# Scale plan — from here to best-at-TB–PB-under-pooled-concurrency

> **Status:** DRAFT roadmap, 2026-10-15 (revised 2026-10-16 after
> `.agent/notes/SCALE_PLAN_REVIEW.md` — 2 blockers + 8 should-fixes resolved;
> blocker [B2] since RESOLVED by the S2 feasibility spike,
> `.agent/notes/S2_SPIKE_FINDINGS.md`). Grounded in the measured
> pg_vectorsearch comparison (`benches/results/vs_pgvectorsearch_20261014/FINDINGS.md`),
> the code-level bottleneck analysis (`.agent/notes/SCALE_RESEARCH_concurrency.md`,
> every claim cited to `file:line`), and the benchmark methodology
> (`benches/METHODOLOGY.md`). Not a release schedule — a direction with
> phases, each gated by the methodology's pooled-concurrency headline and the
> HARD MANDATE in `AGENTS.md` (no corruption, old formats stay readable,
> every change proven by a test that fails-before/passes-after).

## 1. The goal, measured

Be the best vector-search extension for **TB–PB indexes hit constantly by
thousands of pooled clients on a tuned host** (huge pages, large
`shared_buffers`, ZFS + ZIL/SLOG + L2ARC on NVMe), and competitive at every
smaller scale. "Best" is defined by the methodology's headline metrics:
sustained **QPS and QPS/core**, **tail latency (p95/p99/p99.9) under a
pooler**, **memory-per-backend × backend-count**, over a **working set larger
than RAM**. Measured on the three-tier distributed rig (DB on metal, pooler on
its own instance, load fleet of small spot instances).

## 2. Where we are (the honest baseline)

From the measured comparison and verified in code:

| axis | prism (pg_vectorsearch) | pg_turbovec v2.13.0 | gap |
|---|---|---|---|
| CPU/query (1M flat, 0.95) | ~1.0 CPU-ms, 1 thread | ~5.5 CPU-ms, 5 threads | ~5× CPU |
| QPS @ 96 clients (1M) | 49,641 | 2,066–2,108 | **15–24×** |
| QPS @ 96 clients (10M) | 22,294 | 194–229 | **~100×** |
| per-backend resident RAM | ~3 MB | 0.5 GB (1M) … 5 GB (10M) | **100–1000×** |
| cold first-query (fresh backend) | ~40–100 ms | 250–500 ms | ~5× |
| 10M warm p50 @ 0.95 | 2.2 ms | 63 ms | **17–29×** |
| build WAL vs index size | ~1× | ~2× | 2× |

We win: 1M at high recall with the flat index (near-exact ~2 ms), exact
filtered results, deterministic builds, Postgres-native ACID/joins, storage
efficiency vs HNSW.

**The one sentence that explains every loss:** prism serves straight from
`shared_buffers` at ~1 core and a few hundred µs/query, holding ~3 MB of
bounded per-backend state; pg_turbovec holds a **multi-GB private
per-backend decoded copy** of the index and spends **5–16 cores and
40–200 ms/query**. At pooled scale that private cache is disqualifying before
QPS even enters: 1,000 backends × 0.5–5 GB = **0.5–5 TB of duplicated RAM for
one on-disk artefact.**

## 3. Target architecture (the "there")

Three structural shifts, each turning a per-backend/per-query cost into a
per-relation/shared cost:

1. **One decoded index per relation in shared memory, mapped read-only by
   every backend** — not a private copy per backend. This is the keystone.
2. **One core per query by default** under concurrency — intra-query threads
   become an explicit low-concurrency latency knob, not the default.
3. **No per-query index construction** — the IVF path precomputes and shares
   per-cell structures; nothing that is a pure function of the index is
   rebuilt per query (rotation matrix, codebook, blocked layout, compact
   sub-indexes).

Supporting shifts: a rerank window bounded by actual `LIMIT`/recall rather
than dim; bulk-build WAL via `log_newpage` instead of per-page full-page
images.

## 4. Phased roadmap (ordered by impact for the target market)

Each phase is independently shippable, keeps old on-disk formats readable, and
is validated on the pooled rig against the phase before it. Severity ranks map
to the research doc's ranked bottlenecks (#1–#7). **Order:**
S0 → S1 → S4 → S2 → S3 → S5 → S6 — S4 runs *before* the keystone S2 because it
is the cheapest change in the plan and helps at **every** scale, while S2 is
the highest-risk phase and may need a spike before it is even known to be
feasible (see S2 [B2]).

### Phase S0 — honesty + measurement foundation (now, no code risk)

The prerequisite for all of it: we cannot tune what we cannot see.

- Publish the measured comparison honestly in README/docs (done).
- Land `benches/METHODOLOGY.md` (done) and build the **three-tier pooled
  harness** under `benches/scripts/pooled/` (DB metal + pooler instance + spot
  load fleet, open+closed loop, >RAM corpus, tuned storage). This harness is
  the gate for every later phase — a change is only "better" if it moves the
  pooled headline.
- Build a **multi-backend test harness** (two or more real backends against
  one cluster, driven out-of-process), because the HARD-MANDATE rule "every
  logic change ships with a fails-before/passes-after test" is
  **unsatisfiable by a single-backend `#[pg_test]`** for anything
  cross-backend — and S2 is entirely cross-backend. A `#[pg_test]`'s outer
  transaction rolls back before `PreCommit`, so shared-cache behaviour cannot
  be exercised in-process (AGENTS.md). This harness is a hard prerequisite for
  S2, not a detail.
- Establish the **baseline pooled numbers** for v2.13.0 on the rig so every
  phase has a before/after. (The current numbers are from a collapsed
  single-host topology; re-measure on the real rig first.)

### Phase S1 — one core per query under concurrency (bottleneck #2)

Removes the **oversubscription collapse**; a prerequisite, not a gap-closer on
its own.

- Make `scan_parallelism = 1` the effective default under concurrency —
  either document it as the pooled default and ship a tuned `postgresql.conf`
  fragment, or auto-collapse to 1 thread when the backend/active-query count
  indicates a loaded server. Keep multi-thread as an explicit low-concurrency
  latency knob (it already measures correctly at `sp=1`: `FINDINGS.md:254`).
- Remove the per-query pool `install`/futex handoff cost at `sp=1`
  (`cache::bounded`/`single_thread_pool`, `src/cache.rs:1303-1330`).
- *Honest scope:* S1 removes the collapse where ~20 concurrent queries × 5
  threads saturate 96 cores, which is real and matters at concurrency — but it
  does **not** close the single-core CPU/query gap. The equal-CPU row shows
  `sp=1` still loses **23× at 1M** (2,108 vs 49,641 TPS) and **~115× at 10M**
  (194 vs 22,294) because the per-query *fixed cost* (brute-force flat scan,
  always-full exact recheck, per-query IVF build) dominates even at one
  thread (`FINDINGS.md` P5; research §1). S3 and S4 close that gap, not S1.
- *Gate:* pooled QPS/core at high concurrency stops collapsing under
  oversubscription; the CPU/query gap is left to S3/S4.
- *Risk:* minor (default thread count changed) — same class as v2.13.0.

### Phase S4 — bound the exact rerank window by LIMIT, not dim (bottleneck #4)

Promoted ahead of the keystone: the cheapest change in the plan, and the
single biggest per-query CPU lever at `sp=1`, at **every** scale.

- Make the exact-rerank candidate count a function of actual `LIMIT`/recall
  need, not `clamp(dim, 256..=1024)`
  (`hi_dim_rerank_candidate_count`, `src/guc.rs:224-261`,
  `HI_DIM_RERANK_MIN_DIM=256` at `src/guc.rs:214`). A `LIMIT 10` query at
  1024-d should not exact-recompute ~1024 heap tuples by default:
  `hi_dim_rerank=off` + `search_k=16` already measures ~5 ms vs ~42 ms flat
  CPU (~8×) at R@10 0.998 (`p0_findings.md:332`).
- **The in-extension lever is candidate count ONLY.** A "tighter order-by
  lower bound so the reorder queue stops early" is a documented dead end:
  under `xs_recheckorderby`, core's `IndexNextWithReorder` unconditionally
  heap-fetches and exact-recomputes every tuple `amgettuple` returns *before*
  reading our advertised value — investigated and REJECTED as a no-op
  (`src/index/scan.rs:834-857`). Cutting candidate count is the only lever
  short of an upstream core patch to `IndexNextWithReorder`; that core change
  is out of scope for this phase (file it separately if pursued).
- *Release contract:* the exact kernels and scoring stay bit-identical
  (AGENTS.md; `normalise_into` untouched). Changing the *candidate count* is a
  recall/latency trade — a tunable + default change is a **minor**, not a
  scoring-bit change, so it does not break the patch bit-identity rule. It
  *will* shift which rows/order come back for some queries, so prove recall on
  the matched-recall Pareto.
- *Gate:* flat and IVF CPU/query drop ~5–8× at matched recall, at all scales.

### Phase S2 — one decoded index per relation in shared memory (bottleneck #1) — THE KEYSTONE

The structural fix that removes the 0.5–5 TB duplication and the 250–500 ms
cold build. **Feasibility [B2] is RESOLVED (spike, 2026-10-16): buildable, but
only via the turbovec API additions in S2a below — there is no fallback.**
Highest *engineering* risk in the plan; prototype behind a GUC and prove on
the rig before defaulting.

- **What is actually duplicated vs already shared** (the win is precise): the
  *raw relfile pages* already live once in `shared_buffers`; sharing them
  again buys little. What is duplicated per backend — and what S2 shares — is
  the **decoded / SIMD-blocked layout** produced by `prepare()`, plus
  `slot_to_id`, the rotation matrix, the coarse centroids, and the centroid
  graph. That re-decode + re-block *is* the 250–500 ms cold cost
  (`FINDINGS.md:214`) and the 0.5–5 GB private footprint. S2 materialises that
  decoded form **once per index relation** in DSA / a dynamic shared-memory
  segment, mapped read-only by all backends.
- **Cross-backend lifecycle (the OOM/leak class, not corruption).**
  Invalidation keyed on `(relfilenode, freshness)` as today, but freeing a
  superseded DSA allocation is now **cross-process refcounting**: a
  `REINDEX`/`VACUUM`/writer bumps the generation and backends re-map, and the
  old segment is freed only when the last reader unmaps. A single leaked ref
  pins the segment for *every* backend — worse than the historical private
  leaked-`Arc` OOM (AGENTS.md, ~80 min to OOM). S2's gate therefore includes a
  sustained concurrent-write **soak watching backend RSS *and* segment
  occupancy**, not just `turbovec_check`.
- **Corruption posture (HARD MANDATE).** The shared image is **read-only** to
  scanners; the raw relfile stays the source of truth; a map failure falls
  back to the private path, never to a wrong answer. The mutable (`aminsert`)
  arm stays per-backend until S5 — so during the S2 window a **write-heavy
  index gets no sharing benefit** and the memory wall is lifted only for
  read-mostly indexes; state that limit in the gate.
- **NUMA + triple-caching (metal-host first-order concerns).** On a
  multi-socket metal box a single segment pinned to one NUMA node makes every
  cross-node backend pay remote-memory latency per scan — scope first-touch /
  interleave / per-node replication of the shared image. And the raw bytes can
  now live in **three** places — ZFS ARC, `shared_buffers`, and the shared
  decoded segment — so size the shared image against the memory budget the
  whole exercise is trying to shrink; do not let triple-caching eat the win.
- *Gate:* per-backend resident RAM drops from GB to ~MB; cold first-query on a
  fresh pooled backend drops from 250–500 ms toward prism's ~tens of ms; a
  thousand-backend pool fits on the host; the write-soak shows bounded segment
  occupancy. Biggest lever on the "memory-per-backend × backend-count"
  headline — for read-mostly indexes.
- *Risk:* major engineering, gated by the spike; prototype behind a GUC
  (`turbovec.shared_cache = off` default) and prove it on the rig before
  defaulting. Likely a minor when first shipped off-by-default.

### Phase S3 — kill per-query index construction on the IVF path (bottleneck #3)

The fix for the 17–29× 10M loss. **Depends on S2 (or its fallback):** the
per-cell sub-indexes must live in the shared image, or they are just another
private cache.

- Cache the per-cell compact sub-indexes (or at least their blocked layout +
  the shared codebook/rotation) across queries, keyed by cell id, so a probed
  cell is built once per relation (in the S2 shared image), not once per chunk
  per query (`search_compact_chunk` throwaway `from_parts`+`prepare()`,
  `src/cache.rs:1135/1167`). The codebook/rotation are a pure function of
  `(bit_width, dim)` and identical across queries — they must not be rebuilt.
- Longer horizon: evaluate a **posting-list + precomputed-code** IVF layout
  (prism-style). This reorganises the *physical page layout of the codes*
  (contiguous posting lists vs today's cell-range gather) — a genuine wire
  change, **not** a free `kind`-byte add like `KIND_BQ` (which selected a
  decode path over the *same* bytes). Per the HARD MANDATE it needs
  transparent old-format decode **or** an offline in-place page converter
  (a corpus REINDEX is banned); plan it as "**new `kind` + offline converter**,
  not additive," so nobody scopes it as free and discovers otherwise.
- *Gate:* 10M IVF CPU/query and p50 fall toward prism's; the "our flat beats
  our own IVF" inversion (`FINDINGS.md:155-157`) disappears.

### Phase S5 — write path for write-heavy TB-scale (bottleneck #6, then #1-writer)

- **[B1] The `log_newpage` route is BLOCKED by the release gate.**
  drift-check §11a (`scripts/drift-check.sh:289`) hard-`fail`s on a grep that
  lists `log_newpage`/`log_newpage_buffer` *by name*; releases require
  drift-check to pass, so a `log_newpage` call site cannot ship as-is. The ban
  exists for a real portability reason (the comment: "some managed variants
  route custom resource-manager / raw full-page-image records differently").
  **Before S5 is actionable, resolve BOTH:** (a) prove `log_newpage` replays
  correctly on the managed variants §11a protects, then amend §11a with that
  argument answered — **or** (b) stay inside `GenericXLog` and find a
  delta-logging approach (e.g. avoid `GENERIC_XLOG_FULL_IMAGE` where a page
  genuinely changed only in part). Until one is done, S5 is design-only.
- *Goal unchanged:* cut build WAL from ~2× index size (per-page
  `GenericXLog FULL_IMAGE`, 10 sites in `src/index/relfile.rs`) toward ~1×,
  halving ZIL pressure / replication bandwidth. Keep the existing
  skip-unchanged delta for incremental flushes (`write_chain_at`,
  `src/index/relfile.rs:544-612`).
- Revisit the writer-side cache once S2 lands (shared read image + per-backend
  write buffer, flushed to the shared image on commit).
- *Must respect:* durability stays 100% crash-safe; reconcile the §11a
  invariant text and its managed-PG portability intent before touching WAL.

### Phase S6 — residual contention + polish (bottleneck #7)

- Once S3 removes the per-chunk `codebook` call, the process-wide `MEMO` mutex
  (`~/src/turbovec/.../codebook.rs:34`) leaves the hot path. Shard or
  lock-free-read the `CACHE` map (`src/cache.rs:1585`) only if it profiles hot
  on the pooled rig (it is µs-bounded today).
- Reduce the CBOR heap-datum decode cost (5127 B/row vs pgvector 4104 B,
  `FINDINGS.md:365`) — ties into the long-planned raw-vector-varlena work
  (`design/raw-vector-varlena`, already renumbered; reconcile with this plan).

### Phase S7 — sparse bitmaps for tombstones + the IVF probe mask (independent)

pg_turbovec rolls its own **dense** bitmaps in two places that are sparse in
practice and grow with the corpus — so at TB–PB scale they waste on-disk/WAL
bytes and per-query RAM. `~/ws/sparsemap` (our C fork, v5.8.3) is a
compressed sparse bitmap built for exactly this (long runs of set/unset
bits). This phase is **independent of the S2 keystone chain** and can run in
parallel.

- **On-disk tombstone bitmap.** `relfile::read/write_tombstones` +
  `src/index/vacuum.rs:358-371,405-466` persist a dense `Vec<u8>` of
  `n_vectors/8` bytes — one bit per slot, written to the relfile even when few
  slots are dead. At billions of slots this is large and almost all zeros.
  Replacing it with a sparsemap-encoded chunk shrinks the persisted tombstone
  region and its WAL. **Wire-format impact:** the tombstone chain layout
  changes, so this is additive-decode (read old dense, write new sparse) or a
  new `kind`/version per the HARD MANDATE — scope it like any format change,
  with a fails-before/passes-after corruption test and a sustained-load soak.
- **Per-query IVF probe mask.** `ivf.rs:195` `probe_mask() -> Vec<bool>`
  allocates one byte per vector over the whole corpus per query (the in-kernel
  allowlist/cell mask). A sparse representation cuts per-query allocation and
  cache footprint — overlaps the per-query-cost work in S3/S4. In-memory only,
  no wire impact.
- **Prerequisite / scope:** confirm whether we want a Rust binding to the C
  library (FFI/`cc` build, like the turbovec kernel) or a pure-Rust
  equivalent; measure the compressed size + encode/decode cost against the
  dense baseline on a realistically-sparse tombstone set BEFORE adopting —
  sparsemap wins only when the bitmap is actually sparse, so a densely-churned
  index must not regress. **sparsemap currency:** pg_turbovec does **not**
  depend on sparsemap today (no cargo dep, no vendored copy); `~/ws/sparsemap`
  is at v5.8.3. If/when adopted, pin the version and track it.

## 5. Sequencing and why

- **S0 first, always** — no measurement rig and no multi-backend test harness,
  no credible claim of "better" and no way to test S2 at all.
- **S1 then S4** — both ship early as minors with no format change. S1 removes
  the oversubscription collapse (necessary, not sufficient); S4 is the
  cheapest change and the biggest single-core CPU/query lever, at every scale.
  Neither alone closes the gap to prism, and the plan says so plainly (S1
  honest-scope note).
- **S4 before S2** — S4 is cheap, broad, and low-risk; S2 is the highest
  *engineering* effort (its feasibility is now resolved — see [B2] — but it
  needs the S2a turbovec API work first).
- **S2 is the keystone; [B2] feasibility now RESOLVED** (spike 2026-10-16) —
  it is the only fix for the memory-per-backend wall. It is **feasible only
  via turbovec API additions** (S2a: `serialize_prepared` + a borrowed-view
  `search_from_prepared`), which is additive fork work we own; the
  small-structures fallback is a **dead end** (saves ~0) and has been dropped,
  so S2 has a single route. S3/S5 still depend on S2b (the shared image)
  landing. The remaining S2 risk is engineering effort + the cross-backend
  lifecycle/NUMA items, not whether it can be built.
- **S3 then S5** turn the 10M and write-heavy stories around; both depend on
  S2 (or its fallback). S5 is design-only until the [B1] gate conflict with
  drift-check §11a is resolved.
- **S6 is cleanup** gated on profiles, not speculation.
- **S7 (sparse bitmaps) is independent** of the S2 keystone chain and can run
  in parallel with any phase; sequence it by appetite, not dependency. Its
  tombstone half is a wire-format change (additive-decode), so it carries the
  same corruption-test + soak bar as S2b/S3.

## 6. Non-negotiables threaded through every phase

- **No corruption, ever; old formats stay readable; wire bumps are additive
  via a new `kind` byte where possible** (AGENTS.md HARD MANDATE + the v4→v8
  precedent). An offline in-place converter if a break is truly unavoidable;
  never a corpus rebuild.
- **Every logic change ships with a test that fails before / passes after**,
  and persist/format changes get sustained-load validation on the pooled rig,
  not just `turbovec_check`.
- **Bit-identical scoring across patches** stays true; candidate-count and
  thread-count changes are minors, scoring-bit changes are not.
- **Every claim measured on the three-tier rig** per `benches/METHODOLOGY.md`;
  single-host numbers are diagnostics.

## 7. Open questions to resolve before committing engineering

- **[B2 — RESOLVED]** Spike (2026-10-16, `.agent/notes/S2_SPIKE_FINDINGS.md`)
  proved turbovec serialized bytes are position-independent and
  foreign-read-only-mappable, but the decoded index has no borrowed-view
  constructor — so S2 needs turbovec API additions (`serialize_prepared` +
  `search_from_prepared`, phase S2a), which is additive fork work we own. pgrx
  exposes the full DSA/DSM/shmem surface. The small-structures fallback saves
  ~0 and is dropped. No open question remains on *whether* S2 is buildable;
  the open items are the lifecycle/NUMA ones below.
- **[B1, gates S5]** Does `log_newpage` replay correctly on the managed-PG
  variants drift-check §11a protects? If yes, amend §11a with the argument
  answered; if no, stay inside `GenericXLog` with a partial-page delta.
- DSA vs a fixed shared segment for the decoded image; cross-backend eviction
  policy for a shared cache (S2) when multiple large indexes exceed the
  segment — load-bearing for the many-large-indexes target market.
- NUMA placement of the shared segment on a multi-socket metal host
  (first-touch / interleave / per-node replication), and the ARC +
  `shared_buffers` + shared-segment triple-caching memory budget.
- Whether a posting-list IVF (S3 long horizon) is a new `kind` + offline
  converter (most likely) vs an additive `kind`-byte add (unlikely, since it
  reorganises physical page layout).
- The `assign_dups > 1` soft-assignment read-only case (AGENTS.md): a shared
  image must not worsen or mask the non-bijective `slot_to_id` load failure.
- Whether S1's "collapse to 1 thread under load" should be automatic (needs a
  cheap concurrency signal) or operator-set.
- Reconciling S6/CBOR with the `design/raw-vector-varlena` renumbering.
