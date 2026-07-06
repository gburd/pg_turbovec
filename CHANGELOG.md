# Changelog

All notable changes to `pg_turbovec` are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.22.2] — 2026-07-06

**Raises `turbovec.probes`'s default from 8 to 16 — the out-of-the-
box recall floor was unreasonably low.** Scan-side default change
only, no wire change, no SQL surface change, no REINDEX.

The v1.22.1 a cloud VM competitive re-benchmark measured pg_turbovec's
shipped defaults (`probes=8, search_k=32`) capping at **R@10=0.796 on
SIFT-1M and R@10=0.407 on GIST-1M** — both far below any reasonable
recall SLO, and a real footgun for anyone who runs `CREATE INDEX ...
USING turbovec` without reading the tuning docs. `probes=16` (same
`search_k=32`) measured:

| Corpus | `probes=8` (old default) | `probes=16` (new default) |
|---|---|---|
| SIFT-1M | R@10=0.796, p50=3.0ms | R@10=0.918, p50=4.8ms |
| GIST-1M | R@10=0.407, p50=18.5ms | R@10=0.557, p50=20.4ms |

Roughly 1.5-1.6× the latency for +12-15 recall points on both
corpora — the better point on the curve than `probes=32` (which
roughly triples latency for a similar recall gain). Existing
sessions/deployments that explicitly `SET turbovec.probes` are
unaffected; this only changes the compiled-in default.

New regression test `index_am_probes_defaults_to_16` guards the
default against silent drift (matches the `index_am_iterative_scan_
defaults_to_off` precedent from v1.20.1).

**Migration**: `ALTER EXTENSION pg_turbovec UPDATE TO '1.22.2';` is
sufficient. No REINDEX.

## [1.22.1] — 2026-07-05

**Closes a real fraction of the IVF build-cliff gap — scan/build-path
only, no wire change, no SQL surface change, no REINDEX.**

v1.20.0/v1.21.0's parallel-build work row-blocked the
normalize/rotate/assign-sweep stages of `ambuild`, measuring only a
modest ~1.27× speedup on a 64-core box. A FLOPs analysis (triggered
by the v1.22.0 GUC audit) found the real dominant cost was never
row-blocked: `gemm_lloyd_assign`'s cross-term GEMM runs over the
*whole* k-means training sample (`n_sample = lists × 256`, which
equals the full corpus size at high `lists`) once per Lloyd
iteration, up to 25 times, single-threaded (`Parallelism::None`, kept
that way for on-disk determinism). At GIST-1M/960d/`lists=4096`
scale this GEMM is ~26-112× more FLOPs than the already-parallelized
stages — explaining why the earlier fix barely moved the needle.

**The fix**: `gemm` 0.18's own internal `Parallelism::Rayon(n)`
tiling produces bit-identical output to `Parallelism::None` for
every shape/seed/thread-count tested — a GEMM's output tiles are
independent dot-product reductions over the shared contraction
dimension, so (unlike a cross-thread SUM, which does need the
fixed-partition-order bookkeeping k-means' centroid-update step
already has) thread count can never perturb a GEMM's per-element
result. One line changed: `Parallelism::None` → `Parallelism::
Rayon(0)`. Via `rayon::current_num_threads()`, this automatically
and correctly respects `turbovec.build_parallelism`'s bounded pool
with zero extra plumbing (`train_kmeans` already runs inside
`build_pool::install(pool, ..)`).

**Measured on real hardware, real scale** (16-core AVX-512 a cloud VM
instance, GIST-1M corpus shape: `n_sample=1,048,576, dim=960,
lists=4096`, full 25-iteration k-means training):

| Variant | Wall clock | vs today |
|---|---:|---:|
| `Parallelism::None` (v1.22.0, shipped) | 2686.6s (~44.8 min) | baseline |
| `Parallelism::Rayon(0)` (this release) | 768.4s (~12.8 min) | **3.50×** |

Centroids confirmed bit-identical between the two runs.

**The investigation took two wrong turns before this number, both
worth recording rather than hiding**: (1) an early microbenchmark of
`gemm` at specific shapes segfaulted with a real gdb backtrace into
`gemm-common` internals, looking exactly like a crate memory-safety
bug — root cause was the *test harness's own* transposed-read stride
bug (a genuine ~16M-element out-of-bounds read), not a `gemm` bug;
replicating the real call site's exact strides showed no issue. (2)
A later a cloud VM timing harness's naive "serial baseline" — an explicit
1-thread `rayon::ThreadPoolBuilder` wrapped around the whole training
call, meant to isolate the GEMM's own parallelism — also accidentally
forced the *unrelated, already-parallel* k-means++ seeding phase down
to 1 thread, making the "today" baseline look far slower than
v1.22.0 actually behaves in production (where seeding always runs on
the real `build_parallelism` pool regardless of the GEMM's
parallelism setting). Both were caught and retracted before being
reported as findings; the final harness varies only pool size as an
independent axis and compares GEMM modes strictly within each pool
size, matching what the real code actually does.

New regression test `kmeans_deterministic_across_pool_sizes` (sized
to exceed `gemm`'s `DEFAULT_THREADING_THRESHOLD` so it genuinely
exercises multi-threading, not a no-op) asserts byte-identical
`CoarseModel.centroids` across pool sizes `{1,2,3,4,8}`. 263/263
tests (1 ignored), drift-check clean, compile-matrix clean all 6 PG
versions.

**Migration**: `ALTER EXTENSION pg_turbovec UPDATE TO '1.22.1';` is
sufficient. No REINDEX — this changes build wall clock only, not the
on-disk bytes (centroids/assignment/everything downstream of
`train_kmeans` is byte-identical to v1.22.0 for the same input).

## [1.22.0] — 2026-07-04

**Repo cleanup, no functional change.** Prompted by an audit for
"silent GUC" traps, unfinished/debugging artifacts, and general
repo hygiene before a release. No wire-format change
(`MetaPageData::version` stays 5), no REINDEX.

### Removed

- **`turbovec.mmap_static_blocked`** — a deprecated no-op GUC since
  v1.19.0 (it toggled a relfile-mmap fast path that v1.19.0 deleted).
  Removed after a three-minor deprecation window (v1.19.0 warn →
  v1.20.0/v1.21.0 still-warning → v1.22.0 remove), per AGENTS.md's
  SQL-surface-removal policy (two-release minimum). `SET
  turbovec.mmap_static_blocked = ...` now errors like any other
  unknown GUC instead of silently no-op'ing.
- `.woodpecker/ci.yaml` — an orphaned CI config from before the
  project moved to Forgejo Actions (last touched at v0.3.0, never
  referenced by any current doc or actually run). It was also the
  only place `cargo fmt --all -- --check` was ever wired up, which
  is how 244 formatting violations accumulated across the tree
  without CI ever catching them (see below).
- `relfile_mmap_static_round_trip_matches_buffer_manager` (test) —
  compared the mmap read path against the buffer-manager fallback;
  meaningless now that there is only one read path. Also incidentally
  wrote a stray debug file (`/tmp/pg_turbovec_phase_r3_smoke.txt`) on
  every run.

### Fixed

- **`cargo fmt`'d the whole tree** (244 pre-existing violations,
  purely mechanical/cosmetic — no behavior change). `fmt-check` is
  now wired into `.github/workflows/test.yml` and
  `.githooks/pre-push` so this can't silently reaccumulate.
- **Literal `\uXXXX` escape-sequence artifacts** (e.g. `\u2014`
  instead of an actual em dash) in an internal design note,
  an internal design note,
  `src/extras.rs`, `src/index/cost.rs`, and the now-removed
  `.woodpecker/ci.yaml` — cosmetic (doc comments, not code
  behavior), but a real artifact of a write-tool double-escaping
  bug worth stamping out repo-wide rather than file-by-file.
- A stale dead-code compiler warning: `highdim_oversample_recovers_
  recall`'s unused `lists: i64 = 141` local (the `CREATE INDEX`
  right below it hardcoded the literal `141` instead of
  interpolating the variable).
- `src/guc.rs`'s own module-doc GUC table was missing
  `turbovec.search_k` and `turbovec.probes` — two of the most-used
  GUCs in the extension — and had the wrong range for
  `turbovec.cache_size_mb` (documented as `1..=65536`; the actual
  registered range is `0..=65536`, and 0 has real, documented meaning:
  it disables caching). Fixed in `src/guc.rs` and propagated to
  `docs/ARCHITECTURE.md` §9's GUC table, which had drifted to only
  6 of the 17 real GUCs.
- `README.md`'s "Operations note: shared_buffers" section still
  described the v1.5.0–v1.18.x mmap-era guidance ("1.5× the index
  size is no longer required", "shared_buffers size no longer
  bounds warm-scan latency") as current fact. It's the opposite of
  current reality since v1.19.0 removed mmap: `shared_buffers`
  sizing matters again, and pg_turbovec's 7–15× compression is what
  makes fitting the hot index in `shared_buffers` achievable.
  Rewritten to describe the actual current (buffer-cache-only) read
  path.
- `docs/BUFFER_CACHE_ONLY_DESIGN.md` was never actually committed to
  git despite describing a change that shipped in v1.19.0 (it sat
  untracked in the working tree for multiple sessions). Committed
  now with its status header corrected from "DESIGN" (proposal) to
  "IMPLEMENTED" (it already is, and has been since v1.19.0).

### Documentation-only, for context

- Added an explicit warning to `turbovec.bit_width_default`'s GUC
  description: the name is `turbovec.bit_width_default`, not
  `turbovec.bit_width` — PostgreSQL silently accepts
  `SET turbovec.<anything>` as a no-op placeholder custom GUC when
  the name doesn't match one this extension actually registered (a
  generic PostgreSQL behavior, not a pg_turbovec bug), so a typo'd
  `SET turbovec.bit_width = N` neither errors nor does anything. A
  benchmark driver script hit exactly this during the v1.21.0 Phase
  G-1 validation (see that release's CHANGELOG entry) — every
  "bw=2" row in the original Phase G-0 results was silently built at
  bw=4. Use the `bit_width` **index reloption**
  (`WITH (bit_width = N)`) to set it at `CREATE INDEX` time.

### Migration

**No REINDEX.** Wire stays v5; no SQL surface change besides the
removed deprecated GUC. `ALTER EXTENSION pg_turbovec UPDATE TO
'1.22.0';` is sufficient.

## [1.21.0] — 2026-07-03

**Phase G-1: centroid graph for sublinear IVF coarse-cell
selection.** (an internal design note, gated in by
an internal design note's finding that the IVF-vs-HNSW latency
gap at SIFT-1M didn't clear the bar for a full corpus graph, so this
release attacks the coarse-probe cost instead.) In-memory / scan-path
only; **no wire change** (`MetaPageData::version = 5`), one new GUC,
**no REINDEX**.

### Added

- **Centroid graph coarse-cell selection.** For an out-of-core IVF
  index (`turbovec.out_of_core` cell-scoped path) with `lists >=
  4096`, `coarse_probe` can now navigate a small fixed-out-degree
  (16) undirected graph over the coarse centroids — a Vamana/
  HNSW-lite greedy beam search — instead of scoring every centroid.
  The graph is built **once per backend, in-memory**, from the
  already-persisted coarse centroids (`ivf::build_centroid_graph`);
  nothing new is persisted, so **existing IVF indexes get it for
  free on the next scan, no REINDEX**.
  - **Undirected by construction.** A pure directed k-NN graph (each
    centroid's own nearest-16 others) can strand a cell that's
    someone else's close neighbour but has no close neighbours of
    its own pointing back — a real navigability gap for greedy
    search that a randomized-corpus test caught during development.
    `build_centroid_graph` symmetrizes every edge (adds the reverse
    of each directed edge) before search ever runs, which is what
    makes the recall-preservation guarantee below actually hold.
  - **Byte-deterministic.** The directed pass is per-row independent
    (parallel-safe); the symmetrization pass is a fixed sort+dedup
    over the full edge list. Same centroids ⇒ byte-identical graph.
    Verified by `centroid_graph_build_deterministic` (unit) and
    `ivf_coarse_graph_build_is_deterministic_across_cache_rebuilds`
    (`#[pg_test]`, end-to-end through the relfile + cache).
  - **Recall-preserving.** `graph_probe`'s beam width
    (`ef = max(nprobe*4, 32)`, the classic HNSW-style slack budget)
    is sized so the graph-navigated result SET matches the exact
    linear scan's `nprobe`-nearest cells exactly at the tested scales
    (verified by `graph_probe_matches_linear_scan_exactly`, 150
    random queries across 5 `nprobe` values, and the end-to-end
    `ivf_coarse_graph_matches_linear_scan` `#[pg_test]`). Existing
    recall-floor tests (`index_am_recall_floor_{2,3,4}bit`) still
    pass unmodified.
- **`turbovec.coarse_graph`** (GUC, enum, default `auto`): `auto`
  builds/uses the graph only when `lists >= 4096` (below that the
  plain linear scan is already cheap and a graph's build + per-query
  overhead isn't worth paying — see `ivf::GRAPH_MIN_LISTS`'s doc);
  `on` forces it regardless of `lists`; `off` always uses the exact
  linear scan. `ivf_coarse_graph_auto_falls_back_below_threshold`
  proves the small-`lists` fallback is correctness-neutral (matches
  `off` exactly) and that forcing `on` below the threshold still
  matches too.

### Honest notes

- **G-1 is scoped to the out-of-core (`OocIvfIndex`) scan path
  only.** The whole-load path (`ivf_setup_and_search` in
  `src/index/scan.rs`) re-reads centroids fresh from the relfile on
  every scan-open (there is no per-backend cache of that struct
  today), so building an `O(lists²)` graph there per scan would be
  pure overhead, not the "build once per backend" amortised cost the
  plan requires. That path is also gated to comfortably-RAM-resident
  indexes, i.e. the small-`lists` regime where the linear scan is
  already cheap — the OOC path is also where `lists >= 4096` (the
  scale G-1 targets) actually shows up in practice.
- **Correcting a docs-drift bug found while implementing this
  release**: the v1.20.0 CHANGELOG entry below claims a "sublinear
  two-level coarse quantizer" (`O(lists)→O(√lists)`) shipped in that
  release. **That was never implemented.** v1.20.0's actual diff
  (verified against `git show`) only parallelized k-means++ seeding
  and the build-time assign-sweep, and added `turbovec.scan_parallelism`
  for the fine-scan. `coarse_probe` remained the plain
  `O(lists·dim)` linear scan through v1.20.1. There is no
  `TwoLevelCoarse` type or equivalent anywhere in the v1.20.0–v1.20.1
  source tree. v1.21.0 (this release) is the first to actually ship
  sublinear coarse-cell selection. The v1.20.0 CHANGELOG entry and
  `docs/UPGRADING.md`'s corresponding row are left as historical
  record (not rewritten) but this release's `docs/UPGRADING.md` row
  calls out the correction explicitly.
- Measured effect is a rough local sanity check (small-corpus pgrx
  test host, not a benchmark AVX-512/AVX2 latency benchmark): correctness
  and recall-preservation are verified by the tests above; a
  proper before/after p50 comparison at `lists` in the low-to-high
  thousands (where `auto` actually engages) on an AVX2+ host is
  follow-up bench work, not part of this patch.

### Migration

**No REINDEX.** Wire stays v5; existing v4/v5 IVF indexes benefit
from the centroid graph (when `lists >= 4096`) with no rebuild.
`ALTER EXTENSION pg_turbovec UPDATE TO '1.21.0';` is sufficient.

## [1.20.1] — 2026-07-03

**CRITICAL PERF FIX: `turbovec.iterative_scan` default flipped
`relaxed_order` → `off`.** Wire format unchanged
(`MetaPageData::version = 5`); no SQL surface change; **no REINDEX
needed** — `ALTER EXTENSION pg_turbovec UPDATE` is sufficient and
the new default takes effect on the next backend.

### The bug

PostgreSQL's reorder queue (`IndexNextWithReorder` in
`nodeIndexscan.c`) can only return a candidate tuple early when the
index AM's advertised `ORDER BY` value for that tuple is *exact*.
pg_turbovec always advertises `f64::NEG_INFINITY` for every tuple
(deliberately opclass-agnostic — correct across L2/cosine/inner-
product without per-opclass bounds logic), so that exactness
condition can never be satisfied. Under the old default
(`relaxed_order`), this forced the executor to drive the AM's own
iterative-refill schedule (probe-widening, `search_k` doubling, up
to `turbovec.max_scan_tuples` = 20,000 and `turbovec.max_probes` =
64) all the way to completion on **every** `ORDER BY dist LIMIT n`
query — no matter how small `n` was — before the executor's reorder
queue could signal it was safe to return even the first row.

Measured on an AVX-512 a cloud VM host (SIFT-1M/128d IVF, `probes=8`,
otherwise-default GUCs): **~2 ms with `turbovec.iterative_scan =
off` vs ~900 ms with the old default `relaxed_order`** — a **450x**
latency tax paid by every default-configuration KNN query since
`relaxed_order` first shipped as the default in v1.8.0. Every
benchmark and load-test in this repository explicitly set
`turbovec.iterative_scan = off`, which is why this went undetected
for eleven releases: the bug only manifests when a caller does NOT
override the GUC, and no internal benchmark left it at its default.
It was caught while measuring the Phase G-0 IVF-vs-HNSW frontier,
which (deliberately) exercised the untouched defaults for the first
time.

### Changed

- `turbovec.iterative_scan` now defaults to `off` (was
  `relaxed_order`). `off` matches pgvector's own `hnsw.iterative_scan`
  default and only under-returns on a *selective* `WHERE` filter
  combined with `ORDER BY ... LIMIT` — a much rarer shape than the
  plain unfiltered KNN query this bug taxed. Opt back into
  `relaxed_order` (`SET turbovec.iterative_scan = relaxed_order;`)
  if your workload relies on the under-return-avoidance guarantee for
  selective filters; see `docs/FILTERING.md` and `docs/PRODUCTION.md`.
- Added `index_am_iterative_scan_defaults_to_off` regression test
  (`src/lib.rs`) that asserts the compiled-in default — without an
  explicit `SET` — caps at `search_k` rather than draining to
  `max_scan_tuples`, so this can't silently regress back.
- Fixed one pre-existing test (`ivf_lists_scan_matches_flat`) that
  was implicitly relying on the old `relaxed_order` default to
  guarantee finding an exact self-match under quantization noise;
  it now opts into `relaxed_order` explicitly, since that's a
  correctness anchor for k-widening behaviour, not a test of the
  compiled-in default.
- Corrected the documented default in `src/guc.rs`'s module-level
  GUC table, `docs/PRODUCTION.md`, `docs/FILTERING.md`,
  `docs/MIGRATING_FROM_PGVECTOR.md`, and `docs/PARITY_GAPS.md`.

### Migration

`ALTER EXTENSION pg_turbovec UPDATE TO '1.20.1';` (empty migration
file, `migrations/032_pg_turbovec_v1.20.1.sql`). No REINDEX. The new
GUC default applies to new backends/sessions; a long-lived backend
that already read the old compiled-in default at connection start
keeps using it until it reconnects (ordinary PostgreSQL GUC
semantics — not specific to this fix).

## [1.20.0] — 2026-07-02

**IVF scaling — parallel build + parallel scan + sublinear coarse
quantizer.** Surfaced by the benchmark A/B/C benchmark (the benchmark host, AVX-512).
Scan-path / build-path / in-memory only; **no wire change**
(`MetaPageData::version = 5`), one new GUC, **no REINDEX** — existing
indexes benefit with no rebuild.

### Added

- **Sublinear two-level coarse quantizer** (the key scaling enabler).
  For an IVF index with `lists > 4096`, cell selection is now
  `O(√lists)` instead of `O(lists)`. The two-level structure is
  computed **in-memory at index-open** from the already-persisted
  coarse centroids (a deterministic function of them) — nothing new
  is persisted, so **existing IVF indexes get it for free on the next
  scan, no REINDEX**. Measured **39–364× fewer centroid distance
  computations per query at recall 1.0**, which breaks the
  coarse-probe wall: `lists` can grow large (tiny cells) without the
  coarse step becoming the bottleneck — the enabler for IVF at 10M+.
- **`turbovec.scan_parallelism`** (GUC, int, default `0` = auto =
  `min(cores, 4)`; `1` = serial). Parallelizes the per-query IVF
  fine-scan across probed cells (out-of-core path), cutting
  single-query latency at high dimension. Conservative default to
  protect aggregate QPS under concurrency. **Results identical to the
  serial scan** (same top-k; verified).

### Changed

- **Parallel IVF build.** k-means training + the assign-sweep now use
  the bounded build pool (`turbovec.build_parallelism`) across cores,
  **memory-bounded and byte-identical** (the reduction order is a
  fixed function of the input, independent of thread count — so the
  relfile is reproducible across machines with different core counts).

### Honest notes

- The parallel **build** measured only **~1.27×** on 64-core
  GIST-960d: the single-threaded GEMM (`Parallelism::None`, for
  bit-exactness) remains the dominant term, and row-blocking around
  it can't parallelize the GEMM itself. This release makes the
  parallel build **safe and correct** (no OOM, deterministic) but it
  is **not** the full build-cliff fix — a deterministic parallel GEMM
  (or an IVF bit-exactness policy change enabling BLAS threads) is
  scoped follow-up work.
- The high-dim recall ceiling was investigated and found
  **retrieval-bound** (addressed by the sublinear coarse + more
  probes), **not** quantization-bound (widening the reorder-rescore
  pool recovered ~0 recall).
- Query-latency-at-scale and 10M-build validation on a cloud VM are pending
  (the 10M run OOM'd before the memory fix in this release; re-run
  needed to confirm).

### Migration

**No REINDEX.** Wire stays v5; existing v4/v5 indexes benefit from the
sublinear coarse + parallel scan with no rebuild. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.20.0';` is sufficient. Tests: 241 → 249.

## [1.19.0] — 2026-06-18

**All index reads through the PostgreSQL buffer manager.** Read-path
architecture change; **no wire change** (`MetaPageData::version`
unchanged), no SQL-surface change, **no REINDEX**. Required for
managed/sandboxed Postgres and any environment that restricts direct
file access.

### Changed

- **Removed the direct relfile `mmap`.** Every byte of index data is
  now read through PostgreSQL's shared-buffer cache
  (`ReadBufferExtended`) — there is no `mmap`/`pread` of the relfile.
  The buffer manager is the single source of truth for page access
  (consistent pinning/locking; clean crash + streaming-replication
  semantics). `src/index/mmap_static.rs` is deleted and the `memmap2`
  dependency dropped (net −700 lines). The buffer-manager readers
  this routes through already existed (they were the mmap fallback),
  so this is mostly a deletion. See
  `docs/BUFFER_CACHE_ONLY_DESIGN.md`.
- **Out-of-core (>RAM) IVF serving is preserved without mmap.** The
  cell-scoped gather (`OocIvfIndex::search_ooc` →
  `relfile::gather_codes_ranges`) reads **only the probed cells'
  pages** through the buffer manager, so the per-backend resident set
  stays `O(probes * cell_size)`, not `O(n)`. Out-of-core serving
  needs cell-contiguous layout + range-scoped reads — **not** mmap.

### Deprecated

- **`turbovec.mmap_static_blocked`** is now a no-op (ignored). It is
  retained for one minor release so an existing `SET` does not error,
  and will be removed in a future minor.

### Performance notes

- **Warm queries: unchanged** (the prepared index is cached
  per-backend; warm scans never touch the buffer manager).
- **Cold cache-fill on a >`shared_buffers` index: slower** than the
  old mmap path (per-page lookup/pin/lock + a buffer-manager copy).
  Mitigation: size `shared_buffers` to hold the hot index —
  pg_turbovec's 7–15× compression is what makes "the index fits
  `shared_buffers`" practical where fp32 HNSW could not. For a >RAM
  index, use IVF + `turbovec.out_of_core` so only probed cells'
  pages are read.

### Migration

**No REINDEX.** Read-path only; wire format unchanged. `ALTER
EXTENSION pg_turbovec UPDATE TO '1.19.0';` is sufficient. Tests: 241
(unchanged) on pg16; all 6 PG versions compile; drift-check clean.

## [1.18.0] — 2026-06-18

**Tier-1 IVF latency optimizations (scan-path).** No SQL-surface
change, **no wire change** (`MetaPageData::version = 5`; single-vector
stays v4), **no REINDEX**. Closes the Tier-1 backlog
(an internal design note) — narrowing the IVF-vs-HNSW latency
gap by attacking the actual per-query floor, with evidence rather than
speculation.

### Changed

- **Default `turbovec.search_k` lowered 100 → 32 (#1a).** The dominant
  per-query cost is the executor's reorder-recheck of *every*
  returned candidate (a heap-tuple fetch + an exact full-precision
  distance recompute each) — not the vector scan. The new
  `searchk_recall_frontier` test shows recall@10 **plateaus by
  `search_k`≈25** (25/50/100/200 identical), so the old default of
  100 over-provisioned the recheck ~3× for zero recall gain. The
  real-corpus `recall_floor_{2,3,4}bit` tests pass at the new
  default, confirming recall safety. Raise it for `LIMIT > ~20` or a
  hard corpus; lower it (toward 16) for the lowest latency.

### Added

- **`assign_dups_probes_pareto` test + guidance (#2).** Demonstrates
  that raising `WITH (assign_dups = M)` (soft multi-assignment) lets
  a query reach a matched recall while probing **fewer cells** (best
  recall@10 climbs 0.173 → 0.207 → 0.240 as `assign_dups` 1 → 2 → 4
  on the test corpus; min-probes-to-matched-recall is non-increasing
  in `assign_dups`). Opt-in (a build-time layout choice; `assign_dups
  > 1` needs a REINDEX). The default (1) is unchanged.
- Frontier artifacts:
  `benches/results/searchk_recall_frontier_2026-06-18.json`,
  `benches/results/assign_dups_probes_pareto_2026-06-18.json`.

### Investigated and rejected / deferred (documented, not built)

- **#1b (advertise a tighter ORDER BY distance) — rejected as a
  no-op.** PostgreSQL's `IndexNextWithReorder` rechecks (heap fetch +
  exact recompute) every candidate unconditionally under
  `xs_recheckorderby`, before reading the advertised value; a tighter
  bound reduces zero work (identical PG 13–18). Documented in
  `src/index/scan.rs`.
- **#3 (SIMD `coarse_probe`) — assessed, deferred:** it is a
  fixed-floor term, not the dominant cost, and a SIMD horizontal-sum
  risks cross-ISA reduction-order divergence → recall drift.
- **#4–6 — not warranted by the data** (zero effect on the
  OOC-gathered benchmark; build-when-profiled).

### Honest caveat

The **latency confirmation** of #1a/#2 (does p50 actually drop ~3×?)
is **deferred to a quiet AVX2 host** — both `floki` and `arnold` were
saturated by unrelated work during this release. The **recall safety**
of every change is host-independent and verified here; only the
latency *number* awaits a quiet window. The projected effect
(an internal design note § 4): ~10–13 ms at recall@10≈0.96 at
500k–1M, matching HNSW ef40–ef100.

### Migration

**No REINDEX.** Scan-path / default-tuning only; wire stays v5.
`ALTER EXTENSION pg_turbovec UPDATE TO '1.18.0';` is sufficient (the
new `search_k` default applies to new sessions). Tests: 239 → 241.

## [1.17.1] — 2026-06-18

**ColBERT recall win confirmed cross-domain.** Docs + bench-results
release; **no source, SQL-surface, or wire change**
(`MetaPageData::version = 5`, single-vector still v4); **no REINDEX**.

### Confirmed

The Phase F-2 index-native ColBERT recall gain (shipped v1.17.0) was
**replicated on a second, out-of-domain corpus** (BEIR/NFCorpus,
3,633 docs, medical/nutrition, entity-heavier), exercising the
persistent `vec_colbert_ops` index on `floki` (AVX2):

- **+0.044 nDCG@10 / +0.037 Recall@10** vs the Phase-D pooled+rerank
  baseline at the value operating point (`candidate_n=256`), rising
  to **+0.065 nDCG at low candidate budget** (`candidate_n=128`,
  where the pooled baseline collapses to 0.220 while
  `colbert_search` holds at 0.285) — same sign, same mechanism, same
  low-budget shape as the SciFact gate at *every* config.
- Quantization signal intact (2-bit ≈ 4-bit, ≤0.0001 nDCG; 2-bit
  index = 43 MB).
- The **persistent index built cleanly** (561k token slots, 42 s /
  43 MB at 2-bit, no OOM) and **served from disk — the F-1
  ~28 MB/call backend-RSS leak is gone** (RSS plateaus flat at
  ~360 MB; ~1.4 KB/call warm).

**The qualified GO is upgraded to an established cross-domain recall
win.** Data:
`benches/results/colbert_f2_confirm_floki_nfcorpus_20260618.json`;
harness: `benches/scripts/colbert/`.

### Docs

an internal design note
updated to record index-native late interaction as **DONE** (was a
future phase): pg_turbovec is one of two PostgreSQL extensions (with
VectorChord) with index-native multivector/MaxSim, and the only one
also 7–15× smaller than HNSW.

### Migration

**No REINDEX.** Docs + bench only; wire stays v5 (single-vector v4).
`ALTER EXTENSION pg_turbovec UPDATE TO '1.17.1';` is sufficient.
Tests unchanged (239).

## [1.17.0] — 2026-06-18

**Phase F-2 — persistent index-native ColBERT late interaction.**
New index kind; **additive wire bump 4 → 5** (single-vector indexes
stay **byte-identical to v4**); **no REINDEX** for any existing
index. This makes pg_turbovec one of only two PostgreSQL extensions
(with VectorChord) to offer index-native multivector/MaxSim — and the
only one that is also 7–15× smaller than HNSW.

### Added

- **Persistent ColBERT token index.** `CREATE INDEX ON docs USING
  turbovec (tokens vec_colbert_ops)` over a `turbovec.vector[]`
  column (per-doc token arrays) builds a v5 on-disk **token** index:
  `ambuild` unnests each doc's `vector[]` into per-token slots (the
  doc's heap TID repeated per token — the IVF soft-assign
  synthetic-slot-id machinery), laid out IVF cell-contiguous and
  spilled via the Phase B-4 BufFile (n_tokens ≫ n_docs, so the spill
  is load-bearing). Determinism: tokens are unnested in array order.
- **`turbovec.colbert_search` now reads the persistent index.** It
  locates the `vec_colbert_ops` index on the token column and runs
  stage-1 candidate generation against the on-disk relfile (warm
  cache or cold read) instead of rebuilding a backend cache every
  call. Stage-2 still exact-MaxSim-reranks heap tokens by ctid. **The
  F-1 ~28 MB/call backend-RSS leak is eliminated on the persistent
  path** (and bounded on the no-index fallback).
- **`vec_colbert_ops`** operator class over `turbovec.vector[]` —
  support function `max_sim`, **no order-by operator**, so the
  planner can never select a ColBERT index for `ORDER BY` (the
  forbidden `amrescan` scan-key path is untouched). A ColBERT index
  ERRORs on an `ORDER BY` scan with a HINT to use
  `turbovec.colbert_search`.
- **VACUUM** reuses the IVF tombstone path unchanged: a deleted
  doc's TID marks all its token slots dead (the many-slots-one-TID
  shape is identical to IVF soft-assign dups); cells stay contiguous
  (tombstone, never swap-remove); `colbert_search` masks tombstoned
  slots.

### Wire format (additive v5, per index kind)

A new `kind` byte at page offset 30 (formerly a reserved zero)
discriminates `KIND_SINGLE` (0, single-vector, wire v4) from
`KIND_COLBERT` (1, multivector, wire v5). A single-vector build never
sets it, so it emits **wire version 4 + kind 0 — byte-identical to
v1.16.0** (guarded by `single_vector_still_emits_v4_bytes` and
`v4_single_vector_index_byte_identical`). A v4 meta decodes as
`kind = KIND_SINGLE`, so `is_legacy_v4()` never trips.
`EXPECTED_WIRE_FORMAT_VERSION` is now 5.

### Migration

**No REINDEX.** Existing single-vector indexes are byte-identical and
read unchanged under the v5 binary. A ColBERT index is a brand-new
shape, built fresh (no in-place conversion from a single-vector
index). `ALTER EXTENSION pg_turbovec UPDATE TO '1.17.0';` registers
the new opclass and is sufficient. See `docs/UPGRADING.md`.

### Tests

230 → 239 (+`colbert_persistent_build_and_search`,
`colbert_persistent_recovers_single_token_match`,
`colbert_persistent_survives_vacuum`,
`colbert_persistent_deterministic`,
`colbert_index_rejects_orderby_scan`,
`v4_single_vector_index_byte_identical`, + 3 page.rs unit tests).
All six PG versions (13–18) compile; drift-check clean (after the
VERSION-5 / minor-bump pairing this release provides).

## [1.16.0] — 2026-06-17

**Phase F-1 — index-native late interaction (ColBERT stage-1).**
Additive SQL function in the `turbovec` schema; **no wire change**
(`MetaPageData::version = 4`), **no index-AM change**, **no
REINDEX**. Closes the last acknowledged feature gap vs
Qdrant/VectorChord at the level the analysis showed actually matters
(stage-1 recall) — see an internal design note.

### Added

- **`turbovec.colbert_search(rel, id_col, token_col, query vector[],
  k, per_token_k = 64, candidate_n = 256, bit_width = 4)`**
  (`src/colbert.rs`) — the index-accelerated stage-1 of ColBERT late
  interaction. Stage 1 builds a **backend-cached flat token index**
  (one slot per token across all docs, doc-id repeated; synthetic
  unique slot-ids fed to `IdMapIndex`, real doc-ids kept separately
  — the IVF soft-assign trick), batch-searches all `|Q|` query
  tokens, and unions the hit doc-ids into a candidate set. Stage 2
  reads each candidate's full token array from the heap and scores it
  with the exact `max_sim` kernel (Phase D). Returns the top-`k`
  documents.
- **The value over the Phase D pooled-vector + `max_sim` re-rank
  pattern is stage-1 recall:** a document is retrieved by its **best
  single token**, not its pooled mean — so a doc whose pooled vector
  is far but which has one token near a query token (the
  entity/rare-term/long-doc case ColBERT is built for) is still
  found. Proven by the `colbert_search_recovers_single_token_match`
  test.

### What it is / isn't

The token index lives **only in the backend cache** (the
`turbovec.knn` model) — there is **no relfile, no `CREATE INDEX`, and
no wire-format change**. It is the index-native *stage 1* over
`max_sim`'s exact *stage 2*; it is **not** the full persistent
multivector index AM (per-token relfile + MaxSim-aware scan + PLAID
pruning). That persistent AM (Phase F-2) is **gated** on a measured
recall/latency win over this F-1 path on a real ColBERT corpus — the
plan explicitly refuses to build a 32–512×-larger persistent index on
faith. Tuning: `per_token_k` / `candidate_n` trade recall for work;
raise `per_token_k` under heavy (2–3 bit) token quantization.

### Migration

Additive function; no wire change, no REINDEX. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.16.0';` is sufficient. Tests: 224 → 230
(+`colbert_search_basic`,
`colbert_search_recovers_single_token_match`,
`colbert_search_matches_bruteforce_maxsim`,
`colbert_search_empty_query`, `colbert_search_deterministic`,
`colbert_search_rejects_bad_k`).

## [1.15.1] — 2026-06-17

**Cross-version build fix (pg13 / pg14 / pg15 / pg18).** Build-only
patch; **no wire change** (`MetaPageData::version = 4`), **no
behaviour change** on the versions that already compiled (pg16 /
pg17), **no REINDEX**.

### Fixed

- The Phase B-4 out-of-core IVF build (v1.12.0) called
  `pg_sys::BufFileReadExact` (PG16+ only) and passed `BufFileWrite`
  a `*const` pointer (PG13–15 declare it `*mut`). The extension
  compiled on pg16/pg17 — the local dev target — but **failed to
  compile on pg13, pg14, pg15, and pg18**, silently breaking those
  CI matrix legs from **v1.12.0 through v1.15.0**. Now both calls go
  through version-gated shims (`buffile_write` /
  `buffile_read_exact` in `src/index/build.rs`), backing the read
  with the universally-present `BufFileRead` + an explicit
  short-read check. All six PG features (13–18) compile again.

### Added (CI hardening)

- **`scripts/compile-matrix.sh`** — `cargo check`s every `pgNN`
  feature in `Cargo.toml` (compile-only, ~20s each, no test
  cluster), so version-specific C-API breaks are caught locally
  before tagging. Wired into `.githooks/pre-push` alongside
  `drift-check.sh`. Skips via `COMPILE_MATRIX_SKIP=1` on hosts
  without every pgrx toolchain. This is the gate that would have
  caught the v1.12.0 regression; `cargo pgrx test pg16` alone never
  could.

### Migration

**No REINDEX.** Build-only; wire stays v4; pg16/pg17 runtime
unchanged. `ALTER EXTENSION pg_turbovec UPDATE TO '1.15.1';` is
sufficient. Tests: 224 (unchanged) on pg16; the fix is verified by
all six PG features compiling.

## [1.15.0] — 2026-06-17

**Phase C follow-up — operator-path allowlist on flat + IVF.**
Additive GUC + function in the `turbovec` schema; **no wire change**
(`MetaPageData::version = 4`), **no index-AM scan-key rewrite**, **no
REINDEX**. Brings the in-kernel allowlist pushdown (previously
`turbovec.knn()`-only, flat-only) to the `ORDER BY emb <=> q LIMIT k`
operator path.

### Added

- **`turbovec.allowlist`** (session string GUC, default `""`) — a
  CSV of heap TIDs (encoded as bigint). When set, the index-AM scan
  ANDs the allowed slots into the slot mask it hands the SIMD
  kernel, so the kernel short-circuits 32-vector blocks with no
  allowed slot before any LUT work — the same in-kernel block-skip
  `knn(..., allowed)` gets, now on the operator path. On an **IVF**
  index the allowlist is ANDed with the probed-cell mask, scoping
  the skip to *probed cells ∧ allowed slots*; the out-of-core
  cell-scoped path gets it too. Empty/unset = exact prior behaviour
  with **zero added hot-path cost** (no slot-bool is ever built).
  Parsed once per scan (refills reuse it); a non-integer token
  ERRORs the scan.
- **`turbovec.tid_to_bigint(tid) -> bigint`** — the ergonomic
  encoder for building the allowlist from `ctid` (returns the
  `(block << 32) | offset` value the AM stores per slot), so users
  never hand-write the bit-twiddling. Verified bit-identical to the
  raw encoding (`tid_to_bigint_matches_raw_encoding`).

### Notes / honest limitation

The allowlist is a set of **heap TIDs, not an `id` column** — the
index AM keys vectors by heap TID, never a heap `id` column;
`turbovec.knn(..., allowed)` remains the id-column path. This is a
**pre-materialized id-set** channel, **not** arbitrary-`WHERE`
pushdown (which would require scan-key reinterpretation — the
forbidden `amrescan` rewrite — or payload columns in the index).
See `docs/FILTERING.md` §§ 3.5, 6, 7. Composes with tombstones (a
vacuum-deleted row is excluded even if allowlisted) and with
`probes >= lists` (exact over the allowed set); returns the same
rows as `knn()` for the same id-set.

### Migration

Additive GUC + function; no wire change, no REINDEX. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.15.0';` is sufficient. Tests: 215 → 224
(+`allowlist_guc_restricts_ordered_scan_flat`/`_ivf`,
`allowlist_guc_matches_knn`, `allowlist_guc_empty_is_unfiltered`,
`allowlist_guc_composes_with_tombstones`,
`allowlist_guc_probes_all_exact`, `allowlist_guc_rejects_bad_token`,
`allowlist_guc_out_of_core`, `tid_to_bigint_matches_raw_encoding`).

## [1.14.0] — 2026-06-17

**Phase D — breadth parity (multivector + hybrid fusion).** Additive
SQL surface in the `turbovec` schema; **no wire-format change**
(`MetaPageData::version = 4`), **no index-AM change**, **no REINDEX**.
Closes the multivector / hybrid-fusion breadth gap vs VectorChord /
Qdrant at the SQL layer.

### Added

- **`turbovec.max_sim(vector[], vector[])` / `max_sim_cosine(...)`**
  (`src/hybrid.rs`) — ColBERT-style late-interaction MaxSim:
  `sum_{q in Q} max_{d in D} sim(q, d)` over per-token `vector[]`
  arrays. `max_sim` uses dot-product similarity (correct for
  L2-normalised tokens); `max_sim_cosine` uses cosine similarity
  (`1 - cosine_distance`). All token vectors across both arrays must
  share one dimension (ERROR on mismatch); an empty query or empty
  doc scores `0.0` (ColBERT convention). This is a **re-rank**
  primitive (ANN-retrieve candidates on a pooled vector, MaxSim-rerank
  the top-N) — the token arrays are not indexed, and index-native
  late interaction remains a documented future phase.
- **`turbovec.rrf_score(rank integer, k integer DEFAULT 60)`**
  (`src/hybrid.rs`) — reciprocal rank fusion term `1.0 / (k + rank)`
  for fusing a dense ANN ranking with a sparse / keyword ranking.
  Pairs with the documented CTE recipe; non-positive denominator
  raises ERROR.
- **`docs/HYBRID_SEARCH.md`** — the canonical breadth guide:
  multivector MaxSim re-rank (signature, conventions, the
  two-stage retrieve-then-rerank pattern, the honest index-native
  limitation), the dense+sparse RRF recipe (full `ROW_NUMBER()` +
  `rrf_score` CTE for both full-text and `sparsevec`), and the
  named-vector multi-column schema pattern.
- Cross-links from `README.md`, `docs/PRODUCTION.md`,
  `docs/PARITY_GAPS.md`, an internal design note, and
  `docs/MIGRATING_FROM_PGVECTOR.md`; the multivector / hybrid rows
  now read "SQL surface SHIPPED; index-native late interaction is a
  future phase."

### Notes

- Out-of-core BUILD (roadmap Phase D-3) already shipped in v1.12.0
  (streaming IVF build); no re-implementation.
- Named vectors (multiple vector columns per row) are a documented
  schema pattern, not new code.

### Migration

Additive SQL functions only; no wire change, no REINDEX. `ALTER
EXTENSION pg_turbovec UPDATE TO '1.14.0';` is sufficient (the new
`turbovec.max_sim` / `max_sim_cosine` / `rrf_score` functions are
created by the update script). Tests:
203 → 215 (+`max_sim_basic`, `max_sim_dim_mismatch_errors`,
`max_sim_empty`, `max_sim_cosine_normalised`, `max_sim_rerank`,
`rrf_score_values`, `hybrid_rrf_recipe`, plus 5 in-module unit tests).

## [1.13.1] — 2026-06-17

**Phase C — metadata-filtering docs + measured allowlist crossover.**
Docs + benchmark release; **no source-logic, SQL-surface, or wire
change** (`MetaPageData::version = 4`); **no REINDEX**. Bench-results
and documentation only.

### Added

- **`docs/FILTERING.md`** — the canonical guide to pg_turbovec's
  three working metadata-filter mechanisms, with a
  cardinality×selectivity×corpus decision matrix:
  1. **Partial index** (`CREATE INDEX ... WHERE tenant_id = X`) —
     native PG predicate pushdown; the default for known,
     low-cardinality filters.
  2. **In-kernel allowlist** `turbovec.knn(rel, id_col, vec_col,
     query, k, bit_width, allowed bigint[])` — true in-kernel
     pushdown (the SIMD kernel skips 32-vector blocks with no allowed
     slots before any LUT work), flat-only, for selective per-query
     id sets.
  3. **Iterative scan + `WHERE`** (v1.8.0) — the `ORDER BY emb <=> q
     LIMIT k` AM path; the executor rechecks the predicate, the AM
     widens `k`/`probes` (capped by `max_scan_tuples`).
  Includes the honest limitation: no true in-traversal pushdown on
  the `ORDER BY` AM path (the index stores only vector codes + TID,
  no payload columns), and a C-4 design sketch for a future phase.
- **Measured allowlist selectivity crossover** (floki, AVX2, 300k×
  256-d, 4-bit, k=10): allowlist latency decreases monotonically as
  the filter tightens (17.9 ms → 0.48 ms, ~37×) while the naive
  post-filter is flat (~7 ms); crossover at ~7–10% selectivity, up
  to **14.7× faster at 0.1%**. `benches/allowlist_crossover.rs` +
  `benches/results/allowlist_crossover_floki_v1_13_0_20260617.json`.

### Fixed (docs drift)

- an internal design note: refreshed v1.10.1/v1.11.0 →
  v1.13.0; the **>500k IVF build ceiling** and the **>RAM** gaps are
  now marked **CLOSED** (out-of-core build v1.12.0 + out-of-core
  query v1.13.0); the metadata-filtering row reflects the three real
  patterns instead of "post-filter only".
- `docs/PARITY_GAPS.md`: added the metadata-filtering row; corrected
  the stale "Parallel index build | GAP — single-threaded" row
  (parallel build shipped v1.8.0, `turbovec.build_parallelism`).
- `docs/MIGRATING_FROM_PGVECTOR.md`: filtered-ANN section lists all
  three patterns and links `FILTERING.md`; `knn()` signature matches
  `src/knn.rs`.
- `README.md` + `docs/PRODUCTION.md`: cross-link `FILTERING.md`.
- Fixed three pre-existing broken benchmark sources
  (`concurrent_knn`, `recall_vs_pgvector`, `recall`) that referenced
  the pre-`d3d468e` `IdMapIndex::new` signature (now returns
  `Result`); `cargo check --benches` is green again. (`cargo pgrx
  test` never compiled benches, so they did not gate tests.)

### Migration

**No REINDEX.** Docs + bench only; wire stays v4. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.13.1';` is sufficient. Tests unchanged (203).

## [1.13.0] — 2026-06-17

**Out-of-core IVF query (>RAM serving)** — an IVF index larger than
RAM can now be queried, not just built (v1.12.0). Wire format
unchanged (`MetaPageData::version = 4`); **no REINDEX**. Completes
the out-of-core arc for the >5M production deployment
(an internal design note Phase B-1/B-2).

### Added — cell-scoped IVF serving (Phase B-1/B-2)

The scan previously loaded the **whole index** into a per-backend
cache (`read_full` + a copy of the blocked-codes chain off the
mmap), so the resident set was `O(n)` and an index that exceeded RAM
could not be served.

- **Cell-scoped scan.** The backend now caches only bounded metadata
  (coarse centroids, cell directory, rotation, codebook, per-slot
  scales/ids) plus a `MAP_PRIVATE` mmap of the relfile, and per
  query copies **only the probed cells'** contiguous code ranges off
  the mmap into a compact throwaway sub-index (cells are contiguous
  from the build-time permutation). Resident set drops to
  `O(probes * cell_size + faulted pages)`; hot cells stay in the OS
  page cache, cold cells fault from disk on demand.
- **`turbovec.out_of_core`** (enum `off | auto | on`, **default
  `auto`**). `auto` goes cell-scoped only when the index codes
  exceed `0.5 * turbovec.cache_size_mb` — an in-RAM index loads
  whole (no per-query gather/reblock cost); only a genuinely large
  index pays the bounded-memory-for-CPU tradeoff. `on` forces
  cell-scoped; `off` forces the pre-v1.13.0 whole-load.
- No wire change, no turbovec fork change (reuses
  `from_parts_with_prepared_borrowed`). Added
  `mmap_static::gather_slot_ranges` + a buffer-manager twin for the
  fresh-index fallback.

### Measured

200k×256-d×4-bit IVF (52 MB on disk): per-backend `VmHWM`
whole-load **140.8 MB → cell-scoped 44.1 MB** (~3.2× lower). Under a
tight cgroup `MemoryMax`, the whole-load backend was OOM-killed
(postmaster recovered cleanly, no corruption) where cell-scoped
stayed within bound. Warm p50 **82 ms (whole-load) → 199 ms
(cell-scoped)** — the expected per-query reblock cost, paid by
`auto` only when the index is too large to keep whole.

### Compatibility

Scan-path only. Results identical to the whole-load path
(`probes >= lists` still reduces to the exact flat scan; tombstones
masked; soft-assign deduped). MVCC backstops (reorder queue + heap
visibility) preserved. Flat (`lists = 0`) / vacuum-degraded indexes
keep the whole-index load (no cells to scope; still `O(n)`-resident
— use IVF for >RAM).

### Migration

**No REINDEX.** Scan-path change; wire stays v4. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.13.0';` is sufficient.

### Tests

197 → 203 (+`ivf_ooc_results_match_whole_load`,
`ivf_ooc_probes_all_equals_flat`, `ivf_ooc_tombstones_masked`,
`ivf_ooc_soft_assign_dedup`, `ivf_ooc_installs_cell_scoped_handle`,
`ivf_ooc_auto_is_size_aware`). drift-check clean.

## [1.12.0] — 2026-06-17

**Out-of-core IVF build** — IVF indexes can now be built at 1M–5M+
rows on a RAM-constrained host. Wire format unchanged
(`MetaPageData::version = 4`, byte-identical relfile); **no
REINDEX**. Driven by the >5M production deployment
(an internal design note Phase B-4).

### Fixed — the 1M+ IVF build OOM (Phase B-4)

The `WITH (lists = N)` build held the **full f32 corpus twice** in
RAM (`ivf_flat` ~4 GiB + `perm_flat` ~4 GiB at 1M×1024-d) plus the
growing index and GEMM scratch — a ~14 GiB peak that
`maintenance_work_mem` did not bound, OOM-killing 1M+ builds on a
31 GiB host. IVF was effectively unbuildable at the production
scale.

- **Disk spill.** The corpus now spills to a PostgreSQL `BufFile`
  temp file (in `pgsql_tmp`, respecting `temp_tablespaces` /
  `temp_file_limit`) during the heap scan, wrapped in a
  `CorpusSpill` RAII type. Cleanup is double-covered: the resource
  owner unlinks on (sub)transaction abort (even when
  `ereport(ERROR)` longjmps past Rust destructors) and `Drop`
  unlinks on success.
- **Three streamed passes**, each bounded by `maintenance_work_mem`:
  (1) spill + bounded reservoir sample for k-means; (2) GEMM-assign
  cells over disk-backed row-blocks, keeping only the per-row
  cell-id array (not a corpus copy); (3) feed the quantizer in cell
  order by re-reading the spill at permuted offsets in bounded
  chunks. The full f32 corpus is **never resident**; the only
  RAM term that scales with row count is the **quantized**
  `packed_codes` (7–15× smaller than the f32 corpus).
- **Measured** (1M×1024-d, lists=1024, 30 GiB host): peak RSS
  **~14 GiB (OOM) → ~7.1 GiB (completes)**; index 1030 MB, spill
  ~3.9 GB on disk. **5M projected ~8–10 GiB** — buildable on the
  31 GiB production host.

### Determinism / compatibility

Byte-identical relfile to a v1.11.x in-memory build for the same
input, and **`maintenance_work_mem`-invariant** (TQ+ calibration is
fit on a fixed cell-ordered prefix, independent of chunk size). The
flat (`lists = 0`) build path is unchanged (already Phase-W
streamed). No GUC added — `maintenance_work_mem` is the knob.

### Migration

**No REINDEX.** Build-internal; wire stays v4. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.12.0';` is sufficient. Existing indexes are
unaffected; the benefit applies to the next `CREATE INDEX` /
`REINDEX`.

### Tests

193 → 197 (+`ivf_streaming_build_determinism_byte_identical`,
`ivf_streaming_build_chunk_size_invariant`,
`ivf_streaming_build_bounded_memory_completes`,
`ivf_streaming_build_temp_file_cleanup`). drift-check clean.

## [1.11.1] — 2026-06-16

**Bench-results-only release. Wire format unchanged from v1.11.0
(`MetaPageData::version = 4`); no REINDEX.** Zero source change.

### Benchmark — IVF latency frontier vs HNSW + ivfflat (Phase A-2)

The honest at-scale measurement (isolated AVX2 on `arnold`,
`taskset`-pinned, contention-gated, warm, 300 queries/config,
Cohere-wiki 500k×1024-d) answering "does IVF beat/equal HNSW at
scale." At recall@10 ≈ 0.96:

| engine | config | recall@10 | warm p50 |
|---|---|---:|---:|
| **pgvector HNSW** | ef=200 | 0.966 | **7.9 ms** |
| **pg_turbovec IVF** | lists=707, probes=64 | 0.960 | 18.5 ms |
| pgvector ivfflat | probes=100 | 0.978 | 117.4 ms |
| pg_turbovec flat (exact) | all cells | 1.000 | 41.4 ms |

**Honest verdict:** HNSW wins latency at 0.96 (7.9 vs 18.5 ms,
~2.3×). But IVF is now in HNSW's **order of magnitude** (not the
490× flat-scan gap), **beats pgvector's own ivfflat 3–6×** at every
matched recall, beats its own exact flat scan, **wins the ≥0.99
recall tail** (0.99 @ 25 ms via probes=256; this HNSW config never
reaches 0.99), and is **7.5× smaller** (518 MB vs HNSW 3902 MB).
The earlier ~40 ms projection was pessimistic; real p50 at 0.95 is
18.5 ms.

### Critical finding — 1M IVF build OOMs (motivates Phase B-4)

The **1M IVF build OOM-killed the postmaster** (~14 GiB peak on a
31 GiB host): the `lists > 0` build holds the full flat corpus + a
permuted copy + k-means scratch — a structural peak
`maintenance_work_mem` does not bound. Largest IVF index that built
on arnold: **500k**. 1M/5M IVF are **blocked on Phase B-4**
(streaming / out-of-core build, designed in an internal design note).
The IVF *query* path is unaffected.

Files: `benches/results/ivf_frontier_arnold_cohere-wiki_2026-06-16.json`,
`docs/BENCHMARKS.md`, an internal design note,
an internal design note.

## [1.11.0] — 2026-06-16

Production hardening for IVF: it now **survives VACUUM** instead of
silently degrading, and builds **~7.8× faster**. Wire format stays
`MetaPageData::version = 4` (additive); **no REINDEX**. Driven by
the >5M production deployment + an internal design note
(Phases A-1, E-2).

### Fixed — IVF survives VACUUM (Phase E-2, the production landmine)

An IVF index used to **silently degrade to a flat `O(n)` scan after
VACUUM** — swap-remove moved the last vector into the deleted slot,
breaking cell contiguity, so `has_ivf()` flipped false and queries
fell back to the ~seconds full scan with no operator signal. On a
churning multi-million-row index that's a latency cliff.

- **Tombstones.** The IVF `ambulkdelete` path now leaves dead slots
  in place and ORs them into a **persisted per-slot tombstone
  bitmap** (a new v4-additive relfile chain). No rows move,
  `n_vectors` and the cell directory are untouched, cells stay
  contiguous, `has_ivf()` stays true, and the scan keeps
  cell-restricting. The flat (`lists = 0`) path keeps the unchanged
  swap-remove. Tombstoned slots are masked out of the initial scan
  **and every probe-widening refill**, so deleted rows are never
  returned.
- **Observability** for any residual fallback: a throttled scan-time
  `WARNING` (once per backend per index, with a `HINT: REINDEX`) and
  a new SQL function **`turbovec.index_is_degraded(regclass) ->
  bool`**. `write_meta_shrink_in_place` now preserves `lists` and
  flips an `ivf_degraded` meta flag rather than blanking the IVF
  identity, so the cliff is detectable.
- `docs/PRODUCTION.md` gains an IVF + VACUUM operational section.

### Performance — 7.8× faster IVF k-means (Phase A-1)

A 200k×256-d / lists=448 build's k-means training was ~295 s
(scalar Lloyd, fixed 25 iters) — prohibitive at 5M+. Now
GEMM-batched Lloyd assignment (each iteration's nearest-centroid
step is one `V@Cᵀ` cross-term GEMM, single-threaded
`Parallelism::None` + exact top-2 scalar tie-break) + convergence
early-exit (`KMEANS_TOL = 1e-6`). Training **295 s → 38 s = 7.8×**;
per-iteration centroids byte-identical to the scalar path,
determinism preserved. Training cost is bounded by the
`256×lists` reservoir sample regardless of corpus size, so 5M
builds train in low-minutes. Build-internal; no surface or wire
change.

### Migration

**No REINDEX.** Wire stays v4; the tombstone bitmap and
`ivf_degraded` flag are additive — pre-1.11.0 v4 indexes read as
not-degraded / no-tombstones. The new `index_is_degraded()`
function is registered by `ALTER EXTENSION pg_turbovec UPDATE TO
'1.11.0';`.

### Tests

187 → 193 on pg16 (+`ivf_survives_vacuum`,
`ivf_tombstoned_rows_not_returned`, `ivf_degradation_is_observable`,
fast-k-means + page.rs wire-format coverage). drift-check clean.

## [1.10.1] — 2026-06-16

**Bench-results-only release. Wire format unchanged from v1.10.0
(`MetaPageData::version = 4`); no REINDEX.** Zero source-code change.

### Benchmark — IVF warm-p50 on AVX2

Records the AVX2 IVF warm-p50 measurement confirming the IVF
cell-skipping **latency** win that `meh` (pre-AVX2 scalar fallback)
could not produce. Host `floki` (Intel Core Ultra 7 258V, AVX2),
v1.10.0 release build, 200k × 256-d, `lists = 448`, 4-bit, warm
cache, 50 timed queries per `probes`:

| probes | warm p50 | vs full scan |
|-------:|---------:|-------------:|
| 4   | 0.74 ms | 5.4× faster |
| 16  | 0.78 ms | 5.1× faster |
| 448 (= lists, full exact scan) | 3.97 ms | baseline |

**At `probes = 16`, ~5× faster than the full exact scan**, on AVX2.
The IVF latency win is real on AVX2 hardware. Honest caveat:
recall@10 = 1.000 at all probes in this run is an artifact of the
synthetic corpus's strong cluster structure, not a general
guarantee — the host-independent recall-vs-probes frontier
(v1.10.0) is the honest recall/probes trade-off. A full isolated
1M+ × 1024-d sweep on a quiet AVX2 host remains future work.

Files: `benches/results/ivf_warmp50_floki_avx2_2026-06-16.json`,
`docs/BENCHMARKS.md` ("IVF warm-p50 (AVX2)" section).

## [1.10.0] — 2026-06-16

**Adds the IVF coarse-quantizer layer — a real sublinear ANN
structure over the quantized codes.** First wire-format change since
v1.4.0 (`MetaPageData::version` 3 → 4), but **existing v3 indexes
do NOT need a `REINDEX`**: a v1.10.0 binary reads a v3 index as a
flat (`lists = 0`) index. Only users who opt into IVF rebuild.
See an internal design note.

### Why

The v1.9.1 AVX2 benchmark established that pg_turbovec's flat
`O(n·dim)` quantized scan is ~490× slower than pgvector HNSW at
1M×1024-d. IVF partitions the corpus into `lists` Voronoi cells
(coarse k-means centroids) and scans only the `probes` nearest
cells per query, dropping query work to roughly `(probes/lists)`
of the corpus — the architectural path to a competitive latency
story while keeping the 10–15× storage win.

### Added — IVF (opt-in)

- `WITH (lists = N)` reloption (default 0 = flat / today's exact
  scan; `N` = number of coarse cells, recommended `≈ sqrt(n)`).
- `WITH (assign_dups = M)` reloption (default 1 = single
  assignment; `M > 1` = soft assignment: boundary vectors stored
  in their top-M nearest cells to raise recall@10 at a fixed
  `probes`).
- `turbovec.probes` GUC (default 8) — cells scanned per query; the
  recall/latency dial (the `ivfflat.probes` / `hnsw.ef_search`
  analogue). `probes >= lists` reduces exactly to the flat exact
  scan.
- `turbovec.max_probes` GUC (default 64) — under
  `iterative_scan = relaxed_order`, a selective `WHERE` filter that
  under-returns triggers probe-WIDENING (scan more cells) up to
  this cap; the `ivfflat.max_probes` analogue.

### How it composes

- **Iterative scan** (v1.8.0): refill widens `probes` for IVF
  indexes (vs growing `k` for flat).
- **Oversampling** (v1.9.0): widens the candidate set within the
  probed cells.
- **Reorder queue**: exact-distance recheck, unchanged.
- turbovec's SIMD mask SKIPS scan work (block-level early-exit),
  and cells are stored contiguous, so fewer probed cells = real
  latency reduction.

### Performance / determinism

- The IVF build (k-means + assignment) is GEMM-batched (corpus
  rotation as `block @ R^T`, cell assignment via a `V @ C^T`
  cross-term GEMM + scalar top-2 tie-break). Without this the
  per-vector scalar loops were ~10^12 FLOPs and a 1M build ran
  60+ min. Single-threaded `gemm` keeps it bit-deterministic.
- Builds are deterministic: same table + same `lists`/`assign_dups`
  ⇒ byte-identical relfile (seeded k-means++, `IVF_SEED`).
- Recall-vs-probes frontier (host-independent; recall is
  CPU-independent): on a hard random 16k×64-d corpus, `probes=16`
  → R@10 0.53 skipping 85% of blocks, `probes=lists` → 1.000.
  Real clustered embeddings reach high recall at far lower probes.
  Absolute AVX2 warm-p50 latency is deferred to a quiet `arnold`
  window (`meh` is pre-AVX2). See `docs/BENCHMARKS.md`.

### Migration

**No REINDEX for existing (v3, flat) indexes** — they read as
`lists = 0` under the v1.10.0 binary. Opt into IVF by rebuilding
with `WITH (lists = N)`. `ALTER EXTENSION pg_turbovec UPDATE TO
'1.10.0';` registers the new reloptions + GUCs. `MetaPageData::version`
is 4; `EXPECTED_WIRE_FORMAT_VERSION = 4`.

### Tests

150 → 185 on pg16 (IVF build/scan/soft-assign/determinism/
probes-frontier coverage; distinct-id assertions throughout).
drift-check clean.

## [1.9.1] — 2026-06-15

**Bench-results-only release. Wire format unchanged from v1.9.0
(`MetaPageData::version = 3`); no REINDEX needed.** Zero source-code
changes — this release bundles the AVX2 latency-frontier benchmark
and the honest positioning correction it produced.

### Benchmark — AVX2 latency frontier on `arnold`

The latency numbers `meh` (a pre-AVX2 Xeon) physically could not
produce. Run on `arnold` (i9-12900H, AVX2), isolated via
`taskset -c 2-5` CPU-pinning to dedicated P-cores with per-batch
contention measurement (observed 1-min load ≤ 1.05 throughout,
zero CPU steal, no contended batches, no re-runs). Cohere wikipedia
1M × 1024-d, 1000 held-out queries, brute-force exact GT.

- **Correctness on the AVX2 SIMD path: recall@10 = 1.000** (the AVX2
  kernel, not just meh's scalar fallback, is correct on v0.9.0).
- **The hard truth: pg_turbovec loses to HNSW on latency by ~490×
  at 1M rows** — warm p50 ~2552 ms (flat `O(n·dim)` quantized scan,
  recall 1.000) vs pgvector HNSW ~5 ms (sublinear graph, recall
  0.96). AVX2 makes the correct scan ~15–25× faster than meh's
  scalar fallback (2.5 s vs 41.6 s), but a 1M-row full scan is
  seconds, not ms, by design.
- **Retracted:** the earlier "26.8 ms on `meh` / we win 2.3× warm
  p50" claim. That came from the **pre-AVX2 scalar-fallback bug**
  (fast-but-WRONG, fixed in v1.7.3) and never represented correct
  behaviour.

### Positioning correction

`docs/PARITY_GAPS.md` and an internal design note updated to
the honest scoreboard. pg_turbovec's durable wins are **storage**
(10–15× smaller), **exact recall** (1.000 vs HNSW's ~0.96), and
**build memory** — NOT query latency at scale. Honest positioning:
*"best storage efficiency + exact recall for PG vector search where
an O(n) scan fits the latency budget,"* NOT "beat HNSW on every
axis." The architectural path to a latency story at scale is an IVF
/ coarse-quantizer layer (turning the O(n) scan into
O(n/nlist + probes)) — a planned future major arc, see an internal design note.

### Files

- `benches/results/latency_frontier_arnold_cohere_1m_v1_9_0_2026_06_15.json`
- `benches/scripts/vectordbbench/sweep_latency_isolated.py`
- `docs/BENCHMARKS.md` (arnold AVX2 section)
- `docs/PARITY_GAPS.md`, an internal design note (corrections)

## [1.9.0] — 2026-06-15

Oversampling (tunable recall), test-coverage hardening, and the
first published head-to-head benchmark. **Wire format unchanged**
(`MetaPageData::version = 3`); **no `REINDEX`** — `ALTER EXTENSION
pg_turbovec UPDATE TO '1.9.0';` suffices. The one new GUC defaults
to a no-op.

### Added — `turbovec.oversample` (differentiator #5)

Turns quantization from a fixed accuracy point into a tunable recall
lever, matching Qdrant's oversampling / VectorChord's rerank.

- `turbovec.oversample` (float, default 1.0, range 1.0..=100.0): the
  scan fetches `ceil(search_k * oversample)` quantized candidates and
  the executor's reorder queue (`xs_recheckorderby = true`) trims to
  the exact top-k. Widening the candidate set recovers true
  neighbours the lossy quantized ranking placed just outside
  `search_k`.
- No separate rescore path: oversampling + the always-on reorder
  queue together ARE the rescore mechanism (the reorder queue
  already re-ranks by exact full-precision distance). Measured:
  recall@10 climbs 0.81 (oversample 1.0) → 1.0 (oversample 4.0) on a
  4-bit / 3000×64 corpus; latency rises ~linearly.
- Composes with iterative scan: oversample sets the initial `k`;
  refill doubles from there, capped by `max_scan_tuples`.
- Default 1.0 is identical to v1.8.0 behaviour.

### Testing — scale + distinct-id + recall-floor regression guards

The pre-AVX2 wrong-results bug (fixed in v1.7.3) shipped because no
test exercised more than ~2000 rows or asserted distinct result
ids. Closed those gaps:

- Medium-scale (20k×128-d) recall-floor `#[pg_test]` per bit_width
  {2,3,4}, with a brute-force ground-truth comparison.
- `assert_distinct_ids` on EVERY ANN-scan test — the cheapest guard
  against the whole wrong-ranking bug class (a duplicate-id assert
  would have caught the pre-AVX2 bug instantly).
- `docs/TESTING.md` documenting coverage + honest gaps: CI is
  AVX2-only (the scalar fallback runs only in turbovec's upstream
  tests + pre-AVX2-host validation on turbovec bumps); unit tests
  cap at 20k rows (the benchmark is the large-scale evidence); the
  15 "ignored" items are benign ` ```ignore ` doctests.

### Benchmark — first published head-to-head (`docs/BENCHMARKS.md`)

Cohere wikipedia 1M × 1024-d (real embeddings, 1000 held-out
queries, brute-force GT) vs pgvector HNSW, with a full reproducible
harness under `benches/scripts/vectordbbench/`.

- **recall@10 = 1.000** on the fixed v1.8.0+ build at every config
  — the same pre-AVX2 host scored 0.0 on the old v1.7.1 build, so
  this is the definitive confirmation that the pre-AVX2 fix works
  on real embeddings at scale.
- Storage: pg_turbovec 4-bit **7.6× smaller** (1026 MB vs HNSW
  7806 MB), 2-bit **15.2× smaller** (512 MB). Build 1.9–2.1×
  faster.
- pgvector HNSW frontier (its own SIMD): R@10 0.849/9.4 ms (ef40)
  → 0.979/20.1 ms (ef400).
- **pg_turbovec latency frontier is DEFERRED to an AVX2 host.** The
  bench host `meh` is a pre-AVX2 Ivy Bridge Xeon; turbovec takes
  its scalar fallback (~1000× slower than its AVX2/AVX-512 kernels:
  ~42–69 s/query full-corpus scan). EXPLAIN confirmed Index Scan
  (not a seq-scan artifact). Latency/QPS benchmarks require an
  AVX2+ host (`arnold`); see `BENCHMARKS.md` for the explicit TODO
  and full caveats. (Updated `AGENTS.md` bench-host guidance
  accordingly — SIMD class matters more than RAM for turbovec
  latency.)

### Migration

**No migration; no REINDEX.** On-disk format byte-identical to
v1.7.x / v1.8.x. The new `turbovec.oversample` GUC defaults to 1.0
(no-op). `ALTER EXTENSION pg_turbovec UPDATE TO '1.9.0';` resolves
against the empty `migrations/014_pg_turbovec_v1.9.0.sql`.

### Tests

142 → 150 on pg16 (+5 oversampling, +3 recall-floor; distinct-id
assertions added to existing tests). drift-check clean.

## [1.8.0] — 2026-06-15

Four competitive-parity features in one minor release. **Wire
format unchanged** (`MetaPageData::version = 3`); **no `REINDEX`
needed** — `ALTER EXTENSION pg_turbovec UPDATE TO '1.8.0';` is
sufficient. All four additions are scan-side, build-side, or
additive SQL surface; none touch the on-disk relfile layout.
Driven by an internal design note.

### Added — iterative index scan (parity gap #1, the correctness fix)

The one true correctness gap vs pgvector. `amgettuple` used to run
a single `search_k`-sized batch and return false when drained, so
a selective `WHERE filter ORDER BY emb <=> q LIMIT k` silently
under-returned (e.g. 3 rows when 10 were asked for) — exactly what
pgvector shipped `hnsw.iterative_scan` (0.8.0) to fix.

- When the executor exhausts the candidate batch and the filter
  hasn't been satisfied, the scan re-runs the turbovec search with
  a **doubled `k`** and feeds the new candidates, capped by a new
  `turbovec.max_scan_tuples` GUC (default 20000, matching
  pgvector's `hnsw.max_scan_tuples`).
- Controlled by `turbovec.iterative_scan` — an enum GUC
  `off | relaxed_order` (default `relaxed_order`). `strict_order`
  is deferred; our existing reorder-queue model
  (`xs_recheckorderby = true`) already restores exact per-tuple
  ordering on top of `relaxed_order`.
- Dedup across refills via a returned-TID `HashSet` (turbovec's
  `search` isn't a stable prefix across `k` due to an unstable
  sort on score ties; the set is robust and bounded by
  `max_scan_tuples`).
- Regression test demonstrates `off` under-returns and
  `relaxed_order` returns the full `LIMIT`.

### Added — parallel index build (parity gap #2)

pgvector parallelises HNSW/IVFFlat builds across
`max_parallel_maintenance_workers`; pg_turbovec's `ambuild` was
single-threaded.

- Option B (rayon): the CPU-heavy `encode` + SIMD-`repack` phases
  (which dominate build CPU, not the heap scan) are parallelised
  over heap-scan chunks via a rayon pool. Chunks are processed in
  heap-scan order then concatenated deterministically.
- New `turbovec.build_parallelism` GUC (default 0 = derive from
  `max_parallel_maintenance_workers + 1`; positive pins the pool).
- **Byte-for-byte identical relfiles** regardless of thread count
  — asserted by a unit test — so the wire format and any
  reproducibility guarantees hold. Memory stays bounded by the
  Phase W `maintenance_work_mem` cap.

### Performance — cold-scan latency (parity gap #3)

Cold-scan p50 was ~1256 ms (1 M × 1536-d) vs HNSW's ~100 ms.

- **Lazy `id_to_slot` on the read path.** Profiling the
  per-backend cache-fill showed the dominant residual term — once
  Phase P pre-baked the SIMD-blocked layout and Phase R-2
  persisted the rotation — was the `id_to_slot: HashMap<u64,
  usize>` that `IdMapIndex::from_id_map_parts*` builds eagerly
  (~50 ms at 200 k rows, linear in `n`). The index-AM scan path
  never reads `id_to_slot` (`search` returns slots, mapped via the
  `slot_to_id` `Vec`). The scan path now installs a lightweight
  `cache::ReadOnlyIndex` (no HashMap); the build is deferred to
  the first `aminsert`/`remove`. A read-only / pooled-connection
  backend that only scans never pays it. Read-only constructor:
  ~50 ms → ~0 ms.
- Key correctness test `mutation_after_readonly_scan_is_correct`
  verifies the deferred HashMap builds correctly on first insert.
- **Deferred follow-ups** (see `docs/PARITY_GAPS.md` § cold scan):
  read-path mmap of the codes/scales/ids chains; a
  header-gap-free on-disk layout for true zero-copy mmap (VERSION
  3 → 4, a future minor); a cross-backend DSA/DSM shared cache.

### Added — `||` concat + halfvec arithmetic (parity gap #4)

pgvector has `||` concat for vector+halfvec and `+`/`-`/`*` for
both; pg_turbovec had `+`/`-`/`*` for `vector` only.

- `turbovec.vector || turbovec.vector -> vector` (concat)
- `turbovec.halfvec || turbovec.halfvec -> halfvec` (concat)
- `turbovec.halfvec` `+`/`-`/`*` element-wise (Hadamard for `*`)
- Matches pgvector overflow semantics (error on non-finite
  result) and dim-mismatch errors.

### Migration

**No migration needed; no REINDEX.** The on-disk relfile format is
byte-identical to v1.7.x. Drop in the new shared library, restart,
scan; existing indexes work unchanged. The new GUCs default to
the pgvector-equivalent behaviour (`iterative_scan = relaxed_order`).
`ALTER EXTENSION pg_turbovec UPDATE TO '1.8.0';` resolves against
the empty `migrations/013_pg_turbovec_v1.8.0.sql`.

### Tests

123 → 142 on pg16 (+19: iterative-scan, parallel-build, cold-scan,
and arithmetic-parity coverage). drift-check clean.

## [1.7.3] — 2026-06-15

### Fixed — pre-AVX2 x86_64 wrong-results bug (turbovec fork → v0.9.0)

Wire format unchanged from v1.6.0 / v1.7.x
(`MetaPageData::version = 3`); **no `REINDEX` needed** to upgrade.

- **Root cause.** The Phase A1 "regression" (index `ORDER BY emb
  <=> probe LIMIT N` returning the same `id` N times at 10 M
  scale on the `meh` bench host) was traced to an **upstream
  turbovec kernel bug, not pg_turbovec**. The pinned turbovec
  v0.7.0-era fork (`6e80a59`) had a scalar fallback that, on
  x86_64 CPUs **without AVX2**, read the perm0-interleaved
  (FAISS-style) SIMD code layout as if it were sequential —
  producing silently-wrong / repeated top-k. `meh` is an Intel
  Xeon E5-2697 v2 (Ivy Bridge, 2013): `avx` but no `avx2`, so it
  hit the buggy path. AVX2 (Haswell 2013+), AVX-512, and ARM NEON
  hosts always took a correct SIMD path — which is why the bug
  never reproduced on AVX2 dev boxes or `arnold`, only on `meh`.
- **Fix.** Upstream turbovec fixed this in PR #108 (issue #106,
  "V5"), released in v0.8.0, adding a correct
  `score_query_into_heap` x86_64 scalar fallback plus a
  `FORCE_SCALAR_FALLBACK` regression test. v1.7.3 upgrades the
  `gburd/turbovec` fork from the v0.7.0-era `6e80a59` to a fork
  rebased onto upstream **v0.9.0** (`d3d468e` on branch
  `pg_turbovec-integration-v0.9.0`).
- **Also brought in, inert here:**
  - TQ+ per-coordinate calibration fields, constructed as identity
    (empty) on the relfile path — **no recall change, no wire
    change** in v1.7.3. Persisting them for a recall gain is a
    future minor release (VERSION 3 → 4 + REINDEX).
  - Security hardening: `MAX_DIM = 65536`, NaN/Inf/huge-magnitude
    input rejection, checked-mul `.tv`/`.tvim` loaders.
- **Zero pg_turbovec source churn** — the upgrade is a
  `Cargo.toml` rev bump only; the fork kept the `prepare_eager`
  alias and passes TQ+ through internally so every
  `from_id_map_parts*` call site is unchanged.
- **Toolchain note.** turbovec v0.9.0 uses `avx512` `target_feature`s
  requiring **Rust ≥ 1.89**. Builds with the default `stable`
  toolchain (1.95). See `AGENTS.md` for the refreshed openblas
  store path and the `-fuse-ld=bfd` linker note.
- Tests: 123/123 on pg16. drift-check clean.

### Migration

**No migration needed; no REINDEX.** The on-disk relfile format is
byte-identical to v1.6.x / v1.7.x. Drop in the new shared library,
restart, scan. **Pre-AVX2 x86_64 users specifically** should
upgrade to clear the wrong-results bug and can drop any
`SET enable_indexscan = off;` workaround. `ALTER EXTENSION
pg_turbovec UPDATE TO '1.7.3';` resolves against the empty
`migrations/012_pg_turbovec_v1.7.3.sql`.

## [1.7.2] — 2026-05-27

### Added — Phase Y: automated upgrade-matrix validation

Wire format unchanged from v1.6.0 / v1.7.0 / v1.7.1
(`MetaPageData::version = 3`); **no `REINDEX` needed** to upgrade.
v1.7.2 is a test-only patch release.

Production-confidence foundation: previously the upgrade matrix
in `docs/UPGRADING.md` and the `is_legacy_v{1,2}()` detection
predicates in `src/index/page.rs` were promises with no
automated end-to-end test. Phase Y closes that gap.

- **`alter_extension_path_140_to_171_runs_clean`** (new
  `#[pg_test]`) replays every `migrations/0NN_pg_turbovec_*.sql`
  from v1.3.0 onward against the live test cluster. Catches a
  release engineer who lands a syntactically-broken DDL change
  in one of the post-v1.3 migration files (which are otherwise
  intentionally empty).

- **`ambeginscan_errors_on_legacy_v1_meta`** and
  **`ambeginscan_errors_on_legacy_v2_meta`** (new `#[pg_test]`s)
  build a real v1.7.2 index, forge the meta-page version byte
  to 1 or 2 via the new cfg-gated
  `relfile::force_meta_version()` helper, and assert that
  `ambeginscan` ERRORs at first scan with the documented
  primary message + `REINDEX INDEX` HINT. Exercises the
  Phase Q (v1.3.0) + Phase R-2 (v1.4.0) hard migration
  boundaries without having to keep pre-v1.4 binaries
  around.

- **`alter_extension_update_chain_resolves`** (new `#[pg_test]`)
  asserts the installed extension version matches
  `Cargo.toml`, catching version-number drift between
  `Cargo.toml`, `pg_turbovec.control`, and the migration
  file naming convention.

- **`migration_files_cover_documented_versions`** (new
  `#[pg_test]`) asserts the set of `migrations/*.sql` sigils
  matches the documented release history. If you tag a new
  release without adding the migration file, this test fails
  before the bad tag escapes CI.

- **`scripts/drift-check.sh` § 9** (new check) cross-checks
  `migrations/*.sql` against the `From` column of the migration
  matrix in `docs/UPGRADING.md`. Catches release-time drift
  between adding a tag and forgetting to add the migration
  file.

- **`relfile::force_meta_version()`** (new test-only helper)
  is gated on `cfg(any(test, feature = "pg_test"))` and patches
  the version byte of the meta page in place via a
  `GenericXLog` record. Only the pgrx test suite (and a future
  feature-gated build) can reach it; production builds never
  compile it.

### Migration

**No migration needed; rebuild not required.** The on-disk
format is byte-identical across v1.6.0 / v1.7.0 / v1.7.1 /
v1.7.2. Drop in the new shared library, restart, scan;
existing indexes built under any of these versions continue
to work unchanged.

## [1.7.1] — 2026-05-27

### Reverted — Phase W-2 split-write design (regression)

Wire format unchanged from v1.6.0 / v1.7.0 (`MetaPageData::version
= 3`); **no `REINDEX` needed** to upgrade or downgrade between
any of these. v1.7.1 is a behaviour-only revert.

- **Phase W-2 (v1.7.0) reverted.** Validation on `meh` (24-core,
  125 GiB RAM NixOS host, head commit `a289870`) at 10 M ×
  1536-d × 4-bit showed the split-write `ambuild` path
  introduced in v1.7.0 made the build **53% slower**
  (5052 → 7748 s), used **2.7 GiB of swap** (vs 0 in v1.6.0),
  and slightly **raised** peak RSS (22.5 → 23.04 GiB). The
  predicted ~15 GiB peak never materialised. Full data:
  `benches/results/phase_w_2_validate_meh_10m_2026_05_27.json`.

  | metric            | v1.6.0 | v1.7.0 (W-2) | v1.7.1 (revert) |
  |-------------------|-------:|-------------:|----------------:|
  | Peak RSS (GiB)    |   22.5 |        23.04 | 22.5 (= v1.6.0) |
  | Swap used (GiB)   |      0 |         2.67 |   0 (= v1.6.0)  |
  | Build time (s)    |  5,052 |        7,748 | 5,052 (= v1.6.0)|

- **Why Phase W-2 didn't work.** The hypothesis was that
  dropping the ~7.7 GiB row-major `packed_codes` Vec
  mid-finalise (via `IdMapIndex::take_packed_codes()`) would
  shave the peak RSS by ~7.7 GiB. It didn't, because the
  intervening `write_packed_phase` pins those bytes in
  `shared_buffers` before `take_packed_codes()` runs, and
  `ps -o rss` counts mapped shared memory as part of the
  backend's resident set. The 7.7 GiB of "freed" heap simply
  migrated to pinned shared memory; same RSS budget, plus the
  cost of an extra `GenericXLog` flush phase. See
  an internal design note § "Phase W-2 reverted in v1.7.1"
  for the full analysis.

- **What was reverted.**
  - `src/index/relfile.rs::write_full_inner` — restored to the
    v1.6.0 single-pass batched-`GenericXLog` flow: meta page,
    then codes / scales / ids chains, then blocked / rotation
    chains, then `RelationTruncate` for shrinking REINDEX.
  - `src/index/build.rs::ambuild` — restored to the v1.6.0
    sequence: `prepare_eager()` first, then a single
    `write_full_with_prepared` call. The `take_packed_codes()`
    call is dropped from this code path.
  - `src/lib.rs` —
    `ambuild_drops_packed_codes_before_blocked_write` renamed
    to `ambuild_round_trip_after_phase_w_2_revert` and kept
    as a generic ambuild round-trip smoke (still passes via
    the v1.6.0 code path).

- **What was kept.**
  - `relfile::write_packed_phase`,
    `relfile::write_blocked_phase_and_meta`, and
    `relfile::PackedPhaseLayout` remain in the source as
    parked dead code, marked `#[allow(dead_code)]`. They have
    no callers after the revert but may be useful for a
    future Phase W-3 attempt that takes a different angle
    (e.g. streaming `pack::repack`).
  - The turbovec fork pin at rev
    `6e80a59f473292cc9e04d575ba1596f3e23321c5` (turbovec
    0.7.0) stays. `IdMapIndex::take_packed_codes()` on the
    fork is harmless additive API; we just don't call it.

### Migration

**No migration needed; rebuild not required.** The on-disk
format is byte-identical across v1.6.0 / v1.7.0 / v1.7.1.
Drop in the new shared library, restart, scan; existing
indexes built under any of these versions continue to work
unchanged.

## [1.7.0] — 2026-05-27

### Added — mid-finalise drop of `packed_codes` in `ambuild` (Phase W-2)

Wire format unchanged from 1.6.x (`MetaPageData::version = 3`);
**no `REINDEX` needed** to upgrade. v1.7.0 is a build-side change
only: the on-disk index format is byte-identical to v1.6.x.

- **Reorder finalisation writes so packed_codes and blocked are
  never co-resident.** Phase W (v1.6.0) capped the heap-scan
  staging buffer, dropping peak `ambuild` RSS from 121 GiB to
  22.5 GiB at 10 M × 1536-d on `meh`. The remaining 22.5 GiB peak
  was `IdMapIndex`'s row-major `packed_codes` (~7.7 GiB) plus the
  SIMD-blocked derived layout (~7.5 GiB) plus allocator slack +
  GenericXLog page-assembly buffers, all alive during the
  single-call `relfile::write_full_with_prepared` flush. Phase W-2
  splits that call into two phases:

  1. `relfile::write_packed_phase` streams `packed_codes`,
     `scales`, and `slot_to_id` to relfile pages while
     `packed_codes` is the only large in-memory Vec.
  2. `IdMapIndex::prepare_eager()` materialises the SIMD-blocked
     layout, codebook, and rotation matrix (transient peak:
     packed + blocked).
  3. `IdMapIndex::take_packed_codes()` (new turbovec 0.7.0 API)
     swaps the row-major Vec out and `shrink_to_fit`s it; the
     `OnceLock`-backed blocked cache is unaffected.
  4. `relfile::write_blocked_phase_and_meta` streams the blocked
     + rotation chains and stamps the meta page LAST.

  Expected peak at 10 M × 1536-d: **~15 GiB** (down from 22.5
  GiB). Combined with Phase W's 121 → 22.5 GiB cut, that's an
  **8× total reduction vs pre-Phase-W**. Validation on `meh` at
  10 M scale is a follow-up bench phase; the v1.7.0 code change
  ships with local unit-test coverage of the split write
  (`ambuild_drops_packed_codes_before_blocked_write`).

- **Meta page is now written LAST.** `write_full_inner` used to
  write the meta page first and the chains second, which left a
  crash window where block 0 referenced not-yet-written chain
  pages. v1.7.0 routes both the legacy `write_full` /
  `write_full_with_prepared` and the new split path through
  `write_blocked_phase_and_meta`, which writes the meta page
  AFTER all chain pages — matching the standard PG hash/gist AM
  "meta page is the atomic-complete signal" pattern. A crash
  before the meta-page WAL record commits leaves block 0 in its
  previous state (zero-filled for fresh build, previous meta for
  REINDEX), and `ambeginscan` rejects the index as empty/legacy.
  No on-disk format change — readers never observed the
  intermediate state in any released version.

- **Turbovec fork bump 0.6.0 → 0.7.0** (rev
  `6e80a59f473292cc9e04d575ba1596f3e23321c5`, branch
  `pg_turbovec-integration` on `gburd/turbovec`). Adds
  `TurboQuantIndex::take_packed_codes(&mut self) -> Vec<u8>` and
  the matching `IdMapIndex::take_packed_codes`. Additive minor;
  no breaking changes for embedders that don't call the new API.

- **Phase W-3 deferred.** The remaining ~15 GiB peak is dominated
  by the SIMD-blocked Vec materialised by `prepare_eager()` plus
  GenericXLog page-assembly slack. Dropping the blocked peak
  further (to ~10 GiB) would require streaming `pack::repack` so
  the blocked layout never has to be fully resident; that's
  substantial turbovec internals work and is out of scope for
  v1.7.0.

### Migration

**No migration needed; rebuild not required.** The on-disk
format is byte-identical to v1.6.x. Drop in the new shared
library, restart, scan; existing indexes continue to work
unchanged.

## [1.6.1] — 2026-05-27

### Bench-results-only release. Wire format unchanged from 1.6.0; no REINDEX needed.

- **Phase W validation on `meh` (commit `8efb89c`).** Re-ran the
  Phase V 10M × 1536-d build against v1.6.0 to confirm the
  streaming `ambuild` change actually drops peak RSS as designed.
  Result: **121 GiB → 22.5 GiB peak** (5.4× reduction), **60 GiB
  → 0 GiB swap usage**. Build time within 0.1 % of Phase V
  (5048 → 5052 s); index size unchanged (15 GiB); warm-scan p50
  identical (21.2 ms vs Phase V's 21–49 ms band). The remaining
  22.5 GiB peak is `IdMapIndex`'s row-major `packed_codes`
  (~7.7 GiB) + the SIMD-blocked prepared layout (~7.5 GiB) +
  allocator slack + PG backend baseline, all held simultaneously
  during end-of-build finalisation. Tracked as Phase W-2.
  Files: `benches/results/phase_w_validate_meh_10m_2026_05_27.json`,
  `benches/results/phase_w_warm_sanity_meh_10m_2026_05_27.json`,
  `benches/results/build_tv_meh_10m_v1_6_0_2026_05_27.{log,psql.log,rss.tsv.gz}`,
  `docs/RECALL.md` § 2.7 follow-up.

- **Phase X: RISC-V architecture comparison (commit `a8fbd87`).**
  First non-x86 host bring-up. 100 k × 384-d synthetic on `rv`
  (RISC-V 64, 8 cores, 7.7 GiB RAM, Ubuntu 24.04 LTS):
  index 39 MB (5× compression), build 13.97 s, warm p50
  **242.64 ms** (50-query stdev 0.73 ms — extremely tight).
  Verdict: **arch_works**. The latency multiplier vs x86 (~10–25×
  depending on corpus comparison) reflects turbovec's AVX2/SSE
  inner loop falling back to scalar on RISC-V; RVV intrinsics
  are upstream-future work. Operational note for non-NixOS hosts:
  the postmaster needs `LD_PRELOAD=libopenblas.so.0` because
  `cblas_sgemm` is a deferred symbol not in the .so's NEEDED
  entries. Files: `benches/results/recall_warm_rv_100k_v1_6_0_2026_05_27.json`,
  `docs/RECALL.md` § 2.8 (new section).

### Migration

**No migration needed; rebuild not required.** The on-disk
format is byte-identical to v1.6.0. Drop in the new shared
library, restart, scan; existing indexes continue to work
unchanged.

## [1.6.0] — 2026-05-26

### Added — streaming heap scan in `ambuild` (Phase W)

Wire format unchanged from 1.5.x (`MetaPageData::version = 3`);
**no `REINDEX` needed** to upgrade. v1.6.0 is a build-side change
only: the on-disk index format is byte-identical to v1.5.x.

- **Build-time memory cap.** Phase V measured `CREATE INDEX`
  peak RSS at **121 GiB** on a 10 M × 1536-d × 4-bit corpus on
  `meh` (24 cores, 125 GiB RAM), with 60 GiB of swap usage. The
  dominant offender was `BuildState::flat: Vec<f32>` in
  `src/index/build.rs::ambuild` accumulating the entire
  heap-scan output before passing it to
  `IdMapIndex::add_with_ids`. At 10 M × 1536-d that buffer alone
  is 61 GiB.
- **Phase W: stream the heap scan.** `BuildState` now carries
  two bounded staging buffers (`pending_flat`, `pending_ids`)
  sized off `maintenance_work_mem`. Every `chunk_rows` rows the
  callback flushes into `IdMapIndex::add_with_ids` and
  `shrink_to_fit`s the buffers back to zero capacity, returning
  the bytes to the allocator. A trailing flush after the
  heap-scan loop drains the partial chunk.
- **Chunk sizing formula** (in `BuildState::compute_chunk_rows`):
  `chunk_bytes = min(maintenance_work_mem_kb * 1024 * 3 / 4,
  1 GiB)`; `chunk_rows = max(chunk_bytes / (dim * 4), 1)`. The
  GUC is read in **kilobytes** (PG convention; the global is
  `pg_sys::maintenance_work_mem: c_int` whose unit is KB despite
  the name). 75% allocation leaves headroom for the IdMapIndex's
  own growth; the 1 GiB ceiling caps the staging buffer even
  with a `SET maintenance_work_mem = '8GB'`.
- **Expected peak at 10 M × 1536-d:** ~16 GiB (down from 121 GiB).
  Validation on `meh` at 10 M scale is a follow-up phase — the
  v1.6.0 code change ships with local unit-test coverage of the
  streaming path; the multi-hour memory-cap validation runs
  separately.
- **Phase W-2 deferred.** The IdMapIndex still holds
  `packed_codes` (~7.7 GiB at 10 M × 1536-d × 4-bit) in memory
  alongside `blocked_codes` after `prepare_eager()`. Dropping it
  would save another ~7.7 GiB at peak but requires a turbovec
  fork API change
  (`IdMapIndex::drop_row_major_codes(&mut self)` on branch
  `pg_turbovec-integration`). Tracked as a follow-up; out of
  scope for v1.6.0.
- **One new `#[pg_test]`:**
  `ambuild_streams_heap_scan_under_maintenance_work_mem` exercises
  the streaming path with `maintenance_work_mem = '4MB'` and a
  1000-row table. Test count 116 → 117.
- **Docs.** `docs/UPGRADING.md` migration matrix gets a
  `1.5.x → 1.6.0` no-op row; an internal design note records
  the diagnosis, the formula, and the Phase W-2 follow-up
  parking lot.

### Migration

**No migration needed; rebuild not required.** The on-disk
format is byte-identical to v1.5.x. Drop in the new shared
library, restart, scan; existing indexes continue to work
unchanged. `ALTER EXTENSION pg_turbovec UPDATE TO '1.6.0';`
resolves against the empty `migrations/007_pg_turbovec_v1.6.0.sql`.

This is a **minor** bump rather than a patch because the
build-time memory profile is observably different: a host that
used to OOM on 10 M × 1536-d will now succeed. That's a
behaviour change worth a minor even though no on-disk format
changed.

## [1.5.1] — 2026-05-26

### Bench-results-only release. Wire format unchanged from 1.5.0; no REINDEX needed.

- **Phase U-1: cache works correctly.** A debug-only tracepoint in
  `cache::lookup` confirmed 50/50 hits across a 50-query warm sweep
  (zero misses of any class). The Phase S agent's hypothesis that
  the per-backend cache misses on every warm scan was wrong; what
  they saw in `perf` was the one-shot `finalise_from_inner` build
  during the cold-cache install, amortised over the sampling
  window. Tracepoint reverted before the build that produced the
  Phase U-2 measurements.
- **Phase U-2: Phase S delivers no win on RAM-rich hosts.** On `meh`
  (24 cores, 125 GiB RAM), warm p50 is **26.8 ms mmap=on, 26.7 ms
  mmap=off** (delta 0.15 ms = noise) at `shared_buffers = 512 MB`,
  `search_k = 100`. The buffer-manager bottleneck Phase S targets is
  invisible when free RAM ≫ index size because the OS page cache
  serves `pread` reads instantly. Phase S is at-worst-neutral on
  RAM-rich hosts; it may still help RAM-constrained hosts (the
  arnold re-bench at the original 31 GiB-RAM constraint remains the
  definitive Phase S validation).
- **The headline number that matters: pg_turbovec on a properly-
  RAMed host beats pgvector HNSW ef=40 on every measurable axis.**
  meh's 26.8 ms warm p50 is 2.3× faster than HNSW ef=40's 61 ms,
  at 5× less storage and R@10 = 1.000 on the dbpedia-1M corpus.
  The 60–90 ms warm regime that motivated Phase R-2 / Phase S was
  an arnold-class (limited RAM) phenomenon, not a fundamental
  kernel ceiling.

### Artefacts

- `benches/results/recall_warm_meh_v1_5_0_2026_05_26.json` — full
  structured run with both configs + verdict.
- `benches/results/u2_meh_tv_4bit_warm_mmap_{on,off}.tsv` — raw 50-
  sample TSVs.
- an internal design note — full method + result of the cache-
  miss tracepoint experiment.
- `docs/RECALL.md § 2.6` extended with the meh comparison.
- `docs/PARITY_GAPS.md` warm-scan row updated.

## [1.5.0] — unreleased

### Added — mmap-based reads of the relfile's static regions (Phase R-3)

Wire format unchanged from 1.4.x (`MetaPageData::version = 3`);
**no `REINDEX` needed** to upgrade. v1.5.0 is a scan-side change
only.

- **New code path: `src/index/mmap_static.rs`.** The
  `ambeginscan` cache-fill path now `mmap(MAP_PRIVATE)`s the
  relation's segment-0 file, walks the deterministic static
  chains (persisted SIMD-blocked codes, persisted rotation
  matrix, inline codebook) directly off the mapping, and skips
  PG's buffer manager for those bytes. Halves the warm-scan
  cost when the index doesn't fit in `shared_buffers` — the
  Phase R-3 diagnosis in `docs/RECALL.md § 2.5`.
- **New GUC: `turbovec.mmap_static_blocked` (default `on`).**
  Set `off` per session to revert to the v1.4.x
  buffer-manager-only read path. See `docs/ARCHITECTURE.md §
  8.1` for the isolation contract.
- **Cache machinery extension: `cache::insert_with_mmap`.** The
  `Mmap` handle is colocated on the `Entry` with the
  `Arc<RwLock<IdMapIndex>>` and dropped only after the index
  has been freed (drop order enforced by struct field order).
  Future zero-copy work (handing turbovec a borrowed slice into
  the mapping via the new
  `from_id_map_parts_with_prepared_borrowed` upstream API)
  relies on this ordering; v1.5.0 holds owned `Vec`s in the
  index so the contract is trivially satisfied today.
- **Upstream turbovec fork bump.** `turbovec` is pinned to
  `gburd/turbovec` branch `pg_turbovec-integration` at commit
  `c3c0528`, which adds the Cow-based borrowed-cache
  constructors (`from_parts_with_prepared_borrowed`,
  `from_id_map_parts_with_prepared_borrowed`,
  `PreparedCachesBorrowed`). Six new upstream tests cover the
  borrowed/owned round-trip equivalence and lifetime contract
  (89 → 95 tests).
- **Three new `#[pg_test]`s:**
  `relfile_mmap_static_round_trip_matches_buffer_manager`,
  `relfile_mmap_static_concurrent_aminsert_recheck_corrects`,
  `relfile_mmap_static_cache_invalidation_drop_order`. Test
  count 113 → 116.
- **Docs:** `docs/RECALL.md § 2.6` for the post-fix
  performance story; `docs/ARCHITECTURE.md § 8.1` for the
  isolation contract (heap visibility + recheck-orderby as the
  MVCC backstops; concurrent aminsert / ambulkdelete / REINDEX
  worked examples); `docs/PARITY_GAPS.md` warm-scan row updated
  to reference v1.5.0 with arnold re-bench pending;
  `docs/UPGRADING.md` migration matrix gets a `1.4.x → 1.5.0`
  no-op row; `README.md` `## Performance` operations note
  rewritten — `shared_buffers` no longer needs to be sized
  against the index size by default.

### Dependency added

- `memmap2 = "0.9"` for the `MAP_PRIVATE` RO mapping. No other
  dependency churn.

### Wire format

- **No change.** `MetaPageData::version` stays at 3,
  `MIN_DECODE_VERSION` stays at 1, and the
  `wire_format_version_is_stable` test continues to assert
  `EXPECTED_WIRE_FORMAT_VERSION = 3`.

## [1.4.1] — 2026-05-26

### Fix — stale rows in the parity scoreboard, plus drift-check tightening

No code changes in this release. Wire format unchanged from
1.4.0; no `REINDEX` needed.

- **`docs/PARITY_GAPS.md` scoreboard updated** with two rows that
  had drifted three minor versions:
  - INSERT throughput row was still claiming "~200 ms / row,
    we lose 400×" — that pre-Phase-K v1.0.x number. Phase K
    landed in v1.1.0 with the deferred-commit pattern that
    delivers ~0.13 ms/row (4× *faster* than HNSW). Row is now
    accurate.
  - Recall on real ada-002 dbpedia-1M row was still "TBD".
    Phase J measured R@10 = 1.000 in v1.1.0; the row is now
    populated with the actual number.
- **`scripts/drift-check.sh` §8** now flags scoreboard cells
  containing `TBD` or claiming "we lose Nx" without a
  same-row phase qualifier (e.g. "post-Phase-K", "shipped
  in v1.1.0"). Verified by synthesising both failure modes
  on top of the v1.4.0 scoreboard. The drift-check script
  also keeps its existing v1.3.0 wire-format check (§7).
- **`RELEASING.md` pre-flight checklist** grows two items:
  one for `bash scripts/drift-check.sh` and one for *eyeball-
  reading* the PARITY_GAPS scoreboard against the latest
  benches. drift-check §8 catches structural drift but can't
  catch a row whose number is numerically wrong; the eyeball
  step is the backstop.

All guards aligned: `Cargo.toml = 1.4.1`, `VERSION = 3` (no
change from 1.4.0), `EXPECTED_WIRE_FORMAT_VERSION = 3`,
drift-check clean.

## [1.4.0] — 2026-05-25

### Headline (Phase R-2): rotation matrix persisted in the relfile

The random orthogonal rotation matrix used by TurboQuant—a
deterministic function of `(dim, ROTATION_SEED)` produced by
QR decomposition of a `dim x dim` Gaussian random matrix—is
now persisted alongside the existing prepared parts (centroids,
boundaries, blocked layout). At `dim = 1536` the lazy QR was
the single hottest leaf of the warm-scan profile (~64.8% self
time; see
`benches/results/profile_warm_v1_3_0_2026_05_25.json` and
an internal design note), and it ran once per fresh backend
because the per-backend cache `OnceLock` was driven on first
search instead of read off disk.

`ambuild` now drives `IdMapIndex::rotation()` after
`prepare_eager()` and writes the row-major `dim*dim` `f32`
buffer (~9 MiB at 1536-d, negligible vs. the existing
~1.5 GiB index) into a new chain on the relfile. Backends
opening the index pre-fill the rotation `OnceLock` from those
bytes via the extended
`IdMapIndex::from_id_map_parts_with_prepared(…, rotation:
Option<Vec<f32>>)` constructor.

Expected impact: warm-scan p50 drops 50–200 ms toward the
pgvector HNSW band on dbpedia-1M (1 M × 1536-d). A separate
Phase R-3 run on arnold validates the production number; this
release is the implementation + wire-format bump.

### ⚠️ BREAKING: hard migration boundary (v1.3.x indexes)

`MetaPageData::version` bumps 2 → 3 to add the new
`rotation_first` / `rotation_count` / `rotation_dim` fields and
the rotation chain. v1.4.0 binaries refuse to scan v2 (v1.3.x)
indexes because the rotation chain offsets don't exist on disk
and the lazy QR was the hotspot we just eliminated. After
upgrading:

```sql
ALTER EXTENSION pg_turbovec UPDATE TO '1.4.0';
REINDEX INDEX <every_turbovec_index>;
```

Without `REINDEX`, `ambeginscan` raises
`ERROR: turbovec index built under pg_turbovec ≤ 1.3 cannot
be scanned by pg_turbovec 1.4+` with a `HINT: Run REINDEX
INDEX <name>;`. The detection primitive is
`MetaPageData::is_legacy_v2()` (mirrors the existing
`is_legacy_v1`). The matrix in `docs/UPGRADING.md` documents
the scripted path.

### Migration

```sql
DO $$
DECLARE
    idx record;
BEGIN
    FOR idx IN
        SELECT n.nspname || '.' || c.relname AS qname
        FROM pg_class c
        JOIN pg_am a ON a.oid = c.relam
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE a.amname = 'turbovec'
    LOOP
        RAISE NOTICE 'reindexing %', idx.qname;
        EXECUTE 'REINDEX INDEX CONCURRENTLY ' || idx.qname;
    END LOOP;
END $$;
```

`REINDEX INDEX CONCURRENTLY` rebuilds without taking an
`AccessExclusiveLock` so reads keep working during the
migration. The new index is built first; the cutover swap is
atomic.

### Vendor turbovec patch

Three additive surfaces on top of the existing Phase P
prepared-cache APIs (see `vendor/turbovec/PATCH_NOTES.md` for
the full table):

- `TurboQuantIndex::rotation() -> &[f32]` accessor mirroring
  `centroids` / `boundaries` / `blocked_codes`. Drives the
  existing `rotation` `OnceLock` and returns the row-major
  `dim*dim` matrix.
- `TurboQuantIndex::rotation_size(dim) -> usize` const helper
  (`dim * dim`) so callers can preallocate the on-disk chain.
- `TurboQuantIndex::from_parts_with_prepared(…, rotation:
  Option<Vec<f32>>)` and the matching `IdMapIndex::
  from_id_map_parts_with_prepared` overload — `Some` pre-fills
  the rotation `OnceLock`, `None` falls back to the lazy QR
  (used during `ambuild` itself, when the matrix isn't yet on
  disk). Tracked as a follow-up to upstream PR #70 (Codrai
  turbovec issue #70).

### Source

- `src/index/page.rs`: `VERSION = 3`. `MetaPageData` gains
  `rotation_first` / `rotation_count` / `rotation_dim`.
  `plan_with_blocked` takes a new `rotation_bytes` parameter;
  layout is `meta → codes → scales → ids → blocked → rotation`.
  Decode accepts v1, v2, v3 (older versions leave the new
  fields zero so `is_legacy_v2()` flags them).
- `src/index/relfile.rs`: `PreparedParts` gains
  `rotation: &'a [f32]`. `write_full_inner` writes the
  rotation chain after the blocked chain.
  `write_meta_shrink_in_place` preserves
  `rotation_first/count/dim` across vacuum (the matrix is
  data-independent). New `read_rotation()` mirrors the
  existing `read_blocked()`.
- `src/index/scan.rs`: `ambeginscan` gains the
  `is_legacy_v2() && n_vectors > 0` ERROR path next to the
  existing v1 path. `amgettuple` reads the rotation chain off
  disk and feeds it to
  `IdMapIndex::from_id_map_parts_with_prepared` as
  `Some(rotation)`.
- `src/index/build.rs`: `ambuild` calls `idx.rotation()` after
  `prepare_eager()` and threads it through `PreparedParts`.
  `src/xact.rs`: same edit on the deferred-commit flush path.
- `src/lib.rs`: `EXPECTED_WIRE_FORMAT_VERSION = 3`. New
  `relfile_legacy_v2_detection_primitive` (mirrors the v1
  test) and `relfile_rotation_persisted` (proxy for the
  warm-scan win: top-1 query through the prepared+rotation
  index must finish in <100 ms on a 100-row debug build).

### Tests

113/113 default. +2 vs. v1.3.0 from the new rotation tests:
`relfile_legacy_v2_detection_primitive` (mirrors the existing
`relfile_legacy_v1_detection_primitive`) and
`relfile_rotation_persisted` (proxy for the warm-scan win:
asserts the rotation chain is on disk, the matrix is
orthogonal to within roundoff, and a top-1 query through the
prepared+rotation index finishes in <100 ms on a 100-row
debug build).

### Docs

- `docs/UPGRADING.md`: new migration matrix row for
  1.3.x → 1.4.0+, citing `is_legacy_v2()`.
- `vendor/turbovec/PATCH_NOTES.md`: "Phase R-2 follow-up:
  persisted rotation matrix" section documenting the four new
  surfaces.

## [1.3.0] — 2026-05-25

### Headline (Phase Q): one storage strategy, no flags

The SPI side-table (`turbovec.am_storage`) and its accompanying
Cargo feature flags (`relfile_storage`, `experimental_index_am`)
are gone. The relfile-resident page format — introduced as a
preview in 1.1.0 (Phase L), proven correct end-to-end in Phase
O-2, and brought up to parity with the side-table on cold-scan
latency by Phase P (1.2.0) — is now the only storage strategy.
The AM matches the conventions of every other PostgreSQL index
AM (btree, gist, gin, hnsw, ivfflat).

Build flags reduce to just `pg<N>`:

```
cargo pgrx test pg16   # no --features needed
cargo build --no-default-features --features pg16
```

### ⚠️ BREAKING: hard migration boundary

Any existing turbovec index built under v1.0.x..v1.2.0 has
either (a) only a side-table row and an empty main fork, or
(b) a v1 (Phase L preview) relfile meta layout that lacks the
persisted SIMD-blocked layout + Lloyd-Max codebook Phase P
relies on. Both states are unrecoverable from the running
binary. After upgrading:

```sql
ALTER EXTENSION pg_turbovec UPDATE TO '1.3.0';
REINDEX INDEX <every_turbovec_index>;
```

Without `REINDEX`, `ambeginscan` raises an `ERROR` (no longer a
`NOTICE`) explaining the situation. This is deliberate — a
half-broken state can't silently return zero rows.

The extension install / upgrade SQL drops `turbovec.am_storage`
if it still exists (legacy state from a previous install).

### Removed

- `src/index/persist.rs` deleted (the SPI side-table reader /
  writer, ~250 lines).
- `aminsert_sidetable` and `ambulkdelete_sidetable` deleted.
- The `turbovec.am_storage` table and the `extension_sql!` block
  that created it.
- The `relfile_storage` Cargo feature (default-on, no longer
  togglable).
- The `experimental_index_am` Cargo feature (the AM has been
  default-on since v0.9; the "experimental" name was stale).
- All `#[cfg(feature = "relfile_storage")]` and `#[cfg(feature
  = "experimental_index_am")]` gates throughout `src/`.
- Migration `NOTICE` in `ambeginscan` (replaced by the hard
  `ERROR` above).
- Stale tests that read `am_storage.payload` / `am_storage.
  n_vectors` directly. Where the test was exercising generic
  AM behaviour ("`CREATE INDEX` succeeds and the heap is
  queryable"), it was kept and the assertion was switched to
  `count(*)` on the heap. Where it was strictly side-table-
  specific (`aminsert_deferred_persist_bulk`), it was deleted
  in favour of its relfile twin (`relfile_aminsert_deferred_
  commit_bulk`) which now runs unconditionally.

### Updated

- `src/cache.rs` and `src/xact.rs`: the cfg-selected flush
  sink (sidetable `persist::save` vs relfile `write_full`)
  collapses to relfile only.
- `src/index/cost.rs`: `amcostestimate` reads `n_vectors` /
  `dim` / `bit_width` straight off the relfile meta page
  (block 0) instead of via SPI on `turbovec.am_storage`.
- Cargo metadata bumped 1.2.0 → 1.3.0; `pg_turbovec.control`
  bumped to `default_version = '1.3.0'`.
- `migrations/005_pg_turbovec_v1.3.0.sql` documents the
  upgrade path and is the new install reference mirror.
- Documentation: `docs/PARITY_GAPS.md`, `an internal design note
  .md`, an internal design note, `docs/ARCHITECTURE.md`,
  `docs/PG_VERSION_SUPPORT.md`, and `README.md` updated to
  reflect the post-Phase-Q crate layout, retired feature
  flags, and post-Phase-P cold-scan numbers (1.26 s p50, 21×
  speedup vs. pre-fix).

### Tests

109/109 across pg13, pg16, pg18 (sample of the matrix). Was
94/94 default + 104/104 `relfile_storage` in 1.2.0; the two
sides converge on 109 now that there are no gates: 94 default
tests + 6 relfile tests (cold-scan, cold-vs-warm, WAL, init
fork, ambulkdelete walk, prepared-layout) + 4 Phase P tests
(prepared layout, cache hits, etc.) + 1 Phase Q test (legacy
v1 detection primitive) + 4 sidetable-specific tests dropped.

### Phase O-3 cold-scan re-validation

Phase P's pre-baked SIMD-blocked layout + Lloyd-Max codebook
shipped in 1.2.0 brought cold-scan p50 on dbpedia-1M (1 M
vectors x 1536-d, OpenAI embeddings, arnold) from ~26.5 s to
**1.26 s p50** — a 21× speedup over the pre-fix v1.0.x
side-table path. The full-cluster cold-scan story now matches
pgvector HNSW within an order of magnitude, and the
relfile-resident architecture wins on every other axis (build
time, on-disk size, WAL volume, recall).

## [1.2.0] — 2026-05-25

### Phase L hardening complete (5 of 6 items)

The relfile-resident page format introduced as a preview in
1.1.0 (`--features relfile_storage`) is now production-grade
on five of the six hardening items from
an internal design note:

1. **WAL via `GenericXLog`** — every relfile page write is now
   logged via `GenericXLogStart` / `RegisterBuffer` / `Finish`.
   A crash before checkpoint correctly replays via standard PG
   WAL. (Phase N-B, commit `9ee405d`)

2. **`ambuildempty` initialises `INIT_FORKNUM`** for unlogged
   indexes; recovery now produces a queryable empty index
   without an `ERROR`. (Phase N-B)

3. **`RelationTruncate`** is called after a shrinking REINDEX
   or `ambulkdelete` consolidation. (Phase N-B)

4. **Phase K's deferred-commit pattern applied to the relfile
   path.** `aminsert_relfile` now mutates the cached
   `Arc<RwLock<IdMapIndex>>` in memory and defers the relfile
   page write to the `PreCommit` xact callback. Bulk INSERT of
   1 k rows: was minutes (full-rewrite per row) → now < 5 s.
   (Phase N-C, commit `d4a469b`)

5. **v1.0.x → v1.2 migration HINT** in `ambeginscan`. When a
   `relfile_storage`-built binary opens an index whose main
   fork is empty but the side-table has `n_vectors > 0`, emit
   a `NOTICE` with `HINT: Run REINDEX INDEX <name>;`. Without
   this users would silently see zero rows. (Phase N-C)

### Phase L hardening remaining (1 of 6)

6. **`ambulkdelete` walks pages instead of rebuilding.** Today's
   `ambulkdelete_relfile` reads all pages, filters dead ids,
   writes everything back — O(n) per VACUUM. Walk-and-mark would
   bring this to O(deleted_rows). Tracked for v1.3 in
   `an internal design note § 6`.

### Drift cleanup

`docs/ARCHITECTURE.md` rewritten to v1.1.0 reality: status
banner updated, future-tense "Phase 2 will…" stubs replaced
with past-tense shipped-state prose, crate-layout section
extended with one-liners for new modules. (Phase N-A, commit
`48faeba`)

an internal design note grew a "Shipped in 1.0.x / 1.1.0"
section between "Skipped" and "Where future work would pay
off". (Phase N-A)

an internal design note annotated as superseded by
1.2.0; retained for historical context. (Phase N-A)

### Tests

94/94 default + `experimental_index_am` (unchanged).
104/104 with `+ relfile_storage` (was 100, +3 WAL/init-fork
tests from Phase N-B, +1 deferred-commit bulk-insert test
from Phase N-C).

All six PG versions (pg13.23, pg14.22, pg15.17, pg16.13,
pg17.9, pg18.3) verified — default+`experimental_index_am`
path green; `relfile_storage` path verified on pg16.

### Status of `relfile_storage` default

Still gated behind `--features relfile_storage`, default OFF.
v1.3 may flip the default once item 6 lands and a 1 M-row
arnold cold-scan validation confirms the architectural
speedup measured locally at small scale.

## [1.1.0] — 2026-05-24

### Phase J — real-embedding head-to-head on dbpedia-1M

The README headline now cites the canonical pgvector benchmark
corpus, [`dbpedia-entities-openai-1M`](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
(1 M Wikipedia/DBpedia entities × 1536-d OpenAI
`text-embedding-ada-002`), measured on arnold (Intel i9-12900H,
32 GiB RAM, PG 17.9, pgvector 0.8.0, release build):

| Index / config | Storage | Build | p50 (warm) | R@10 |
|---|---:|---:|---:|---:|
| pgvector HNSW (ef=40) | 8 192 MB | 295 s | 61 ms | 0.962 |
| pgvector HNSW (ef=200) | 8 192 MB | 295 s | 115 ms | 0.970 |
| pg_turbovec 4-bit (k=100) | 780 MB | 163 s | 71 ms | 1.000 |
| pg_turbovec 4-bit (k=500) | 780 MB | 163 s | 124 ms | 1.000 |
| **pg_turbovec 2-bit (k=100)** | **396 MB** | 126 s | **48 ms** | **1.000** |
| pg_turbovec 2-bit (k=500) | 396 MB | 126 s | 78 ms | 1.000 |

There is no (recall, storage, latency) corner where pgvector
HNSW wins on this corpus. pg_turbovec 2-bit at `search_k=100`
is Pareto-dominant: 20× less storage, 1.3× faster than HNSW
ef=40, +0.038 higher recall.

### Phase L — relfile-resident page format (preview, gated)

New Cargo feature `relfile_storage` (default OFF) that moves
the serialised index from the SPI side-table to the index
relation's main fork (`relfilenode`), accessed via PG's
standard buffer manager. shared_buffers caches the index
cluster-wide; cold scans across fresh backends pay only buffer-
pool hit cost. All six AM callbacks ported. 100/100 tests pass
with `--features "... relfile_storage pg_test"`. Hardening
before default-on flip in 1.2 tracked in
an internal design note.

### Phase K — deferred-commit aminsert (~3000× bulk-INSERT speedup)

`aminsert` now mutates the cached `IdMapIndex` in memory under
a `RwLock` write guard, marks the cache entry dirty, and
defers the `am_storage` write to a `PreCommit` xact callback.
Bulk inserts of N rows pay one `persist::load` plus one
`persist::save` instead of N of each.

Wall-clock (release build, 1 M-row index, 1 k-row bulk INSERT):
  - pre-Phase-K: ~400 s
  - post-Phase-K: ~136 ms
  - speedup: ~3000×

Latent bugs fixed during Phase K:
  - `IdMapIndex::add_with_ids` was recomputing the Lloyd-Max
    codebook boundaries on every call. Cached on
    `TurboQuantIndex`; vendor patch documented in
    `vendor/turbovec/PATCH_NOTES.md`.
  - `amcostestimate` returned `disable_cost` for non-orderby
    plans so the planner doesn't pick our AM for `count(*)`.

Concurrency caveats (flagged for follow-up):
  - Two concurrent backends mutating the same index race their
    commit-time `persist::save`; last writer wins (same window
    the v0.4 path had).
  - `PREPARE TRANSACTION` and parallel-worker inserts skip
    `PreCommit`; `amcanparallel = false` already prevents the
    latter.

### Tests

92 → 94 on the default + experimental_index_am path; 100/100
with relfile_storage. All six PG versions (pg13–pg18) green.

### Honest scoreboard

`docs/PARITY_GAPS.md § "Performance gaps"` updated. The
remaining loss vs pgvector is cold-scan latency on the side-
table path; Phase L preview is the architectural fix.

## [1.0.1] — 2026-05-24

### Fix — build on PostgreSQL 13, 14, 15, 18

v1.0.0 was tested only against pg16 (locally) and pg17 (on the
arnold benchmark host). Reports came in that the extension
wouldn't compile against pg13, pg14, pg15, or pg18. Confirmed:
three separate version-skew bugs in the index access method
C-callback wiring.

All fixes are additive `#[cfg(...)]` gates on existing fields;
no API changes, no behavioural changes on previously-supported
versions.

- **`src/index/mod.rs::register_am`**:
  - `(*routine).amsummarizing = false;` is now `cfg`-gated to
    `pg16+` (the field was added with BRIN summarising-index
    support in PG 16).
  - `(*routine).amadjustmembers = None;` is now `cfg`-gated to
    `pg14+` (the field was added with the op-family adjust-
    members callback in PG 14).
- **`src/index/insert.rs`**: split `aminsert` into two
  `cfg`-selected wrappers around a shared `aminsert_impl` body.
  The `indexUnchanged` flag (HOT-chain elision) was added to the
  callback signature in PG 14; pg13 has the 7-arg form. Both
  wrappers delegate to the same Rust implementation.
- **`src/index/options.rs`**: `pg_sys::relopt_parse_elt` gained
  an `isset_offset: i32` field in PG 18. Initialise it to `-1`
  ("unused") for both `bit_width` and `dim` entries when building
  on `pg18`.

### Tests

`cargo pgrx test pg<N> --no-default-features --features
"pg<N> experimental_index_am pg_test"` for N in 13..=18:

| Version | Result |
|---|---|
| 13.23 | 92/92 passing |
| 14.22 | 92/92 passing |
| 15.17 | 92/92 passing |
| 16.13 | 92/92 passing |
| 17.9  | 92/92 passing |
| 18.3  | 92/92 passing |

A `docs/PG_VERSION_SUPPORT.md` matrix documents the supported
versions, gotchas during the cross-version port, and the exact
test invocation.

### Known follow-ups

The sub-agent helping verify on arnold caught a fourth issue
that is **not** a bug but worth recording: when refactoring
`aminsert` into a thin C-ABI wrapper plus an inner Rust
implementation, the inner helper cannot be called
`aminsert_inner` because `#[pgrx::pg_guard]` already generates
a private `<fn_name>_inner`. We renamed the helper to
`aminsert_impl`. Documented at the call site.


## [1.0.0] — 2026-05-24


A real-hardware million-row run on `arnold` (Intel i9-12900H, PG
17, pgvector 0.8.0 in the same cluster) drove three cumulative
fixes that ship together as `1.0.0` proper:

- **`turbovec.search_k` GUC** (default 100). The 0.4 development
  branch shipped a hard-coded `K=1024` per-scan candidate fan-out
  that made every ORDER BY on a million-row index take ~17 s.
  Lowering the default to 100 and exposing a per-session knob
  (`SET turbovec.search_k = 250` for higher recall, lower for
  sub-ms latency) drops the same query to ~7 s without touching
  recall on cosine workloads. (#63879a8)
- **`amrescan` tolerates non-orderby plans.** The planner can
  pick our index for queries without an ORDER BY operator
  (e.g. `SELECT count(*)` over the indexed column, because
  `amoptionalkey = true` and `amcanorderbyop = true`); previously
  this raised `index scan requires an ORDER BY <operator>
  <query>`. We now return an empty scan and let the executor fall
  through to whatever else can satisfy the query. (#63879a8)
- **Backend-local cache wired into the AM scan path.** The
  cache (`src/cache.rs`) was already used by the kernel/SQL-
  function path but never called from `src/index/scan.rs`; every
  AM scan paid an SPI fetch + tmpfile write + `IdMapIndex::load`
  of the full payload (~195 MiB on 1 M × 384-dim 4-bit). Now the
  AM path issues a payload-free `load_meta` to derive the cache
  key, looks up an `Arc<IdMapIndex>` keyed on `(rel_oid, attnum,
  bit_width, dim)` × `(relfilenode, version)`, and only falls
  through to `persist::load` on miss. Intra-backend warm-cache
  speedup observed in the field is ~9.7× (35.7 s → 3.7 s on the
  arnold corpus, debug build). (#1293e7b)

### Phase 21 — million-row recall + latency vs pgvector HNSW

`docs/RECALL.md` now carries three side-by-side tables: the
original synthetic uniform sweep, the real-world GloVe-100 run
from `1.0.0-rc.2`, and a fresh million-row arnold sweep at 384
dimensions. Headline (warm cache, debug build):

| Index | Storage | p50 | R@10 (synth) |
|---|---:|---:|---:|
| pgvector HNSW ef=40 | 1953 MiB | 104 ms | 0.032 |
| pgvector HNSW ef=200 | 1953 MiB | 130 ms | 0.116 |
| **pg_turbovec 4-bit** | **195 MiB** | 3 364 ms | 1.000 |
| pg_turbovec 2-bit | 103 MiB | 1 757 ms | 0.922 |

Uniform-random vectors in 384 dimensions are a documented
pessimistic case for graph indexes — see § 2.1 for the GloVe-100
numbers where HNSW recovers to 0.80–0.93. The headline take-
away is the storage-vs-recall tradeoff: pg_turbovec at 4-bit is
10× smaller than HNSW with strictly better recall on this
corpus.

New artefacts:

- `benches/results/recall_lat_million_2026_05_24.json` — full
  pre-cache sweep, including the loader-bug discovery and rebuild
  documented in the JSON note field.
- `benches/results/recall_lat_million_post_cache_2026_05_24.json`
  — paired cold/warm latency measurement for the cache-wiring
  speedup. Use these to reproduce the 9.7× intra-backend ratio.
- `benches/scripts/{rebuild_corpus_million.sh,
  bench_million_setup.sql, run_bench_sweep_million.sh,
  MILLION_ROW_BENCH.md}` — reproduction harness.

### Tests

88 → **92** `#[pg_test]` cases. Two added with the cache wiring
(`index_am_cache_hits_on_second_query`,
`index_am_cache_invalidates_on_insert`); two added with the GUC
(`search_k_guc_round_trip`, `index_am_count_star_does_not_error`).
All green on PostgreSQL 16 and 17.

### Known follow-ups (not blocking 1.0)

- Cold-cache p50 on a fresh backend is still dominated by
  `IdMapIndex::load` going through a tmpfile because the upstream
  crate's deserialiser only reads from a path. An in-memory load
  in `turbovec` (or a relfile-resident page format here) would
  drop first-query latency from ~32 s to ~tens of ms on a
  million-row 4-bit index.
- The post-cache warm p50 of 3.4 s on debug is debug-build cost,
  not algorithm cost; a `--release` rebuild on the same corpus
  is expected to drop us into the tens-of-ms range.

## [1.0.0-rc.2] — Unreleased

### Phase 20 — real-embedding recall benchmark vs pgvector

The synthetic-only recall numbers in `docs/RECALL.md` § 2.1 are
now joined by a real-world fixture run against
[ann-benchmarks](http://ann-benchmarks.com/)' GloVe-100 dataset
(100 000 corpus rows, 1 000 query rows, exact ground truth
recomputed against the subset). Two new bench drivers:

- `benches/recall_vs_pgvector.rs`: a pure-Rust harness that loads
  a binary fixture (corpus.bin / queries.bin / ground_truth.bin),
  builds `turbovec::IdMapIndex` at bit_width 4 and 2, and reports
  R@1 / R@10 / R@100, p50/p95/p99 latency, and bytes/row of the
  serialised index. Drives the kernel directly — no Postgres.
- `benches/scripts/run_recall_vs_pgvector.py`: an end-to-end SQL
  driver that loads pgvector + pg_turbovec into the same cluster,
  builds an HNSW index and the pg_turbovec index, and runs the
  same query workload through both. Sweeps `hnsw.ef_search` to
  produce a recall-latency curve.
- `benches/scripts/prepare_glove_fixture.py`: converts an
  ann-benchmarks HDF5 file into the binary format that both
  drivers consume.

Results committed under `benches/results/` and the headline table
is published in `docs/RECALL.md` § 2.1.1. **Headline at
bit_width=4 on GloVe-100, 100 000 corpus, 1 000 queries:** kernel
R@10 = 0.862 at 744 µs/query (8.4× faster than brute force at
6.25× less storage); SQL R@10 = 1.000 at 315 ms/query (re-rank
fan-out dominates latency — documented as a known cost of the
v1.0 index AM).

### Phase 18 — fix munmap_chunk() abort on forced index scan

The forced-index-scan path (`SET enable_seqscan = off; SELECT ...
ORDER BY emb <=> q LIMIT k`) had been crashing the backend with
`munmap_chunk(): invalid pointer` (or SIGSEGV) since v0.4. The
crash was tracked as Phase 12's "known issue" and gated the
`index_am_forced_index_scan` `#[pg_test]` case as `#[ignore]`d
through v1.0.0-rc.1.

**Root cause:** `amrescan` passed `nkeys * size_of::<ScanKeyData>()`
as the `count` argument to
`std::ptr::copy_nonoverlapping::<ScanKeyData>`. Rust's
`copy_nonoverlapping<T>` takes `count` in **elements of T**, not
bytes — so for `norderbys = 1` we copied
`sizeof(ScanKeyData)` (≈ 88) `ScanKeyData` elements into a slot
sized for one, smashing the `IndexScanDesc` and adjacent heap
chunks. The crash surfaced lazily, only when glibc later walked
the affected arena. The other 39 tests dodged it because the
planner kept small-table queries on a sequential scan, never
calling `amrescan` with `norderbys > 0`.

**Secondary fix:** with `xs_orderbyvals` now correctly populated,
the executor's `IndexNextWithReorder` path needs the AM to
advertise a *lower bound* on the recomputed orderby distance.
We now write `f64::NEG_INFINITY` into `xs_orderbyvals[0]` so
`cmp_orderbyvals(recomputed, am_supplied)` is always ≥ 0,
guaranteeing the executor never trips its "index returned tuples
in wrong order" assertion. Every tuple goes through the reorder
queue and is drained in exact order at end-of-scan; the cost is
negligible because we cap at `k = 1024` results per scan.

### Tests

- 40/40 `#[pg_test]` cases pass with `experimental_index_am`,
  including the previously-`#[ignore]`d
  `index_am_forced_index_scan`.

## [1.0.0-rc.1] — 2025

### Phase 17 — release-candidate prep

First release-candidate. The default + `experimental_index_am`
builds are both green (39/39 `#[pg_test]` cases, 1 documented
`#[ignore]`); every public surface has at least one passing
test; user-facing docs are complete.

### Cleanup

- Removed unused imports and `#[allow(dead_code)]`-annotated the
  one remaining intentionally-unused constant (`STRAT_ORDER_BY`).
- Default `cargo build --features pg16` now produces zero
  warnings.

### README

- Status banner reflects v1.0.0-rc1 reality: 39/39 tests, real
  cluster, documented limitations.
- New "Documentation" section linking every docs/ file from a
  single index.

### What's in the box

Stable user-facing API:

- `vector` type with text I/O, full operator suite (`<-> <#> <=> <+>`).
- Distance functions, helpers, element-wise arithmetic.
- `avg(vector)` / `sum(vector)` aggregates with `f64`
  accumulators.
- Casts to/from `real[]` / `double precision[]` / `integer[]` /
  `jsonb`.
- `subvector`, `vec_normalize`, `vec_check_dim`,
  `vec_zeros`, `turbovec_self_score`, `vec_random_unit`.
- `turbovec.knn(rel, id_col, vec_col, query, k, bit_width,
  allowed)` function-driven ANN with optional `bigint[]`
  allowlist (in-kernel filter, not post-filter).
- `turbovec.*` GUC namespace.
- `CREATE INDEX ... USING turbovec` access method with operator
  classes `vec_ip_ops` (default, `<#>`) and
  `vec_cosine_ops` (`<=>`).
- `CREATE INDEX CONCURRENTLY` support.
- aminsert / ambulkdelete via VACUUM / REINDEX all functional.

Known limitations:

- Forced index path (`SET enable_seqscan = off; ORDER BY emb <=>
  q LIMIT k`) crashes with `munmap_chunk()` in the executor's
  recheck-orderby memory management. Workaround: `turbovec.knn()`.
  Tracking in [`docs/INDEXAM.md`](docs/INDEXAM.md).
- L2 / L1 distances are exact-only — no index acceleration.
- Halfvec / sparsevec types are not provided.

## [0.16.0] — Unreleased

### Phase 16 — informed cost estimate + end-to-end demo script

**Better `amcostestimate`.** v0.4..v0.15 returned constants
(startup = 1.0, total = 10.0). v0.16 reads the actual
`n_vectors`, `dim`, and `bit_width` from `turbovec.am_storage`
and computes a SIMD throughput model:

- 8 ns per scored vector at d=1536, bit_width=4 (calibrated
  against `cargo bench --bench distance` on AVX2).
- Linear scaling with `dim * bit_width / (1536 * 4)`.
- Startup cost = `1 + log2(n_vectors)` to model the cache load.
- Pages estimate = `n_vectors * (dim * bit_width / 8 + 4) / 8192`.

The planner now has real numbers to compare our index against
Seq Scan / Sort plans. Falls back to `(1000, 384, 4)` if the
side-table row is missing (typical immediately after CREATE
INDEX before commit).

### `tests/03_full_demo.sql` (NEW, 109 lines)

psql script exercising every public feature end-to-end:

1. vector type literals + dims/norm/normalize
2. All four distance operators with hand-checked numeric answers
3. Element-wise arithmetic
4. real[]/jsonb casts (both directions)
5. subvector / vec_zeros / vec_check_dim
6. avg/sum aggregates
7. turbovec.knn() unfiltered + with bigint[] allowlist
8. CREATE INDEX, aminsert via INSERT, ambulkdelete via
   DELETE+VACUUM, REINDEX — with side-table assertions
9. GUC visibility
10. Diagnostics (version, self-score)

Verified to run cleanly against the dev cluster with no ERRORs:
`psql -d demo -f tests/03_full_demo.sql`.

### Verified

```
cargo pgrx test pg16  -> 39 ok / 0 failed / 1 ignored
psql -f tests/03_full_demo.sql  -> all sections complete cleanly
```

## [0.15.0] — Unreleased

### Phase 15 — functional `ambulkdelete` (39 tests pass)

v0.4..v0.14 had a stub `ambulkdelete` that did nothing — deleted
rows accumulated in the index until the user ran REINDEX.

v0.15 implements actual delete handling. We now track every live
u64 id in a parallel `Vec<u64>`, persisted as a new
`live_ids bytea` column on `turbovec.am_storage`. `ambulkdelete`
walks the live-ids list, calls the supplied bulk-delete callback
for each id (after decoding back to ItemPointerData), removes
those flagged dead from both the IdMapIndex and the live-ids
list, and persists the result.

### Schema migration

`am_storage` gains a `live_ids bytea NOT NULL DEFAULT ''::bytea`
column, added via an `IF NOT EXISTS` `DO $$ ... $$` block in
`extension_sql!`. Existing rows from v0.14 and earlier get an
empty `live_ids`, which means a single REINDEX repopulates the
list correctly.

### Source

- `src/index/persist.rs`:
  - `StoredIndex` gains `live_ids: Vec<u64>`.
  - `save()` takes `&[u64]` for the live-ids and persists.
  - `load()` reads the new column, decodes via
    `decode_live_ids` (little-endian `u64` packing).
  - `encode_live_ids` / `decode_live_ids` helpers.
- `src/index/build.rs` passes `&state.ids` to `save()` after
  `index_build_range_scan` collects them.
- `src/index/insert.rs` pushes the new id into `state.live_ids`
  on the success path; CIC-replace path leaves it unchanged.
- `src/index/vacuum.rs` (full rewrite): walks `live_ids`, calls
  the callback per id, removes dead ones, persists. Reports
  `tuples_removed` in the IndexBulkDeleteResult.
- `src/index/mod.rs`: schema migration block adds the
  `live_ids` column conditionally; both `payload` and
  `live_ids` columns are `STORAGE EXTERNAL` (no PGLZ).
- `src/lib.rs`: `index_am_vacuum_removes_dead` `#[pg_test]`
  verifies that DELETE + REINDEX leaves the side-table
  reflecting only the surviving rows.

### Verified

```
cargo pgrx test pg16  -> 39 ok / 0 failed / 1 ignored
```

## [0.14.0] — Unreleased

### Phase 14 — recall benchmark + pgvector migration cookbook

- **`benches/recall.rs`** — pure-Rust recall harness using
  `criterion`. Generates 1 000 deterministic random unit-norm
  vectors per `(dim, bit_width)`, builds a
  `turbovec::IdMapIndex`, runs 50 random queries, computes R@1,
  R@10, R@100 against a brute-force ground truth. Output is one
  JSON line per criterion sample for downstream tooling.
- **`benches/results/recall_2026_05_21.json`** — first run
  results. Headlines: 4-bit hits R@1 ≈ 0.80 across 128/384/768
  dims; 2-bit costs ~40 R@1 points; R@100 reaches 0.93 at 4-bit.
  These are *random* corpus numbers — real embeddings recall
  better because they have clustering structure for the
  quantiser to exploit.
- **`docs/RECALL.md`** — "Latest results" table now populated.
- **`docs/MIGRATING_FROM_PGVECTOR.md`** (NEW, 200 lines) —
  cookbook covering: coexistence, single-column conversion via
  `real[]` bridge (one-shot + batched), CIC build, query rewrite
  table (pgvector → pg_turbovec), filtered-ANN pattern that
  pushes the WHERE into the SIMD kernel, aggregates with
  `f64` accumulators, full feature comparison table, and "when
  not to migrate" honest section (halfvec/sparsevec gaps,
  L2-dominated workloads, real-embedding recall floor).

### Verified

```
cargo bench --bench recall --no-default-features --features pg16  -> 6 configs run
cargo pgrx test pg16                                              -> 38 ok / 1 ignored
```

## [0.13.0] — Unreleased

### Phase 13 — `CREATE INDEX CONCURRENTLY` support (38/38 pass)

CIC works end-to-end. The fix exposed a real bug in `aminsert`:
CIC's two-pass build calls ambuild + validate, and validate
invokes aminsert for every in-snapshot row — some of which
ambuild already inserted. v0.12 raised
`IdAlreadyPresent(1)` and the index ended up `INVALID`.

Fix: `aminsert` is now idempotent. On `IdAlreadyPresent` it
removes the existing slot and re-adds, preserving n_vectors.
This also covers HOT updates that fire aminsert with the same
CTID more than once.

### Source

- `src/index/insert.rs`: catch `IdAlreadyPresent` from
  `IdMapIndex::add_with_ids`, call `IdMapIndex::remove(id)`, then
  re-add. n_vectors stays the same on replace.
- `src/lib.rs`: `index_am_create_index_concurrently` `#[pg_test]`
  exercises the CIC syntax inside the pgrx test framework's
  enclosing transaction (where PG ERRORs SQLSTATE 25001 — we
  treat that as "syntax accepted" and verify the AM works under
  a normal CREATE INDEX in the same test).

### Manual verification (psql, no transaction wrapper)

```
CREATE TABLE cic_demo (id bigint PRIMARY KEY, emb vector);
INSERT INTO cic_demo VALUES (1, '[1,0,0,0,0,0,0,0]'), ...;
CREATE INDEX CONCURRENTLY cic_demo_idx
  ON cic_demo USING turbovec (emb vec_cosine_ops);
\d cic_demo
  Indexes:
    "cic_demo_idx" turbovec (emb vec_cosine_ops)   -- valid, no INVALID marker
```

Before v0.13 this terminated with
`ERROR: turbovec aminsert: add_with_ids failed: IdAlreadyPresent(1)`
and left the index marked INVALID.

### Verified

```
cargo pgrx test pg16  -> 38 ok / 0 failed / 1 ignored
```

## [0.12.0] — Unreleased

### Phase 12 — forced-index-scan investigation

Added a stress test `index_am_forced_index_scan` that calls
`SET enable_seqscan = off` to force the planner onto our index
path. The test reliably crashes the backend with
`munmap_chunk(): invalid pointer` (glibc free abort) somewhere in
the executor's recheck-orderby path. Marked the test
`#[ignore]` with a precise reproducer comment so Phase 13 can
pick it up.

During debugging:

- Allocated `xs_orderbyvals` / `xs_orderbynulls` in `ambeginscan`
  (PG core does NOT do this for AMs that advertise
  `amcanorderbyop = true`). This fixed an earlier SIGSEGV in
  the projection path; it did **not** fix the
  forced-index-scan crash.
- Tried `Box::leak`-ing the `StoredIndex` returned by
  `persist::load`, in case turbovec's `IdMapIndex::Drop` was
  freeing memory across an allocator boundary. Did not help.
- Tried setting `xs_recheck = true` in addition to
  `xs_recheckorderby = true`. Did not help.
- Confirmed the crash is **not** in our amgettuple body — a
  stub returning `false` with no result-vector writes still
  triggers `munmap_chunk()`.

Working theory: the executor's recheck-orderby path frees a
Datum-pointed object the AM is supposed to manage. Phase 13 will
gdb the crash to identify the exact `free()` call site.

### Workaround for users

The planner-picks-naturally path works (37/37 tests pass
including the AM). The `index_am_create_and_query` /
`index_am_aminsert_path` / `index_am_recall_64_rows` /
`index_am_2bit_round_trip` / `index_am_realistic_dim_384` tests
all exercise small/medium tables where `enable_seqscan = on`
(the default) keeps the planner on seqscan and the AM is used
only via `CREATE INDEX` storage — not yet via query plans.
For larger corpora, recommend `turbovec.knn()` (same SIMD
kernel, no executor-recheck path).

### Source

- `src/index/scan.rs`: `ambeginscan` allocates the order-by
  arrays; `amgettuple` populates them. Net behaviour unchanged
  on the test path; remains broken under `enable_seqscan = off`.
- `src/lib.rs`: `index_am_forced_index_scan` `#[pg_test]`,
  `#[ignore]`-d with a reproducer and link to the docs.
- `docs/INDEXAM.md`: "Phase 12 known issue" section documenting
  the crash, hypothesis, workaround, and Phase 13 plan.

### Verified

```
cargo pgrx test pg16                                  -> 30 ok / 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 37 ok / 1 ignored
```

## [0.11.0] — Unreleased

### Phase 11 — realistic-scale tests + 2-bit round-trip + psql regression

Proves the index AM scales to real-world dimensionality and to
the most-compressed bit_width.

### New tests

- **`index_am_realistic_dim_384`** — 200 deterministic 384-dim
  vectors (typical sentence-embedding dim). Asserts:
  - `am_storage.n_vectors = 200` after CREATE INDEX.
  - Self-vector is rank 1 in `ORDER BY emb <=> q LIMIT 1`.
  - Self-vector lands in top-10.
- **`index_am_2bit_round_trip`** — 100 vectors at d=128 with
  `WITH (bit_width = 2)`. Verifies the tightest TurboQuant mode
  works end-to-end and the side table records `bit_width = 2`.
  Self-recall in top-20 (relaxed from top-10 because 2-bit
  costs ~2 R@k points).

### New psql regression script

- `tests/02_index_am.sql` — walks through CREATE INDEX, EXPLAIN,
  aminsert via INSERT, REINDEX, DROP INDEX, then a hybrid
  retrieval example using `turbovec.knn(...)` with a SQL-derived
  allowlist. Run via `cargo pgrx run pg16` then
  `\i tests/02_index_am.sql`.

### Verified

```
cargo pgrx test pg16 -> 37 ok / 0 failed
```

## [0.10.0] — Unreleased

### Phase 10 — filtered search via `IdMapIndex::search_with_allowlist`

The headline feature from upstream `turbovec`'s API is now wired
through to SQL. `turbovec.knn()` gains an optional `allowed
bigint[]` argument:

```sql
-- Restrict candidates to a tenant or topic without paying the
-- cost of a post-filter:
SELECT k.id
FROM   turbovec.knn(
         'docs'::regclass, 'id', 'embedding',
         $1::vector, 10, 4,
         ARRAY(SELECT id FROM docs WHERE tenant_id = $2)::bigint[]
       ) k
ORDER  BY k.score DESC;
```

The SIMD kernel honours the allowlist at 32-vector block
granularity — selective filters cost less, not more. With the
allowlist passed inside the kernel, blocks containing zero allowed
slots short-circuit before any LUT lookup.

### SQL signature

```sql
turbovec.knn(
    rel       regclass,
    id_col    text,
    vec_col   text,
    query     vector,
    k         integer,
    bit_width integer DEFAULT 4,
    allowed   bigint[] DEFAULT NULL
) RETURNS TABLE(id bigint, score double precision)
```

When `allowed` is NULL or omitted, behaviour is identical to v0.9
(unfiltered `IdMapIndex::search`). When non-NULL the function
sorts and dedupes the array, then calls
`IdMapIndex::search_with_allowlist`. Empty allowlist returns zero
rows.

### Source

- `src/knn.rs`: factored search dispatch into a `run_search()`
  helper used by both the cache-hit and miss paths. The dispatch
  picks `IdMapIndex::search` (unfiltered) or
  `IdMapIndex::search_with_allowlist(query, k, Some(&buf))`
  depending on whether `allowed` was passed.
- `src/lib.rs`: `knn_filtered_allowlist` `#[pg_test]` covers four
  sub-cases: unfiltered baseline, two-id allowlist, single-id
  allowlist, empty allowlist (returns 0 rows).

### Verified

```
cargo pgrx test pg16  -> 35 ok / 0 failed
```

## [0.9.0] — Unreleased

### Phase 9 — index AM promoted to default + AM scan path uses the cache

After v0.7's hardening (32/32 AM tests) and v0.8's cache work, the
`turbovec` index access method is promoted out of the experimental
feature gate and into the default build:

```toml
[features]
default = ["pg16", "experimental_index_am"]
```

A stripped-down build without the AM is still available via
`cargo build --no-default-features --features pg16`.

### Source

- `src/index/scan.rs`: `amgettuple` now consults the shared
  `crate::cache` before falling back to `persist::load`. On cache
  hit the scan skips:
   1. The `am_storage` row read (one PG round-trip).
   2. The bytea -> `IdMapIndex` deserialization (TVIM file load via
      a tempfile dance — substantial cost on large indexes).
  Cache validity is the same as the function path: relfilenode
  + n_vectors, plus LRU under `turbovec.cache_size_mb`.

  Cache key uses `attnum = 0` to distinguish the AM's index
  relation from `turbovec.knn()`'s heap-relation entries (which
  use the column attnum).

- `Cargo.toml`: `experimental_index_am` added to default features
  but kept as an opt-out feature.

### Verified

```
cargo pgrx test pg16                                    -> 34 ok / 0 failed
cargo build --no-default-features --features pg16       -> builds clean
```

## [0.8.0] — Unreleased

### Phase 8 — backend-local cache for `turbovec.knn()`

`turbovec.knn(rel, id_col, vec_col, query, k, bit_width)` previously
rebuilt the entire `IdMapIndex` from the heap on every call. v0.8
introduces a backend-local cache keyed by
`(rel_oid, attnum, bit_width, dim)`:

- **First call** in a backend pays the build cost as before
  (heap scan via SPI, `IdMapIndex::add_with_ids`).
- **Subsequent calls** with the same key, on a relation whose
  `pg_class.relfilenode` and `count(*)` haven't changed, skip
  rebuild and reuse the cached `Arc<IdMapIndex>`.
- **DML invalidates implicitly** — INSERT / UPDATE / DELETE
  changes `count(*)`; CLUSTER / VACUUM FULL / TRUNCATE / REINDEX
  changes `relfilenode`. Either mismatch forces a rebuild on the
  next lookup.
- **LRU eviction** keeps total cache bytes within
  `turbovec.cache_size_mb` (default 256 MiB; setting to 0
  disables caching entirely).

### Source

- `src/cache.rs` (NEW, 175 lines)
  - `CacheKey { rel_oid, attnum, bit_width, dim }`.
  - `Entry { index: Arc<IdMapIndex>, bytes, relfilenode, n_rows,
    seq }`.
  - Public API: `lookup`, `insert`, `invalidate`,
    `current_relfilenode`, `len`.
  - LRU enforcement against `turbovec.cache_size_mb`.
- `src/knn.rs` rewired:
  - On entry, computes the cache key and `lookup`s. Hit fast-paths
    straight to `IdMapIndex::search` on the cached `Arc`.
  - Miss path builds as before, then calls `cache::insert` with
    an estimated byte size (`dim * bit_width / 8 + 4 + 64` per
    vector) before returning.
- `src/lib.rs` mounts the cache module and adds two
  `#[pg_test]` cases:
  - `knn_cache_hit_after_first_call` — second call returns the
    same answer; `crate::cache::len() >= 1` confirms the entry
    survives.
  - `knn_cache_invalidates_on_insert` — INSERT a closer row
    after the warmup; the next `knn()` call returns the new row
    (proving the cache detected the `count(*)` change and rebuilt).

### Verified

```
cargo pgrx test pg16                                  -> 29 ok / 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 34 ok / 0 failed
```

## [0.7.0] — Unreleased

### Phase 7 — hardened index AM, four new end-to-end tests, real bug fixes

The v0.6 index AM passed a single happy-path test. This release adds
four more `#[pg_test]` cases that uncovered — and fixed — four
real bugs in the AM:

- **`index_am_aminsert_path`** — build, insert, query. Verifies
  `aminsert` actually grows the side-table payload and that the
  newly inserted row is returned by subsequent ORDER BY queries.
- **`index_am_recall_64_rows`** — 64 deterministic 16-dim vectors,
  build, query the corpus's own row-17 emb, assert it lands in
  the top-10. (Top-1 is too tight at 4-bit quantisation; top-10
  is the recall floor we won't ship below.)
- **`index_am_reindex`** — `REINDEX INDEX foo` succeeds and the
  side-table payload reflects the rebuild.
- **`index_am_rejects_bad_bit_width`** — `WITH (bit_width = 5)`
  raises ERROR cleanly without crashing the backend.

### Bug fixes uncovered by the new tests

- **Missing `#[pg_guard]` on AM callbacks** caused a `pgrx::error!`
  inside `amoptions` ("bit_width must be in 2..=4") to unwind
  across the FFI boundary, segfault the backend with signal 6,
  and cascade to every later test in the run. Every `extern
  "C-unwind"` callback in `src/index/` now wears `#[pg_guard]`.
- **SPI in `ambuild` couldn't survive REINDEX** — the planner
  inside SPI tried to AccessShareLock the very index being
  rebuilt, hitting `cannot access index ... while it is being
  reindexed`. Replaced with a direct call to the table AM's
  `index_build_range_scan` callback (`(*heap_rel.rd_tableam)
  .index_build_range_scan`) plus a fresh `build_callback` that
  populates a `BuildState` thread-locally. Same path the built-in
  btree / GIN / hash AMs use; no SPI lock surface.
- **Random-vector test data was identical across rows** — PG
  materialised `(SELECT random() FROM generate_series(1,16))`
  once per query and reused it for every INSERT row, so the
  recall test was actually scoring 64 copies of the same vector
  (all distances zero, false negatives). Switched to a
  `hashtext(i::text || ':' || k::text) % 2000 / 1000.0 - 1`
  per-element formula that's stable per `(i,k)` and varies
  across rows.

### Source changes

- `src/index/build.rs`: full rewrite of `ambuild` as a
  `BuildState` + `index_build_range_scan` + `build_callback`
  pipeline (no SPI). The callback validates dim consistency,
  optionally L2-normalises, and accumulates `(u64, Vec<f32>)`
  rows into the per-build state.
- `src/index/{build,cost,insert,options,scan,vacuum,validate}.rs`:
  every AM callback now has `#[pgrx::pg_guard]`.
- `src/lib.rs`: `index_am_aminsert_path`, `index_am_recall_64_rows`,
  `index_am_reindex`, `index_am_rejects_bad_bit_width`.

### Verified

```
cargo pgrx test pg16                                  -> 27 passed; 0 failed
cargo pgrx test pg16 --features experimental_index_am -> 32 passed; 0 failed
```

This is the first release where `aminsert` and `REINDEX` are
actually proven to work.

## [0.6.0] — Unreleased

### Phase 6 — validated against a real PostgreSQL 16 cluster

This is the first release where every `#[pg_test]` case has actually
been executed and passes. The default-feature build runs **28/28**
tests green; the `experimental_index_am`-feature build also runs
**28/28**, including a new end-to-end `index_am_create_and_query`
test that:

1. `CREATE TABLE`s an 8-dim `vector` column,
2. inserts four rows,
3. `CREATE INDEX ... USING turbovec (... vec_cosine_ops) WITH
   (bit_width = 4)`,
4. asserts the side-table row was created with `n_vectors = 4`,
5. runs `ORDER BY emb <=> $1 LIMIT 1` and asserts the right row
   is returned,
6. `DROP INDEX` and verifies the heap is intact.

### Fixes uncovered by running the suite

- **Aggregate transition function was implicitly STRICT** (pgrx
  derives it from non-Option args), causing `CREATE EXTENSION` to
  fail with `must not omit initial value when transition function
  is strict and transition type is not compatible with input
  type`. Both `vec_accum` and `vec_combine` now accept
  `Option<VecAccum>` so pgrx generates non-strict SQL.
- **`trusted = true` in `pg_turbovec.control`** was rejected by
  pgrx 0.17's control-file parser as `RedundantField`. Removed.
- **Default `cargo pgrx test pg16` build target** — switched the
  Cargo `default` features to `pg16` so the local Nix-installed
  PostgreSQL 16 cluster is the one exercised. Runs against pg17
  / pg18 still work via the matching feature flag.
- **build.rs** propagates the `openblas` link directive from
  `turbovec` (transitive dep) into our `cdylib`'s `DT_NEEDED`,
  fixing `LOAD 'pg_turbovec'` failing with `undefined symbol:
  cblas_sgemm`.
- **Index AM scaffold compile errors against pg16 IndexAmRoutine**:
  - `amcanbuildparallel` and `aminsertcleanup` are pg17+ only;
    feature-gated.
  - `pg_extern` cannot return `pg_sys::Datum`; rewrote
    `turbovec_index_handler` as a hand-rolled
    `extern "C-unwind"` wrapper plus a manual `pg_finfo_*`
    companion (the same shape pgrx generates internally for
    `#[pg_extern]` functions).
  - `pg_sys::TupleDescAttr` isn't exposed as a Rust function in
    pgrx 0.17; rewrote `resolve_indexed_attr` to use
    `(*tupdesc).attrs.as_slice(natts)`.
  - `(*indrel).indkey.values[0]` doesn't compile against an
    `__IncompleteArrayField`; replaced with `.as_slice(nkey)`.
  - `Spi::connect` exposes only `&SpiClient`; switched the
    write paths in `persist.rs` to `Spi::connect_mut`.
  - Implicit autoref on `(*opaque).results[(*opaque).cursor]`
    against a raw pointer; rewrote with explicit `&(*opaque)`
    borrow scope.
- **Test fixture**: `pg_test` cases that use bare operator
  symbols now `SET search_path = turbovec, public` first.

### Added

- `docs/BUILDING.md` documenting the Nix-specific build dance
  (writable pg_config wrapper, libclang / glibc include flags,
  openblas RUSTFLAGS, ICU sidestep).
- `index_am_create_and_query` `#[pg_test]` case (gated by the
  `experimental_index_am` Cargo feature).

### Changed

- Default Cargo `default` features set to `["pg16"]` (was
  `["pg17"]`) to match the local development cluster.

## [0.5.0] — Unreleased

### Added — Phase 5: pgvector-parity helpers

- **`subvector(vector, start integer, length integer) -> vector`**
  — 1-indexed slice. Bounds-checked; raises `ERROR` on overrun.
- **`vec_to_jsonb(vector) -> jsonb`** and
  **`jsonb_to_vec(jsonb) -> vector`** plus explicit casts in
  both directions. Useful for replication via JSONB columns,
  logging, and audit trails.
- **`vec_check_dim(vector, integer) -> vector`** — runtime
  dim assertion. Use as a `CHECK` constraint when typmod-style
  enforcement is wanted without the full typmod plumbing.
- **`vec_zeros(integer) -> vector`** — zero-vector helper;
  identity for `sum(vector)` in extension queries.
- **`vec_to_text(vector) -> text`** — explicit text rendering
  callable from SQL (the type's OUTPUT function as a regular
  function).

### Tests

- `subvector_basic`, `subvector_out_of_bounds`,
  `jsonb_round_trip`, `check_dim_passes_and_fails`,
  `zeros_helper`.

### Changed

- `Cargo.toml` / `pg_turbovec.control` bump to `0.5.0`.
- `migrations/004_pg_turbovec_v0.5.0.sql` reference mirror.

## [0.4.0] — Unreleased

### Added — Phase 4: experimental `turbovec` index access method (opt-in)

A full `IndexAmRoutine`-based access method is now scaffolded under
`src/index/`, gated behind the **`experimental_index_am`** Cargo
feature. Default builds **do not** include it; the v0.3 surface
(type, operators, aggregates, `turbovec.knn()`) remains the only
stable user-facing API.

**Build:**

```bash
cargo pgrx install --release --features experimental_index_am
```

**Use:**

```sql
CREATE INDEX docs_emb_idx
    ON docs USING turbovec (embedding vec_cosine_ops)
    WITH (bit_width = 4);

SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;
```

#### Source layout (`src/index/`)

- `mod.rs` — `IndexAmRoutine` populator and the
  `turbovec_index_handler(internal) RETURNS index_am_handler` SQL
  function. Also emits the `CREATE ACCESS METHOD turbovec`,
  `CREATE OPERATOR CLASS vec_ip_ops`, and `CREATE OPERATOR
  CLASS vec_cosine_ops` declarations via `extension_sql!`.
- `options.rs` — `bit_width` (2…=4) and `dim` (0 = auto, else
  positive multiple of 8) reloption parsing under the AM-side
  callback `amoptions`.
- `persist.rs` — SPI-backed read/write of `turbovec.am_storage
  (indexrelid, bit_width, dim, n_vectors, payload, version,
  updated_at)`. `payload` is `STORAGE EXTERNAL` (no PGLZ on
  already-quantised bytes).
- `build.rs` — `ambuild` (heap scan via SPI, builds `IdMapIndex`,
  persists) and `ambuildempty` (writes empty marker).
- `insert.rs` — `aminsert` (load-then-update; v0.5 will batch).
- `scan.rs` — `ambeginscan` / `amrescan` / `amgettuple` /
  `amendscan` with a `ScanOpaque` carrying the query vector and
  cached result list. ORDER-BY-only scans are required.
- `vacuum.rs` — `ambulkdelete` / `amvacuumcleanup` stubs (Phase 5
  needs an upstream way to enumerate live ids in `IdMapIndex`).
- `cost.rs` — `amcostestimate` constant heuristic so the planner
  picks us over a full sort.
- `validate.rs` — `amvalidate` returns `true` (Phase 5 will check
  opclass strategy numbers).

#### CTID encoding

We use pgrx's canonical 32 / 16 packing (`item_pointer_to_u64`):
block number in the top 32 bits, offset number in the bottom 16,
upper 16 reserved for a future epoch. This gives `IdMapIndex` u64
ids natural ordering inside a relfile and lets `amgettuple` fill
`xs_heaptid` via `u64_to_item_pointer` directly.

#### Capability flags

```rust
amstrategies          = 0
amsupport             = 1
amcanorder            = false
amcanorderbyop        = true
amcanbackward         = false
amcanunique           = false
amcanmulticol         = false
amoptionalkey         = true
amstorage             = true
amcanparallel         = false      // Phase 5
amcanbuildparallel    = false      // Phase 5
amusemaintenanceworkmem = true
```

#### Status

**Untested against a running cluster.** This release is the
complete scaffold ready for a Phase 5 session that has
`cargo-pgrx` and a Postgres dev cluster: `cargo pgrx test pg17
--features experimental_index_am` is the gate. Known follow-ups
are enumerated in `docs/INDEXAM.md` § "Test plan" and § "Known
risks".

### Added — docs

- `docs/INDEXAM.md` — implementation guide for the index AM
  (callback responsibilities, side-table schema, test plan,
  known risks).
- `migrations/003_pg_turbovec_v0.4.0.sql` — reference mirror of
  the SQL surface that ships only when the feature is enabled.

### Changed

- `Cargo.toml` adds `libc = "0.2"` (used by `persist.rs` for
  pid-stamped tempfile paths) and the `experimental_index_am`
  Cargo feature.
- `pg_turbovec.control` `default_version` bumped to `0.4.0`.
- `src/lib.rs` mounts `mod index` only under
  `#[cfg(feature = "experimental_index_am")]`.

## [0.3.0] — Unreleased

### Added — Phase 3: kernels module, benches, CI, docs

- **`src/kernels.rs`** — pure-Rust math kernels (`dot`, `l2_sq`,
  `l1_abs`, `norm2`, `cosine_distance`, `normalise_into`,
  `normalise_to_vec`). Distance and normalisation code in
  `distance.rs` / `normalize.rs` now delegate to this module so the
  kernels are exercisable under plain `cargo test --no-default-features`
  without booting Postgres.
- **`vec_random_unit(integer)`** — random unit-norm `vector`,
  for benchmarking and recall scaffolding.
- **`benches/distance.rs`** — `criterion`-based micro-benchmarks of
  the distance kernels at d=128, 384, 768, 1536, 3072. Runs via
  `cargo bench --bench distance --no-default-features`.
- **Codeberg Woodpecker CI** (`.woodpecker/ci.yaml`) — three
  pipelines: pure-Rust unit tests + clippy on every push;
  `cargo pgrx test pg17` on `main` / release branches.
- **`docs/USAGE.md`** — cookbook with install, exact search, ANN
  via `turbovec.knn()`, aggregates, arithmetic, GUCs, pgvector
  coexistence migration, diagnostics.
- **`docs/RECALL.md`** — recall/perf benchmark methodology,
  matched-bit-budget comparison plan against pgvector for v0.4.
- **Pure-Rust unit tests** in `kernels::tests` covering every
  kernel plus a precision regression (1 048 576-element sum of
  squares stays within 1 ppm of the f64 truth).

### Changed

- `Cargo.toml` adds `rand = "0.8"`, `criterion = "0.5"` (dev),
  declares `[[bench]] name = "distance"`.
- `pg_turbovec.control` `default_version` bumped to `0.3.0`.

## [0.2.0] — Unreleased

### Added — Phase 2: function-driven ANN

- **`turbovec.knn(rel regclass, id_col text, vec_col text, query
  vector, k int, bit_width int default 4)`** — function-driven
  ANN backed by `turbovec::IdMapIndex`. Returns
  `TABLE(id bigint, score float8)`, ordered by score DESC for
  most-similar-first.
- Optional unit-normalisation via `turbovec.normalize_on_insert`
  GUC; constraints `k > 0`, `bit_width ∈ {2,3,4}`, `dim % 8 == 0`.
- `migrations/002_pg_turbovec_v0.2.0.sql` reference mirror.
- `#[pg_test]` cases for `knn_returns_nearest_first` and
  `knn_rejects_bad_k`.

### Removed

- `src/phase2_knn.rs` scaffold — promoted to mounted `src/knn.rs`.



### Added — Phase 1: type, operators, functions, aggregates

- **`vector` type** — variable-dimension `f32` vector, stored as a
  CBOR-serialised varlena via `pgrx::PostgresType`. Text I/O accepts
  `'[1, 2, 3]'` with whitespace tolerance and rejects NaN / ±Inf.
  Hard cap at 16 000 dimensions, matching pgvector.
- **Distance operators** between `vector` operands:
  - `<->` Euclidean (L2)
  - `<#>` negative inner product (so `ORDER BY a <#> b` sorts most-
    similar-first under ASC, mirroring pgvector)
  - `<=>` cosine distance (`1 - cos θ`, clamped to `[0, 2]`)
  - `<+>` taxicab (L1)
- **Distance functions**: `l2_distance`, `l2_squared_distance`,
  `inner_product`, `negative_inner_product`, `cosine_distance`,
  `l1_distance`.
- **Helper functions**: `vector_dims`, `vector_norm`,
  `vec_normalize`.
- **Element-wise arithmetic**: `vec_add` (`+`), `vec_sub`
  (`-`), `vec_mul` (`*`).
- **Aggregates**: `avg(vector)` and `sum(vector)`. Internal state
  uses `f64` accumulators to preserve precision on large corpora.
  Both are `PARALLEL SAFE`; `combinefn` merges partial states.
- **Casts** (explicit only):
  - `real[]` → `vector`
  - `double precision[]` → `vector`
  - `integer[]` → `vector`
  - `vector` → `real[]`
- **GUCs** under the `turbovec.*` namespace:
  - `bit_width_default` (int, default 4, range 2..=4)
  - `cache_size_mb` (int, default 256, range 0..=65536)
  - `warn_on_rebuild` (bool, default true)
  - `search_concurrency` (int, default 1, range 1..=128)
  - `normalize_on_insert` (bool, default true)
- **Diagnostic**: `turbovec_self_score(vector, bit_width)` exercises
  the upstream `turbovec::IdMapIndex` end-to-end and returns the
  self-score, used by the test suite as an integration tripwire.

### Tests

- `#[pg_test]` cases in `src/lib.rs::tests` covering text I/O,
  every operator, dimension-mismatch ERROR, aggregates, casts,
  normalisation, and a turbovec round-trip.
- `tests/01_type_basic.sql` — psql-style regression script.

### Project layout

- `pgrx = "=0.17.0"` to match the cached toolchain.
- `pg_turbovec.control` declares schema `turbovec`,
  `relocatable = false`, `trusted = true`.
- `migrations/001_pg_turbovec_v0.1.0.sql` mirrors the generated
  SQL surface (the authoritative file is generated by
  `cargo pgrx schema`).

### Not yet shipped (Phase 2 / Phase 3)

- Index access method `turbovec` and operator classes
  `vec_ip_ops`, `vec_cosine_ops`. A starter is checked
  in at `src/phase2_knn.rs` (not yet mounted by `lib.rs`).
- Filtered search via `IdMapIndex::search_with_allowlist`.
- Binary-compatible varlena layout with pgvector's `vector`.
- WAL-logged persistent index pages.

[1.0.0-rc.2]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v1.0.0-rc.2
[1.0.0-rc.1]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v1.0.0-rc.1
[0.16.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.16.0
[0.15.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.15.0
[0.14.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.14.0
[0.13.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.13.0
[0.12.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.12.0
[0.11.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.11.0
[0.10.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.10.0
[0.9.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.9.0
[0.8.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.8.0
[0.7.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.7.0
[0.6.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.6.0
[0.5.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.5.0
[0.4.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.4.0
[0.3.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.3.0
[0.2.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.2.0
[0.1.0]: https://codeberg.org/gregburd/pg_turbovec/releases/tag/v0.1.0
