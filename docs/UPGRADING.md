# Upgrading pg_turbovec

`pg_turbovec` follows [SemVer](https://semver.org/) with a strict
data-format compatibility contract:

- **Patch releases** (`X.Y.Z` → `X.Y.Z+1`) **never change the on-disk
  index format.** Drop in the new shared library, restart, scan; no
  `REINDEX` required.
- **Minor releases** (`X.Y` → `X.Y+1`) **may bump the on-disk format
  version** when there's a clear performance / correctness win that
  can't be expressed at the existing version. When they do, the new
  binary detects the old format on first open and emits a
  `REINDEX`-pointed `ERROR` rather than silently corrupting or
  returning bad results.
- **Major releases** (`X` → `X+1`) **may make breaking changes** to
  the SQL surface in addition to the on-disk format.

Every release that bumps the on-disk format **ships with**:

- A clear `ERROR` from `ambeginscan` saying "this index was built
  under pg_turbovec ≤ X.Y; run `REINDEX INDEX <name>;` to migrate".
- A row in the migration matrix below.
- A test under `src/lib.rs` (search for `legacy_v1`-style names) that
  exercises the detection primitive.

## Migration matrix

| From | To | Required action | Notes |
|---|---|---|---|
| 1.0.x (side-table only) | 1.3.0+ | `REINDEX INDEX <name>;` per index | The 1.0.x indexes have an empty main fork and a `turbovec.am_storage` row; v1.3.0 drops that table during `ALTER EXTENSION pg_turbovec UPDATE` (`migrations/005_pg_turbovec_v1.3.0.sql`). The index is unscannable until reindexed. |
| 1.1.x (side-table only) | 1.3.0+ | `REINDEX INDEX <name>;` | Same as 1.0.x. |
| 1.2.x with `--features relfile_storage` | 1.3.0+ | `REINDEX INDEX <name>;` | The 1.2 relfile preview wrote `MetaPageData::version = 1`. v1.3.0 introduced the pre-baked SIMD-blocked layout + codebook, bumping `VERSION` to 2. The detection primitive in `src/index/page.rs::MetaPageData::is_legacy_v1()` fires on these. |
| 1.2.x without `relfile_storage` | 1.3.0+ | `REINDEX INDEX <name>;` | Same boat as 1.0/1.1. |
| 1.3.x | 1.4.0+ | `REINDEX INDEX <name>;` per index | v1.4.0 introduced the persisted rotation matrix in the relfile (`MetaPageData::version` 2→3). v1.3.x indexes have an empty rotation chain so the new binary detects them via `MetaPageData::is_legacy_v2()` and ERRORs out cleanly. The lazy QR was the warm-scan hotspot (~64.8% self time on dbpedia-1M), so persisting the matrix closes the gap to pgvector HNSW. |
| 1.3.x → 1.3.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.4.x → 1.4.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.4.x | 1.5.0+ | _none_ | v1.5.0 (Phase R-3) is a scan-side change only: the `ambeginscan` cache-fill path now mmaps the deterministic static regions of the relfile (blocked codes + rotation matrix + inline codebook) instead of pulling them through the buffer manager. The on-disk format (`MetaPageData::version = 3`) is byte-identical to v1.4.x. No REINDEX. The fall-back GUC `turbovec.mmap_static_blocked = off` reverts to the v1.4.x scan path on a per-session basis. See `docs/ARCHITECTURE.md` § "Index AM · mmap isolation contract" for the consistency story. |
| 1.5.x → 1.5.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.5.x | 1.6.0+ | _none_ | v1.6.0 (Phase W) is a build-side change only: `ambuild` now streams the heap scan into `IdMapIndex::add_with_ids` in chunks bounded by `maintenance_work_mem` (capped at 1 GiB) instead of accumulating the entire heap-scan output in a single `Vec<f32>`. Peak `CREATE INDEX` memory drops from ~121 GiB to ~16 GiB at 10 M × 1536-d × 4-bit. The on-disk format (`MetaPageData::version = 3`) is byte-identical to v1.5.x. **No REINDEX required.** Existing v1.5.x indexes continue to work unchanged; the v1.6.0 binary's ambuild path simply uses less memory on the next CREATE INDEX / REINDEX. |
| 1.6.x → 1.6.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.6.x | 1.7.0+ | _none_ | v1.7.0 (Phase W-2) is a build-side change only: `ambuild` now writes `packed_codes` to relfile pages, materialises the SIMD-blocked layout via `prepare_eager()`, drops `packed_codes` via the new `IdMapIndex::take_packed_codes()` (turbovec 0.7.0), then writes the blocked + rotation chains and stamps the meta page LAST. Peak `CREATE INDEX` memory drops from ~22.5 GiB to ~15 GiB at 10 M × 1536-d × 4-bit (8× total reduction vs pre-Phase-W). The on-disk format (`MetaPageData::version = 3`) is byte-identical to v1.6.x. **No REINDEX required.** Existing v1.6.x indexes continue to work unchanged; the v1.7.0 binary's ambuild path simply uses less memory on the next CREATE INDEX / REINDEX. |
| 1.7.x → 1.7.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.7.1 specifically reverts the v1.7.0 (Phase W-2) build-path reordering after the 10 M × 1536-d validation on `meh` showed the split-write design made the build 53% slower (5052 → 7748 s) and used 2.7 GiB of swap (vs 0 in v1.6.0) without actually lowering peak RSS — the pinned-shared-buffer component of `ps -o rss` ate the predicted savings. v1.7.2 is a test-only patch that adds automated `#[pg_test]`s for the upgrade matrix (Phase Y): forged-meta-page detection of pre-v1.4 wire formats and migration-file drift checks. **v1.7.3** upgrades the turbovec kernel fork from the v0.7.0-era `6e80a59` to a fork rebased onto upstream **v0.9.0** (`d3d468e`), fixing a kernel bug where x86_64 CPUs WITHOUT AVX2 returned silently-wrong / repeated top-k from indexed ANN scans (the perm0-interleave scalar-fallback bug, upstream PR #108 / issue #106). TQ+ calibration is adopted as identity (no recall or wire change); security hardening (`MAX_DIM`, NaN/Inf rejection) comes along. The on-disk format is byte-identical across v1.6.0 / v1.7.0 / v1.7.1 / v1.7.2 / v1.7.3; **no REINDEX needed** when upgrading or downgrading among them. Pre-AVX2 x86_64 users specifically should take v1.7.3 to clear the wrong-results bug. See an internal design note § "Phase W-2 reverted in v1.7.1" and `docs/PRODUCTION.md` § "Known issues". |
| 1.7.x | 1.8.0+ | _none_ | v1.8.0 is a competitive-parity minor: iterative index scan (`turbovec.iterative_scan`, `turbovec.max_scan_tuples`), parallel index build (`turbovec.build_parallelism`), cold-scan latency cut (lazy `id_to_slot` on the read path), and additive `\|\|` concat + halfvec `+`/`-`/`*` arithmetic. All changes are scan-side, build-side, or additive SQL surface; **the on-disk relfile format is byte-identical to v1.7.x** (`MetaPageData::version = 3`). **No REINDEX needed.** The new GUCs default to pgvector-equivalent behaviour (`iterative_scan = relaxed_order`). The additive operators/functions are created by the generated `1.7.3--1.8.0` upgrade script; `ALTER EXTENSION pg_turbovec UPDATE TO '1.8.0';` is sufficient. See an internal design note. |
| 1.8.x → 1.8.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.8.x | 1.9.0+ | _none_ | v1.9.0 adds `turbovec.oversample` (tunable recall — fetch `search_k * oversample` quantized candidates, reorder-queue trims to exact top-k), plus test-coverage hardening and the first published benchmark. The GUC is additive and defaults to 1.0 (no-op). On-disk format byte-identical to v1.7.x / v1.8.x (`MetaPageData::version = 3`). **No REINDEX.** `ALTER EXTENSION pg_turbovec UPDATE TO '1.9.0';` is sufficient. |
| 1.9.x → 1.9.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.9.1 is bench-results-only (the AVX2 latency-frontier run on `arnold` + the honest positioning correction it produced); no source or wire change. |
| 1.4.x – 1.9.x (flat) | 1.10.0+ | _none_ | v1.10.0 adds the IVF coarse-quantizer layer and bumps `MetaPageData::version` 3 → 4 — BUT a v1.10.0 binary reads any existing v3 (flat) index as a `lists = 0` flat index, byte-compatible. **No REINDEX needed** to upgrade. `ALTER EXTENSION pg_turbovec UPDATE TO '1.10.0';` registers the new `lists` / `assign_dups` reloptions and `turbovec.probes` / `turbovec.max_probes` GUCs. Opt into IVF only by rebuilding a chosen index with `WITH (lists = N)` (recommended `N ≈ sqrt(n)`); that index is then v4-IVF, while un-rebuilt indexes stay v4-flat. See an internal design note. |
| 1.10.x → 1.10.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.10.1 is bench-results-only (the AVX2 IVF warm-p50 measurement confirming the ~5×-vs-full-scan latency win); no source or wire change. |
| 1.10.x | 1.11.0+ | _none_ | v1.11.0 hardens IVF for production: it survives VACUUM via tombstones (a v4-ADDITIVE per-slot bitmap chain) instead of silently degrading to a flat scan, and adds the `turbovec.index_is_degraded(regclass)` function + a throttled degradation WARNING; IVF k-means builds ~7.8× faster (build-internal). Wire stays `MetaPageData::version = 4` — the tombstone bitmap + `ivf_degraded` flag are additive, so pre-1.11.0 v4 indexes read as not-degraded/no-tombstones. **No REINDEX.** `ALTER EXTENSION pg_turbovec UPDATE TO '1.11.0';` registers the new function. |
| 1.11.x → 1.11.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.11.1 is bench-results-only (the Phase A-2 IVF latency frontier measurement + Phase B out-of-core design); no source or wire change. |
| 1.11.x | 1.12.0+ | _none_ | v1.12.0 makes the IVF build out-of-core (spills the corpus to a PG temp file; peak RSS at 1M×1024-d ~14 GiB → ~7.1 GiB, 5M buildable on a 31 GiB host). Build-internal only — the on-disk relfile is byte-identical to a v1.11.x in-memory build for the same input, `MetaPageData::version` stays 4. **No REINDEX**; the benefit applies to the next `CREATE INDEX` / `REINDEX`. `ALTER EXTENSION pg_turbovec UPDATE TO '1.12.0';` is sufficient. |
| 1.12.x → 1.12.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.12.x | 1.13.0+ | _none_ | v1.13.0 adds out-of-core IVF *query* (`turbovec.out_of_core` enum, default `auto`): an IVF index larger than RAM can now be served cell-scoped (caches only bounded metadata + an mmap; copies just the probed cells' code ranges per query). Scan-path only — results are identical to the whole-load path, `MetaPageData::version` stays 4. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.13.0';` is sufficient. |
| 1.13.x → 1.13.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.13.1 is docs + bench only (Phase C metadata-filtering guide + allowlist crossover measurement + drift fixes); no source-logic or wire change. |
| 1.13.x | 1.14.0+ | _none_ | v1.14.0 (Phase D) adds the multivector / hybrid SQL surface: `turbovec.max_sim` / `max_sim_cosine` (ColBERT-style MaxSim re-rank over `vector[]`) and `turbovec.rrf_score` (reciprocal rank fusion). Additive functions only — no index-AM change, `MetaPageData::version` stays 4. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.14.0';` is sufficient. |
| 1.14.x → 1.14.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.14.x | 1.15.0+ | _none_ | v1.15.0 (Phase C follow-up) adds the operator-path allowlist: a session GUC `turbovec.allowlist` (CSV of heap TIDs) that flows a pre-materialized id-set into the `ORDER BY emb <=> q` scan for in-kernel block-skip pushdown on flat + IVF, plus the `turbovec.tid_to_bigint(tid)` encoder. Additive GUC + function only — no index-AM scan-key rewrite, `MetaPageData::version` stays 4. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.15.0';` is sufficient. |
| 1.15.x → 1.15.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.15.1 is a build-only fix (the Phase B-4 BufFile spill didn't compile on pg13/14/15/18 from v1.12.0–v1.15.0); no wire or runtime change on pg16/pg17. |
| 1.15.x | 1.16.0+ | _none_ | v1.16.0 (Phase F-1) adds `turbovec.colbert_search` — index-accelerated stage-1 ColBERT late interaction (backend-cached token index + exact `max_sim` stage-2 rerank). Additive function only; the token index is backend-cache-only (no relfile), `MetaPageData::version` stays 4. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.16.0';` is sufficient. |
| 1.16.x → 1.16.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
| 1.17.x → 1.17.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. v1.17.1 is docs + bench only (the F-2 ColBERT recall win confirmed cross-domain on NFCorpus); no source or wire change (single-vector stays v4, colbert v5). |
| 1.17.x | 1.18.0+ | _none_ | v1.18.0 (Tier-1 IVF latency) lowers the default `turbovec.search_k` 100 → 32 (the recheck floor, not the scan, dominates per-query latency; recall@10 plateaus by ~25) and documents that raising `WITH (assign_dups = M)` reaches matched recall at fewer probes. Scan-path / default-tuning only — no SQL surface or wire change, `MetaPageData::version` stays 5. **No REINDEX** (the `assign_dups` lever is opt-in and only takes effect on a fresh build). `ALTER EXTENSION pg_turbovec UPDATE TO '1.18.0';` is sufficient. |
| 1.18.x | 1.19.0+ | _none_ | v1.19.0 removes the direct relfile `mmap`: all index data is now read through PostgreSQL's buffer manager (`ReadBufferExtended`). Required for managed/sandboxed Postgres. Read-path only — no SQL surface or wire change, `MetaPageData::version` unchanged; out-of-core IVF serving is preserved (the cell-scoped gather reads only probed cells' pages via the buffer manager). `turbovec.mmap_static_blocked` is deprecated to a no-op. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.19.0';` is sufficient; size `shared_buffers` to hold the hot index for best cold-fill latency. |
| 1.19.x | 1.20.0+ | _none_ | v1.20.0 lands IVF scaling: a sublinear two-level coarse quantizer (O(lists)→O(√lists) cell selection, computed in-memory from the persisted centroids, auto for lists>4096), a `turbovec.scan_parallelism` GUC (parallel per-query fine-scan, opt-in), and a memory-bounded byte-identical parallel build. In-memory / scan-path / build-path only — no wire change (`MetaPageData::version` stays 5), one new GUC. **No REINDEX** — existing v4/v5 IVF indexes get the sublinear coarse + parallel scan on the next scan with no rebuild. `ALTER EXTENSION pg_turbovec UPDATE TO '1.20.0';` is sufficient. |
| 1.20.0 | 1.20.1+ | _none_ | **v1.20.1 is a CRITICAL PERF FIX, not a feature change.** `turbovec.iterative_scan` default flips `relaxed_order` → `off`. Root cause: PostgreSQL's reorder queue (`IndexNextWithReorder`) can only return a tuple early when the AM's advertised `ORDER BY` value is exact; pg_turbovec always advertises `NEG_INFINITY` (opclass-agnostic safety), so under the old default the executor was forced to drive the AM's own iterative-refill schedule to completion on EVERY `ORDER BY ... LIMIT` query — measured **~450x** slower (SIFT-1M/128d: ~2ms vs ~900ms) than with `off`. Scan-side GUC-default-only — no wire change, no SQL surface change, `MetaPageData::version` stays 5. **No REINDEX**. If your workload relies on `relaxed_order`'s under-return-avoidance for a selective `WHERE` filter, opt back in with `SET turbovec.iterative_scan = relaxed_order;`. `ALTER EXTENSION pg_turbovec UPDATE TO '1.20.1';` is sufficient. |
| 1.20.x | 1.21.0+ | _none_ | v1.21.0 (Phase G-1, an internal design note) adds a small in-memory undirected graph (Vamana/HNSW-lite, fixed out-degree 16, symmetrized for recall-safe greedy search) over the IVF coarse centroids: `coarse_probe` on the out-of-core cell-scoped path can navigate the graph instead of scoring every centroid once `lists >= 4096`. The graph is built ONCE PER BACKEND, IN-MEMORY, from the already-persisted coarse centroids (deterministic; see `centroid_graph_build_deterministic` / `ivf_coarse_graph_build_is_deterministic_across_cache_rebuilds`) — nothing new is persisted, `MetaPageData::version` stays 5. One new GUC, `turbovec.coarse_graph` (off/auto/on, default `auto`). **No REINDEX** — existing v4/v5 IVF indexes get the graph (when `lists >= 4096`) on the next scan with no rebuild. `ALTER EXTENSION pg_turbovec UPDATE TO '1.21.0';` is sufficient. Note: the v1.20.0 row above (and its CHANGELOG entry) describes a "sublinear two-level coarse quantizer" that was never actually implemented in that release — v1.20.0 shipped parallel k-means seeding/build and `turbovec.scan_parallelism` only; `coarse_probe` stayed the plain O(lists·dim) linear scan until this release's graph. v1.21.0 is the first release to actually ship sublinear coarse-cell selection. |
| 1.21.x | 1.22.0+ | _none_ | v1.22.0 is a repo-cleanup release, no functional change. `turbovec.mmap_static_blocked` — a deprecated no-op since v1.19.0 (it toggled a relfile-mmap fast path deleted that release) — is **removed** after a three-minor deprecation window (v1.19.0 warn → v1.20.0/v1.21.0 still-warning → v1.22.0 remove), per AGENTS.md's SQL-surface-removal policy. `SET turbovec.mmap_static_blocked = ...` now errors like any unknown GUC instead of silently no-op'ing. Also: fixed a stale dead-code warning, deleted a test made meaningless by the mmap removal, `cargo fmt`'d the whole tree (244 pre-existing formatting violations, purely cosmetic — `fmt-check` was never wired into real CI before this release, only into an already-dead `.woodpecker/ci.yaml`, which is also removed), and fixed literal `\uXXXX` escape-sequence artifacts in several doc files. No wire-format change (`MetaPageData::version` stays 5), no other SQL surface change. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.22.0';` is sufficient. |
| 1.22.0 | 1.22.1+ | _none_ | v1.22.1 closes a real fraction of the IVF build-cliff gap: `gemm_lloyd_assign`'s Lloyd-loop cross-term GEMM (dominant k-means training cost at high `lists`) now runs `Parallelism::Rayon(0)` instead of `Parallelism::None`, respecting `turbovec.build_parallelism` automatically. Bit-identical centroid output confirmed (empirically and via a new regression test). Measured on real GIST-1M-scale k-means training (16-core AVX-512 a cloud VM): 2686.6s → 768.4s, a 3.50× speedup. Scan/build-path only — no wire-format change, no SQL surface change, no new/changed GUC or reloption. **No REINDEX**: this changes build wall clock only, not the on-disk bytes. `ALTER EXTENSION pg_turbovec UPDATE TO '1.22.1';` is sufficient. |
| 1.22.1 | 1.22.2+ | _none_ | v1.22.2 raises `turbovec.probes`'s default from 8 to 16 — the old default capped out-of-the-box recall at R@10=0.796 (SIFT-1M) / R@10=0.407 (GIST-1M), well below any reasonable SLO. `probes=16` measures R@10=0.918 / 0.557 respectively for ~1.5-1.6× the latency. Existing sessions/deployments that explicitly `SET turbovec.probes` are unaffected. Scan-side default only — no wire-format change, no SQL surface change. **No REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.22.2';` is sufficient. |
| 1.25.1 | 1.26.0+ | _none_ | v1.26.0 (Phase G-2d(a)) adds a **partitioned/merge PARALLEL build** for the graph index kind (`WITH (graph = true)`) so it scales past the single-pass serial ceiling (which didn't complete at 5M rows). Partition into P shards → build each in parallel → stitch via a parallel cross-shard refinement + reverse-edge pass. New GUC `turbovec.graph_build_partitions` (int, default auto: derives P from corpus size + build-pool budget; 0/1 forces single-pass; N forces N shards). **Build-time change only — emits the IDENTICAL on-disk v6 CSR shape**, so no wire-format change, no new operators/types/functions, and **no REINDEX** (existing graph indexes unaffected; only NEW graph builds use the parallel path). Verified: recall parity (partitioned matches or BEATS single-pass, 0.958→0.996 R@10 in a findable regime), ~8× build speedup (P=16, 200k rows, 8-core box), bit-identical determinism across (corpus, seed, P) and rayon pool sizes. `ALTER EXTENSION pg_turbovec UPDATE TO '1.26.0';` is sufficient. |
| 1.27.0 | 1.27.1+ | _none_ | v1.27.1 (Phase Q-4a) parallelizes the IVF k-means build — a build-SPEED change only. The persisted IVF centroids + assignment are BYTE-IDENTICAL to v1.27.0 for a fixed (corpus, seed, lists, dim); no wire change (stays v7), no SQL-surface change, **no REINDEX**. Two remaining serial hot loops (`gemm_lloyd_assign`'s per-row argmin, `rotate_corpus_into`'s GEMM) were parallelized bit-identically; measured ~1.91× faster IVF builds (sub-linear — Lloyd iterations are sequentially dependent). Only NEW `WITH (lists = N)` builds are affected (faster); existing indexes unchanged. `ALTER EXTENSION pg_turbovec UPDATE TO '1.27.1';` is a no-op upgrade. |
| **1.4.x – 1.26.x (ANY kind)** | **1.27.0+** | **`REINDEX INDEX <name>;` per index** | **v1.27.0 (Phase Q-0) de-duplicates the on-disk quantized-codes storage, roughly halving the per-vector index footprint** — the storage blocker cleared for the large-index storage target. Prior versions persisted each vector's codes TWICE: the row-major bit-plane `packed_codes` chain AND the SIMD-`blocked` chain (`pack::repack` output). Since the blocked layout is a pure function of the packed codes, v7 drops the blocked chain from disk and recomputes it once per backend at index-open (per-query latency unchanged; scan results bit-identical). **This bumps `MetaPageData::version` 6 → 7 and is NOT additive** — a v7 relfile has no blocked chain, so it is not byte-compatible with any prior version for ANY kind (single-vector, ColBERT, IVF, graph); all kinds now emit v7 (the `kind` byte still discriminates). A pre-v7 index (v1..v6) is detected by `MetaPageData::is_legacy_v6()` and `ambeginscan` ERRORs with `HINT: REINDEX INDEX <name>;` at first scan (never silent corruption). No SQL-surface change. **Migration:** `ALTER EXTENSION pg_turbovec UPDATE TO '1.27.0';` then `REINDEX INDEX <name>;` once per index. Until reindexed, scans ERROR with the hint (they do NOT return wrong results). |
| 1.25.0 | 1.25.1+ | _none_ | v1.25.1 is a **release-tooling + docs/benchmark patch** — no shippable code change (binary byte-identical to v1.25.0). Adds the tag-triggered PGXN + postgresql.org-news publish pipeline and the Qdrant/ANN-Benchmarks competitive benchmark (which validated v1.25.0's `hi_dim_rerank` at scale: GIST-960-1M recall 0.876→0.953). No wire-format change (stays v6), no SQL-surface change, **no REINDEX**. `ALTER EXTENSION pg_turbovec UPDATE TO '1.25.1';` is a no-op upgrade. |
| 1.24.0 | 1.25.0+ | _none_ | v1.25.0 adds **`turbovec.hi_dim_rerank`** (enum off\|auto\|on, default auto) — a dimension-aware exact-L2 rerank-window widening that recovers high-dimensional recall. The offline Gap-B investigation (an internal design note) showed the high-dim gap (GIST-1M/960d ~0.86) is an in-cell quantized-RANKING loss, NOT a retrieval ceiling — the true NNs land in the probed cells (cell recall 0.98-0.996 at probes 64-128); a wider exact-L2 recheck recovers them (an SQ4 analog: R@10 0.666→0.978 at 960d). `auto` applies a `clamp(dim, 256..=1024)` candidate floor only for `dim >= 256` (SIFT-128 untouched; an explicit `search_k`/`oversample` override past the floor wins). One new GUC, additive; **no wire-format change** (stays v6), no new operators/types/functions, **no REINDEX**. The new default improves high-dim recall out of the box at a small high-dim-only latency cost; `SET turbovec.hi_dim_rerank = off` restores exact pre-1.25.0 candidate behaviour. `ALTER EXTENSION pg_turbovec UPDATE TO '1.25.0';` is sufficient. |
| 1.23.0 | 1.24.0+ | _none_ | v1.24.0 (Phase G-2b) adds **VACUUM + incremental INSERT support for the graph index kind** (`WITH (graph = true)`). Both previously raised a clear `ERROR` (v1.23.0 was build+scan only); they now work. **NO wire-format change** — wire format stays v6, byte-identical to v1.23.0; existing v4/v5/v6 indexes all decode unchanged, **no REINDEX**. VACUUM uses the same per-slot tombstone bitmap IVF already uses; aminsert is a deliberate O(n)-per-row whole-relfile rewrite (build-then-serve model; heavy churn should still REINDEX). Two real bugs fixed en route: a tombstone-chain/graph-adjacency-chain block-offset collision that corrupted a graph index on insert-after-VACUUM, and a VACUUM entry-point fallback that missed the "entry point survives but all its neighbors got tombstoned" dead-end. Both are binary fixes; neither changes the on-disk format. Still deferred: G-2c (SIMD traversal), G-2d (5M-scale HNSW-latency gate). `ALTER EXTENSION pg_turbovec UPDATE TO '1.24.0';` is sufficient. |
| 1.22.2 | 1.23.0+ | _none_ | v1.23.0 adds `WITH (graph = true)`, a new opt-in Vamana-style graph index kind (Phase G-2a). Wire format bumped to v6, ADDITIVE per kind — existing v4 (single-vector) and v5 (ColBERT) indexes decode byte-identical under the v6 binary (verified by dedicated tests). **No REINDEX for any existing index.** A graph index is a brand-new on-disk shape only a v6 binary produces; there is no in-place migration into it — build one explicitly. Correctness-first release: real Vamana build + scan with verified recall, but VACUUM/aminsert against a graph index raise a clear `ERROR` (not yet supported — rebuild after bulk changes), and the real HNSW-latency gate measurement has not yet been run (no latency/recall-vs-HNSW claim made). See CHANGELOG.md and an internal design note. `ALTER EXTENSION pg_turbovec UPDATE TO '1.23.0';` is sufficient. |
| 1.4.x – 1.16.x (single-vector) | 1.17.0+ | _none_ | v1.17.0 (Phase F-2) adds the **PERSISTENT ColBERT token index**: a new `vec_colbert_ops` opclass over a `turbovec.vector[]` column builds a v5 on-disk token index, and `turbovec.colbert_search` reads stage-1 from the relfile instead of rebuilding a backend cache every call. The wire bump 4 → 5 is **strictly additive per index kind**: a single-vector index (`vec_*_ops` over a `vector` column) still emits wire version 4 with a zeroed `kind` byte (page offset 30) — its relfile is **byte-identical to v1.16.0** (verified by the `single_vector_still_emits_v4_bytes` unit test and the `v4_single_vector_index_byte_identical` `#[pg_test]`). A v4 index decodes under the v5 binary as `kind = KIND_SINGLE` (the kind byte was a reserved zero on v4), so `MetaPageData::is_legacy_v4()` **deliberately never trips** and existing single-vector indexes need **no REINDEX**. Only an index built `USING turbovec (col vec_colbert_ops)` over a `vector[]` column is v5 (`kind = KIND_COLBERT`); that index is a brand-new shape (per-token slots, doc TID repeated in the ids chain) with NO `ORDER BY` semantics — `ambeginscan` ERRORs on an ORDER BY scan against it with a HINT to use `turbovec.colbert_search`. There is no in-place migration of a v4 single-vector index into a v5 ColBERT index; a ColBERT index is built fresh. `ALTER EXTENSION pg_turbovec UPDATE TO '1.17.0';` registers the new opclass. |
| 1.9.x | (IVF minor, version TBD by the parent) | _none for flat indexes_ | The IVF layer (an internal design note) bumps `MetaPageData::version` 3→4 to add an opt-in inverted-file index, but the bump is **flat-readable**: a v3 index decodes under the v4 binary as a flat index (`lists = 0`), and `MetaPageData::is_legacy_v3()` deliberately never trips, so `ambeginscan` does NOT error on v3 or v4-flat indexes. Existing indexes keep working with **no REINDEX** — `ALTER EXTENSION pg_turbovec UPDATE` only. The new format is strictly opt-in: only an index built `WITH (lists = N)` (`N > 0`) uses the IVF layout (cell-contiguous codes + persisted coarse centroids + cell directory). A `WITH (lists = 0)` build (the default) is byte-identical to the v3 flat layout modulo the version byte. **IVF-1 ships the build path + on-disk layout only; the scan path is still flat** (cell-restricted search is IVF-2), so building `WITH (lists > 0)` today persists cells but does not yet change query latency. Recommended `lists ≈ √n`. |

If you maintain pg_turbovec for a fleet of clusters, scripting the
migration looks like:

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
`AccessExclusiveLock` so reads keep working during the migration.
The new index is built first; the cutover swap is atomic.

## How the format-version contract is enforced

Three guardrails:

1. **`src/index/page.rs::VERSION`** is the single source of truth. Any
   change to this constant in a patch release is a release-process bug
   and must be reverted before tagging.

2. **`scripts/drift-check.sh` includes a wire-format check.** It
   reads the most recent tag's `VERSION` constant from git history and
   compares it to the working tree. If the working tree has a higher
   `VERSION` than the last tag and the difference between tags is a
   patch bump, drift-check fails the push. (See § 11 of the script.)

3. **`#[pg_test] wire_format_version_is_stable_across_patches`** in
   `src/lib.rs` reads the tag list from git, finds the most recent
   patch line (e.g. `1.3.0` → `1.3.x` for any `x`), and asserts the
   compiled `VERSION` matches what the most-recent patch tag in that
   line emitted. Out-of-tree work (between tags) is allowed to bump
   freely; the gate is at tag time.

## Adding a new minor release that bumps VERSION

Checklist for the release engineer (see `RELEASING.md` for the full
release flow):

1. **Decide the new version layout.** Add fields to `MetaPageData` if
   needed; bump `VERSION` to the next integer. Keep `MIN_DECODE_VERSION`
   at the oldest version still in production.
2. **Write the detection primitive.** Add a method like
   `MetaPageData::is_legacy_vN(&self) -> bool` and a `#[pg_test]
   relfile_legacy_vN_detection_primitive` covering it.
3. **Wire the ERROR in `ambeginscan`.** Match on the legacy-version
   detection and emit a `pgrx::ereport!(ERROR, FEATURE_NOT_SUPPORTED,
   "this index was built under pg_turbovec ≤ X.Y; run \`REINDEX INDEX
   <name>;\`)`.
4. **Add a row to the migration matrix above.**
5. **Add a `migrations/0NN_pg_turbovec_vX.Y.0.sql`** if there's any
   SQL-level change (drop a table, rename a function, etc).
6. **CHANGELOG entry** must include a "Migration" section with the
   exact REINDEX scripts users need to run.
7. **`scripts/drift-check.sh`** will start nagging until `VERSION`
   updates land alongside the version bump in `Cargo.toml`. That's
   intentional — both must change together.

## What "patch release" actually means

A patch release fixes a bug, plugs a security hole, or improves
performance without changing the on-disk format or the SQL surface.
Patch releases include:

- Bug fixes that don't change the page layout.
- Performance improvements to the search kernel that produce
  bit-identical scoring (e.g. SIMD width upgrade, allocator tuning).
- New SQL helper functions that don't change existing ones.
- Documentation, CI, and bench-script changes.

Patch releases explicitly DO NOT include:

- Changes to `MetaPageData` field order or sizes.
- New `MetaPageData` fields (those bump `VERSION` and need a minor).
- Changes to the page-allocation layout (chain offsets, `rows_per_*_page`).
- Changes to the SIMD-blocked layout produced by `pack::repack`
  (would change `blocked_codes_first` / `_count` semantics).
- Changes to the codebook serialisation.
- Changes to existing operator definitions (`<=>`, `<#>`, etc.) or
  function signatures.

If a "bug fix" requires touching any of those, it's a minor release,
not a patch.
