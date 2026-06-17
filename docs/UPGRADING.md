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
| 1.13.x → 1.13.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |
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
