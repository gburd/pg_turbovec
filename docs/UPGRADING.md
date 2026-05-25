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
| 1.3.x → 1.3.x+1 (patch) | _none_ | none | Wire format is frozen across patch releases. |

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
