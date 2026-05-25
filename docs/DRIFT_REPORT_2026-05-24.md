# pg_turbovec docs drift report — 2026-05-24

> **Superseded by 1.1.0** — the ARCHITECTURE.md and
> ROADMAP_DECISIONS.md drift items flagged below were resolved in
> the same commit that adds this note. Retained for historical
> context.

First run of the new `.pi/skills/drift-check/SKILL.md` against the
live tree at commit `2753e22` (post-v1.0.1).

## All clear ✅

- **Versions consistent**. `Cargo.toml = 1.0.1`,
  `pg_turbovec.control = 1.0.1`, `CHANGELOG.md` newest entry =
  `[1.0.1]`, latest tag = `v1.0.1`.
- **Test count consistent**. 83 `#[pg_test]` + 10 `#[test]` = 93
  annotations; runner reports `92 passed` after cfg-gating; docs
  cite `92/92 passing` consistently.
- **PG version matrix consistent across three sources**:
  - `Cargo.toml` features: `pg13, pg14, pg15, pg16, pg17, pg18`
  - `docs/PG_VERSION_SUPPORT.md` table: 6 rows
  - `.forgejo/workflows/test.yml` matrix: `[13, 14, 15, 16, 17, 18]`
- **CHANGELOG.md vs git log**. The `[1.0.1]` entry covers every
  user-visible change since `v1.0.0`.
- **Vendored deps** documented. `vendor/turbovec/PATCH_NOTES.md`
  describes the deviation from upstream with an upstreaming plan.

## Drift found 🟡

### `docs/ARCHITECTURE.md` is partially stale

The doc was written during the Phase-2 design phase and describes
some things in the future tense that have shipped:

- §1 still calls `src/index/` "Phase 2 will introduce" — it's
  shipped and contains `build.rs`, `cost.rs`, `insert.rs`,
  `mod.rs`, `options.rs`, `persist.rs`, `scan.rs`, `vacuum.rs`,
  `validate.rs`.
- §3.2 "Phase 2 representation (binary-compatible with pgvector)"
  is documented but the actual decision (skip it) is in
  `docs/ROADMAP_DECISIONS.md`. The two docs disagree.
- §10 "Done so far" milestones list ends at Phase 9 (or so) but
  we shipped Phase 21 (`docs/CHANGELOG.md` § [1.0.0]).
- The doc doesn't mention `vendor/turbovec/`, `src/cache.rs`,
  `src/aggregate.rs`, `src/halfvec*.rs`, `src/sparsevec*.rs`,
  `src/bitvec.rs`, or `src/cast.rs`.

**Suggested fix:** rewrite §1, §3.2, §10 in past tense; add
sub-sections for cache, aggregates, halfvec/sparsevec/bitvec, and
the vendor patch. ~30 min of work; not blocking.

### `docs/ROADMAP_DECISIONS.md` should reference v1.0.1 fixes

The "Where future work would pay off" section (§ Cold-path fix)
describes the Phase H learning but doesn't call out the
turbovec.search_k GUC (Phase D) or the per-backend cache
(Phase E) as shipped Phase 21 items. They live in the
`CHANGELOG.md` only.

**Suggested fix:** add a "Shipped in 1.0.x" section above
"Where future work would pay off", duplicating one-liners from
the changelog. Helps readers navigate "what's done vs what's
next" without bouncing across files.

## Suggestions (not drift, just observations)

- **Add a CONTRIBUTING.md note about the `pg_guard` `_inner`
  collision** (pgrx generates `<fn>_inner` for any `#[pg_guard]`
  function; user-defined helpers must use a different suffix).
  We learned this the hard way during v1.0.1; capturing it
  saves the next contributor's afternoon.
- **`docs/PHASE19_PROGRESS.md`** describes a binary-compat
  varlena handoff, but `docs/ROADMAP_DECISIONS.md` (and
  `CHANGELOG.md` [1.0.0]) explicitly skip it. Either delete
  PHASE19_PROGRESS.md or annotate the top with "deferred per
  ROADMAP_DECISIONS.md §1; this doc is preserved for the next
  attempt".
- **`docs/USAGE.md`** wasn't audited (skill checklist item didn't
  open it); next pass should verify that every `CREATE INDEX
  USING turbovec` example still uses the v1.0.x reloption names.

## Actions

- (none committed in-place this round; report only)
- Two follow-up commits worth doing if the parent agrees:
  1. Refresh `docs/ARCHITECTURE.md` against the actual tree.
  2. Add a "Shipped" section to `docs/ROADMAP_DECISIONS.md`.

The skill itself was tweaked twice while running this audit:
once to count `#[test]` alongside `#[pg_test]` (the original
formula reported 83 vs 92 because pgrx surfaces both binaries
under one runner), once to use `--sort=-creatordate` for tag
listing (semver pre-release tags sort *ahead* of their final tag
under `--sort=-v:refname`, which surprised the audit).
