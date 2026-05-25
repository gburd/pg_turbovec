# Phase Q: remove side-table storage

The user's call (post-Phase O-2): the side-table storage path
(`turbovec.am_storage` SPI bytea) was always the shortcut to ship
v1.0.0 fast, not the architecture we want long-term. Other PG
index AMs (btree, gist, gin, hnsw, ivfflat) all use the relfile
mechanism for a reason: it integrates cleanly with the buffer
manager, WAL, replication, REINDEX, VACUUM, and pg_upgrade.

Phase L (1.2.0) shipped the relfile path as a feature-gated
preview. Phase O-2 confirmed it's correct end-to-end. Phase P
(pre-baked SIMD-blocked layout + codebook persistence) closes
the cold-scan gap that was blocking default-on flip.

Phase Q is the cleanup: rip out the side-table path entirely so
the codebase has exactly one storage strategy, the feature flag
disappears, and the AM matches the conventions of every other
PG index AM.

## Scope

### Delete

- `src/index/persist.rs` — the SPI side-table reader/writer.
  Every function in this file goes away.
- `aminsert_sidetable` in `src/index/insert.rs`.
- `ambulkdelete_sidetable` in `src/index/vacuum.rs`.
- The `relfile_storage` Cargo feature in `Cargo.toml`. Promote
  what it gated to default-on.
- The `experimental_index_am` Cargo feature is also up for
  reconsideration: in v1.3.0 the index AM has been default-on
  for many releases, the "experimental" name is stale. Either
  drop the feature flag entirely (recommended) or rename it
  to something like `index_am_off` for users who want the
  smaller .so without AM support.
- All `#[cfg(feature = "relfile_storage")]` and
  `#[cfg(not(feature = "relfile_storage"))]` gates throughout
  `src/`. After this commit grep `cfg.*relfile` should
  return zero hits.
- `migrations/` SQL files that create or alter
  `turbovec.am_storage`. Replace with a drop migration that
  removes any stale row, since v1.3.0 indexes won't write to
  it.
- All `am_storage` references in tests in `src/lib.rs`.
- The migration HINT in `ambeginscan` — it's no longer needed
  because there's no ambiguity between v1.2 (gated) and v1.3.

### Update

- The deferred-commit cache (`src/cache.rs` /
  `src/xact.rs`) currently has cfg-selected flush sinks
  (sidetable persist::save vs relfile write_full). Remove
  the cfg branch; the sink is always relfile.
- `pg_turbovec.control`, `Cargo.toml`: bump 1.2.0 → 1.3.0.
- `CHANGELOG.md`: write the [1.3.0] entry covering Phase P + Q
  + any other in-flight items.
- `docs/PARITY_GAPS.md`: cold-scan row gets the post-Phase-P
  number, INSERT row stays the post-Phase-K number, no more
  feature-flag asterisks.
- `docs/ROADMAP_DECISIONS.md`: move the relfile + Phase P/Q
  items from "Where future work would pay off" to "Shipped".
- `README.md`: status banner gets the post-Q test counts;
  drop any "experimental" / "preview" language in the AM
  description.
- `docs/PG_VERSION_SUPPORT.md`: re-run the test matrix with
  the new build flags (which are now just `pg<N>` since
  `experimental_index_am` and `relfile_storage` are gone).

### Migration story for users

Anyone with a v1.0.x → v1.2 turbovec index has a side-table
row and an empty main fork. After upgrading to v1.3.0:

1. The extension upgrade SQL drops the side-table row.
2. Any existing index relations have empty main forks; the
   first `ambeginscan` after the upgrade hits an empty
   relfile and emits an `ERROR`-not-`NOTICE`: "this index
   was built under pg_turbovec ≤ 1.2; run `REINDEX INDEX
   <name>;` to migrate".
3. Once REINDEX runs, the index is in the v1.3 wire format
   and queryable.

This is a hard migration boundary, not a soft one. Document
loudly in `CHANGELOG.md`.

## What NOT to do in Q

- Don't change the on-disk wire format. Phase P already did
  that; Q is mechanical removal only.
- Don't touch `vendor/turbovec/` (its API surface is fine).
- Don't change anything user-visible besides what the bullet
  list above calls out.

## Tests

After Q, every test in `src/lib.rs` should pass under the new
default flags:

```bash
cargo pgrx test pg<N>   # no --features needed
```

Expected count: ~107/107 (v1.2.0's 105/105 relfile tests + 2
that were sidetable-only and need to be ported or deleted).

Drop tests that exclusively exercise side-table-only behaviour
(e.g. anything that read `am_storage.payload` directly). Port
tests that exercised generic AM behaviour and happened to be
gated on the side-table feature.

## Heartbeat convention reminder

Read `.pi/skills/long-running-bench/SKILL.md` first. Use
`benches/scripts/lib/with-heartbeat.sh` for any command that
takes > 60 s, and the parent will poll
`benches/scripts/poll-heartbeat.sh <log>` to verify liveness.

## Don't

- Don't push to origin (parent merges).
- Don't bump the version in this commit alone — Q is paired
  with the v1.3.0 tag, which the parent does.
- Don't delete the v1.2.0 tag or rewrite history.

## Reply with

- Lines of code deleted vs added (net should be heavily
  negative — this is a removal pass).
- Test count after Q.
- Any user-visible behaviour change you can't justify by the
  scope above.
- Whether v1.3.0 is ready to tag (depends on Phase P + O-3
  validation).
