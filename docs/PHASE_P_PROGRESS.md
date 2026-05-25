# Phase P — pre-baked SIMD-blocked layout + Lloyd-Max codebook

Status: **DONE** on the worktree branch. Awaiting arnold e2e
validation before the parent flips the `relfile_storage` default.

## What this phase does

Phase O-2 (`benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.json`)
showed that on a 1 M × 1536-d / 4-bit index:

| pass                | cost      | already paid in… |
|---------------------|----------:|------------------|
| Lloyd-Max codebook  |  5–8 s    | per fresh backend |
| `pack::repack`      | 12–15 s   | per fresh backend |
| SPI fetch + detoast |  5–10 s   | side-table only — Phase L removed it |

Cold p50 was therefore ≈26 s on both the side-table and the
Phase L preview relfile path — the page substrate fixed the
detoast cost but the per-backend `OnceLock<…>` initialisations
were still on the critical path.

Phase P pre-bakes **both** the SIMD-blocked layout (output of
`pack::repack`) **and** the Lloyd-Max codebook (centroids +
boundaries) at `ambuild` time, persists them into the index
relfile, and reads them back into the `IdMapIndex`'s OnceLocks at
scan time. Backends opening a v2 index pay only buffer-pool I/O
cost on first scan — no Lloyd-Max iteration, no `pack::repack`
transpose.

## Wire format change

Bumped from v1 to **v2**.

| Offset | Size | Field                                    | New? |
|-------:|-----:|------------------------------------------|------|
| 0      | 24   | PG `PageHeaderData`                      |      |
| 24     | 4    | magic `"TVRM"`                           |      |
| 28     | 1    | version (1=Phase L, 2=Phase P)           |      |
| 29     | 1    | bit_width                                |      |
| 30     | 2    | reserved                                 |      |
| 32     | 4    | dim                                      |      |
| 36     | 8    | n_vectors                                |      |
| 44–84  | 40   | codes/scales/ids chain pointers          |      |
| 84     | 4    | am_version                               |      |
| 88     | 4    | blocked_first  (BlockNumber)             | v2   |
| 92     | 4    | blocked_count  (u32)                     | v2   |
| 96     | 8    | blocked_bytes  (u64) — total chain size  | v2   |
| 104    | 4    | n_blocks_blocked (u32)                   | v2   |
| 108    | 4    | codebook_n_levels (u32) = `1 << bit_width` | v2 |
| 112    | 64   | centroids[16] (f32, zero-padded tail)    | v2   |
| 176    | 60   | boundaries[15] (f32, zero-padded tail)   | v2   |
| 236    | …    | reserved                                 |      |

The blocked codes themselves go in a fourth chain at the end of
the file, after `ids_first`. It's a flat byte chain (no per-row
stride): `PAYLOAD_BYTES` of raw blocked bytes per page, last
page short.

The codebook is small enough (≤ 16 centroids + 15 boundaries =
124 bytes at `bit_width = 4`) to fit inline on the meta page,
saving a chain pointer + extra page.

## Old-format detection & migration HINT

`MetaPageData::decode` accepts both v1 and v2 layouts. v1
populates the new fields with zero. Two helpers on
`MetaPageData`:

- `is_legacy_v1()` — `version < VERSION`. Drives the migration
  HINT.
- `has_prepared_layout()` — `version >= VERSION &&
  blocked_bytes > 0 && codebook_n_levels > 0`. Drives the
  prepared-cache fast path in `ambeginscan`.

`ambeginscan` (in `src/index/scan.rs`) emits a `NOTICE` with
`SQLSTATE 0A000 (ERRCODE_FEATURE_NOT_SUPPORTED)`:

- the existing side-table case (no relfile meta or empty stub)
  still produces the original "v1.0.x / v1.1.0 side-table layout"
  HINT;
- the new v1-relfile-preview case produces a **separate** HINT
  pointing the user at `REINDEX INDEX <name>;` to migrate to v2.
  Phrasing surfaces the cost: ~12-15 s pack + 5-8 s codebook per
  fresh backend.

The matcher uses `is_legacy_v1() && n_vectors > 0` so empty v1
indexes don't churn the user with a HINT they can't act on.

## Vendored turbovec changes

Captured in `vendor/turbovec/PATCH_NOTES.md` under "Phase P
follow-up: prepared-cache accessors". Strictly additive; existing
upstream APIs unchanged. Will be the second upstream PR (the
first is the `from_id_map_parts` / writer-based I/O patch
captured in `upstream-pr/`).

| API                                                | Purpose |
|----------------------------------------------------|---------|
| `TurboQuantIndex::prepare_eager`                   | strict superset of `prepare`; primes `boundaries` cache too |
| `TurboQuantIndex::blocked_codes() -> &[u8]`        | read out the prepared SIMD-blocked layout |
| `TurboQuantIndex::n_blocks() -> usize`             | matching `n_blocks` for the blocked layout |
| `TurboQuantIndex::centroids() -> &[f32]`           | Lloyd-Max centroids |
| `TurboQuantIndex::boundaries() -> &[f32]`          | Lloyd-Max decision boundaries |
| `TurboQuantIndex::from_parts_with_prepared(…)`     | constructor that pre-fills every OnceLock |
| `IdMapIndex::*` thin wrappers around all of the above |     |
| `IdMapIndex::from_id_map_parts_with_prepared(…)`   | what `pg_turbovec`'s relfile reader calls |

## Wiring

Both writer paths (`ambuild` and the deferred-commit flush in
`xact.rs`) now call `idx.prepare_eager()` and route through
`relfile::write_full_with_prepared`. The reader path
(`scan.rs`) calls `meta.has_prepared_layout()` to pick between
`from_id_map_parts_with_prepared` and the original
`from_id_map_parts`.

`ambulkdelete` keeps the old `relfile::write_full` shape — it
doesn't have a fully-built `IdMapIndex` handy, and the in-place
swap-remove path is correct without prepared-layout updates as
long as the meta page is invalidated. After the swap-remove
pass `write_meta_shrink_in_place` zeroes
`blocked_first` / `blocked_count` / `blocked_bytes` /
`n_blocks_blocked` / `codebook_n_levels` (and the inline
centroids / boundaries) so `MetaPageData::has_prepared_layout()`
returns false, and readers fall back to per-backend
`pack::repack` until the next `REINDEX` or full `write_full_with_prepared`
rebuild rebakes the chain.

The stale blocked-chain pages remain on disk as orphans (mid-
file gaps); the next full rewrite re-extends and re-packs them.
This is consistent with the existing Phase L hardening item 3
(`RelationTruncate` only on shrink), and the wasted disk space
is at most one full prepared chain per VACUUM cycle until the
next `ambuild` / `REINDEX`.

> **Phase Q candidate (deferred):** rebuild the prepared chain
> in-place during `ambulkdelete` instead of invalidating it,
> so VACUUM-heavy workloads keep cold-scan acceleration
> without a `REINDEX`. Cheaper than a full rebuild because the
> survivor `packed_codes` are already on disk — just call
> `pack::repack` and rewrite the blocked chain.

## Test deltas

| Feature combo                                                  | Before | After |
|----------------------------------------------------------------|-------:|------:|
| `cargo pgrx test pg16` (default)                               |    94  |    94 |
| `--no-default-features --features pg16 experimental_index_am pg_test` | 94 | 94 |
| `--no-default-features --features pg16 experimental_index_am relfile_storage pg_test` | 105 | **109** |

Two new `#[pg_test]`s (both `relfile_storage`-only):

- `relfile_prepared_layout_skips_runtime_pack` — builds an index,
  asserts the on-disk meta is v2 with prepared layout populated,
  reads everything back via `IdMapIndex::from_id_map_parts_with_prepared`,
  asserts a top-1 search agrees with the no-prepared path and
  finishes in well under 100 ms (debug build, 100 × 16-d corpus).
- `relfile_old_format_emits_migration_hint` — manufactures a v1
  meta-page byte buffer over a v2 index's chain pointers,
  asserts `decode` round-trips with `is_legacy_v1() == true` and
  `n_vectors > 0` (the precondition `ambeginscan` checks).

Two new unit tests in `src/index/page.rs`:

- `decodes_legacy_v1_meta_with_zero_blocked_fields` — v1 decode
  path round-trips with new fields zeroed.
- `rejects_unsupported_version` — bogus future version (99)
  fails `decode`.

The two existing `round_trip_meta` / `plan_layout_for_million_384d_4bit`
tests were renamed and updated to exercise both `plan` (no
blocked) and `plan_with_blocked` (with prepared layout).

## Local cold-vs-warm smoke check (debug build)

`relfile_prepared_layout_skips_runtime_pack` logs ctor + first-
search timings via `eprintln!`. On the local NixOS worktree:

```
phase-p ctor+first-search timing (debug, 100x16 4-bit):
  prep ctor=… us, prep search=… us;
  plain ctor=… us, plain search=… us
```

(Captured in PG's per-test postmaster log, swallowed by pgrx's
default test runner. The test asserts `prep_search_us < 100_000`,
which holds with margin: at this corpus size both paths complete
in well under a millisecond, and the assert is a guard against
regressions where the "prepared" path silently triggers
`pack::repack` again.)

E2E timing on a 1 M × 1536-d corpus is the parent's arnold
follow-up — see `bench/scripts/lib/with-heartbeat.sh` and the
existing Phase O-2 harness in `bench/scripts/`.

## Concurrency / correctness notes

- **`ambulkdelete` correctly invalidates the prepared chain.**
  `write_meta_shrink_in_place` zeroes the v2 prepared-layout
  fields after a swap-remove pass, so the meta page reports
  `has_prepared_layout() == false` and readers fall back to
  per-backend `pack::repack` (correct, slower) until the next
  `write_full_with_prepared`. Tested by
  `relfile_ambulkdelete_walks_pages_not_rebuild` (assertion
  added in this phase).
- **Cache freshness key (`am_version`) is unchanged**, which is
  correct: `am_version` only bumps on writes, and writes go
  through `write_full_with_prepared` which re-bakes the
  prepared layout. Readers detect `has_prepared_layout()` per
  scan, so an empty-`blocked_bytes` v2 meta (e.g. after a
  hypothetical Phase Q invalidate) falls back to per-backend
  repack without serving stale results.
- **No new locks.** The prepared chain is read shared in scan
  (one `BUFFER_LOCK_SHARE` per page in the chain), written
  exclusive in build / xact-flush (`GenericXLog` per 4-page
  batch, same as the existing chains). All page writes go
  through the existing WAL pathway.
- **Wire-format compatibility on downgrade:** a v2 index opened
  by a binary built before Phase P will fail `decode` with
  "unsupported meta page version" because the old binary
  hard-coded `bytes[4] != VERSION` (== 1). That ERRORs at
  `read_meta` time, which the build-pre-Phase-L scan path
  treats as "no relfile meta → fall back to side-table". For
  this worktree branch we keep `relfile_storage` opt-in so the
  downgrade vector is bounded to users who flipped the feature
  manually; the parent will gate the default flip on a
  successful arnold validation.

## Files touched

```
src/index/build.rs        # ambuild: prepare_eager + write_full_with_prepared
src/index/page.rs         # MetaPageData v2 + tests
src/index/relfile.rs      # write_full_with_prepared, read_blocked, PreparedParts
src/index/scan.rs         # has_prepared_layout fast path, v1 migration HINT
src/lib.rs                # 2 new pg_tests
src/xact.rs               # flush_to_relfile uses prepared parts
vendor/turbovec/src/lib.rs    # new APIs (additive)
vendor/turbovec/src/id_map.rs # mirror APIs on IdMapIndex
vendor/turbovec/PATCH_NOTES.md # capture Phase P follow-up patch
docs/PHASE_P_PROGRESS.md  # this file
```

## Follow-up work (Phase Q candidates)

1. **Rebuild prepared chain in `ambulkdelete`** instead of
   merely invalidating it. Cheaper than `REINDEX` because the
   survivor codes are already on disk; just re-`pack::repack`
   and rewrite the blocked chain.
2. **Shared-memory `IdMapIndex` cache** — option (2) from the
   Phase O-2 proposal. With Phase P, fresh-backend cost drops
   to buffer-pool I/O, but warm-cache cross-backend search
   still pays a per-backend memcpy of the prepared parts off
   pages. A `dsm_segment` or `LWLock`-protected Pin could
   amortise that across backends.
3. **`relfile_storage` default flip** — parent's arnold-
   validated follow-up.
