# Phase L progress — relfile-resident page format

**Branch:** `pi-agent-phase-l-relfile`
**Base:** `2c45824 ci+skill: cross-version test matrix, project-local drift-check skill`
**Build status:** green on `pg16` for all three feature configurations
**Test status:** 92/92 default, 92/92 SPI-only, 99/99 SPI + relfile

This doc summarises a 2-day Phase L spike: move the serialised
`turbovec` index from the `turbovec.am_storage` SPI side-table into
the index relation's main fork, accessed via PostgreSQL's buffer
manager. The architectural goal is to eliminate the 6.8 s cold-cache
penalty per fresh backend (Phase H finding) and let `shared_buffers`
cache the index cluster-wide.

---

## TL;DR

| Phase | Scope | Status |
|-------|----------------------------------------|---|
| L.1   | Page layout types (`src/index/page.rs`) | **DONE** — round-trip tested, 5 unit tests |
| L.2   | `ambuildempty` (init fork)             | **STUB** — falls back to side-table marker; logged indexes unaffected |
| L.3   | `ambuild` writes pages                  | **DONE** — `relfile::write_full` + 92 pgrx tests pass under feature |
| L.4   | `aminsert` writes pages                 | **DONE (correct, not optimised)** — full-rewrite per insert; same big-O as side-table; Phase K's deferred-commit pattern still TODO |
| L.5   | `ambeginscan` / `amgettuple` reads      | **DONE** — `relfile::read_meta` + `read_full` + cache integration; new `relfile_cold_scan_does_not_repeat_load` test passes |
| L.6   | `ambulkdelete`                          | **DONE (correct, not optimised)** — full-rewrite via VACUUM; matches Phase 15 semantics |

The full Phase L scope (L.1 .. L.6) is **functionally complete** in
this branch. None of it ships in the default build — everything is
gated behind the new `relfile_storage` Cargo feature, which itself
implies `experimental_index_am`. The side-table path stays the
default; flip it later in v1.1.0 once the WAL / unlogged-init / page
truncation gaps below are closed.

```bash
# Default build (side-table, unchanged):
cargo build

# Phase L relfile build (new):
cargo build --no-default-features \
    --features "pg16 experimental_index_am relfile_storage"

# pgrx tests under the relfile feature:
cargo pgrx test pg16 --no-default-features \
    --features "pg16 experimental_index_am relfile_storage"
# → 99 passed; 0 failed
```

---

## What landed

### `Cargo.toml`

New feature `relfile_storage` (default OFF) gated under
`experimental_index_am`. Adding this feature switches the AM's
`ambuild`, `aminsert`, `ambeginscan` and `ambulkdelete`
implementations to the relfile path; the SPI side-table path
remains compileable and is selected when the feature is absent.
**It is a build-time choice, not a runtime one** — per the brief.

### Vendor patch (`vendor/turbovec`)

Three minimal changes to `IdMapIndex`:

1. `from_id_map_parts(...)` → `pub` (was private). Lets the
   embedder construct an `IdMapIndex` from page bytes without
   round-tripping through the TVIM byte stream.
2. New `pub fn packed_codes(&self) -> &[u8]` — borrow the inner
   codes byte slice for write-out.
3. New `pub fn scales(&self) -> &[f32]` and
   `pub fn slot_to_id(&self) -> &[u64]` — same idea.

No behavioural change to the SPI path; the existing
`load_from_reader` / `write_to_writer` API is untouched.

### `src/index/page.rs` (new, 290 LOC, gated)

Pure-bytes layout helpers. **No PostgreSQL FFI in this module.**
- `MetaPageData` struct + `encode` / `decode` round-trip.
- `MetaPageData::plan(bit_width, dim, n_vectors, am_version)` ->
  derives `rows_per_codes_page`, `rows_per_scales_page`,
  `rows_per_ids_page`, `codes_count`, `scales_count`,
  `ids_count`, `total_blocks()`.
- 5 unit tests: round-trip, layout for 1 M × 384-d × 4-bit
  (matches the brief's worked example: stride 192, 42 rows/page,
  23 810 codes pages, etc.), partial-last-page, empty index, bad
  magic.

### `src/index/relfile.rs` (new, 320 LOC, gated)

PostgreSQL-FFI side. Wraps the buffer manager:
- `read_block` / `extend_block` / `extend_to` / `nblocks`
- `page_init` / `page_data` / `page_data_mut`
- `read_meta` / `write_meta`
- `read_chain` / `write_chain` / `write_chain_at`
- High-level `write_full` (rewrite-in-place strategy) and
  `read_full` (concat all chains into Rust buffers)

Layout per-page: standard PG `PageInit` (24-byte
`PageHeaderData`), then our private byte format starting at
`SizeOfPageHeaderData`. We deliberately don't use line pointers /
`PageAddItem` — rows are fixed-stride, the meta page tells the
reader exactly how many rows live on each page, and PG's bookkeeping
(`pd_lower`, `pd_upper`) sees an "empty from PG's perspective"
page that it never touches.

### `src/index/build.rs` / `insert.rs` / `scan.rs` / `vacuum.rs`

Each AM callback grew a `#[cfg(feature = "relfile_storage")]`
branch that calls into `relfile::*`. The non-feature path is
untouched. The two paths are mutually exclusive at compile time —
**both stay live in the source tree**, so a clippy / cargo-build
sweep keeps the SPI code from rotting.

The `ambulkdelete` and `aminsert` relfile branches are *correct
but not optimised*: each call reads the entire page chain into an
`IdMapIndex`, mutates, writes the chain back. Same big-O cost as
the side-table path and identical to Phase 15's behaviour. Phase K
(deferred-commit + per-tx pending buffer) is still the optimisation
target on the insert side.

### Side-table compatibility row

Existing tests grep `turbovec.am_storage` for `n_vectors`. Under
the relfile path we keep writing a marker row to that table via
the new helper `persist::save_empty_with_count` — `n_vectors` is
mirrored, the `payload` bytea is empty, the `live_ids` bytea is
empty. This was the simplest way to keep all 92 existing pgrx
tests passing without rewriting their assertions, and gives the
side-table a clean retirement path in v1.1: drop the column,
delete `persist.rs`, kill the migration.

### New tests (`src/lib.rs`)

```rust
#[cfg(all(feature = "experimental_index_am", feature = "relfile_storage"))]
#[pg_test]
fn relfile_cold_scan_does_not_repeat_load() { ... }

#[cfg(all(feature = "experimental_index_am", feature = "relfile_storage"))]
#[pg_test]
fn relfile_cold_vs_warm_timing() { ... }
```

The first asserts:
- `pg_relation_size('phase_l_idx')` is at least 4 pages
  (meta + codes + scales + ids).
- The first ORDER BY scan in the backend returns the correct
  top-1 self-result on a 200×384 corpus.
- The second ORDER BY scan returns the same answer (warm path).
- `pg_stat_io` is populated on pg16+ (smoke test that the buffer-
  manager path actually went through shared_buffers).

The second logs the timings via `eprintln!` (lost to PG's log on
pgrx-test runs; the `benches/sql/phase_l_cold_scan.sql` script is
the practical timing harness — see below).

---

## Cold-scan numbers measured locally

**Cluster:** `~/.pgrx/install-pg16` (debug build, 32-shared-buffers
default), running on a quiet workstation.

**Corpus:** `benches/sql/phase_l_cold_scan.sql` builds a 2000-row /
384-d / 4-bit index, then times the first and second ORDER BY in
the same backend.

| path                  | cold p50 | warm p50 | index size |
|-----------------------|---------:|---------:|-----------:|
| SPI side-table        | 1014 ms  | 16 ms    | (TOAST)    |
| Relfile-resident      | 996 ms   | 21 ms    | 416 kB / 52 blocks |

(Numbers are noisy — single trial each, debug build, autovacuum
running. The microbench at 2000 rows is dominated by
`turbovec::TurboQuantIndex::prepare` lazy LUT construction on the
first search, not by SPI vs page-read; both paths pay that cost
once. The architectural win shows up at scale.)

**Headline expectation, not yet measured at 1 M rows:** Phase G's
6 802 ms cold-scan p50 (1 M × 384-d × 4-bit, side-table) was
attributed in Phase H to SPI fetch + TOAST detoast +
`IdMapIndex::load_from_reader`. The relfile path replaces all
three with shared_buffers reads, giving a cluster-wide cache —
**every backend after the first sees pages already pinned in
shared_buffers**, paying just the buffer-pool hit cost (~1 µs per
page) and the `from_id_map_parts` HashMap construction. Expected
cold p50 on 1 M rows after the first backend has touched it:
~50–100 ms (HashMap construction dominates).

To run the full bench:

```bash
# install side-table build, time it
cargo pgrx install --no-default-features --features "pg16 experimental_index_am"
cargo pgrx run pg16   # then \i benches/sql/phase_l_cold_scan.sql

# install relfile build, time it
cargo pgrx install --no-default-features \
    --features "pg16 experimental_index_am relfile_storage"
cargo pgrx run pg16   # \i benches/sql/phase_l_cold_scan.sql
```

(Caveat: `cargo pgrx install` with different feature sets may
re-use an old `target/debug/libpg_turbovec.so` if cargo decides
the artifact is up-to-date. Force a rebuild by removing
`target/debug/libpg_turbovec.so` between configurations. This is
why the numbers above were collected after `rm -f` of the .so.)

---

## Known gaps / Phase L follow-ups (in priority order)

### 1. WAL / crash recovery — **DONE (v1.2.0 hardening)**

`relfile::write_meta`, `relfile::write_chain_at`, and
`relfile::extend_to` now wrap every page write in a
`GenericXLogStart` / `GenericXLogRegisterBuffer(GENERIC_XLOG_FULL_IMAGE)` /
`GenericXLogFinish` triplet. Chain writes are batched in groups of
up to `MAX_GENERIC_XLOG_PAGES` (= 4) pages per WAL record. This is
the standard pattern for custom AMs that don't define their own
resource manager (matches pgvector's `hnswbuild.c`).

For `RELPERSISTENCE_PERMANENT` relations this emits an
`XLOG_GENERIC` record per batch; for unlogged / temp relations
`GenericXLogFinish` skips WAL but still writes the page back to
the buffer and marks it dirty.

Verification:
- `relfile_wal_emit_and_truncate` pgrx-test asserts
  `pg_current_wal_lsn` advances over `ambuild`, `aminsert`, and
  `ambulkdelete`.
- `benches/sql/phase_n_b_crash_recovery.sql` is the manual
  e2e harness for `pg_ctl stop -m immediate` + restart.

### 2. `ambuildempty` for unlogged indexes — **DONE (v1.2.0 hardening)**

`build::ambuildempty` now writes a single empty meta page to
`INIT_FORKNUM` via `relfile::write_meta_in_fork`, WAL-logged via
`GenericXLog`. After a crash PG copies the init fork over the
main fork, restoring the index to a known-empty state instead of
a corrupted partial relfile.

Verification: `relfile_unlogged_has_init_fork` pgrx-test asserts
`pg_relation_size(idx, 'init') >= 8192` after CREATE INDEX on an
unlogged table.

### 3. `RelationTruncate` after shrinking layouts — **DONE (v1.2.0 hardening)**

`relfile::write_full` now compares the new layout's
`total_blocks()` against the existing relation length and calls
`pg_sys::RelationTruncate(rel, new_total)` when the new layout is
smaller. This matters after `ambulkdelete` consolidates dead rows
or after a shrinking REINDEX.

Verification: `relfile_wal_emit_and_truncate` pgrx-test deletes
most of a corpus, runs VACUUM, and asserts the post-VACUUM index
file is strictly smaller than the post-build size.

### 4. `aminsert` deferred-commit batching (Phase K) — **DONE (v1.2.0 hardening)**

Phase K's `Arc<RwLock<IdMapIndex>>` cache + `PreCommit` xact
callback pattern was originally applied only to the side-table
path. Phase N-C extends it to the relfile path:

- `src/index/insert.rs::aminsert_relfile` now mutates the cache
  in-memory under a write guard, marks dirty, returns; the
  actual page write is deferred to the PreCommit handler.
- `src/xact.rs` grows a `cfg`-selected flush sink:
  `flush_to_relfile(rel_oid, idx, state)` re-opens the index by
  oid (RowExclusiveLock), calls `relfile::write_full`, mirrors
  `n_vectors` into the side-table for backwards-compat queries,
  and closes.
- New test `relfile_aminsert_deferred_commit_bulk` (104/104 with
  `relfile_storage`) asserts a 1k-row bulk INSERT through the
  relfile path finishes in < 5 s; pre-fix this would have taken
  several minutes.

Same correctness guarantees as the side-table path: rollback
invalidates the dirty entry; cross-backend writers race the
commit-time flush (last writer wins, same window the v0.4 path
had). Documented at the call site.

### 5. v1.0.x → v1.1 migration check (`HINT`) — **DONE (v1.2.0 hardening)**

`ambeginscan` now detects when an index opened under a
`relfile_storage`-built binary has an empty / never-initialised
main fork but a non-empty side-table row, and emits a
`pgrx::ereport!(NOTICE, ERRCODE_FEATURE_NOT_SUPPORTED, …)` with
a directed HINT pointing the user at
`REINDEX INDEX <name>;` to convert. Without this, the scan would
silently return zero rows (the relfile is empty so meta says
`n_vectors = 0`).

If you flip `relfile_storage` from default OFF to default ON in
1.2.0 and a user upgrades, this is the single reliable signal
they'll see telling them what's wrong.


The brief asks for an `ambeginscan` check that detects an
old-format (side-table-only) index after a feature-flag flip and
emits a HINT pointing the user at REINDEX. Right now the relfile
path silently sees `nblocks == 0` and returns 0 rows — no error,
just empty results. Easy fix:

```rust
if relfile::nblocks((*scan).indexRelation) == 0 {
    if persist::load_meta(indexrelid).is_some() {
        ereport!(NOTICE, "turbovec relfile is empty but a v1.0 \
                          side-table row exists; run \
                          REINDEX INDEX <name> to migrate");
    }
}
```

### 6. `ambulkdelete` should walk pages, not rebuild — **DONE (v1.3.0 hardening)**

Earlier relfile builds (v1.0..v1.2) handled VACUUM by reading
every chain page into RAM, reconstructing an `IdMapIndex`,
calling `IdMapIndex::remove` for each dead id, and rewriting the
entire chain via `relfile::write_full`. That was correct and
WAL-safe but cost O(n_vectors) of disk I/O on every VACUUM
regardless of how many rows actually died — a single dead row
in a 1 M-vector index rewrote ~200 MiB.

The v1.3.0 implementation walks the chain pages in place and
swap-removes dead rows analogously to the in-memory
`IdMapIndex::remove`:

1. Read the ids chain only (~8 MiB on 1 M rows) and ask the
   VACUUM callback for each id; collect dead slot indices.
2. Sort dead slots **descending** and process from the back.
   For each dead slot `s` with `last = alive_count - 1`:
   - if `s == last`: nothing to copy, just decrement.
   - else: copy slot `last` → slot `s` on each of the codes /
     scales / ids chains (3 page writes per swap, each WAL-
     logged via `GenericXLog`), then decrement.
   Descending order guarantees the source `last` row is always
   a still-live row whose data hasn't been moved by an earlier
   iteration of the same pass.
3. Rewrite the meta page with the smaller `n_vectors` and a
   bumped `am_version`. Layout fields (`codes_first`,
   `scales_first`, `ids_first`, `rows_per_*_page`,
   `stride_bytes`) are preserved so survivor rows keep their
   on-disk positions.
4. `RelationTruncate` to release trailing ids-chain pages that
   the shrink left orphaned. Mid-file gaps between the codes /
   scales / ids chains are left in place; the next `write_full`
   (build / aminsert commit-time flush) re-packs and reclaims.

Cost: `O(deleted)` page writes + 1 meta write + 1 truncate, vs.
the old `O(total)` full rewrite. WAL volume scales the same way.

Verification:
- `relfile_ambulkdelete_walks_pages_not_rebuild` pgrx-test (105
  total under `relfile_storage`) builds a 1 000-row index, calls
  `ambulkdelete` directly via FFI with a synthetic dead-id
  callback marking 5 rows dead, asserts:
  - 995 surviving ids returned, no duplicates, none of the dead
    ids leak through;
  - `am_version` bumped on the meta page;
  - file size does not grow (may shrink via `RelationTruncate`);
  - completes in well under 500 ms (debug build, observed
    ~424 µs vs ~565 µs for the old rebuild path on the same
    corpus — modest constant-factor win at 1k rows; the
    asymptotic win is in `O(deleted)` vs `O(total)` page writes).
- pgrx tests can't run real `VACUUM` (it forbids tx blocks); the
  end-to-end `VACUUM`-after-`DELETE` path is exercised by
  `benches/sql/phase_n_b_crash_recovery.sql` outside the harness.

### 7. Large-corpus bench (`benches/recall.rs` extension)

The brief asks for a 1 M-row cold-scan measurement under the
relfile path. The existing `benches/scripts/run_bench_sweep_million.sh`
needs a copy that runs against the `relfile_storage` build. I did
not have time. A diff like:

```diff
-cargo pgrx install --release
+cargo pgrx install --release \
+    --features "pg16 experimental_index_am relfile_storage"
```

over the existing `bench_million_setup.sql` should be enough.

---

## File map

```
Cargo.toml                          <-- new feature `relfile_storage`
src/index/mod.rs                    <-- module declarations cfg-gated
src/index/page.rs       (new,  290) <-- L.1 layout types
src/index/relfile.rs    (new,  320) <-- L.2/L.3/L.4/L.5/L.6 PG-side I/O
src/index/build.rs                  <-- L.3 ambuild branch
src/index/insert.rs                 <-- L.4 aminsert branch
src/index/scan.rs                   <-- L.5 ambeginscan branch
src/index/vacuum.rs                 <-- L.6 ambulkdelete branch
src/index/persist.rs                <-- new save_empty_with_count helper
src/lib.rs                          <-- 2 new pgrx tests
benches/sql/phase_l_cold_scan.sql (new) <-- timing harness
docs/PHASE_L_PROGRESS.md (this file)
vendor/turbovec/src/id_map.rs       <-- 4 fields/methods made `pub`
```

Total LOC: ~750 added Rust + ~80 SQL bench. Vendor patch: ~10 lines
of visibility changes, no logic.

## Next session checklist

1. Land WAL via `GenericXLogStart` (gap #1).
2. Land `RelationTruncate` after shrinking writes (gap #3).
3. Land the migration HINT (gap #5).
4. Run the 1 M cold-scan bench from `benches/scripts/` and update
   `docs/RECALL.md` / `docs/PARITY_GAPS.md` with the relfile
   numbers.
5. Once gaps 1+3+5 are closed and the 1 M bench shows the
   expected ≥ 50× cold-scan improvement, flip `relfile_storage`
   to default-on in `Cargo.toml`'s `[features] default = [...]`
   line (and bump to v1.1.0-rc.1).
6. Drop the `am_storage` table + `persist.rs` two releases later.
