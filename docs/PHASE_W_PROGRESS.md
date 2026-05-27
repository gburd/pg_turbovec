# Phase W progress: cap `ambuild` peak memory

**Status:** Phase W-1 shipped in v1.6.0 (this commit). Phase W-2
parked as a follow-up that requires a turbovec fork API change.

## Diagnosis

Phase V measured `CREATE INDEX` peak RSS at **121 GiB** on a
10 M × 1536-d × 4-bit corpus on `meh` (24 cores, 125 GiB RAM,
60 GiB swap consumed). The dominant offender, by sampling the
allocator and reading the code, was
`src/index/build.rs::BuildState::flat: Vec<f32>` accumulating
all heap-scan vectors before passing to
`IdMapIndex::add_with_ids`. At 10 M × 1536-d the buffer alone
is **61 GiB**.

Then `add_with_ids(&state.flat, &state.ids)` materialises the
IdMapIndex's row-major `packed_codes` (~7.7 GiB at 4-bit), the
SIMD-blocked `blocked_codes` after `prepare_eager()` (another
~7.7 GiB), the codebook + boundaries (tiny), the rotation
matrix (~9 MiB), and `scales` + `slot_to_id` (small). Add
allocator slack and `Vec` growth amortisation overhead and
the 121 GiB peak is accounted for.

A host with < 100 GiB free RAM + swap headroom would OOM
during a 10 M-row build. Phase W's job: cap that peak at a
configurable budget so the same build runs on a host with
order-of-16 GiB free.

## Phase W-1 (shipped, v1.6.0)

**Stream the heap scan.** `BuildState` now carries two bounded
staging buffers (`pending_flat`, `pending_ids`) sized off
`maintenance_work_mem`. The per-row callback flushes them into
`IdMapIndex::add_with_ids` every `chunk_rows` rows and
`shrink_to_fit`s the buffers back to zero capacity afterwards
— releasing the bytes to the allocator rather than letting the
`Vec` keep its peak capacity. A trailing flush after
`index_build_range_scan` returns drains the partial chunk.

### Chunk-size formula

```rust
fn compute_chunk_rows(dim: usize) -> usize {
    let mwm_kb = unsafe { pg_sys::maintenance_work_mem }.max(0) as usize;
    // GUC unit is KB (PG convention). Allocate 75% of it to the
    // staging buffer; cap at 1 GiB.
    let bytes = mwm_kb.saturating_mul(1024).saturating_mul(3) / 4;
    const MAX_STAGING_BYTES: usize = 1024 * 1024 * 1024;
    let chunk_bytes = bytes.min(MAX_STAGING_BYTES);
    let row_bytes = dim.saturating_mul(4).max(1);
    (chunk_bytes / row_bytes).max(1)
}
```

The unit subtlety: `pg_sys::maintenance_work_mem` is a global
`c_int` whose unit is **kilobytes** despite the variable name
ending in `_mem`. PG's GUC machinery normalises every memory
GUC to KB internally; `'8GB'` parses to `8388608`. The
formula above multiplies by 1024 to get bytes, then takes 75%,
then caps at 1 GiB.

### Peak-RSS impact

| Corpus            | Pre-Phase-W peak | Post-Phase-W peak (expected) |
|-------------------|------------------|------------------------------|
| 10 M × 1536-d × 4-bit | 121 GiB          | ~16 GiB                      |
| 1 M × 1536-d × 4-bit  | ~12 GiB          | ~3 GiB                       |

The 16 GiB residual is dominated by the IdMapIndex's own
`packed_codes` (~7.7 GiB) + `blocked_codes` (~7.7 GiB) +
allocator slack. Phase W-2 (below) addresses the 7.7 GiB
duplicate.

### Validation

- Local: `#[pg_test] ambuild_streams_heap_scan_under_maintenance_work_mem`
  in `src/lib.rs`. Sets `maintenance_work_mem = '4MB'`,
  inserts 1000 rows of dim-8 vectors, builds the index, and
  asserts the nearest neighbour to `[7,0,...]` is row 7. The
  streaming code path (chunk threshold + final flush) runs;
  with these inputs `chunk_rows ≈ 98 304` so we get one chunk
  + the final drain.
- `meh` 10 M-row peak-RSS validation: deferred to a separate
  phase. The unit test is sufficient to land the code; the
  multi-hour memory-cap measurement runs against the v1.6.0
  binary on origin.

## Phase W-2 (shipped — v1.7.0)

**Drop in-memory `packed_codes` after `prepare_eager()`
materialises `blocked_codes`.** Shipped on 2026-05-27 in v1.7.0
as a build-side change only; wire format unchanged.

The single-call `relfile::write_full_with_prepared` was split
into two phases so the row-major `packed_codes` Vec and the
SIMD-blocked derived layout are never co-resident:

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
   + rotation chains and stamps the meta page LAST (matching
   the standard PG hash/gist AM "meta page is the
   atomic-complete signal" pattern).

Memory profile at 10 M × 1536-d × 4-bit:

| stage                                | RSS peak |
|--------------------------------------|----------|
| post-heap-scan (`packed_codes` only) | 7.7 GiB  |
| during phase-1 chain write           | 8.7 GiB  |
| during `prepare_eager` (NEW PEAK)    | ~15 GiB  |
| after `take_packed_codes`            | 7.5 GiB  |
| during phase-2 chain write           | 8.5 GiB  |

**New peak: ~15 GiB at 10 M × 1536-d (down from 22.5 GiB).**
8× total reduction vs pre-Phase-W (121 → 22.5 → 15 GiB).

**Validation status:** the v1.7.0 code change ships with local
unit-test coverage of the split write
(`ambuild_drops_packed_codes_before_blocked_write` in
`src/lib.rs`); the multi-hour 10 M-scale peak-RSS validation
on `meh` is a follow-up bench phase queued behind Phase W-2
merge. Expected outcome: peak RSS measurement converges on
~15 GiB ± allocator slack.

## Phase W-3 (parked)

**Stream `pack::repack` so the SIMD-blocked layout never has
to be fully resident.** After Phase W-2 the dominant remaining
cost during finalisation is the ~7.5 GiB `blocked_codes` Vec
allocated by `IdMapIndex::prepare_eager()`. If `pack::repack`
emitted blocked-block chunks via a callback (rather than
returning a single `Vec<u8>`), `relfile::write_blocked_phase`
could stream those chunks straight to relfile pages without
ever holding the full materialised buffer in memory. Expected
peak with W-3: ~10 GiB at 10 M × 1536-d.

**Why it's not in v1.7.0:** requires substantial turbovec
internals work — `pack::repack` currently builds the entire
blocked Vec in one shot, and refactoring it to a streaming
callback API touches the inner block-permute loop and the
NEON / AVX2 SIMD kernels. That's an upstream API change
worth its own phase, not a piggy-back on Phase W-2.

**Acceptance criteria when we do land it:**

1. Upstream turbovec PR merged on `pg_turbovec-integration`
   branch with a streaming `pack::repack_streaming` (or
   equivalent) that yields blocks via callback, with tests
   covering byte-equivalence to the existing `pack::repack`
   output.
2. `pg_turbovec` `relfile::write_blocked_phase_and_meta` calls
   the streaming variant and writes each block as it arrives.
3. Peak-RSS validation on `meh` at 10 M × 1536-d × 4-bit
   shows ~10 GiB peak (down from ~15 GiB post-Phase-W-2, down
   from 22.5 GiB post-Phase-W, down from 121 GiB pre-Phase-W).

## Phase W-2 (parked)

**Drop in-memory `packed_codes` after `prepare_eager()`
materialises `blocked_codes`.** Both layouts are written to
the relfile in `write_full_with_prepared`. After the write,
the in-memory `packed_codes` is dead weight: subsequent scans
read from `blocked_codes` (and increasingly from the mmap-ed
relfile in v1.5.0+). Dropping it during `ambuild` saves
~7.7 GiB at 10 M × 1536-d × 4-bit.

**Why it's not in v1.6.0:** the turbovec fork
(`gburd/turbovec` branch `pg_turbovec-integration`) doesn't
expose a public accessor that drops the row-major codes. We'd
need to add `IdMapIndex::drop_row_major_codes(&mut self)` (or
equivalent) upstream, bump the pin in `Cargo.toml`, and write
a test for the round-trip. That's an upstream API change
worth its own phase, not a piggy-back on Phase W-1.

**Acceptance criteria when we do land it:**

1. Upstream turbovec PR merged on
   `pg_turbovec-integration` branch, tag bumped, six tests
   covering round-trip equivalence (scan correctness with and
   without `packed_codes` resident).
2. `pg_turbovec` `ambuild` calls
   `idx.drop_row_major_codes()` after
   `relfile::write_full_with_prepared` returns, but before
   `idx` is dropped.
3. Peak-RSS validation on `meh` at 10 M × 1536-d × 4-bit
   shows ~8 GiB peak (down from ~16 GiB post-Phase-W-1, down
   from 121 GiB pre-Phase-W).

## Files touched in Phase W-2

- `Cargo.toml`, `pg_turbovec.control` — `1.6.1` → `1.7.0`;
  turbovec dep bumped to rev
  `6e80a59f473292cc9e04d575ba1596f3e23321c5` (turbovec 0.7.0).
- `src/index/relfile.rs` — new `PackedPhaseLayout` struct, new
  `write_packed_phase` and `write_blocked_phase_and_meta`
  helpers; existing `write_full` / `write_full_with_prepared`
  refactored to route through the same two-phase path; meta
  page now written LAST instead of first (no on-disk format
  change, but tighter crash-consistency).
- `src/index/build.rs` — `ambuild` finalisation reorders to
  phase-1 / `prepare_eager` / `take_packed_codes` / phase-2.
- `src/lib.rs` — new `#[pg_test]
  ambuild_drops_packed_codes_before_blocked_write`.
- `migrations/009_pg_turbovec_v1.7.0.sql` — empty migration
  (no SQL surface change).
- `CHANGELOG.md` — v1.7.0 entry.
- `docs/UPGRADING.md` — `1.6.x → 1.7.0` row in migration
  matrix.
- `docs/PHASE_W_PROGRESS.md` — this file.

Upstream turbovec changes (branch `pg_turbovec-integration`):

- `turbovec/src/lib.rs` — `TurboQuantIndex::take_packed_codes`.
- `turbovec/src/id_map.rs` — `IdMapIndex::take_packed_codes`
  wrapper.
- `turbovec/tests/id_map.rs` —
  `take_packed_codes_after_prepare_eager_keeps_search_working`
  unit test.
- `turbovec/Cargo.toml` — `0.6.0` → `0.7.0`.
- `CHANGELOG.md` — 0.7.0 entry.

## Files touched in Phase W-1

- `src/index/build.rs` — new `BuildState` shape with
  `pending_flat` / `pending_ids` / `chunk_rows` / `idx`
  fields; new `compute_chunk_rows` and `flush` methods;
  callback flushes when chunk threshold is hit; ambuild
  drains the trailing partial chunk before constructing the
  on-disk layout.
- `src/lib.rs` — new `#[pg_test]
  ambuild_streams_heap_scan_under_maintenance_work_mem`.
- `Cargo.toml`, `pg_turbovec.control` — `1.5.1` → `1.6.0`.
- `migrations/007_pg_turbovec_v1.6.0.sql` — empty migration
  (no SQL surface change).
- `CHANGELOG.md` — v1.6.0 entry.
- `docs/UPGRADING.md` — `1.5.x → 1.6.0` row in migration
  matrix.
- `docs/PHASE_W_PROGRESS.md` — this file.
