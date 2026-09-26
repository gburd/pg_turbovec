# Upstream proposal #3 — parallelize `pack::repack` (byte-identical)

**Fork carry:** `47a26a3` (branch `pgtv-2.0.0-port`, on top of turbovec 1.0.0 `ccab9f3`)
**Target file:** `turbovec/src/pack.rs`
**Process note:** issue-first. This is the **most likely to be accepted** of the
three — it's a pure perf win, byte-identical, uses machinery (rayon +
`par_chunks_mut` + a `PAR_THRESHOLD`) that already exists in the *same file*, and
touches no public API. Depends on / composes with proposal #1.

---

## (a) Motivation — from turbovec's own perspective

`repack` is the row-major → SIMD-blocked transform. It's O(n·dim) and single-
threaded today, so it's a serial floor on any load path that has to build the
blocked layout — the larger the corpus, the more it dominates load latency.
Every consumer that reconstructs an index from persisted codes pays it once per
open. turbovec already parallelizes the *sibling* transform in the same file
(`apply_native_transform`, which uses `rayon` `par_chunks_mut` above a 4 MiB
`PAR_THRESHOLD`); `repack` is structurally identical — each output block depends
only on its own input rows and lands in a disjoint output byte range — so it's
the same pattern applied to the one hot transform that wasn't yet parallel. This
is general-utility: it speeds turbovec's *own* `from_parts`/load path on any
multicore host, not just an embedder's.

## (b) Exact diff summary

`+52` lines in `pack.rs` body of `repack` + a `+81`-line `#[cfg(test)]` block.
No public signature change. No behaviour change (byte-identical output).

- Adds `use rayon::prelude::*` (rayon already a turbovec dep: `rayon = "1.12"`).
- `const PAR_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;` (mirrors the existing
  `apply_native_transform` PAR_THRESHOLD in the same file) and
  `const BLOCKS_PER_TASK: usize = 64;`.
- Below the threshold (or `n_blocks <= 1`): unchanged serial path
  (`extract_codes_flat` → `pack_blocked_native!`).
- Above the threshold: pre-allocate the full `blocked` buffer and fill it with
  `par_chunks_mut(BLOCKS_PER_TASK * bytes_per_block)`, each task computing its
  block range via the (crate-internal) `repack_block_range` and `copy_from_slice`
  into its slot. Because block `i`'s output occupies exactly the disjoint range
  `[i·n_byte_groups·BLOCK, (i+1)·n_byte_groups·BLOCK)` and depends only on rows
  `[i·BLOCK, (i+1)·BLOCK)`, concatenation is byte-identical to serial.
- Test `parallel_repack_is_byte_identical_to_serial` asserts `par == seq`
  **byte-for-byte** against a serial reference oracle kept in the test module,
  across 11 shapes: sub-threshold (serial path), above-threshold (rayon path),
  `n` not a multiple of `BLOCK` (tail padding), `n` not a multiple of
  `BLOCKS_PER_TASK·BLOCK` (partial last task), and every supported bit width
  (2/3/4-bit).

## Measured win

- **Micro (repack alone), 250k × 1024d × 4-bit:** ~6 s serial → ~250 ms on
  8 cores (~24×).
- **End-to-end pg_turbovec cold-scan, 1M × 1024d × 4-bit flat** (the transform is
  paid once per backend at cold index-open;
  `benches/results/rebench_20260925/coldscan_{old_serial,new_parallel}.log`):
  cold-backend p50 **1765.9 ms → 566.2 ms** (min 1753 → 562, p95 1777 → 570,
  n=20). Warm-backend p50 unchanged (30.4 → 30.6 ms) — as expected, warm doesn't
  re-repack. This is the item-2 cold penalty called out in
  `benches/results/rebench_20260925/FINDINGS.md`.

## (c) API-stability / maintenance concerns the maintainer would raise

- **Determinism / correctness is the whole risk, and it's the one turbovec
  cares about most.** A wrong byte here is a silently mis-scored index — no
  crash, no error, just degraded recall at cold-open. The guard test asserts
  byte-identity directly (not via a round-trip), across the boundary shapes
  (threshold crossing, tail padding, partial last task, all bit widths). That is
  exactly the "assert the property the fast path preserves" the mutation gate
  demands, and it's the argument that should carry the PR.
- **`repack_block_range` is `pub(crate)`** — this diff relies on it. Fine
  internally; no new public surface.
- **Thread-pool interaction.** Using the global rayon pool inside `repack` means
  a caller that wants to bound parallelism (pg_turbovec runs it inside a bounded
  build pool via `install`) inherits whatever pool is current. That matches
  `apply_native_transform`'s existing behaviour, so it's consistent, but the
  maintainer may want a note that `repack` now touches the rayon pool.
- **`BLOCKS_PER_TASK = 64` and the 4 MiB threshold are tuning knobs**, not
  derived constants. They mirror the existing sibling transform's choices; the
  maintainer may want them justified or unified with `apply_native_transform`'s.
- **Interaction with proposals #1/#2:** independent of #2. Composes with #1 —
  if `repack` isn't public (#1 rejected), turbovec still benefits from a parallel
  internal `repack` on its own load path, so #3 stands alone on upstream's own
  merits even if #1 doesn't land.

## (d) Proposed issue/PR text

**Title:** `Parallelize pack::repack (byte-identical, mirrors apply_native_transform)`

**Body:**

> `pack::repack` (row-major → SIMD-blocked) is single-threaded and O(n·dim), so
> it's a serial floor on the load path that dominates as the corpus grows. The
> sibling transform `apply_native_transform` in the same file is already
> parallelized with `rayon` `par_chunks_mut` above a 4 MiB `PAR_THRESHOLD`;
> `repack` is structurally identical — each output block is a disjoint byte range
> depending only on its own input rows — so the same pattern applies.
>
> This parallelizes `repack` over block-aligned ranges above the same threshold,
> serial below it (and for `n_blocks <= 1`). Output is **byte-identical** to the
> current serial path — pinned by `parallel_repack_is_byte_identical_to_serial`,
> which asserts `par == seq` byte-for-byte across 11 shapes covering the
> threshold crossing, tail padding, a partial last task, and 2/3/4-bit.
>
> Measured: 250k × 1024d × 4-bit repack ~6 s → ~250 ms on 8 cores. End-to-end in
> pg_turbovec (which pays this once per backend at index-open), a 1M × 1024d
> flat cold-scan drops from p50 1766 ms → 566 ms with warm latency unchanged.
>
> rayon is already a dependency. No public API change. Flagging as an issue per
> CONTRIBUTING; happy to open the PR if you're open to it. It'd get an
> `## [Unreleased]` line (behaviour-preserving perf change to shipped code, so
> the changelog gate applies even though it's not new surface).

**Test plan (if invited to PR):**
`cargo test -p turbovec --release` **and the debug leg** (the debug CI leg is
where the block-alignment `debug_assert!`s in this diff actually execute — the
release matrix elides them), `parallel_repack_is_byte_identical_to_serial`
included; before/after repack timing in the PR body.
