# Upstream PR: in-memory I/O + pub `from_parts` (issue #70)

> **Status (2026-06-15):** issue
> [#70](https://github.com/RyanCodrai/turbovec/issues/70) is still
> OPEN. Upstream has been very active (v0.6 → v0.9.0) and merged a
> large security-audit PR (#108) that, among other things, **fixed
> a pre-AVX2 x86_64 scalar-fallback bug** that was producing
> silently-wrong top-k — a bug `pg_turbovec` hit in production on a
> pre-AVX2 bench host. So the upstream relationship is paying off
> even without #70 being merged.
>
> The `pg_turbovec` fork now tracks **upstream v0.9.0**
> (`gburd/turbovec` branch `pg_turbovec-integration-v0.9.0`,
> commit `d3d468e`). The diffs below were written against
> turbovec 0.5.0 and are **stale** — the current fork patch is
> re-applied by hand on top of v0.9.0 and additionally threads the
> new TQ+ calibration fields through `from_parts`. If the
> maintainer invites a PR, it should be regenerated against v0.9.0;
> the API shape (pub `from_parts` / `packed_codes` / `scales`,
> `Read`/`Write` IO, `from_id_map_parts*`) is unchanged in intent.

This directory captures the additive patches `pg_turbovec` carries on top of [`turbovec`](https://crates.io/crates/turbovec) so the maintainer can review and accept them without diffing through our project history.

## Files (stale — written against turbovec 0.5.0)

- `01-lib-rs.diff` — `TurboQuantIndex::{from_parts, packed_codes, scales}` made `pub`, plus a new `boundaries: OnceLock<Vec<f32>>` field that caches the Lloyd-Max decision boundaries alongside the existing `centroids: OnceLock<Vec<f32>>`.
- `02-io-rs.diff` — new `pub fn write_to<W: Write>`, `load_from<R: Read>`, `write_id_map_to<W: Write>`, `load_id_map_from<R: Read>` mirroring the existing path-based functions. The path-based functions are refactored to be thin wrappers.
- `03-id_map-rs.diff` — `IdMapIndex::write_to_writer<W: Write>` and `load_from_reader<R: Read>` thin wrappers around the new `io::*_from`/`*_to` functions; shared post-decode validation extracted into `from_id_map_parts`.

302 lines total; all additive (no behavioural changes to existing APIs).

## Suggested PR title

> Public `Read`/`Write` I/O API + cached codebook boundaries

## Suggested PR body

(copy-paste into the GitHub PR form)

```markdown
This PR adds two additive features useful for embedders that work
with already-in-memory payloads (databases, RPC servers, vector
stores) and inserts at sub-second cadence.

## 1. In-memory I/O surface

Upstream today has `IdMapIndex::load(path)` and `write(path)`
taking `impl AsRef<Path>`. The internal parsers are already
generic over `Read` / `Write` — only the public surface needs
widening.

Adds:

- `io::write_to<W: Write>`, `io::load_from<R: Read>`
- `io::write_id_map_to<W: Write>`, `io::load_id_map_from<R: Read>`
- `IdMapIndex::write_to_writer`, `IdMapIndex::load_from_reader`

Refactors the existing path-based functions into thin wrappers
around the new generic ones; no wire-format change.

## 2. Cached Lloyd-Max codebook boundaries

`TurboQuantIndex` already caches `centroids: OnceLock<Vec<f32>>`.
This PR adds a sibling `boundaries: OnceLock<Vec<f32>>` for the
decision boundaries computed alongside the centroids during the
codebook build.

Why: `TurboQuantIndex::add_with_ids` was recomputing the
Lloyd-Max boundaries on every single-row insert. At
`bit_width=4` / `dim=8` that's ~47 ms per row in our
measurements; for a 1 k-row bulk insert that's 47 s of pure
codebook recomputation.

Caching brings it to amortised constant. Same boundaries are
produced; pure perf fix.

## 3. Visibility

Three previously-`pub(crate)` items on `TurboQuantIndex` are
made `pub` so external embedders can construct the index from
their own deserialised parts:

- `from_parts(dim, bit_width, n_vectors, packed_codes, scales)`
- `packed_codes()`
- `scales()`

These are exactly what the new `IdMapIndex::load_from_reader`
helper uses internally; making them public lets consumers do the
same when they want a different deserialisation strategy.

## Tests

All upstream tests (`cargo test -p turbovec`) continue to pass:
- `tests::id_map`, `concurrent_search`, `distortion`,
  `kernel_correctness`, `lazy_init`, `rotation`, `swap_remove`,
  `io_versioning`, `filtering`, `codebook`, `encode` — all
  green.

Additive patches; no behavioural changes to existing APIs; no
wire-format or version bump needed.

## Caller perspective

`pg_turbovec` (the PostgreSQL extension that motivated this PR)
has been carrying these patches as `vendor/turbovec/` since
v1.0.0. Real-world numbers from a 1 M × 1536-dim OpenAI
embedding corpus on Intel i9-12900H:

- Pre-cache boundary recompute: 1 k-row bulk INSERT took ~400 s.
- Post-cache: ~136 ms (~3000× speedup), commit-time serialise
  dominates the new floor.

The in-memory `Read`/`Write` API also lets database extensions
read TVIM payloads straight from a database column without an
intermediate tmpfile. Cold-path scan latency dropped from
6.8 s to 6.8 s (no measurable change at our scale because the
SPI fetch dominated, but the API hygiene win was worth shipping
on its own).

If accepted, `pg_turbovec` will drop its `vendor/turbovec/`
directory and depend on the next `turbovec` release directly
from crates.io.
```

## How to submit

```bash
# After confirming all three diffs apply cleanly to upstream HEAD:
gh repo clone RyanCodrai/turbovec ~/oss/turbovec-fork
cd ~/oss/turbovec-fork
git checkout -b in-memory-io-and-cached-boundaries
patch -p1 < /home/gburd/ws/pg_turbovec/upstream-pr/01-lib-rs.diff
patch -p1 < /home/gburd/ws/pg_turbovec/upstream-pr/02-io-rs.diff
patch -p1 < /home/gburd/ws/pg_turbovec/upstream-pr/03-id_map-rs.diff
cargo test
git commit -am "Add public Read/Write I/O API + cached codebook boundaries"
git push -u fork in-memory-io-and-cached-boundaries
gh pr create --repo RyanCodrai/turbovec --title "Public Read/Write I/O API + cached codebook boundaries" --body-file PR_BODY.md
```

## Status

- Diffs captured: complete (this directory; targets upstream `turbovec 0.6.0`)
- Upstream tests verified locally: complete (`cargo test --tests --release` against
  the draft branch on `RyanCodrai/turbovec` v0.6.0; all upstream test
  files green)
- Upstream issue opened: complete — [issue #70](https://github.com/RyanCodrai/turbovec/issues/70)
- Draft PR branch published: complete — [`gburd/turbovec` `in-memory-io-and-pub-from-parts`](https://github.com/gburd/turbovec/tree/in-memory-io-and-pub-from-parts)
- PR submitted: pending the by-invitation gate per upstream `CONTRIBUTING.md`
- Upstream merged + released: pending

## Relationship to `pg_turbovec-integration`

This PR (the `in-memory-io-and-pub-from-parts` branch) is now the
**parent / subset** of the broader [`gburd/turbovec` `pg_turbovec-integration`](https://github.com/gburd/turbovec/tree/pg_turbovec-integration)
branch. Starting with `pg_turbovec` v1.4.0, the extension depends on
that fork branch directly via `git = ...` in `Cargo.toml` instead of
vendoring `turbovec` 0.5.0 under `vendor/turbovec/`.

The fork branch is a strict superset of this PR — it adds Phase P
prepared-cache accessors (`prepare_eager`, `blocked_codes`, `n_blocks`,
`centroids`, `boundaries` getter, `from_parts_with_prepared`,
`from_id_map_parts_with_prepared`) and the Phase R-2 rotation
accessor (`rotation`, `rotation_size`) that `pg_turbovec` needs to
persist Lloyd-Max + QR + SIMD-blocked layouts into its relfile
and pre-fill the inner `OnceLock`s on scan startup.

When upstream merges any portion of issue #70, the fork branch
rebases onto the released version and `pg_turbovec` follows. The
Phase P/R-2 additions stay on the fork until upstream is ready
for a separate PR cycle (or the user's mmap-resident reads in
Phase S settle the API surface enough to upstream as one chunk).

## Notes

Upstream `turbovec 0.6.0` already absorbed the cached Lloyd-Max boundaries
(`boundaries: OnceLock<Vec<f32>>` on `TurboQuantIndex`) that we'd been
carrying separately — they arrived at the same fix independently. The
remaining items in the issue are the in-memory `Read`/`Write` API and
promoting `TurboQuantIndex::from_parts` (plus its `packed_codes()` /
`scales()` accessors) from `pub(crate)` to `pub`. The diff captured
here targets `v0.6.0` (not the original `v0.5.0` we vendored), so when
upstream merges we can `cargo update -p turbovec` and drop
`vendor/turbovec/` cleanly.
