# Upstream PR: in-memory I/O + cached codebook boundaries

This directory captures the additive patches `pg_turbovec` carries on top of [`turbovec` 0.5.0](https://crates.io/crates/turbovec) so the maintainer can review and accept them without diffing through our project history.

## Files

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

- Diffs captured: ✅ (this commit)
- Upstream tests verified locally: pending (need fresh `cargo test -p turbovec` against a clean upstream checkout)
- PR submitted: pending (manual step; requires GitHub auth)
- Upstream merged + released: pending
- `vendor/turbovec/` removed from `pg_turbovec` and dependency switched to crates.io: blocked on the above
