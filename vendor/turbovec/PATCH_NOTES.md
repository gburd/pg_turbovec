# Vendored `turbovec` 0.5.0 with in-memory I/O surfaced

This directory is a copy of the upstream
[`turbovec` 0.5.0](https://crates.io/crates/turbovec) crate with a
small additive patch that exposes the existing reader/writer-based
internals as a public API.

## Why we vendor

Upstream's `IdMapIndex::load(path)` and `IdMapIndex::write(path)`
take `impl AsRef<Path>` only. When the index payload is already in
RAM (for us: read out of a PostgreSQL `bytea` column via SPI),
going through a path means: write to `/tmp`, fsync, re-open, read
back, parse. On a 1 M × 384-dim × 4-bit index that's ≈195 MB of
needless I/O, dominating cold-cache `amgettuple` latency
(measured at ~32 s p50 in `benches/results/recall_lat_million_post_cache_2026_05_24.json`).

The internal parsers are already generic over `Read`/`Write` — only
the public surface needs widening.

## What we changed

Four additive surfaces (no behavioural changes to existing APIs):

| Location | Added |
|---|---|
| `src/lib.rs` | `TurboQuantIndex::from_parts` made `pub` (was `pub(crate)`). `packed_codes()` and `scales()` made `pub` (were `pub(crate)`). |
| `src/lib.rs` | Added a `boundaries: OnceLock<Vec<f32>>` field on `TurboQuantIndex` next to the existing `centroids: OnceLock<Vec<f32>>`. `add_with_ids` now caches the Lloyd-Max codebook boundaries on the first add and reuses them across subsequent adds in the same backend. Pre-cache, the boundaries were recomputed on every single-row insert (~47 ms / row at bit_width=4 / dim=8); caching brings that to amortised constant. Pure performance fix; the same boundaries are produced. |
| `src/io.rs` | `pub fn write_to<W: Write>(…)` mirroring `write`. `pub fn load_from<R: Read>(…)` mirroring `load`. `pub fn write_id_map_to<W: Write>(…)` mirroring `write_id_map`. `pub fn load_id_map_from<R: Read>(…)` mirroring `load_id_map`. The path-based functions are now thin wrappers around the generic ones. |
| `src/id_map.rs` | `IdMapIndex::write_to_writer<W: Write>` and `IdMapIndex::load_from_reader<R: Read>` thin wrappers around the new `io::*_from`/`*_to` functions. Refactored shared post-decode validation into `from_id_map_parts`. |

All upstream tests (`cargo test -p turbovec`) still pass.

## Upstreaming plan

We intend to submit the same diff as a PR to
<https://github.com/RyanCodrai/turbovec> once it has soaked here.
When upstream merges and publishes a release with these APIs we
will switch `pg_turbovec`'s dependency back to crates.io and remove
this directory.

## License

`turbovec` is MIT-licensed. We retain the upstream `Cargo.toml`,
`README.md`, `examples/`, and `tests/` unchanged. Our additive
patches are also MIT, matching upstream.
