# Vendored `turbovec` 0.5.0 with in-memory I/O surfaced

This directory is a copy of the upstream
[`turbovec` 0.5.0](https://crates.io/crates/turbovec) crate with two
additive patches that expose the existing reader/writer-based
internals as a public API and let callers persist the prepared
search caches alongside the raw codes.

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

### Phase P follow-up: prepared-cache accessors

The `from_parts(…)` constructor leaves the search caches
(`blocked`, `centroids`, `boundaries`) empty, so the first call to
`search` on a freshly-loaded index pays the full one-time
`pack::repack` + Lloyd-Max codebook cost. On a 1 M × 1536-d × 4-bit
index that's measured at ~26 s of wall-clock per fresh backend
(`benches/results/recall_relfile_cold_scan_v1_3_0_2026_05_25.json`).
For `pg_turbovec`'s relfile path that means every backend opening
the index for the first time burns 26 s before serving its first
query, even though the prepared layout is a deterministic function
of the on-disk codes.

Phase P adds a second additive layer that lets the embedder
persist the prepared layout out-of-band and feed it back at load
time. Backend startup then drops to buffer-pool I/O cost only.

| Location | Added |
|---|---|
| `src/lib.rs` | `TurboQuantIndex::blocked_codes() -> &[u8]`, `n_blocks() -> usize`, `centroids() -> &[f32]`, `boundaries() -> &[f32]`. Public read-out of the OnceLock contents — forces compute if not yet cached. |
| `src/lib.rs` | `TurboQuantIndex::prepare_eager(&self)`. Strict superset of `prepare`: also primes the `boundaries` cache so all four prepared parts are readable. |
| `src/lib.rs` | `TurboQuantIndex::from_parts_with_prepared(…)` constructor that takes blocked codes + n_blocks + centroids + boundaries and pre-fills the OnceLocks. Subsequent `search` calls skip `pack::repack` and the codebook compute entirely. |
| `src/id_map.rs` | `IdMapIndex` thin wrappers: `prepare_eager`, `blocked_codes`, `n_blocks`, `centroids`, `boundaries`, `from_id_map_parts_with_prepared`. Shared post-decode validation refactored into a private `finalise_from_inner` helper. |

The new APIs are strictly additive: existing callers see no
behaviour change, and `from_parts` / `from_id_map_parts` continue
to leave the caches empty (which is the correct default when the
embedder doesn't have a persisted prepared layout).

All upstream tests (`cargo test -p turbovec`) still pass.

### Phase R-2 follow-up: persisted rotation matrix

The rotation matrix (`rotation::make_rotation_matrix(dim)`) is
a deterministic function of `(dim, ROTATION_SEED)` produced by
QR decomposition of a `dim x dim` Gaussian random matrix. At
`dim = 1536` the QR alone is the dominant warm-scan hotspot
on a fresh backend (~64% self time on the dbpedia-1M profile;
see `benches/results/profile_warm_v1_3_0_2026_05_25.json`).
Phase R-2 in `pg_turbovec` persists the matrix in the relfile
so the OnceLock comes back pre-populated when a backend opens
the index.

| Location | Added |
|---|---|
| `src/lib.rs` | `TurboQuantIndex::rotation() -> &[f32]` accessor that drives the existing `rotation` `OnceLock` and returns the prepared matrix. Mirrors `centroids()` / `boundaries()` / `blocked_codes()`. |
| `src/lib.rs` | `TurboQuantIndex::rotation_size(dim) -> usize` const helper (`dim * dim`) so callers can preallocate the on-disk chain without instantiating an index. |
| `src/lib.rs` | `TurboQuantIndex::from_parts_with_prepared(…)` extended with a final `rotation: Option<Vec<f32>>` parameter. `Some(buf)` pre-fills the rotation `OnceLock`; `None` falls back to the lazy QR (existing behaviour, used during `ambuild` when the matrix isn't yet on disk). |
| `src/id_map.rs` | `IdMapIndex` thin wrappers: `rotation`, `rotation_size`, and the matching `rotation: Option<Vec<f32>>` parameter on `from_id_map_parts_with_prepared`. |

This is a follow-up to upstream PR #70 (Phase P prepared
caches). Existing callers that pass `None` for `rotation` see
no behaviour change; the lazy QR runs on first search exactly
as before.

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
