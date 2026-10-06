# Fork carry #4 — parallel `pack::planes_repack` (cold-open), DRAFT, not filed

Branch: gburd/turbovec@pgtv-2.11.0-port, commits ba92958 + 455549f (on 1.1.1).

## Problem

1.1.0's planes layout builds the search cache via `BlockedCache::build` ->
`pack::planes_repack`, which is serial. `pack::repack` (the classic layout's
cache build) is still serial upstream too, but an embedder that rebuilds the
cache per process from packed codes (pg_turbovec: once per PostgreSQL backend)
pays it on every cold open. Measured on Graviton4 (c8gd.8xlarge, 32 vCPU),
1M x 1024-d:

| | 4-bit | 2-bit |
|---|---|---|
| `planes_repack` serial (1.1.1) | 279 ms | 1,092 ms |
| parallel, block-range tasks written in place | 18-21 ms | 32 ms |

## Change

Split the block space into 64-block ranges; each task calls the existing
`planes_repack_block_range` and copies its sign bytes and low rows straight into
disjoint chunks of the two output regions (`par_chunks_mut` zip). Serial below
4 MiB of packed codes. Serial body kept as `planes_repack_serial` (test oracle).

## Proof

`parallel_planes_repack_is_byte_identical_to_serial`: 2/4-bit, sub/above the
threshold, tail padding (n % 32 != 0), partial last task; passes on x86_64 and
aarch64. The planes test module (33 tests) still passes on aarch64.

## Ask

Pair with #545 (parallel `repack`). Same shape, same byte-identity pin.
