# Fix B: decode the constant operand once per expression (2026-10-07)

EC2 c7i.4xlarge (Xeon Platinum 8488C, AVX-512), PostgreSQL 16.15 built
`-O2 -g -fno-omit-frame-pointer` without cassert (a private copy at
`/work/pg16rel_b`), postmaster pinned to CPUs 6,7. 20,000 rows of random
1024-d vectors. `t` uses the type's default storage (values TOASTed out of
line); `tp` is the same data with `SET STORAGE PLAIN`. Base is `3ff7768`;
new is `perf/fix-b-qcache`. Both are release builds and the two `.so` files
are swapped into one cluster, alternating base/new for 5 rounds. Each cell
is the median over rounds of the median of 5 timed runs (`EXPLAIN (ANALYZE,
TIMING OFF)`; the first of 6 runs is discarded).

## Per-call cost, `SELECT sum(tv <op> '<1024-d literal>') FROM tab` (µs/row)

Every function aligned to 64 bytes in both builds (`ab64.*`, see below):

| table | op | base | new | Δ |
|---|---|---:|---:|---:|
| t (TOASTed) | `<=>` | 10.50 | 8.21 | −2.29 |
| t | `<->` | 9.49 | 7.08 | −2.41 |
| t | `<#>` | 9.33 | 6.91 | −2.41 |
| t | `<+>` | 9.38 | 7.06 | −2.32 |
| t | `vector_dims` (control: 1 decode, code untouched) | 5.88 | 5.86 | −0.01 |
| tp (PLAIN) | `<=>` | 7.65 | 5.33 | −2.32 |
| tp | `<->` | 6.55 | 4.21 | −2.35 |
| tp | `<#>` | 6.47 | 4.12 | −2.35 |
| tp | `<+>` | 6.59 | 4.18 | −2.41 |
| tp | `vector_dims` (control) | 3.11 | 3.11 | −0.00 |

The saving is about 2.3 µs per call, the same on every operator and both
storages. That is about one CBOR decode of a 1024-d value on this host
(about 2.5 µs, inferred from `vector_dims` on `tp`, which is one decode plus
the scan). So the constant is no longer decoded per row, and the cache's own
cost (detoast-packed, a ~5 KB memcmp, a ~5 KB memcpy on the row side) is
small next to a decode.

## Default build alignment (`ab1.*`)

The same A/B with default function alignment showed the same −2.0 to −2.1
µs/call on the distance ops, but also +0.36 µs/row on the `vector_dims`
control, which this change does not touch. The CBOR decoder's machine code
is byte-identical between the two builds; only its start address moved
(`parse_value` at offset 0 mod 64 in base, 32 mod 64 in new). With every
function 64-byte aligned (`-C llvm-args=-align-all-functions=6`) the control
difference went to zero and the distance-op saving went to −2.3. So the
+0.36 is a code-placement effect on the shared decoder, not a cost of this
change. Any release build can land on either side of that. Treat ±0.4
µs/decode as the layout noise floor on this host.

## ORDER BY recheck, flat 4-bit cosine index, `search_k = 1024`, LIMIT 10 (`ab2.*`)

20 distinct queries (query vector taken from a row by subquery), 3 reps
each, per-query minimum, median over queries, 5 alternated rounds:

| table | base ms | new ms | Δ ms | Δ per candidate |
|---|---:|---:|---:|---:|
| t (TOASTed) | 18.52 | 16.12 | −2.40 | −2.34 µs |
| tp (PLAIN) | 15.20 | 12.88 | −2.32 | −2.26 µs |

Default-alignment builds. The per-candidate saving matches the per-call
micro-measure.

## Correctness

`cargo pgrx test pg16` on the same host: 458 passed, 0 failed, 8 ignored
(the 448-test baseline plus the 10 `distance_cache_*` tests). Before the
fix, `distance_cache_decodes_constant_once_per_expression` fails (2 decodes
per call); the other nine pass both before and after, which pins behaviour.
Two mutations of the cache key were run against the tests:
length-only comparison (3 tests fail) and ignoring the last 16 payload
bytes (2 tests fail).

Not measured: Graviton/aarch64, other dimensions, and parallel query
(each parallel worker has its own FmgrInfo, so each decodes the constant
once).
