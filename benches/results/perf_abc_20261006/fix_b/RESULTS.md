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

Both rows include a cost this change does not remove: the query vector
comes from `(SELECT tv FROM t WHERE id = ...)`, and `t` stores it as an
external TOAST pointer. The cache compares detoasted bytes, so that
pointer is still fetched from TOAST on every call (the reviewer measured
300 toast block reads for 300 rows with `(SELECT tv FROM src WHERE id =
7)`). A query vector sent by the client as a parameter or literal arrives
inline and does not pay this. Follow-up, not done here: avoid the
per-call fetch, e.g. by detoasting the constant once before the scan.
Keying the cache on the TOAST pointer is not safe: the cache can outlive
a statement (PL/pgSQL simple expressions), and a toast value OID can be
reused after VACUUM.

## Correctness

`cargo pgrx test pg16` on the same host: 462 passed, 0 failed, 8 ignored
(the 448-test baseline plus 14 `distance_cache_*` tests;
`full_suite_pg16.txt`). Before the fix,
`distance_cache_decodes_constant_once_per_expression` fails (2 decodes
per call); the others pass both before and after, which pins behaviour.

The stale-cache tests evaluate their expression below an `OFFSET 0`
fence, so rows reach the cache in physical order. Without the fence PG16+
sorts the input of `array_agg(... ORDER BY id)` first, and the row that
differs from row 1 only in its last coordinate never directly follows
row 1. Further tests cover a LATERAL kNN join (index-scan rescans reuse
one FmgrInfo while the Param changes per outer row), a PL/pgSQL simple
expression (its FmgrInfo lives for the whole transaction), the cache
being freed (a test-only live-cache counter returns to its starting
value at statement end, cursor close and error), and a parallel plan
matching the serial sum.

Two deliberately broken builds were run against the tests
(`broken_build_review.txt`). With a cache key that ignores the last 16
payload bytes, 4 tests fail: inline and toasted values (every one of 30
op/shape cases per table: constant left and right, subplan, cross join),
the LATERAL kNN join and the PL/pgSQL test. In the previous round, before
the `OFFSET 0` fence, only the subplan shape caught this mutation. With a
cache stored in `fn_extra` but never freed (no reset callback), the
freed-cache test fails. An earlier round also tried a length-only key
comparison (3 tests failed).

Not measured: Graviton/aarch64, other dimensions, and parallel query
timing (each parallel worker has its own FmgrInfo, so each decodes the
constant once; correctness is covered by
`distance_cache_parallel_matches_serial`).
