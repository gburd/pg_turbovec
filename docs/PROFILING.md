# Profiling pg_turbovec's warm-scan path

Item 5 from the v1.2.0 audit proposal:

> A `perf record` profile would tell us whether the bottleneck is
> the kernel itself, the executor's recheck-orderby reorder queue,
> or heap page reads. A single targeted optimisation could push
> the multiplier from 1.3× to 3×.

## Why 1.3× and not more

On `dbpedia-entities-openai-1M` (Phase J), the warm-cache p50
breakdown is:

| Index | p50 |
|---|---:|
| pgvector HNSW ef=40 | 61 ms |
| pg_turbovec 4-bit `search_k=100` | 71 ms |
| pg_turbovec 2-bit `search_k=100` | **48 ms** |

We win on 2-bit and lose by 16% on 4-bit. Both indexes do the
same plan shape:

1. AM `amgettuple` returns 100 candidates (ranked by quantised
   cosine distance against `IdMapIndex::search`).
2. Executor's reorder queue re-evaluates the orderby expression
   for each candidate against the heap tuple — exact f32 cosine
   distance via `cosine_distance(emb, q)`.
3. Top-10 popped off the queue.

There are three plausible bottlenecks:

- **(K) The kernel `IdMapIndex::search`** itself, ~50 ms on 1 M ×
  1536-d 4-bit. Bound by SIMD-LUT scoring throughput.
- **(R) The recheck-orderby phase**, which reads 100 heap pages
  and computes 100 exact cosine distances. f32 cosine on 1 536-d
  is ~6 µs/op via `arch_simd::dot_product`, so ~600 µs total —
  but heap-page misses can blow that up to 10 ms+ if the heap
  isn't in `shared_buffers`.
- **(E) Executor overhead** (reorder queue insertion, tuple
  formation, expression evaluation). Usually negligible but
  worth measuring.

If (K) dominates, raise `bit_width` to 4 (we already use it) or
explore SIMD width upgrades. If (R) dominates, make the heap
fit in shared_buffers or switch to in-index payload (we already
keep the f32 vector in `am_storage` but the heap path is what
the executor uses). If (E) dominates, that's a pgrx + index-AM
question.

## How to run

```bash
# Connect to a PG cluster with pg_turbovec installed and a
# bench_dbpedia database holding the 1M corpus.
bash benches/scripts/profile_warm_scan.sh
```

Output:

- `/tmp/turbovec-warm-scan.perf` — raw `perf record` data.
- `/tmp/turbovec-warm-scan-symbols.txt` — top-50 hot symbols.
- `/tmp/turbovec-warm-scan-flame.svg` — FlameGraph (if
  `/opt/FlameGraph` is present).

## What to look for

If `IdMapIndex::search` and its inner `score_block_*` /
`fast_scan` symbols dominate (e.g. >70% of cycles), the win is
in the kernel — look at the upstream `turbovec` SIMD path or the
batched-LUT optimisations.

If `cosine_distance` / `dot_product` symbols dominate, the win
is in the recheck-orderby (R). Possible fixes:
- Pre-compute and cache the f32 vectors in the AM payload so we
  can recheck without a heap read (would double our index size,
  but bring cold + warm scan latency together).
- Switch to a tighter recheck function that uses the quantised
  scores plus a small-correction term computed inline.

If `slot_getsomeattrs`, `BufferGetPage`, or other executor
internals dominate, you're looking at heap-fetch overhead —
make sure shared_buffers is large enough to hold the heap, or
consider a covering index path.

## Status

- Script committed: ✅ (this commit)
- First profile run: pending (manual; needs `perf` perms on the
  bench host)
- Optimisation identified + landed: pending the profile run
