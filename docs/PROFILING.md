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

- Script committed: yes (`benches/scripts/profile_warm_scan.sh`).
- First profile run: 2026-05-25 on arnold against v1.3.0 (head
  `99e5dd7`) + dbpedia-1M, 4-bit `docs_tv_4bit`, 50-iteration warm
  loop @ 999 Hz for 10 s. Artefacts:
    - `benches/results/profile_warm_v1_3_0_2026_05_25.perf.gz`
    - `benches/results/profile_warm_v1_3_0_2026_05_25.symbols.txt`
    - `benches/results/profile_warm_v1_3_0_2026_05_25.flame.svg`
    - `benches/results/profile_warm_v1_3_0_2026_05_25.json`
- Verdict: **K** — the kernel.
- Optimisation identified + landed: **identified, not landed.** See
  *Next steps* below for Phase R-2.

## Verdict from the 2026-05-25 profile

Top-3 hot symbols by self time:

| Rank | Symbol | Self % |
|---:|---|---:|
| 1 | `gemm_f64::microkernel::fma::f64::x2x6` | 64.77 |
| 2 | `crossbeam_epoch::default::with_handle::<…,Guard>` | 7.50 |
| 3 | `<crossbeam_deque::deque::Stealer<…>>::steal` (rayon) | 2.24 |

By layer (sum of distinguishable self-pct):

| Layer | Share |
|---|---:|
| Kernel **setup** — f64 GEMM + rayon machinery | ~81% |
| Scheduler / futex / yield (kernel-mode side-effect of rayon) | ~14.5% |
| SIMD-LUT scoring (`score_block_*`, `fast_scan`, NEON-LUT build) | **0%** |
| Recheck-orderby (`cosine_distance`, `dot_product`, `slot_getsomeattrs`) | **0%** |
| Executor internals (`BufferGetPage`, `index_form_tuple`) | **0%** |

The profile is dominated by `gemm_f64::microkernel::fma::f64::x2x6`,
which the codebase only invokes from one place —
`turbovec::rotation::make_rotation_matrix` building a `dim x dim`
orthogonal matrix via `faer::Mat::<f64>::qr()`. This is **not** the
SIMD-LUT scoring loop the v1.2 audit predicted would dominate option
K; it is the kernel **setup** path, specifically the rotation matrix
QR construction. The scoring kernel itself, the recheck-orderby
phase, and the executor internals contribute essentially nothing to
the warm-scan profile on this corpus.

Mechanism: `from_id_map_parts_with_prepared` populates the prepared
`OnceLock`s for `centroids`, `boundaries` and `blocked` from the
relfile, but **leaves `rotation` empty**. The first `search()` call
on a freshly loaded `IdMapIndex` therefore runs the full f64 QR
decomposition on a 1536×1536 random Gaussian matrix — ~5 GFLOPs of
f64 work, dispatched across the rayon pool, and visible in the
profile as `gemm_f64::*` plus the surrounding rayon synchronisation.

## Next steps — Phase R-2

**Persist the rotation matrix in the relfile alongside the existing
prepared parts.** The rotation is a deterministic function of `(dim,
ROTATION_SEED)`, so `ambuild` can build it once via
`make_rotation_matrix(dim)` and write its `dim*dim` f32 buffer
(about 9 MiB at 1536-d) into the meta-page chain. The scan-side
`from_id_map_parts_with_prepared` then pre-fills the `rotation`
`OnceLock` from those bytes, eliminating the f64 QR cost from every
scan path.

Expected gain: ~50-200 ms removed from cold-cache scans and from
any scan whose backend has not yet warmed the per-backend cache.
On this corpus that closes the 1.16× (and now 1.6×, see below) gap
to HNSW ef=40 outright.

Risk: minor. Adds ~9 MiB to the on-disk index footprint at 1536-d
(negligible vs. the 1527 MiB blocked-layout body) and bumps the
relfile wire format. Backward-compatible at read time: a relfile
without rotation persisted falls back to the current
QR-on-first-search path.

Diagnostic side-question to settle in the same phase: the v1.3.0
warm-p50 on this measurement is ~96-105 ms (vs. the 71 ms quoted
above from v1.2), which suggests rotation may be running on
*every* scan rather than once per backend — i.e. the per-backend
cache may be missing more often than expected. Persisting the
rotation removes the symptom either way; if the cache is in fact
being missed, that is its own bug to fix in the same patch.
