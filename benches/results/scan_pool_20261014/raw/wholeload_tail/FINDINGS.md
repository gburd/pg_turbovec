# Whole-load IVF latency tail — root cause (2026-10-15)

A follow-up to the 2.13.0 qualification "surprise 1": whole-load IVF
(`out_of_core = off`) has a per-query latency tail that the out-of-core path
does not, on the SAME 1M × 1024-d index, and 2.13.0 does not change it. Run on
a c7i.8xlarge (AVX-512 VBMI+VNNI), pg_turbovec `fix/scan-concurrency-2.12.2`
(see `pgtv_sha.txt`), 1M × 1024-d Cohere wikipedia-2023-11, flat/IVF 4-bit
lists=1000, 400 held-out queries, warm. Raw: `A_planes_on.jsonl`, `A.txt`,
`B.txt`, `C.txt`, `D.txt`; scripts `tail_*.py`, `tail_all.sh`.

## It is NOT the thread pool, NOT the data distribution, NOT recall

- `A.txt`: whole-load p50 74.3 ms, p95 133 ms; out-of-core p50 75.9, p95 135.
  (This 400-query sample was dominated by slow queries — a different draw than
  the 1,000-query qualification set where whole-load p50 was 35 ms — but the
  tail and its cause are the same.) Returned top-10 set identical for 400/400
  queries between the two paths.
- `C.txt`: `scan_parallelism = 1` changes nothing (62.7 vs 67.5 ms slow-set
  median). So it is not the per-query thread pool 2.13.0 bounded.
- Whole-load and out-of-core return the same ids, so it is not recall.

## It IS the classic masked kernel's O(candidates × k) top-k heap

`D.txt` (perf, 30 slowest whole-load queries): **52% of CPU is
`turbovec::search::avx512_post_flush_heap_update`**, the classic kernel's
per-block top-k maintenance; `materialize_rotation_matrix` is ~5% (a separate
cost, below).

The decisive knob (`C.txt`): **`hi_dim_rerank = off` cuts the slow-set median
from 67.5 ms to 13.8 ms**; `search_k = 10` with it off, 12.8 ms. `probes = 64`
(4× the cells) barely moves it (63.4 ms) — it is not the probed-row count.

Mechanism, read from `turbovec/src/search.rs`:
- A whole-load IVF scan calls `search_with_mask` over the full 1M-row index
  with a mask covering the probed cells, requesting `k` = the `hi_dim_rerank`
  widened candidate count (several hundred at 1024-d), not 10.
- The classic (non-planes) kernel always runs with `buffered = false`, so each
  block that clears the threshold inserts into a `k`-entry heap with an O(k)
  `rescan_min` per insertion (`avx512_post_flush_heap_update`,
  `heap_min_idxs != HEAP_BUFFERED`). Cost is O(candidates × k); with k in the
  hundreds over 16,000 probed rows that is the tail.
- The planes path uses a capacity-`2k` unordered buffered collector
  (`HEAP_BUFFERED`) instead, which is why flat planes queries have no such
  tail. A masked scan never reaches it (`buffered = planes… && mask.is_none()`
  / the classic masked kernel hard-codes `buffered = false`).
- The out-of-core path avoids it structurally: it gathers the probed cells
  into a compact index of a few thousand rows and runs the SAME k-heap over
  that small n, so O(candidates × k) is tiny.

## Consequences / recommendation

- **Not a correctness bug and not the 2.13.0 bug.** Scores and recall are
  unaffected; it is a performance limitation of the whole-load masked path.
- **Does not block the pg_vectorsearch benchmark**: the IVF cells use the
  out-of-core path (the recommended serving mode at scale), which is already
  fast. Whole-load IVF latency should not be published until this is fixed.
- **Fix (turbovec, separate change):** give the classic masked kernel the same
  buffered collector the planes path uses when `k` is large, or route a masked
  whole-load scan through the compact-gather path. Sized work, not a one-liner;
  filed as a follow-up, not a 2.13.0 blocker.
- **Second, smaller cost:** `materialize_rotation_matrix` rebuilds the
  dim×dim rotation per query (~5–10% here; ~5 ms at 1024-d, ~10 ms at 1536-d
  measured separately). It is a pure function of `dim`; cache it per index.
  Also a follow-up.
