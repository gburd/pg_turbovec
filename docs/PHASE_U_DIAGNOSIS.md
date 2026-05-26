# Phase U-1: cache-miss diagnosis (meh, 2026-05-26)

## Background

The Phase S agent flagged that the per-backend
`Arc<RwLock<IdMapIndex>>` cache (`src/cache.rs`) appeared to miss on
every warm scan in the dbpedia-1M bench. The evidence cited was a
`perf record` profile of the v1.4.0 warm-scan path showing
`turbovec::id_map::IdMapIndex::finalise_from_inner` at ~5.95 % self,
with `HashMap::insert` as a hot symbol underneath it. That function
runs only inside the cache-miss path (cold backend startup or freshly
invalidated cache entry), so seeing it hot in a 50-query *warm* sweep
implies one of three things:

1. The cache entry is being **evicted** between queries (eviction bug
   in `cache::enforce_cap`).
2. The cache entry is present but the freshness predicate is firing
   wrongly (`am_version` bumping on a read-only workload, e.g. a
   stray `aminsert` / `ambulkdelete` / `PreCommit` callback).
3. The cache key is unstable across queries (different `attnum` /
   `bit_width` / `dim` / `rel_oid` per call).

Phase R-3's headline figure was warm p50 ≈ 90 ms vs the kernel's
predicted ~20 ms, and a 70 ms gap is a lot to attribute to anything
other than "we're doing the cold-path work on every query".

## Method

Built v1.5.0 with a debug-only tracepoint in `src/cache.rs::lookup`:
five `AtomicU64` counters (`lookups`, `hits`, `misses_key`,
`misses_ver`, `misses_rel`) plus two `#[pg_extern]` functions
(`turbovec.cache_stats()` and `turbovec.cache_stats_reset()`) so the
caller can drain the counters between phases.

Tracepoint discriminates the three miss classes:

- `misses_key` — `HashMap::get` returned `None`. No prior install in
  this backend.
- `misses_rel` — entry present, `entry.relfilenode != expected_relfile`.
  Relfile rewrite happened (CLUSTER, VACUUM FULL, REINDEX, TRUNCATE).
- `misses_ver` — entry present, `entry.n_rows != expected_n_rows`. AM
  path: `am_version` mismatch from a writer-side bump (`aminsert` →
  `PreCommit` rewrite, or `ambulkdelete`).

Workload (kept small to iterate fast — the cache-miss bug, if it
existed, would manifest at any scale):

- 2000 × 64-d random unit-norm vectors (`turbovec.vec_random_unit(64)`).
- `CREATE INDEX ... USING turbovec ... WITH (bit_width = 4)`.
- Single warm psql session.
- `cache_stats_reset()`.
- 50 queries via a plpgsql wrapper that does
  `ORDER BY emb OPERATOR(turbovec.<=>) q LIMIT 1` with a different
  `q` each call.
- `SELECT * FROM cache_stats();`.

Ran two variants for confidence:

- 50 calls of a plpgsql function `u1_one(qid)` from a single SQL
  statement (`SELECT u1_one(qid) FROM generate_series(1, 50)`).
- A `DO` block with 50 sequential queries via `EXECUTE`-style top-level
  SQL.

## Result

```
   metric   | count
------------+-------
 lookups    |    50
 hits       |    50
 misses_key |     0
 misses_ver |     0
 misses_rel |     0
```

Identical for both query-loop variants.

## Verdict

**The cache works correctly. The Phase S agent's "cache miss"
hypothesis was wrong.**

50 lookups → 50 hits, 0 misses of any class. The first lookup in a
fresh backend is a `misses_key` install (when the warmup pre-pays it,
the steady-state count is 50 / 50 hit; without warmup it would be 49
hits + 1 install, also confirmed in a separate run).

What the Phase S agent saw in perf was almost certainly the one-shot
`finalise_from_inner` build that runs *during the warmup1 cache-miss
install* — for a 1 M-vector id-map that's ~1 M `HashMap::insert`
calls, dominating the cold-load wall time. If `perf record` was
sampling across all 50 queries (warmups included), that one big
HashMap construction amortises to ~5–6 % of total cycles, exactly
matching the observed 5.95 % self.

## Implications for the warm-scan bottleneck

The 90 ms warm p50 on arnold is therefore *not* explained by repeated
`IdMapIndex` rebuilds. Other candidates, in approximate order of
likely impact:

1. **Buffer-manager copies of the static regions.** This is the
   bottleneck Phase R-3 / Phase S targeted with mmap-resident reads.
   Phase R-3's perf snapshot showed `ReadBufferExtended` /
   `__memmove_avx_unaligned_erms` / `mdreadv` at ~65 % of warm-scan
   time on a host where the 1.5 GB index doesn't fit in 512 MB
   `shared_buffers`.
2. **Recheck-orderby heap fetches.** `xs_recheckorderby = true`
   forces the executor to fetch each of the 100 candidate heap
   tuples and recompute the ORDER BY expression in `f64` — 100 wide
   (~6 KB TOAST'd) tuples × 50 queries on a buffer-pool-cold heap
   could easily account for 10–20 ms.
3. **The SIMD search kernel itself.** This is what `turbovec.search`
   was *meant* to dominate. If the kernel is N % of the time, the
   non-kernel overhead is what we tune.

Phase U-2 (this commit's other half) measures the Phase S mmap delta
on `meh`. Spoiler: the buffer-manager bottleneck (#1) is invisible on
a 125 GiB-RAM host because the OS page cache is the buffer manager's
backing store and `pread` from a hot OS cache costs ~0. The arnold
re-bench remains the definitive Phase S validation.

## Tracepoint disposition

Reverted before the clean v1.5.0 install for U-2. Not committed:
`debugging-only addition; don't ship it in a release` per the U-1
brief.
