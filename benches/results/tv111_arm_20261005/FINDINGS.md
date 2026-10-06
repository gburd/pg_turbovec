# turbovec 1.1.1 adoption — qualification on Graviton4 (2026-10-05)

pg_turbovec **v2.11.0** = v2.10.3 + upstream turbovec **1.1.1** (staged 2/4-bit
"planes" search) + our fork carries #1–#3 cherry-picked unchanged + **new carry
#4** (parallel planes cold-open repack). This directory is the evidence.

Host: EC2 `c8gd.8xlarge`, **Graviton4 (Neoverse-V2: dotprod, i8mm, sve2)**, 32
vCPU, 61 GiB, local NVMe, **Debian 13 arm64**, account hotdog (170848442262),
us-east-2. PostgreSQL **16.15 built `-O2` without cassert** (pgrx's default
download is `--enable-cassert`; a first pass on it is kept in `raw/*_cassert.jsonl`
and shows the same direction with ~25% higher absolute numbers). Second box
`c8gd.4xlarge` (same CPU family) ran the full `cargo pgrx test`. Both
terminated; see `INSTANCE.md`.

Why aarch64: turbovec 1.1's staged search engages only on aarch64 (dotprod) and
on x86 with AVX-512 VBMI+VNNI. **arnold (i9-12900H) and meh are AVX2/AVX-only
and take the unchanged whole-index scan** — on those hosts 2.11.0 searches
exactly like 2.10.3.

Corpus: CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (en), first
**1,000,000 × 1024-d** rows (L2-normalized, cosine), 1,000 held-out queries
(rows 1,000,000–1,000,999, not indexed), exact top-100 GT by numpy dot. Same
corpus family as `rebench_20260925`.

A/B method: ONE index (bytes identical for both arms — see §1), the
`pg_turbovec.so` swapped between arms with a postmaster restart + page-cache
drop, arms **alternated** old → new → new+planes_off, **3 rounds**, medians
reported. 200 queries per arm, 10 discarded warm-ups, one psql session per arm,
latency = top-level `EXPLAIN (ANALYZE)` **Execution Time** (whole query,
including the exact heap recheck), query vectors inlined as literals,
`oversample=1.0`, `hi_dim_rerank=off`.

## 1. Wire format: byte-identical (the safety claim, proven on the real host)

Same heap, `CREATE INDEX … WITH (bit_width=4)` once with v2.10.3 and once with
v2.11.0; sha256 over the meta page (minus the `am_version` counter) and the
codes/scales/ids chain payloads, read via `pageinspect.get_raw_page`:

```
v2.10.3  docs_tv_old n=1000000 stride=512 chains_sha256=7b1ade79112f1595 meta_sha256=0cf27b940851e4ee
v2.11.0  docs_tv     n=1000000 stride=512 chains_sha256=7b1ade79112f1595 meta_sha256=0cf27b940851e4ee
```

Identical (`raw/hash_old.txt`, `raw/hash_new.txt`; re-confirmed after moving to
the non-assert server). Wire stays **v8**; no REINDEX. Consistent with
turbovec's own encode goldens, whose codebook/calibration/codes/scales columns
are unchanged 1.0.0 → 1.1.1 (only its file-container column moved, which we
never use).

## 2. Warm end-to-end latency (what a user sees)

p50 ms, flat index, 1M × 1024-d, whole query incl. recheck. Recall@10 vs exact GT.

| bit_width | search_k | v2.10.3 | **v2.11.0** | speedup | v2.11.0 planes off | p95 old → new | R@10 old / new |
|---|---|---|---|---|---|---|---|
| 4 | 32 | 7.88 | **6.72** | **1.17×** | 7.70 | 10.6 → 9.4 | 1.000 / 1.000 |
| 4 | 100 | 10.88 | **9.52** | **1.14×** | 10.44 | 15.4 → 14.3 | 1.000 / 1.000 |
| 4 | 256 | 18.32 | **15.96** | **1.15×** | 17.64 | 26.1 → 24.0 | 1.000 / 1.000 |
| 4 | 1024 | 62.17 | **54.57** | **1.14×** | 59.36 | 94.4 → 87.4 | 1.000 / 1.000 |
| 2 | 32 | 7.22 | **6.86** | 1.05× | 7.07 | 9.9 → 9.6 | 0.993 / 0.993 |
| 2 | 100 | 10.45 | **9.78** | 1.07× | 10.10 | 14.7 → 14.2 | 1.000 / 1.000 |
| 2 | 256 | 17.81 | **16.51** | 1.08× | 16.98 | 25.5 → 24.0 | 1.000 / 1.000 |
| 2 | 1024 | 61.67 | **55.11** | 1.12× | 60.00 | 93.3 → 87.3 | 1.000 / 1.000 |

**Recall is unchanged at every cell** (the exact heap recheck re-ranks the
candidate set; see §4 for the candidate-set agreement). The "planes off" column
(same v2.11.0 binary, `TURBOVEC_{4,2}BIT_PLANES=0`) isolates the staged search:
most of the gain is the staged route, a small remainder is 1.1's other kernel
work.

Index Scan node startup time (`raw/kernel.jsonl`, `LIMIT 1`) tracks end-to-end
almost exactly (4-bit 7.72 → 6.58 ms at k=32, 62.13 → 54.26 at k=1024). It is
NOT kernel-isolated: pg_turbovec advertises a `-inf` order-by lower bound, so
the executor's reorder queue drains every candidate (heap fetch + exact
recheck) before it emits the first row even under `LIMIT 1`.

### Why end-to-end is ~1.15×, not upstream's "1.87×"

Upstream measures the turbovec kernel alone. The pure kernel on this host, same
1M × 1024-d real corpus, no PostgreSQL (`raw/kbench.jsonl`, `kbench/`):

| bits | threads | 1.0.0 k=10 | **1.1.1 k=10** | 1.0.0 k=100 | **1.1.1 k=100** | 1.0.0 k=1024 | **1.1.1 k=1024** |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 22.67 | **6.57 (3.45×)** | 23.16 | **6.92 (3.35×)** | 30.89 | **9.88 (3.13×)** |
| 4 | 32 | 1.84 | **0.62 (2.97×)** | 2.32 | **0.76 (3.06×)** | 8.29 | **1.28 (6.48×)** |
| 2 | 1 | 12.46 | **6.59 (1.89×)** | 12.96 | **7.24 (1.79×)** | 20.64 | **10.35 (1.99×)** |
| 2 | 32 | 0.94 | **0.58 (1.62×)** | 1.40 | **0.70 (1.99×)** | 7.74 | **1.18 (6.56×)** |

**The kernel really is 3–6.5× faster.** A PostgreSQL backend runs the search
on rayon's global pool (32 threads here), so the 32-thread row is the
apples-to-apples one, and the **end-to-end saving equals the kernel saving**:
4-bit k=1024 saves 7.6 ms end-to-end vs 7.0 ms in the kernel; k=32 saves
1.16 ms vs 1.36 ms. What remains of a query is backend-side and unchanged by
this release: per-query overhead plus, per candidate, the heap fetch + vector
deserialize + exact recheck. A straight-line fit of §2 from k=32 to k=1024
gives about 4.5 ms + ~48 µs per candidate.

**Correction (2026-10-06):** an earlier version of this paragraph presented
that ~48 µs slope as the cost of PostgreSQL's per-candidate recheck. The slope
is real wall-clock marginal cost on this host at 1M rows, but the attribution
was wrong. A real-query attribution
([`../recheck_20261006/FINDINGS.md`](../recheck_20261006/FINDINGS.md))
measured ~17–18 µs per candidate at 1024-d on x86: ~1–2 µs PostgreSQL core
(heap fetch, reorder queue, executor), the rest pg_turbovec's own code (two
serde-CBOR decodes ~8 µs, TOAST fetch ~5 µs, scalar distance kernel ~3 µs).
The per-candidate cost was not measured on this Graviton4 host. With the scan
now ~1 ms here, the per-candidate recheck dominates a flat-index query at
large `search_k`; in the x86 AVX2 run (200k rows, `search_k = 1024`) the scan
was still roughly half of the query. The 48 µs here and the 17–18 µs there are
different measurements (wall-clock at 1M rows vs backend CPU at 200k rows); a
1M-row per-candidate breakdown is part of the follow-up benchmark.
The next lever is fewer candidates at equal recall or a cheaper recheck, not a
further kernel speedup.

## 3. Cold-backend latency (connection-pool reality)

Fresh backend per query (per-backend cache cold → relfile read + search-cache
build repaid), OS page cache warm, 20 queries × 3 rounds, p50 ms
(`raw/cold.jsonl`, `raw/cold_bw2.jsonl`):

| index | v2.10.3 | **v2.11.0** | v2.11.0 planes off |
|---|---|---|---|
| 4-bit flat, 1M × 1024-d | 513.7 | **476.9** (−7%) | 512.5 |
| 2-bit flat, 1M × 1024-d | 327.0 | **319.8** (−2%) | 326.1 |

**This needed fork carry #4.** Stock turbovec 1.1.1 builds the planes cache
with a *serial* `planes_repack`, bypassing our carry #3 (the parallel repack
that gave v2.10.3 its 3.1× cold-scan cut). Measured in isolation on this host,
1M × 1024-d (`raw/hostinfo_and_repack_probe.txt`):

| | 4-bit | 2-bit |
|---|---|---|
| carry #3 `repack` (classic layout, parallel) | 53 ms | 31 ms |
| upstream `planes_repack` (serial) | **279 ms** | **1,092 ms** |
| carry #4 `planes_repack` (parallel, in-place) | **18–21 ms** | **32 ms** |

Without carry #4 a 2-bit cold backend would have paid ~1 s extra and the
v2.10.3 cold-scan win would have regressed. A first cut of carry #4
(concatenating per-task buffers serially) still regressed 4-bit cold by +15%
(508 → 587 ms, `raw/cold_carry4v1.jsonl`); the shipped version writes each
task's output in place, as carry #3 does. Both are byte-identical to the serial
upstream body, pinned by `parallel_planes_repack_is_byte_identical_to_serial`
(2/4-bit, sub/above threshold, tail padding, partial last task; x86_64 and
aarch64).

## 4. Candidate-set agreement (the behaviour change that makes this a minor)

The staged search returns **exact scores** but an **approximate id set**: a
vector whose sign bits alone rank it outside the shortlist is not seen. Measured
on this corpus, pure kernel, staged vs whole-index scan, 200 queries
(`kbench/`, `ids_*` dumps):

| bits | threads | k | queries with identical id set | mean id overlap |
|---|---|---|---|---|
| 4 | 1 | 10 | 98.5% | 99.85% |
| 4 | 32 | 10 | 100.0% | 100.00% |
| 4 | 1 | 100 | 73.5% | 99.64% |
| 4 | 32 | 100 | 82.0% | 99.75% |
| 2 | 1 | 10 | 96.5% | 99.60% |
| 2 | 32 | 10 | 100.0% | 100.00% |
| 2 | 1 | 100 | 86.5% | 99.81% |
| 2 | 32 | 100 | 99.0% | 99.99% |

(1.0.0 and 1.1.1 whole-index scans agree on 100% of queries.) At k = 100 a
quarter of queries swap one or two of their *100* candidates, typically at the
tail. pg_turbovec feeds `search_k` candidates to the exact recheck, so what
matters is whether the true top-10 survive — **recall@10 is unchanged in every
cell of §2**. Agreement is lower than upstream's 99.92–100% (measured on OpenAI
embeddings); Cohere multilingual-v3 is a different corpus. Operators who need
the whole-index scan's exact candidate set can set
`TURBOVEC_4BIT_PLANES=0` / `TURBOVEC_2BIT_PLANES=0` in the postmaster
environment.

## 5. Corruption gates (AGENTS.md HARD MANDATE #1)

The risk 1.1 introduces: after an `add`/`remove`, turbovec **reconstructs
`packed_codes()` — the bytes we persist — from the planes cache**. A wrong
reconstruction would write silently different codes.

1. **`persist_is_byte_exact_through_planes_layout`** (new `#[pg_test]`): 40k
   rows (past turbovec's 32,768 planes gate), the real
   `from_id_map_parts` → `prepare` → `remove` ×300 → `add_with_ids` ×200 →
   PreCommit flush path, then every row's in-memory AND persisted
   (code row, scale) must equal the pre-mutation bytes or a same-calibration
   fresh encode. Run at 2 and 4 bits. **Verified to actually take the planes
   layout on Graviton4** (probe: `planes=true` at 40k × 64-d for both widths) —
   it is not a classic-layout test passing by default. On x86/AVX2 it exercises
   the classic layout.
2. **Full suite on aarch64**: `cargo pgrx test pg16` — 446 / 0 / 8 before the
   two fix tests (`raw/pgtest_pg16_aarch64.log`); on the release branch see
   `raw/pgtest2_pg16_aarch64.log`. x86_64 local: **448 / 0 / 8**.
3. **Sustained-insert soaks** with a byte-level oracle: see §6 (found and
   fixed two pre-existing bugs; the A/B re-run is clean on v2.11.0).

turbovec's own suite on the fork branch: **523 passed / 0 failed** on Graviton4
(planes tests run, not skipped); 35/35 planes + byte-identity tests under
`qemu-aarch64 -cpu max` locally.

## 6. Soak — found two pre-existing bugs; both fixed; A/B re-run clean

`soak2.py` (final form): flat 4-bit index seeded with 60k real 1024-d vectors
(past the planes gate); 3 writer backends commit COPY batches of 1/16/128 new
rows plus UPDATE churn; the orchestrator `pg_terminate_backend`s a writer every
30–90 s (mid-flush); VACUUM every 2 min; every 60 s `turbovec_check` must be
clean and a scan must return 10 distinct ids. At the end, **byte-level oracle**:
`CREATE INDEX` again on the same heap and compare every entry (root TID →
code row + scale) of the soaked index against it via `pageinspect`.

### Bug 1 — per-scan memory leak (OOM), since v1.8.0

Run 1 (v2.11.0 pre-fix), 81 min, 18,005 commits, 42 kills, 16 VACUUMs,
60k → 933k rows, every `turbovec_check` clean — then the **OOM killer took a
long-lived scanning backend at 52 GB** and the postmaster went through crash
recovery (`raw/soak_run1_oom.log`). Cause: `amendscan` was a no-op, so every
scan leaked its `ScanOpaque`, including an `Arc` to the cached whole index;
after a cache replacement those leaked `Arc`s pin every superseded copy
(`raw/memrepro*.py`, `raw/memrepro_results.txt`):

| binary | scanner RssAnon over 30 commits (200k × 1024-d) |
|---|---|
| v2.10.3 | 265 → **6,083 MB** (+200 MB/commit) |
| v2.11.0 pre-fix | 241 → **6,057 MB** |
| v2.11.0 fixed | 241 → **248 MB** |

Fix: `amendscan` drops the opaque. Test `amendscan_releases_cached_index_handle`
(6 live refs after 5 scans before; 1 after). Run 2 with the fix: 90 min,
19,553 commits, peak backend RSS ~2 GB at 1M rows (tracks index size, no
per-commit growth — `raw/rss.log`).

### Bug 2 — stale `touched_ids` re-splice (silent wrong entries), since v1.29.1

Run 2's byte oracle (`raw/soak_run2_fixed.log`, forensics in
`raw/soak_run2_forensics.txt` via `verify2.py` / `classify.py`) found **0 live
rows missing, but 818 live rows whose entry carried a DIFFERENT live row's
exact bytes and 3,402 stale entries** (pointing at heap-only or unused TIDs).
`turbovec_check` was clean the whole time — ids stay unique. An A/B soak (both
binaries concurrently, same host, same workload, 40 min) proved it
**pre-existing**:

| run | binary | commits | wrong codes | stale entries | missing |
|---|---|---|---|---|---|
| A/B #1 | v2.10.3 | 12,518 | **530** | **2,369** | 0 |
| A/B #1 | v2.11.0 (leak fix only) | 12,545 | **721** | **2,564** | 0 |
| A/B #2 | v2.10.3 | 12,203 | **406** | **2,073** | 0 |
| **A/B #2** | **v2.11.0 (both fixes)** | **12,387** | **0** | **0** | **0** |

Cause: the deferred flush splices only this transaction's upserted ids
(`touched_ids`) onto the current on-disk index, but the list was never cleared
after a successful flush and the cache entry outlives the transaction. A
long-lived writer therefore re-spliced every id it had ever written, from its
stale in-memory copy, on every commit. After VACUUM removes such an id and the
heap reuses its TID for another row, the re-splice overwrites that row's entry
with old codes (wrong codes) or appends the vacuumed entry back (stale entry).
Fix: `clear_dirty` empties `touched_ids`. Test
`flush_does_not_resplice_previous_txn_ids` (two transactions through the real
cache path with a VACUUM-style on-disk removal between them: fails before —
the removed id comes back — passes after).

**A/B #2 is the qualifying run: v2.11.0 matched a fresh CREATE INDEX of its
heap byte-for-byte on all 658,221 entries after 40 min, 12,387 commits, 40
mid-flush kills and 19 VACUUMs, while v2.10.3 on the identical workload beside
it corrupted 406 entries and left 2,073 stale ones.**

## Not claimed

- **No x86 AVX-512 VBMI+VNNI measurement.** The staged path also engages
  there (Ice Lake server+, Zen 4); upstream reports similar kernel gains. Not
  measured here.
- **No change on AVX2-only x86** (arnold, most consumer Intel, Zen 3 and older):
  planes do not engage; 2.11.0 searches exactly as 2.10.3 (§2 "planes off"
  column is that path on ARM).
- **IVF and ColBERT** not re-benchmarked. IVF serves through the same kernel
  (masked search, now staged per 1.1.1's #557 fix) but cell-scoped
  out-of-core chunks are typically below the 32,768-vector planes gate.
- **Indexes under 32,768 rows** are unchanged (upstream's own gate).
