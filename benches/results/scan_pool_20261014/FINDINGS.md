# pg_turbovec 2.13.0 qualification: bounded scan pool vs 2.12.1 (x86, i7i.metal-24xl)

Everything below was measured on this host unless it is explicitly labelled an estimate.
The raw files are under `raw/q/`, the scripts that ran are under `scripts/`, and the
per-cell summaries are `raw/q/warm/*.summary.json` plus per-query `*.jsonl`.

## Setup

- **RUN `q2130-202610142349`.** kiwi account 830118047484, us-east-2c, instance `i-06a1442cbdad13f33`,
  i7i.metal-24xl **on-demand**, launched 2026-10-14T23:49:57Z. AMI ami-06b91dc29c29304cb (Fedora 44, kernel
  7.2.10-200.fc44). Tags: Owner=gburd, Purpose=pgtv-2.13.0-qualify, Expires=2026-10-16, run=RUN. SG
  `sg-008270f5ab6b30982` allows tcp/22 from 73.4.58.126/32 only. Launched with
  `--instance-initiated-shutdown-behavior terminate`, and user-data ran `shutdown -h +480` (confirmed scheduled for
  07:50:35 UTC).
- **Host** (`raw/q/host.txt`): Xeon Platinum 8559C (Emerald Rapids), 96 vCPU = 48 cores × SMT2, 1 socket,
  1 NUMA node, 755 GiB RAM. avx512bw, avx512vbmi, avx512_vnni and amx_tile are present. Storage: 6 × 3.4 TB
  instance-store NVMe in mdadm RAID0 → XFS at /data. Governor `performance`, `idle-set -D 2`, swap off, THP at
  the AMI default (madvise). P0's `setup_host.sh` minus its hard-coded hugepage count.
- **PostgreSQL 18.6**, built with P0's configure line (non-cassert, `-O2 -g -fno-omit-frame-pointer`).
  `postgresql.conf` = P0's `postgresql.bench.conf` (shared_buffers 128GB, huge_pages on,
  max_parallel_workers_per_gather 0, jit off). `vm.nr_hugepages` = 67,496, taken from
  `postgres -C shared_memory_size_in_huge_pages` (66,984) plus 512.
- **No CPU pinning anywhere.** The postmaster's affinity is `0-95` (recorded at every swap in
  `raw/q/warm/swaps.txt` and `raw/q/pgb/swaps.txt`). pgbench ran on the same host, also unpinned.
  `RAYON_NUM_THREADS` and `GLIBC_TUNABLES` were never set: every swap checked `/proc/<postmaster>/environ`
  and recorded "no RAYON*/GLIBC*".
- **Two .so files**, same PG, rustc 1.98.0 (LLVM 22.1.8), cargo-pgrx 0.19.3,
  `cargo pgrx install --release --no-default-features --features pg18` (`raw/q/versions.txt`):
  - **OLD** = tag v2.12.1 (95d0fdd), turbovec pin 29de559. md5 `66fcaeee66dd1483c590303bf94e4fd6`.
  - **NEW** = branch `fix/scan-concurrency-2.12.2` at **8cf1642** (the lead's review-fix head, which pins turbovec
    7873157). It is not b85b252: the lead's commit landed before the clone. md5
    `9a157a6ce35fda2cfffcaf903807156b`.
  - **Arm swap** (`scripts/swap.sh`): `pg_ctl stop -m fast`, copy the .so over `pg_turbovec.so`, start, then
    `pg_prewarm` of docs, docs_tv_flat, docs_ivf, docs_ivf_tv and queries. `turbovec_version()` reported 2.12.1 /
    2.13.0 on every swap, and `turbovec.search_concurrency` was present under OLD and absent under NEW. The catalog
    `extversion` stayed 2.12.1 for both arms (no `ALTER EXTENSION UPDATE`; the 2.12.1→2.13.0 script is
    comment-only).
- **Data.** CO1024 1M from P0's `prep_co1024.py`. The 11 shard md5s and all six `.npy` md5s are **identical to P0**
  (`raw/datasets.lock.json`), so it is the same 1,000,000 corpus vectors, the same 5,200 held-out queries and the
  same exact k=100 GT. Two LOGGED, STORAGE PLAIN `turbovec.vector` tables were loaded: `docs` (flat index) and
  `docs_ivf` (IVF index), one vector index per table per P0's harness finding. 16-stream COPY took 9.9 s per
  table. Checksums: count 1,000,000, sum(id) 499,999,500,000, and component sums equal to ~1e-15 relative
  (`raw/q/load_checksums.txt`).
- **Indexes were built once, under OLD** (`turbovec.build_parallelism = 32`):
  - flat 4-bit `docs_tv_flat`: 8.9 s, 559,947,776 B.
  - IVF lists=1000 `docs_ivf_tv`: 28.7 s, 564,076,544 B.

  Both sizes are byte-equal to P0's. After a CHECKPOINT, the sha256 of each index relfile was taken after the
  build and again after each of the 8 measured arm-passes. All 9 snapshots agree: flat `4ed3adbd…53ad63`, IVF
  `57ef8893…edd66cd` (`raw/q/index_sha.txt`). Neither arm changed an index byte.

## Method (single client)

`scripts/run_rounds.sh` and `scripts/bench_q.py`. bench_q.py is P0's `bench.py` plus a few counters, all
collected outside each query's timing window: max live thread count, thread names, the first warm-up query's time,
and counts of queries > 60 / > 100 ms.

- **Rounds.** Rounds 0 (discarded), 1, 2 and 3. Each round runs OLD then NEW, and each arm-pass is a swap,
  restart and prewarm.
- **Cells.** Every cell is a fresh connection: 50 warm-up queries (qids 0–49; the first one builds the per-backend
  cache) followed by **1,000 timed queries** (qids 50–1049), prepared, over a unix socket, timed as client wall
  time.
- **Reported values.**
  - p50, p95, p99 and CPU: the median over rounds 1–3 of each round's value. The p50 min–max across rounds is
    shown in parentheses.
  - CPU-ms/query: the `/proc/<pid>/stat` thread-group utime+stime delta over the 1,000 queries (as P0).
  - threads: the maximum of `/proc/<pid>/task` sampled after every timed query.
  - Slow-query counts: summed over the three rounds (3,000 queries).
- No order violations or short results in any cell. The server logs had 0 ERROR/WARNING/FATAL/PANIC lines,
  checked after every arm-pass.

### a. Flat 4-bit, `hi_dim_rerank = off`, by `search_k`

| search_k | arm (scan_parallelism) | R@10 | p50 ms (min–max) | p95 | p99 | CPU-ms/q | threads |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | OLD | 0.9980 | 1.15 (1.14–1.16) | 1.32 | 1.43 | 62.0 | 97 |
| 16 | NEW default (=4) | 0.9980 | 1.85 (1.79–1.91) | 1.88 | 1.90 | 5.2 | 5 |
| 16 | NEW sp=1 | 0.9980 | 5.16 (5.06–5.49) | 5.19 | 5.22 | 4.9 | 2 |
| 16 | NEW sp=88 | 0.9980 | 1.03 (1.01–1.05) | 1.20 | 1.27 | 46.1 | 89 |
| 16 | NEW sp=96 | 0.9980 | 1.10 (1.10–1.11) | 1.29 | 1.38 | 51.2 | 97 |
| **32** | **OLD** | 1.0000 | **1.20** (1.20–1.21) | 1.28 | 3.44 | **45.9** | **97** |
| 32 | OLD with sp=1 (no effect in 2.12.1) | 1.0000 | 1.16 (1.16–1.17) | 1.25 | 3.42 | 38.2 | 97 |
| **32** | **NEW default (=4)** | 1.0000 | **2.06** (1.99–2.08) | 2.19 | 3.61 | **5.7** | **5** |
| **32** | **NEW sp=1** | 1.0000 | **5.26** (5.20–5.61) | 5.34 | 9.93 | **5.0** | **2** |
| **32** | **NEW sp=88** | 1.0000 | **1.07** (1.06–1.07) | 1.16 | 3.01 | **32.9** | **89** |
| 32 | NEW sp=96 | 1.0000 | 1.12 (1.11–1.12) | 1.20 | 3.35 | 37.5 | 97 |
| 128 | OLD | 1.0000 | 2.04 (2.03–2.05) | 2.25 | 8.82 | 87.8 | 97 |
| 128 | NEW default | 1.0000 | 2.58 (2.58–2.66) | 2.77 | 4.81 | 6.5 | 5 |
| 128 | NEW sp=1 | 1.0000 | 5.93 (5.85–6.27) | 6.01 | 10.85 | 5.7 | 2 |
| 128 | NEW sp=88 | 1.0000 | 1.81 (1.80–1.83) | 1.96 | 8.91 | 60.7 | 89 |
| 128 | NEW sp=96 | 1.0000 | 1.95 (1.94–1.95) | 2.11 | 8.66 | 71.8 | 97 |
| 1024 | OLD | 1.0000 | 7.22 (7.20–7.24) | 7.53 | 7.67 | 109.8 | 97 |
| 1024 | NEW default | 1.0000 | 8.00 (7.85–8.08) | 8.43 | 8.66 | 13.1 | 5 |
| 1024 | NEW sp=1 | 1.0000 | 12.02 (11.93–12.40) | 12.53 | 12.66 | 11.8 | 2 |
| 1024 | NEW sp=88 | 1.0000 | 6.92 (6.92–6.93) | 7.23 | 7.41 | 76.0 | 89 |
| 1024 | NEW sp=96 | 1.0000 | 7.13 (7.10–7.13) | 7.40 | 7.56 | 87.6 | 97 |

- **Threads.** NEW's count is exactly 1 + scan_parallelism (names `turbovec-scan-N`). OLD is 1 + 96 (rayon's global
  pool) at every setting, including `scan_parallelism = 1`, which confirms that 2.12.1 did not bound flat scans.
- **NEW default vs OLD.** CPU per query is 7.9–12× lower. p50 is higher by 0.5–0.9 ms (+26% to +61% at
  search_k ≤ 128, +11% at 1024).
- **sp=88 and sp=96.** Both beat OLD on p50 and on CPU at every search_k: a bounded pool sized to the cores is not
  slower than the global pool it replaces.
- **sp=1 vs P0.** sp=1 matches P0's `RAYON_NUM_THREADS=1` rows (5.33 ms / 5.1 CPU-ms at search_k 32).
- **Flat p99.** The p99 bump at search_k 32 and 128 comes from the same 12 (search_k 32) or 20 (search_k 128) qids
  in every round and in both arms. It is query-dependent, not scheduling.
- **Memory.** Pss_Anon per backend: OLD 1,114–1,129 MB, NEW default 995–1,010 MB, NEW sp=1 992–1,000 MB.

### b. Flat, all GUC defaults (search_k 32, hi_dim_rerank auto, scan_parallelism 0)

| arm | R@10 | p50 ms (min–max) | p95 | p99 | CPU-ms/q | threads | Pss_Anon MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD | 1.0000 | 7.23 (7.23–7.24) | 7.54 | 7.69 | 112.8 | 97 | 1,129 |
| NEW | 1.0000 | 7.99 (7.86–8.09) | 8.43 | 8.63 | 13.0 | 5 | 1,010 |

At defaults, NEW's p50 is +0.76 ms (+10.5%) and its CPU per query is 8.7× lower.

### c. IVF lists=1000, `out_of_core = on` (the ~100 ms tail)

| probes (sp) | arm | R@10 | p50 ms (min–max) | p95 | p99 | CPU-ms/q | threads | queries > 60 ms / > 100 ms (of 3,000) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 (default 4) | OLD | 0.9248 | 16.52 (16.30–17.70) | 103.44 | 105.07 | 62.6 | 5 | **910 / 909** |
| 16 (default 4) | NEW | 0.9248 | 15.86 (15.86–15.87) | 18.34 | 18.96 | 38.5 | 5 | **0 / 0** |
| 64 (default 4) | OLD | 0.9826 | 48.30 (48.07–49.14) | 120.11 | 122.23 | 131.8 | 5 | **418 / 418** |
| 64 (default 4) | NEW | 0.9826 | 47.65 (47.64–47.76) | 53.16 | 54.68 | 121.2 | 5 | **0 / 0** |
| 16 (sp=8) | OLD | 0.9248 | 98.95 (98.92–99.02) | 100.62 | 101.28 | 113.2 | 9 | 1,905 / 365 |
| 16 (sp=8) | NEW | 0.9248 | 12.47 (12.46–12.47) | 13.88 | 14.36 | 40.2 | 9 | 0 / 0 |

- **The tail is gone on NEW** at every setting: 0 of 3,000 queries over 60 ms in each NEW cell.
- **OLD still shows it.** At probes 16, 30% of queries land at ~103 ms. The discarded round 0 showed the same:
  OLD 343/1,000, NEW 0/1,000.
- **On OLD the slow set changes between rounds.** Rounds 0 and 1 share only 151 of their 343 and 423 slow qids,
  which fits the codebook-memo race described in the CHANGELOG.
- **More threads now help OOC.** With the fix, sp=8 is faster than sp=4 (12.5 vs 15.9 ms). On OLD, sp=8 was the
  worst case at 99 ms.

### d. IVF whole-load (`out_of_core = off`, `cache_size_mb = 2048`), probes 16

| arm | R@10 | p50 ms (min–max) | p95 | p99 | CPU-ms/q | threads | > 60 / > 100 ms (of 3,000) | Pss_Anon MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD | 0.9248 | 34.70 (34.69–34.87) | 115.45 | 137.29 | 85.1 | 97 | 760 / 229 | 1,118 |
| NEW | 0.9248 | 35.47 (35.11–35.60) | 117.15 | 138.97 | 46.5 | 5 | 771 / 228 | 998 |

- **Threads are bounded on NEW:** 5, down from 97. CPU per query drops 45%.
- **The whole-load path has its own tail, unchanged by 2.13.0.** About 25% of queries take > 60 ms on both arms.
  - It is the same queries: per-qid latency correlates 1.000 between OLD and NEW, and 250–253 qids are slow in
    all three rounds.
  - It is data-dependent, not the pool: the same qids run in a median 17.3 ms on the NEW OOC path, against
    15.5 ms for the others.
  - **Root-caused** (`raw/wholeload_tail/`): the classic masked kernel keeps a `k`-entry top-k heap with an
    O(k) `rescan_min` per candidate (`avx512_post_flush_heap_update` = 52% of the slow queries' CPU), and a
    whole-load IVF scan runs it over the full 1M-row index at `k` = the `hi_dim_rerank` widened candidate count
    (several hundred), so the cost is O(candidates × k). `hi_dim_rerank = off` cuts the slow-set median
    67.5 → 13.8 ms. The out-of-core path runs the same heap over a few-thousand-row compact index, so it is
    cheap. Filed as [#7](https://github.com/gburd/pg_turbovec/issues/7); a separate per-query
    `materialize_rotation_matrix` rebuild (~5–10% here) is [#8](https://github.com/gburd/pg_turbovec/issues/8).
    Neither is a 2.13.0 regression, and the IVF serving mode at scale is out-of-core, which has no such tail.

### First query on a connection (per-backend cache cold-open from shared_buffers)

This is the first of the 50 warm-up queries in each cell, median of rounds 1–3.

| path | OLD | NEW default | NEW sp=1 | NEW sp=88 |
|---|---:|---:|---:|---:|
| flat | 426–440 ms | 511–525 ms | 800–820 ms | 413–430 ms |
| OOC IVF | 128–167 ms | 132–168 ms | | |
| whole-load IVF | 435 ms | 522 ms | | |

The cold-open repack now runs on the bounded pool. That is the design, and it costs ~85 ms per new connection at
the default and ~380 ms at sp=1. This matters for workloads that churn connections; no churn cell was run.

## Multi-client sanity (pgbench)

`scripts/run_pgb.sh` and `scripts/pgb_summ.py`, with P0's `:qv` script form.

- **Workload.** Flat `docs`, all GUC defaults. Closed loop, `-M prepared`, 16 and 96 clients (`-j` = min(clients,
  32)), `-T 75`.
- **Window.** The first 15 s is warm-up: every backend cold-opens its cache. The measured window is [15, 75) s,
  taken from the 1 s aggregate logs.
- **Server CPU.** utime+stime delta of all postmaster children over that window, read from `/proc`. Host
  CPU-busy comes from `/proc/stat` and includes pgbench itself.
- **Arm order.** Two rounds, each OLD → NEW → NEW with `PGOPTIONS='-c turbovec.scan_parallelism=1'`.

| arm | clients | TPS r1 / r2 | mean latency ms | host CPU busy % | server CPU cores | server CPU-ms / txn | threads / backend (mid-run) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD | 16 | **869 / 866** | 18.4 | 91.6 / 91.8 | 88.3 | 101.6 / 102.1 | 97 |
| NEW default | 16 | **1,118 / 1,114** | 14.3 | 35.7 / 35.7 | 34.1 | 30.6 | 5 |
| NEW sp=1 | 16 | 727 / 727 | 22.0 | 16.7 | 15.9 | 21.9 | 2 |
| OLD | 96 | **1,090 / 1,092** | 88.0 | 99.8 / 99.9 | 95.3 | 87.4 / 87.3 | 97 |
| NEW default | 96 | **1,432 / 1,430** | 67.1 | 99.3 / 99.2 | 94.6 | 66.1 | 5 |
| NEW sp=1 | 96 | 1,557 / 1,558 | 61.6 | 98.7 / 98.6 | 94.1 | 60.4 | 2 |

- **TPS.** NEW at defaults gives +29% at 16 clients and +31% at 96 clients.
- **CPU at 16 clients.** OLD saturates the host (91.6% busy, ~88 cores) for 869 TPS. NEW does 1,118 TPS on 34
  cores and leaves 64% idle (mpstat: OLD usr 82.2 / sys 8.6, NEW usr 35.1 / sys 0.4).
- **sp=1.** Fastest at 96 clients (1,557 TPS, +43% over OLD), but slower than OLD at 16 clients: one thread per
  query cannot use the idle cores.
- **Thread column.** The thread counts come from the +15 s snapshot. The +75 s snapshot lands as pgbench
  disconnects and the backends are exiting, so it reads 1–2 threads. `raw/q/pgb_table.md` prints that end-of-run
  value. Use this table's column.
- **Not investigated.** Under 16-way load, NEW uses 30.6 server CPU-ms per transaction, against 13.0 CPU-ms per
  query single-client.

## What this does and does not show

- **Shown, at 1M × 1024-d (CO1024) on one AVX-512 VBMI+VNNI x86 host:**
  - 2.13.0 bounds threads for flat and whole-load IVF scans, and `scan_parallelism` now takes effect on them.
  - The out-of-core IVF ~100 ms tail is gone.
  - Recall is identical to 4 decimals per cell across arms.
  - Index bytes are untouched.
  - Closed-loop TPS improves at 16 and 96 clients.
- **Not shown:**
  - ColBERT, 1-bit, `turbovec.knn()`, the graph kind.
  - Graviton / aarch64.
  - Other corpus sizes or dimensions.
  - Cold reads (no drop_caches cell), connection-churn workloads, filtered queries.
  - pgbench on IVF.
  - Pinned or NUMA-split configurations.
- **Returned ids** were compared to GT (recall), not byte-compared OLD vs NEW.
- **Planes path.** The index is ≥ 32,768 rows and the host has VBMI+VNNI, so the turbovec 1.1 "planes" scan path
  is expected. It was not separately confirmed.
- **Whole-load tail.** It is real and unchanged by this release (§ d). It still needs a root cause before IVF
  whole-load latencies are published.

## Step 6: `cargo pgrx test pg16` on the branch head (appended by the collection step)

Run on this host: fresh clone of `fix/scan-concurrency-2.12.2`, debug profile, pgrx-managed PG 16.15 (`cargo pgrx init --pg16 download`), started after every measurement had finished. Poll status: `DONE rc=0`. Full log: `raw/q/pgtest_pg16.txt`.

```
8cf1642c09cbb6a8ca02119578ff3b9f5a200d63
2026-10-14 20:01:23 -0400 scan: review fixes -- cfg(test) GUC read, turbovec carry #6 test handshake (7873157), docs
47:turbovec = { git = "https://github.com/gburd/turbovec.git", rev = "78731577c79b18f264b4604381fe7451acb5c20d", features = ["mask-skip-counter"] }
96
test result: ok. 475 passed; 0 failed; 9 ignored; 0 measured; 0 filtered out; finished in 150.14s
test result: ok. 0 passed; 0 failed; 22 ignored; 0 measured; 0 filtered out; finished in 0.00s
EXIT 0
```

Benchmark server logs (all arm-passes + pgbench): 0 ERROR/WARNING/FATAL/PANIC lines (`raw/q/pglog/`).

## Teardown

All artefacts were pulled first (`raw/q/`, including hb/, host/ and pglog/). Then `.agent/notes/q2130_teardown.sh` ran (selects by tag run=q2130-202610142349 only), finishing 2026-10-15T01:27:50Z. Instance wall time 1.63 h on-demand ≈ $15 at the $9.06/h list price P0 used (an estimate; the pricing API is denied). Output, ending with the by-tag verification (the instance may still be listed as `terminated`; volumes, SG and key must be empty):
```
TERMINATINGINSTANCES	i-06a1442cbdad13f33
CURRENTSTATE	32	shutting-down
PREVIOUSSTATE	16	running
{
    "Return": true,
    "GroupId": "sg-008270f5ab6b30982"
}
{
    "Return": true,
    "KeyPairId": "key-003057d7f1f43eaaa"
}
--- verify (instances must all be terminated; volumes/SG/key empty):
i-06a1442cbdad13f33	terminated
```
