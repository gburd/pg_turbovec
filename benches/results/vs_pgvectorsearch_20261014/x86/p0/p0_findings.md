# P0 smoke findings — pg_turbovec v2.12.1 vs pg_vectorsearch (x86, i7i.metal-24xl)

Plan: `.agent/notes/VS_BENCH_PLAN.md` (rev v2). Worker: P0 bench agent. All numbers below
are MEASURED on this host unless explicitly marked "estimate".

## Run / host
- RUN = `pgvs-202610141901`, account kiwi 830118047484, us-east-2b, instance `i-0d2d9f4fc515b0ef3`,
  launched 2026-10-14 19:03 UTC. Tags Owner/Purpose/Expires=2026-10-17/run. SG `sg-029a1bfb036d5524c`
  tcp/22 from 73.4.58.126/32 only. `--instance-initiated-shutdown-behavior terminate`, user-data
  `shutdown -h +600` (verified: "Shutdown scheduled for Thu 2026-10-15 05:03:56 UTC").
- **Market: ON-DEMAND, not spot.** Spot launch failed in all three AZs with
  `MaxSpotInstanceCountExceeded`: kiwi's quota L-34B43A08 ("All Standard Spot Instance Requests") is
  **32 vCPU**; i7i.metal-24xl is 96. **P1+ all on-demand; spot quota 32 vCPU.** (On-demand quota
  L-1216C47A = 2048 vCPU.) Fallback approved by lead. P0 cost is therefore ~$9.06/h (list estimate,
  pricing API denied) instead of ~$1/h.
- Host: see `remote/host.txt`. Fedora 44 kernel 7.2.10-200.fc44.x86_64, INTEL(R) XEON(R) PLATINUM 8559C
  (Emerald Rapids), 96 vCPU / 48 cores, SMT2, 755 GiB. ISA gate: avx512bw, avx512vbmi, avx512_vnni all present.
  6 × 3.4 TB instance-store NVMe → mdadm RAID0 (512K chunk) → XFS `-K`, /data noatime,nodiratime,logbufs=8 (21 TB).
  cpupower governor performance, idle-set -D 2. swap off; sysctls per plan B.4.
- huge pages: PG18 needs `shared_memory_size_in_huge_pages = 66984` for shared_buffers=128GB with this
  conf (the plan's "(sb + 2 GB)/2 MiB" = 66,560 was too small → first start FATAL "could not map anonymous
  shared memory"). Set `vm.nr_hugepages = 67048`. **Plan B.4 fix: derive nr_hugepages from
  `postgres -C shared_memory_size_in_huge_pages`, not from a formula.**

## Versions (`remote/versions.txt`)
- PostgreSQL **18.6**, `./configure --prefix=/opt/pg18 --with-liburing --with-lz4 --with-zstd --with-openssl
  --with-icu CFLAGS="-O2 -g -fno-omit-frame-pointer"` (non-cassert), gcc 16.2.1.
- pgvector **v0.8.7** (f37c13f), plain `make` with default flags.
- pg_vectorsearch **main 7a9716b** (2026-10-13 "Update README.md"; = 1 commit after tag v0.1.0 36b75d1,
  README-only diff), VERSION 0.2.0-dev. `meson setup build --buildtype=release -Dsimd=full -Dnative=false
  -Dblas=enabled -Dtools=disabled -Dc_args=-I/usr/include/openblas`. **Plan B.8 fix:** on Fedora 44
  `openblas-devel` puts cblas.h in /usr/include/openblas and ships no openblas.pc, so `-Dblas=enabled`
  FAILS ("BLAS required but not found") without `-Dc_args=-I/usr/include/openblas`; with it meson reports
  "BLAS: OpenBLAS (optimized matrix operations enabled)". CREATE EXTENSION warns "pg_vectorsearch 0.2.0-dev
  is a prerelease".
- pg_turbovec **v2.12.1** (95d0fdd), rustc 1.98.0 (LLVM 22.1.8), cargo-pgrx 0.19.3,
  `cargo pgrx install --release --pg-config /opt/pg18/bin/pg_config --no-default-features --features pg18`.
- .so md5: pg_turbovec 2e1bdd7095a62a1bd503cc4aa0931a26, pg_vectorsearch-0.2.0-dev 62eb442fb4a9432fdf799343965635db,
  vector 0e347694b106ae3ad7de18465d60850f.
- postgresql.conf additions: `scripts/postgresql.bench.conf` (plan B.10 verbatim + socket/logging lines).
  Postmaster env: `env | grep -ci GLIBC` = 0 (no GLIBC_TUNABLES).

---

## Q9 — prism refuses UNLOGGED; GUC defaults  ✅ ANSWERED
Evidence: `evidence/q9_prism_unlogged_and_gucs.txt`.
- `CREATE INDEX ON u USING prism (emb vec32_cosine_ops)` on an UNLOGGED table →
  `ERROR:  prism indexes do not support unlogged tables` / `HINT:  Use a logged table, or run ALTER TABLE
  ... SET LOGGED before creating the index.` (README quote).
- `USING prism (emb vector_cosine_ops)` → `ERROR: operator class "vector_cosine_ops" does not exist for
  access method "prism"` (confirms the hard failure noted in plan §0).
- Defaults (pg_settings boot_val): `prism.query_limit = 0`, `prism.leaf_refine_threshold = 0`,
  `prism.recent_buffers = on`, `prism.nprobe = 0`, `prism.rerank = on`, `prism.rerank_pool = 0`,
  `prism.fastscan_bits = 16`, `prism.centroid_beam_scale = 0.5`, `prism.probe_expand = 2`,
  `prism.distance_mode = default`, `prism.centroid_error_scale = 0`, `prism.log_build_stats = off` (12 GUCs).
  Note: plan §0 says build reloption `distance_mode=asymmetric`; the GUC's value is `default`.
- turbovec GUC defaults (20) also dumped; match plan §0 (`cache_size_mb=256`, `search_k=32`,
  `hi_dim_rerank=auto`, `out_of_core=auto`, `max_scan_tuples=20000`, `ivf_max_delta_pct=10`).

## Q8 — THP / sockets / SNC  ✅ ANSWERED
Evidence: `remote/host.txt`.
- THP: `enabled = always [madvise] never`, `defrag = always defer defer+madvise [madvise] never` → AMI
  default is **madvise / madvise**, left unchanged (R18b as expected).
- `lscpu`: Socket(s) = **1**, Core(s)/socket 48, Thread(s)/core 2; NUMA node(s) = **1** (node0 CPUs 0-95) →
  no SNC, no numactl binding needed.

## Q6 — pgvector default build: x86 distance kernel  ✅ ANSWERED
Evidence: `evidence/q6_pgvector_kernel.txt`, `remote/versions.txt`.
- pgvector v0.8.7's Makefile default is `OPTFLAGS = -march=native` (plus `-ftree-vectorize -fassociative-math
  -fno-signed-zeros -fno-trapping-math -ffp-contract=fast`); all 20 TUs compiled with `-march=native` (grep of
  the make log). So the "default" build is NOT portable: it is tuned for the build host (Emerald Rapids).
- With `-march=native`, `USE_TARGET_CLONES` only fires when `!__FMA__` → no target_clones; the auto-vectorised
  loop is inlined into `cosine_distance`. objdump of `cosine_distance` in the installed vector.so: 19 ×
  `vfmadd231ps` on **ymm** registers incl. `%ymm19` (EVEX = AVX-512VL encoding, 256-bit width — GCC's default
  `prefer-vector-width=256` on this target), 0 zmm. → pgvector runs an AVX-512VL/FMA 256-bit kernel, the best
  its default build produces on this host. **Not hobbled; the HNSW reference arm is valid.**
- Asymmetry worth a README footnote: pgvector's default is host-native while pg_vectorsearch (`-Dnative=false`)
  and pg_turbovec are portable runtime-dispatch builds. Since HNSW is reference-only, no change needed.

## Load + builds (CO1024 1M) — inputs for Q10 calibration
Evidence: `remote/load_times.txt`, `remote/load_checksums.txt`, `remote/build/*.summary`, `remote/datasets.lock.json`.
- Corpus prep (`scripts/prep_co1024.py`): 11 shards en/0000..0010 (md5s in datasets.lock.json), 1,006,041 rows
  read, **841 exact-duplicate corpus vectors dropped**, 0 null/zero; 5,200 held-out queries, **0 query↔corpus
  exact duplicates, 0 queries with distance-0 top-1**; max |‖v‖−1| = 1.45e-7 (corpus) / 1.22e-7 (queries) → unit
  vectors (< 1e-6 OK). GT: fp32 sgemm top-200 → fp64 rescore → top-100; tie-ambiguous (gap < 1e-6) k=10: 3
  queries, k=100: 24. Prep wall 85 s (shards cached by then; first fetch ~20 s via hf_transfer).
  Python 3.14 note: multiprocessing default is forkserver → the Pool re-ran the module; use `get_context("fork")`.
- Both tables LOGGED (`relpersistence = p`), `STORAGE PLAIN` (attstorage `p`), TOAST 0 bytes. 16-stream text COPY
  from identical files ('%.9g', exact fp32 round-trip): **db_pv 11.5 s, db_tv 10.5 s**; VACUUM (FREEZE, ANALYZE)
  5.6 s / 5.5 s. Cross-table checksums (plan C(b)): count 1,000,000 / sum(id) 499,999,500,000 equal; sum of
  component 1 = 2920.7662003634596 (pv) vs …573 (tv), component 1024 = 6826.938138479035 vs …039 — equal to
  ~1e-15 relative (fp64 summation order), queries table sum identical. Heap 8,192,450,560 B (pv) vs 8,192,131,072
  B (tv) — both 1 row/page. Avg `pg_column_size(emb)`: pgvector 4,104 B, **turbovec.vector 5,127 B (+25%)**;
  page counts equal at 1024-d (caveat 5 confirmed: ~5 B/element CBOR).
- Index builds, 32 threads, serial (one at a time), `wal_level=replica`:

  | index | wall s | index bytes | WAL bytes | peak Σ RssAnon (postmaster children) |
  |---|---:|---:|---:|---:|
  | prism defaults (nlist auto=3969, 31 workers+leader) | 43.2 | 316,080,128 | 317,563,608 | 21.4 GB (VmRSS sum†) |
  | turbovec flat 4-bit (`build_parallelism=32`) | 8.6 | 559,947,776 | 1,126,839,816 | 2.30 GB |
  | turbovec IVF lists=1000 4-bit | 28.4 / 28.5 (2 builds) | 564,076,544 (both) | 1,135,168,392 | 1.95 / 2.01 GB |
  | pgvector HNSW m=16 efc=64 (31+1) | 48.7 / 49.2 | 8,192,008,192 | 4,449,782,904 | 0.11–0.15 GB |

  † prism's peak was sampled as summed VmRSS (includes each worker's touched shared_buffers pages, so it
  over-counts); the later samplers use RssAnon. Re-measure prism peak with RssAnon in P2.
  WAL for turbovec ≈ 2 × index size (it full-page-writes the relfile, then a second time? — to check in P2, not
  a P0 blocker). `prism_index_settings('docs_prism')`: nlist 3969 (auto), fan_out 63, nlevels 2, target_pages 3,
  posting_pages 38455, centroid_format fastscan, fastscan **off** (auto), soar_lambda 1, boundary_epsilon 0.35,
  kmeans_nredo 1, distance_mode asymmetric, **nprobe 32 (auto)**.
- **Harness finding:** with two vector indexes on one column the planner silently picks one (with flat + IVF on
  db_tv.docs it chose the IVF index; with prism + HNSW on db_pv.docs it chose prism). So **each index gets its
  own table copy** (`docs`, `docs_ivf`, `docs_h`), and `bench.py --expect-index` asserts the EXPLAIN index name.
  P1/P2 must do the same (or drop/create per arm). This costs one heap copy (~8.2 GB at 1M, ~82 GB at 10M) per
  extra index kind — fits in RAM/disk; plan § A sizing should add it.
- Plan fixes discovered: turbovec opclass lookup needs `search_path = public, turbovec` (README § 519/919);
  `ALTER DATABASE db_tv SET search_path = public, turbovec` used. psql `-c "A; B"` runs as one transaction, so
  VACUUM must be its own `-c`.

## Q3 — turbovec residency route: buffer manager, not mmap  ✅ ANSWERED
Evidence: `remote/q13/q3_flat.txt`, `remote/q13/q3_strace.txt`, `remote/q13/q3_cold_query.txt`, script `scripts/q13_probe.sh`.
Method: `pg_ctl stop -m fast; sync; echo 3 > drop_caches; start`; one cold flat query (`docs_tv_flat`, 68,353 blocks =
559,947,776 B) under `strace -f -e mmap,munmap,pread64,preadv,read,openat,io_uring_enter,madvise` on the backend.
- Before: `fincore` 0 B resident, `pg_buffercache` 0 buffers. After the ONE cold query: `fincore` 534M/534M
  resident, `pg_buffercache` **68,138** buffers of the index (5 hit + 69,161 read reported by EXPLAIN BUFFERS,
  "I/O Timings: shared read=1953.123" ms; Execution Time 2,559.6 ms).
- strace: index relfile opened via `openat(... O_RDWR)` (PG smgr), **69,177 `pread64(…, 8192, …)`** calls, 0 preadv,
  0 io_uring_enter (single-block ReadBuffer path, not read_stream), **0 file-backed mmaps of any relfile** (the only
  4 fd-mmaps are `pg_turbovec.so` being dlopen'ed). Big anonymous mmaps: one 512,004,096 B (= 1M × 1024 × 4 bit / 8
  codes buffer, the per-backend cache) and ~95 × 128 MiB PROT_NONE arenas (one per rayon worker thread = glibc
  per-thread malloc arenas; see Product findings).
- After `pg_prewarm('docs_tv_flat')` from cold: `pg_buffercache` = 68,353 = every block.
- **Answer:** all turbovec index reads go through `ReadBufferExtended` → shared_buffers; no relfile mmap
  (matches PRODUCTION.md "no relfile mmap since v1.19.0"). So the residency gate for turbovec = `pg_buffercache`
  ≥ 99% of index blocks (same as prism), plus the per-backend cache being built (first query on the connection).
  `drop_caches` + `pg_prewarm(index)` apply to turbovec exactly as to prism. Cold loads are synchronous 8 KB
  preads (69k of them, ~2 s at 1M on this NVMe RAID0) — relevant to the cold cell (D4.2), not a P0 blocker.

## Q1 — flat index cached WHOLE per backend at default cache_size_mb = 256?  ✅ YES
Evidence: `remote/q13/q1_pss_trace.txt`, `remote/q13/q1_flat_cap{256,2048}_sk32.ids.summary.json`.
- Index codes 512,000,000 B (512 MB) ≫ cap 256 MB. One connection, prewarmed shared_buffers, default GUCs:
  PSS before any query 4.9 MB → after query 0: **Pss_Anon 1,147,473 kB (1.12 GiB)**, then flat (+0.1 MB over 11
  more queries). Query 0 = 426.5 ms (cache build from shared_buffers), queries 1–11 = 7.6–8.9 ms. No re-load.
- 500-query batches at cap 256 vs 2048 MB: p50 7.74 vs 7.24 ms, p95 9.24 vs 7.69 ms, recall 1.000 both, PSS_Anon
  1,155,729 vs 1,155,771 kB (identical), first-half vs second-half p50 7.83/7.60 vs 7.26/7.20 ms. → the cap changes
  nothing for a single flat index (enforce_cap evicts only while `map.len() > 1`). The 256-vs-2048 p50 gap (0.5 ms)
  is within run-to-run noise at this n (different backends, loadavg 4.7–7.1 from the parallel Q7 download);
  not attributed to the cap.
- Per-backend resident cost of the flat 1M index = **~1.12 GiB PSS/backend** (512 MB codes + slot→id + per-thread
  arenas etc.). At 96 clients that is ~108 GiB → fine at 1M; at 10M flat (~11 GiB/backend) the plan's G-row cap
  ("memory-bound at N clients") applies.
- **No D3 rewrite needed:** headline flat stays at default `cache_size_mb = 256`.

## Q2 — IVF path per cache_size_mb, and how to read it  ✅ ANSWERED
Evidence: `remote/q13/q2_ivf.txt`, `remote/q13/q2_ivf_*.ids.summary.json`. IVF lists=1000 (`docs_ivf_tv`), probes=16,
500 queries, 1 connection each, prewarmed.

| cache_size_mb | out_of_core | path taken | backend threads | Pss_Anon | p50 / p95 ms | CPU-ms/q | recall@10 |
|---:|---|---|---:|---:|---:|---:|---:|
| 256 | auto | cell-scoped (OOC) | 5 | 83 MB | 16.1 / 103.4 | 55.3 | 0.9246 |
| 256 | on | OOC | 5 | 83 MB | 16.4 / 104.0 | 64.1 | 0.9246 |
| 256 | off | whole-load | 97 | 1,118 MB | 29.7 / 128.0 | 91.5 | 0.9246 |
| 1024 | auto | **cell-scoped** (codes 512 MB = 0.5 × 1024 MB, test is `>` → … see note) | 97 | 1,118 MB | 29.7 / 128.1 | 90.8 | 0.9246 |
| 2048 | auto | whole-load | 97 | 1,118 MB | 29.3 / 127.3 | 88.2 | 0.9246 |
| 1024/2048 | on | OOC | 5 | 83 MB | 16.3 / 104.7, 16.3 / 103.6 | 61.4 / 58.9 | 0.9246 |
| 1024/2048 | off | whole-load | 97 | 1,118 MB | ~29.6 / 128 | 91–94 | 0.9246 |

  Note (1024/auto): measured = whole-load (97 threads, 1.1 GB). codes_bytes = 1M × 1024 × 4/8 = 512,000,000 B;
  threshold = 0.5 × 1024 MiB = 536,870,912 B; 512,000,000 > 536,870,912 is false → whole-load. Consistent with
  `guc::out_of_core_decide` (`codes_bytes > 0.5 × cap`).
- **How to label:** the path is NOT logged; read it from the backend: OOC = 5 threads (1 + scan_parallelism auto 4)
  and ~83 MB PSS; whole-load = 97 threads (1 + 96 rayon global pool) and ~1.1 GB PSS at 1M. `bench.py` records
  `threads_before/after` + `smaps_rollup` per batch, so every IVF cell is self-labelling. Rule: auto → OOC iff
  `n·dim·bit_width/8 > 0.5 × cache_size_mb MiB`.
- Surprising result for the plan: at 1M **OOC is FASTER than whole-load at p50** (16 vs 30 ms) and uses less CPU
  (55–64 vs 88–94 ms/q). Both paths have a heavy p95 (~100–130 ms, ≈ 6–8× p50): a long tail on some queries,
  to be characterised in P3 (not seen on flat: p95/p50 = 1.2). Recall identical across paths (same candidate set).
- IVF lists=1000 probes=16 recall@10 = 0.9246 at 1M (search_k 32, hi_dim_rerank auto).

## Product findings (pg_turbovec; lead-owned fix — the bench agent does NOT modify pg_turbovec source)
1. **`turbovec.search_concurrency` is a no-op GUC.** `git grep -n SEARCH_CONCURRENCY v2.12.1 -- src` →
   only `src/guc.rs:39` (the static) and `src/guc.rs:716` (its registration); `search_concurrency` only at
   guc.rs:12 (doc table) and :711 (name). Nothing reads it. Its description ("…This GUC caps that fan-out. 1
   disables intra-query parallelism.") is false. Evidence: `evidence/product_search_concurrency_grep.txt`.
2. **Flat and IVF whole-load single-query search runs on rayon's GLOBAL pool = all 96 host threads.** turbovec
   (rev 29de559) `search.rs:168 single_query_workers()` returns `rayon::current_num_threads()` when ≥
   `SINGLE_QUERY_PARALLEL_MIN_BLOCKS = 1024` blocks (≈ 32k rows) are visited; pg_turbovec never installs a
   bounded pool around `IdMapIndex::search`/`search_masked` (cache.rs:243/262). Evidence:
   `evidence/product_turbovec_single_query_workers.txt`.
3. **Measured (CO1024 1M × 1024-d, i7i.metal-24xl, 96 vCPU, defaults, 1 client, warm):**
   - flat 4-bit: backend thread count **97** (`/proc/<pid>/task`), wall p50 7.74 ms, **CPU 127.5 ms/query**
     (utime+stime of the thread group from `/proc/<pid>/stat` fields 14+15, delta over 500 queries; task-sum
     127.2 ms) → ≈ **16.5 cores busy per query** (`remote/q13/q1_flat_cap256_sk32.ids.summary.json`).
   - IVF lists=1000 whole-load: 97 threads, p50 29.7 ms, CPU 88–94 ms/q (≈ 3 cores avg over a ~30 ms query).
   - IVF out-of-core: 5 threads (bounded `scan_parallelism` pool works), CPU 55–64 ms/q.
   - Side-effect: each rayon worker gets a glibc malloc arena (strace shows ~95 × 128 MiB PROT_NONE reservations
     on the first query), so the global pool also inflates per-backend VSZ and RSS.
   - Cross-check with `perf stat -p` and the RAYON_NUM_THREADS=1/4 equal-CPU rows: Q4 below.

## Benchmark treatment of the product finding (lead directive, 2026-10-14)
1. **HEADLINE = defaults** (what a user gets today). Every latency/QPS number for BOTH systems carries
   **CPU-ms/query** and **QPS/core** beside it. README caveat: *pg_turbovec's default single-query latency uses
   intra-query parallelism across all cores (rayon global pool; measured ~16 cores/query on flat 1M), while
   pg_vectorsearch uses one core per query.* The 1-client latency ratio is never shown without the CPU column.
2. **EQUAL-CPU row** (P0 measured below; planned for P3 and P5): pg_turbovec with **`RAYON_NUM_THREADS=1` in the
   postmaster environment** (recorded exactly, verified via `/proc/<postmaster>/environ`; env setting, same class
   as GLIBC_TUNABLES) → 1 query = 1 core for both systems. Also `RAYON_NUM_THREADS=4`. Flat 4-bit at the same
   search_k operating points: p50/p95, CPU-ms/q, recall.
3. **P5 multi-client QPS runs BOTH default and RAYON_NUM_THREADS=1 for pg_turbovec**, both reported in the table;
   neither is picked after seeing results.

## Calibration grids (P0, 1,000 timed qids 50..1049 after 50 warm-up, 1 connection per point, restart+drop_caches+prewarm per arm) — recall selection data; latencies are P0 timing info only, NOT headline
Evidence: `remote/calib/*.summary.json|jsonl`, `remote/calib/arm_times.txt`, `scripts/calib_p0.sh`. Client wall (psycopg, prepared, unix socket) incl. ~0.1 ms Python overhead. All rows: 0 order violations / 0 short results unless shown.

**prism** — prism defaults, knob prism.nprobe (0 = auto)

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prism_0 | 1000 | 0.9679 | 1.28 | 1.72 | 1.1 | 1 | 3 | 1.3 |
| prism_4 | 1000 | 0.7620 | 0.55 | 0.64 | 0.3 | 1 | 3 | 0.6 |
| prism_6 | 1000 | 0.8260 | 0.62 | 0.74 | 0.3 | 1 | 3 | 0.6 |
| prism_8 | 1000 | 0.8658 | 0.67 | 0.83 | 0.4 | 1 | 3 | 0.7 |
| prism_10 | 1000 | 0.8920 | 0.73 | 0.92 | 0.5 | 1 | 3 | 0.7 |
| prism_14 | 1000 | 0.9242 | 0.85 | 1.08 | 0.6 | 1 | 3 | 0.9 |
| prism_20 | 1000 | 0.9478 | 1.01 | 1.32 | 0.8 | 1 | 3 | 1.0 |
| prism_28 | 1000 | 0.9627 | 1.24 | 1.63 | 1.0 | 1 | 3 | 1.3 |
| prism_40 | 1000 | 0.9750 | 1.46 | 1.98 | 1.2 | 1 | 3 | 1.5 |
| prism_56 | 1000 | 0.9826 | 1.79 | 2.49 | 1.6 | 1 | 3 | 1.8 |
| prism_80 | 1000 | 0.9887 | 2.31 | 3.15 | 2.1 | 1 | 3 | 2.4 |
| prism_112 | 1000 | 0.9929 | 3.03 | 3.91 | 2.8 | 1 | 3 | 3.1 |
| prism_160 | 1000 | 0.9956 | 4.05 | 5.00 | 3.8 | 1 | 3 | 4.1 |
| prism_224 | 1000 | 0.9969 | 5.43 | 6.41 | 5.1 | 1 | 3 | 5.4 |
| prism_320 | 1000 | 0.9978 | 7.47 | 8.47 | 7.1 | 1 | 3 | 7.4 |
| prism_448 | 1000 | 0.9989 | 10.07 | 11.23 | 9.7 | 1 | 3 | 10.0 |
| prism_640 | 1000 | 0.9991 | 13.90 | 15.37 | 13.6 | 1 | 3 | 13.8 |
| prism_1024 | 1000 | 0.9993 | 21.58 | 23.49 | 21.2 | 1 | 3 | 21.5 |

**tvflat** — turbovec flat 4-bit, knob search_k, hi_dim_rerank=off

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tvflat_10 | 1000 | 0.9399 | 1.15 | 1.37 | 59.1 | 97 | 1114 | 1.2 |
| tvflat_16 | 1000 | 0.9980 | 1.18 | 1.41 | 58.8 | 97 | 1114 | 1.2 |
| tvflat_24 | 1000 | 0.9999 | 1.25 | 1.46 | 62.2 | 97 | 1115 | 1.3 |
| tvflat_32 | 1000 | 1.0000 | 1.21 | 1.32 | 41.9 | 97 | 1115 | 1.2 |
| tvflat_48 | 1000 | 1.0000 | 1.25 | 1.36 | 42.2 | 97 | 1117 | 1.4 |
| tvflat_64 | 1000 | 1.0000 | 1.56 | 1.75 | 67.9 | 97 | 1117 | 1.7 |
| tvflat_96 | 1000 | 1.0000 | 1.80 | 2.00 | 78.5 | 97 | 1121 | 2.0 |
| tvflat_128 | 1000 | 1.0000 | 2.03 | 2.20 | 81.9 | 97 | 1121 | 2.2 |
| tvflat_192 | 1000 | 1.0000 | 2.44 | 2.62 | 88.8 | 97 | 1128 | 2.9 |
| tvflat_256 | 1000 | 1.0000 | 2.81 | 2.99 | 88.5 | 97 | 1128 | 3.2 |
| tvflat_384 | 1000 | 1.0000 | 3.56 | 3.76 | 93.6 | 97 | 1128 | 4.0 |
| tvflat_512 | 1000 | 1.0000 | 4.28 | 4.52 | 95.8 | 97 | 1128 | 4.7 |
| tvflat_768 | 1000 | 1.0000 | 5.74 | 5.98 | 104.9 | 97 | 1128 | 6.0 |
| tvflat_1024 | 1000 | 1.0000 | 7.15 | 7.48 | 102.8 | 97 | 1129 | 7.4 |

**tvflatdef** — turbovec flat 4-bit, ALL defaults

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tvflatdef_32 | 1000 | 1.0000 | 7.69 | 8.50 | 127.7 | 97 | 1129 | 8.1 |

**tvivf** — turbovec IVF lists=1000, knob probes, other GUCs default (path auto → see threads)

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tvivf_1 | 1000 | 0.4894 | 9.01 | 11.01 | 8.6 | 97 | 33 | 8.8 |
| tvivf_2 | 1000 | 0.6556 | 10.95 | 13.06 | 10.8 | 99 | 42 | 11.1 |
| tvivf_3 | 1000 | 0.7377 | 12.30 | 13.95 | 13.0 | 100 | 62 | 12.4 |
| tvivf_4 | 1000 | 0.7875 | 11.93 | 13.99 | 15.5 | 100 | 59 | 12.6 |
| tvivf_6 | 1000 | 0.8421 | 11.86 | 12.91 | 20.5 | 101 | 66 | 13.1 |
| tvivf_8 | 1000 | 0.8699 | 12.20 | 99.05 | 38.8 | 5 | 77 | 27.5 |
| tvivf_12 | 1000 | 0.9055 | 14.11 | 101.11 | 48.1 | 5 | 70 | 30.3 |
| tvivf_16 | 1000 | 0.9261 | 16.02 | 102.64 | 50.0 | 5 | 83 | 27.6 |
| tvivf_24 | 1000 | 0.9484 | 20.11 | 105.86 | 61.4 | 5 | 102 | 29.1 |
| tvivf_32 | 1000 | 0.9607 | 24.02 | 107.90 | 78.9 | 5 | 110 | 36.5 |
| tvivf_48 | 1000 | 0.9761 | 31.07 | 106.57 | 108.2 | 5 | 162 | 45.8 |
| tvivf_64 | 1000 | 0.9838 | 47.59 | 117.00 | 130.4 | 5 | 195 | 51.3 |
| tvivf_96 | 1000 | 0.9905 | 67.58 | 127.88 | 193.7 | 5 | 38 | 74.3 |
| tvivf_128 | 1000 | 0.9943 | 76.35 | 136.61 | 171.0 | 5 | 33 | 73.9 |
| tvivf_192 | 1000 | 0.9972 | 75.87 | 147.69 | 138.4 | 5 | 264 | 79.7 |
| tvivf_256 | 1000 | 0.9987 | 114.93 | 121.67 | 199.7 | 5 | 287 | 111.1 |

**hnsw** — pgvector HNSW m=16 efc=64 (reference), knob ef_search

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hnsw_40 | 1000 | 0.8799 | 1.41 | 1.91 | 1.1 | 1 | 2 | 1.4 |
| hnsw_100 | 1000 | 0.9519 | 2.21 | 3.15 | 1.9 | 1 | 2 | 2.2 |
| hnsw_200 | 1000 | 0.9770 | 3.21 | 4.88 | 3.0 | 1 | 2 | 3.3 |
| hnsw_400 | 1000 | 0.9891 | 5.29 | 8.46 | 5.2 | 1 | 2 | 5.5 |

**Preliminary bracketing (plan D1 rule, on these 1,000 qids — NOT operating_points.json; P3 re-derives on ≥5,000):**

- prism T=0.9: bracket [8, 10, 14, 20]; headline (smallest ≥T) = 14
- prism T=0.95: bracket [14, 20, 28, 40]; headline (smallest ≥T) = 28
- prism T=0.98: bracket [28, 40, 56, 80]; headline (smallest ≥T) = 56
- prism T=0.99: bracket [56, 80, 112, 160]; headline (smallest ≥T) = 112
- tvflat T=0.9: bracket [10, 16]; headline (smallest ≥T) = 10
- tvflat T=0.95: bracket [10, 16, 24]; headline (smallest ≥T) = 16
- tvflat T=0.98: bracket [10, 16, 24]; headline (smallest ≥T) = 16
- tvflat T=0.99: bracket [10, 16, 24]; headline (smallest ≥T) = 16
- tvivf T=0.9: bracket [6, 8, 12, 16]; headline (smallest ≥T) = 12
- tvivf T=0.95: bracket [16, 24, 32, 48]; headline (smallest ≥T) = 32
- tvivf T=0.98: bracket [32, 48, 64, 96]; headline (smallest ≥T) = 64
- tvivf T=0.99: bracket [48, 64, 96, 128]; headline (smallest ≥T) = 96

## Q10 — per-cell wall-time calibration  ⚠️ PARTIAL
Measured: warm single-client cell only (all four arms) + equal-CPU rows. NOT timed in P0: cold/new-connection, closed/open-loop QPS, churn, filtered, adoption, build-variance (builds themselves ARE timed above). For those the plan's § F estimates stand as ESTIMATES.

Warm single-client headline cell, CO1024 **1M**, per system = 2 passes (ids + EXPLAIN) × 5 rounds (round 0 + ABBA×4) × 5,200 queries × Σ(per-query wall over the unique bracket points); restarts ≈ 5 s/arm-pass (arm_times.txt) are negligible:

- prism: 10 unique points, Σ per-query wall 17.4 ms → **0.25 h** measured-basis
- tvflat: 3 unique points, Σ per-query wall 3.7 ms → **0.05 h** measured-basis
- tvivf: 10 unique points, Σ per-query wall 409.3 ms → **5.91 h** measured-basis
- 1M CO1024 headline warm cell, both systems: **6.22 h** (turbovec arm counted = flat @1M; IVF row shown for the 10M basis)
- 10M CO1024 (prism + turbovec IVF) EXTRAPOLATION ×√10 (not measured): ~19.5 h; flat-10M secondary ×10 of the flat row: ~0.5 h

## Equal-CPU rows — RAYON_NUM_THREADS in the postmaster env (lead directive)  
Evidence: `remote/q45/rayon*.summary.json`, `remote/q45/rayon_*_start.txt` (shows /proc/<postmaster>/environ RAYON line + affinity: postmaster pinned to cpus 0-43,48-91 — note: server affinity mask = 88 vCPUs for these rows, 4 whole cores reserved for pgbench).

| label | n | R@10 | p50 ms | p95 ms | CPU-ms/q | threads | PssAnon MB | batch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rayon1_flat_default | 1000 | 1.0000 | 12.18 | 12.79 | 12.0 | 2 | 1000 | 12.3 |
| rayon1_flat_hdroff_sk16 | 1000 | 0.9980 | 5.22 | 5.28 | 5.0 | 2 | 992 | 5.2 |
| rayon1_flat_hdroff_sk32 | 1000 | 1.0000 | 5.33 | 5.37 | 5.1 | 2 | 992 | 5.4 |
| rayon1_flat_hdroff_sk64 | 1000 | 1.0000 | 5.54 | 5.59 | 5.4 | 2 | 992 | 5.7 |
| rayon1_flat_hdroff_sk128 | 1000 | 1.0000 | 6.01 | 6.10 | 5.8 | 2 | 993 | 6.1 |
| rayon1_flat_hdroff_sk256 | 1000 | 1.0000 | 6.96 | 7.13 | 6.8 | 2 | 994 | 7.1 |
| rayon1_flat_hdroff_sk512 | 1000 | 1.0000 | 8.72 | 8.99 | 8.6 | 2 | 996 | 8.9 |
| rayon1_flat_hdroff_sk1024 | 1000 | 1.0000 | 12.20 | 12.75 | 12.0 | 2 | 1000 | 12.3 |
| rayon1_ivf_p16_ooc | 1000 | 0.9261 | 16.20 | 103.27 | 49.8 | 5 | 83 | 27.2 |
| rayon1_ivf_p16_whole | 1000 | 0.9261 | 33.59 | 119.11 | 44.7 | 2 | 997 | 45.1 |
| rayon4_flat_default | 1000 | 1.0000 | 7.96 | 8.81 | 13.3 | 5 | 1006 | 8.2 |
| rayon4_flat_hdroff_sk16 | 1000 | 0.9980 | 1.90 | 1.93 | 5.5 | 5 | 996 | 1.9 |
| rayon4_flat_hdroff_sk32 | 1000 | 1.0000 | 2.03 | 2.32 | 5.7 | 5 | 996 | 2.1 |
| rayon4_flat_hdroff_sk64 | 1000 | 1.0000 | 2.25 | 2.40 | 6.1 | 5 | 996 | 2.3 |
| rayon4_flat_hdroff_sk128 | 1000 | 1.0000 | 2.63 | 2.80 | 6.8 | 5 | 997 | 2.7 |
| rayon4_flat_hdroff_sk256 | 1000 | 1.0000 | 3.42 | 3.66 | 7.8 | 5 | 998 | 3.5 |
| rayon4_flat_hdroff_sk512 | 1000 | 1.0000 | 4.95 | 5.28 | 9.8 | 5 | 1000 | 5.1 |
| rayon4_flat_hdroff_sk1024 | 1000 | 1.0000 | 8.01 | 8.45 | 13.6 | 5 | 1007 | 8.2 |
| rayon4_ivf_p16_ooc | 1000 | 0.9261 | 16.17 | 103.29 | 50.5 | 5 | 83 | 27.9 |
| rayon4_ivf_p16_whole | 1000 | 0.9261 | 33.16 | 118.64 | 44.4 | 5 | 1000 | 44.2 |
| rayondefault_flat_default | 1000 | 1.0000 | 7.08 | 7.64 | 92.2 | 89 | 1118 | 7.4 |
| rayondefault_flat_hdroff_sk16 | 1000 | 0.9980 | 1.07 | 1.24 | 52.0 | 89 | 1104 | 1.1 |
| rayondefault_flat_hdroff_sk32 | 1000 | 1.0000 | 1.12 | 1.21 | 36.6 | 89 | 1105 | 1.1 |
| rayondefault_flat_hdroff_sk64 | 1000 | 1.0000 | 1.51 | 1.68 | 55.3 | 89 | 1107 | 1.6 |
| rayondefault_flat_hdroff_sk128 | 1000 | 1.0000 | 1.93 | 2.10 | 71.4 | 89 | 1111 | 2.1 |
| rayondefault_flat_hdroff_sk256 | 1000 | 1.0000 | 2.72 | 2.89 | 81.7 | 89 | 1118 | 3.1 |
| rayondefault_flat_hdroff_sk512 | 1000 | 1.0000 | 4.17 | 4.38 | 84.5 | 89 | 1118 | 4.6 |
| rayondefault_flat_hdroff_sk1024 | 1000 | 1.0000 | 7.06 | 7.37 | 90.3 | 89 | 1118 | 7.3 |
| rayondefault_ivf_p16_ooc | 1000 | 0.9261 | 16.18 | 103.00 | 53.7 | 5 | 83 | 30.8 |
| rayondefault_ivf_p16_whole | 1000 | 0.9261 | 33.39 | 119.03 | 80.5 | 89 | 1108 | 44.4 |

## Q4 — CPU accounting /proc vs perf stat
Evidence: `remote/q45/q4_perf.txt`, `q4_perfstat_db_*.txt`. 500 queries, default env, flat (db_tv) and prism (db_pv):
```
db_tv wall=3.920s /proc thread-group cpu=50.530s task-sum cpu=50.500s per-query tg=101.06ms nthreads=89
# started on Wed Oct 14 19:59:11 2026


 Performance counter stats for process id '44390':

         51,218.07 msec task-clock                                                            
           186,304      context-switches                                                      
            12,890      cpu-migrations                                                        

       4.934386711 seconds time elapsed


db_pv wall=0.688s /proc thread-group cpu=0.550s task-sum cpu=0.550s per-query tg=1.10ms nthreads=1
# started on Wed Oct 14 19:59:17 2026


 Performance counter stats for process id '44502':

            559.22 msec task-clock                                                            
               504      context-switches                                                      
                 0      cpu-migrations                                                        

       1.867795892 seconds time elapsed
```
- db_tv: /proc thread-group 50.530 s vs perf task-clock 51.218 s (perf window includes ~1.2 s idle around the batch) → ratio 0.987
- db_pv: /proc thread-group 0.550 s vs perf task-clock 0.559 s (perf window includes ~1.2 s idle around the batch) → ratio 0.984

## Q5 — pgbench \gset script + pinning
Evidence: `remote/q45/q5_pgbench.txt`, `q5_compare.txt`, `pinning.txt`, `pgb_*.sql`.
```
client cpus=44,92,45,93,46,94,47,95 server cpus=0-43,48-91
+ pg_ctl -D /data/pgdata -m fast -w stop
waiting for server to shut down..... done
server stopped
+ sync
+ echo 3
+ sudo tee /proc/sys/vm/drop_caches
+ env taskset -c 0-43,48-91 pg_ctl -D /data/pgdata -l /home/fedora/pg.log -w start
waiting for server to start.... done
server started
++ head -1 /data/pgdata/postmaster.pid
+ PM=44287
++ taskset -pc 44287
+ echo 'postmaster 44287 affinity: pid 44287'\''s current affinity list: 0-43,48-91'
postmaster 44287 affinity: pid 44287's current affinity list: 0-43,48-91
+ tr '\0' '\n'
+ grep -E 'RAYON|GLIBC'
+ echo 'no RAYON/GLIBC env'
no RAYON/GLIBC env
+ for db_r in db_tv:docs db_tv:docs_tv_flat db_tv:queries db_pv:docs db_pv:docs_prism db_pv:queries
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d d
tv captured=100 identical_order=100 identical_set=100
pv captured=100 identical_order=100 identical_set=100
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_tv -c 'DROP TABLE IF EXISTS pgb_out' -c 'CREATE TABLE pgb_out(qid int, ids bigint[])'
+ echo '== plan form (tv)'
== plan form (tv)
pgbench: error: Run was aborted; the above results are incomplete.
+ echo '== :qv form capture (tv)'
== :qv form capture (tv)
+ grep -E 'latency|tps|failed|error'
number of failed transactions: 0 (0.000%)
latency average = 11.718 ms
tps = 85.339159 (without initial connection time)
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_tv -Atc 'EXPLAIN (COSTS OFF) INSERT INTO pgb_out SELECT 1, array_agg(id) FROM (SELECT id FROM docs ORDER BY emb <=> (SELECT q FROM queries WHERE qid=1) LIMIT 10) s'
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_tv -Atc 'COPY (SELECT qid, array_to_string(ids, '\'','\'') FROM pgb_out) TO STDOUT'
+ echo '== :qv form 30 s 4 clients pinned (tv)'
== :qv form 30 s 4 clients pinned (tv)
+ grep -E 'progress|latency|tps|failed|error'
progress: 10.0 s, 370.8 tps, lat 10.777 ms stddev 15.415, 0 failed
progress: 20.0 s, 386.1 tps, lat 10.358 ms stddev 3.169, 0 failed
progress: 30.0 s, 387.8 tps, lat 10.313 ms stddev 3.084, 0 failed
number of failed transactions: 0 (0.000%)
latency average = 10.480 ms
tps = 381.482364 (without initial connection time)
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_pv -c 'DROP TABLE IF EXISTS pgb_out' -c 'CREATE TABLE pgb_out(qid int, ids bigint[])'
+ echo '== plan form (pv)'
== plan form (pv)
pgbench: error: Run was aborted; the above results are incomplete.
+ echo '== :qv form capture (pv)'
== :qv form capture (pv)
+ grep -E 'latency|tps|failed|error'
number of failed transactions: 0 (0.000%)
latency average = 2.323 ms
tps = 430.564944 (without initial connection time)
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_pv -Atc 'EXPLAIN (COSTS OFF) INSERT INTO pgb_out SELECT 1, array_agg(id) FROM (SELECT id FROM docs ORDER BY emb <=> (SELECT q FROM queries WHERE qid=1) LIMIT 10) s'
+ psql -h /tmp -X -v ON_ERROR_STOP=1 -d db_pv -Atc 'COPY (SELECT qid, array_to_string(ids, '\'','\'') FROM pgb_out) TO STDOUT'
+ echo '== :qv form 30 s 4 clients pinned (pv)'
== :qv form 30 s 4 clients pinned (pv)
+ grep -E 'progress|latency|tps|failed|error'
progress: 10.0 s, 3670.3 tps, lat 1.089 ms stddev 1.103, 0 failed
progress: 20.0 s, 3754.8 tps, lat 1.065 ms stddev 0.161, 0 failed
progress: 30.0 s, 3751.2 tps, lat 1.066 ms stddev 0.163, 0 failed
number of failed transactions: 0 (0.000%)
latency average = 1.073 ms
tps = 3725.655710 (without initial connection time)
```
Note: the plan's `:'qv'` form is psql syntax; pgbench supports only `:qv` (prepared → bound as $1). Result of each form is in the excerpt above.

## Q7 — dataset URLs / pins / counts
Evidence: `remote/q7/*`.
```
http://ann-benchmarks.com/cohere-wikipedia-22-12-1M-angular.hdf5 404
https://ann-benchmarks.com/cohere-wikipedia-22-12-1M-angular.hdf5 404
http://ann-benchmarks.com/glove-100-angular.hdf5 200
https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings 401
https://huggingface.co/api/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3 200
ed8a95682c184a5ad3de2538e64e189c  cohere_large_10m/neighbors.parquet
293629678416c4e1b8076e45d17ba569  cohere_large_10m/test.parquet
d1aca052763a292b10e80bbda964f55d  cohere_medium_1m/neighbors.parquet
c63a0efd0e611d9f35345af1fc6f7966  cohere_medium_1m/test.parquet
51f7420cc71b6de7f7218d2dc1eb983f  cohere_medium_1m/train.parquet
072e4fa27dcb4c116485367d06c6f1a4  openai_large_5m/neighbors.parquet
392273730a5f6a8893acca3f4d6d8a9b  openai_large_5m/test.parquet
/data/vdb/cohere_large_10m/neighbors.parquet rows 1000 schema id: int64
neighbors_id: large_list<item: int64>
  child 0, item: int64
/data/vdb/cohere_large_10m/test.parquet rows 1000 schema id: int64
emb: large_list<item: float>
  child 0, item: float
/data/vdb/cohere_medium_1m/neighbors.parquet rows 1000 schema id: int64
neighbors_id: large_list<item: int64>
  child 0, item: int64
/data/vdb/cohere_medium_1m/test.parquet rows 1000 schema id: int64
emb: large_list<item: float>
  child 0, item: float
/data/vdb/cohere_medium_1m/train.parquet rows 1000000 schema id: int64
emb: large_list<item: float>
  child 0, item: float
/data/vdb/openai_large_5m/neighbors.parquet rows 1000 schema id: int64
neighbors_id: list<item: int64>
  child 0, item: int64
-- schema metadata --
pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' + 491
/data/vdb/openai_large_5m/test.parquet rows 1000 schema id: int64
emb: large_list<item: double>
  child 0, item: double
cohere_medium_1m: test (1000, 768) train (1000000, 768)
norms test min/max 12.342849 16.224667 train min/max 12.258187 17.474176
corpus exact-dup rows (after normalise): 81  test queries exactly equal to a corpus row: 1
test queries with max cos >= 1-1e-6: 1  >= 0.9999: 1
```
- **CW768-1M source BROKEN (plan R7):** `ann-benchmarks.com/cohere-wikipedia-22-12-1M-angular.hdf5` returns 404 (http and https); the name is not in ann-benchmarks' `datasets.py` registry (checked main); pg_vectorsearch's bench_ann.py expects it pre-placed in `../ann-benchmarks/data/`. HF `Cohere/wikipedia-22-12-en-embeddings` = 401. **Proposed replacement:** VDBBench `cohere_medium_1m` (same Cohere wikipedia-22-12 768-d model, 1M train + 1,000 test, reachable) — convert it ONCE to an ann-benchmarks-layout HDF5 (train/test/neighbors from our own GT) so their bench_ann.py cross-check reads identical bytes. Needs lead/user sign-off (matrix change).
- glove-100-angular.hdf5 resolves (200, 485,413,888 B). CO1024 shards pinned in `remote/datasets.lock.json` (11 shards, md5s).
- **zilliz bucket throughput ≈ 6 MB/s single-stream** (cohere_medium_1m/train.parquet: 1.18 → 2.68 GB in 4 min). P1 needs ~31 GB (cohere_large_10m) + ~45 GB (openai_large_5m) per host → ~3.5 h single-stream; use ≥8 parallel curl -C - streams (one per shard) or the 3 h P1 budget slips.

## Status / teardown
- At 2026-10-14 20:07 UTC the instance has run 1.07 h on-demand ≈ **$10** (list estimate $9.06/h). **P1+ all on-demand; spot quota 32 vCPU** — plan § F already prices P1–P7 on-demand (≈ $850 incl. P0), so the estimate is unchanged except P0 (~$7 → actual on-demand, see above).
- **Instance LEFT RUNNING** (Q10 incomplete: non-warm cell types untimed). Safety net re-armed AFTER the final rsync to `shutdown -h +180` (terminate-on-shutdown) — see `evidence/rearm.txt`. All artefacts pulled (`remote/`).
- Teardown (by tag only): `bash .agent/notes/pgvs_p0_teardown.sh` (instances, spot reqs, available volumes, SG sg-029a1bfb036d5524c, key pair pgvs-202610141901; then verifies). NOT yet run.
- `scripts/bench.py` = the exact version that ran on the host; `scripts/bench_sweep_UNTESTED.py` = a local refactor (multi-point sweep in one connection) that was never run.

## Lead follow-up (2026-10-14 ~20:20–20:40 UTC, same host): the IVF out-of-core ~100 ms tail

Evidence: `remote/tailprobe/`, `scripts/tail_probe.sh`, `scripts/tail_strace.sh`, `scripts/tail_perf2.sh`.
IVF lists=1000, `out_of_core=on`, `cache_size_mb=256`, probes=16, 300 queries, 1 connection per row:

| `turbovec.scan_parallelism` | p50 ms | p95 ms | queries > 60 ms | median of those |
|---:|---:|---:|---:|---:|
| 1 (serial) | 18.1 | 22.8 | **0** | — |
| 2 | 22.6 | 27.9 | **0** | — |
| 4 (auto default) | 16.5 | 105.1 | 80 / 300 | 103.2 |
| 8 | 98.9 | 100.8 | 166 / 300 | 99.5 |

- The tail is **not the data**: it doesn't follow the query (the slow sets at probes 8/16/32 barely
  overlap, 14–24 of ~150), and it disappears entirely at `scan_parallelism` 1 or 2. It **is the
  intra-query thread pool**: slow queries sit in a tight 99–107 ms band, and `strace -f` on a slow backend
  shows the scan-pool worker threads blocked in `futex` for 50–93 ms per query.
- That points at a wake-up/hand-off stall between pg_turbovec's per-backend `turbovec-scan-N` pool
  (`build_pool::scan_pool_with`) and turbovec's own rayon use inside the chunk search. turbovec
  calls `rayon::current_num_threads()` / spawns into "the current pool", so the nested search runs on the
  scan pool's own threads, and with ≥ 4 chunks the pool appears to deadlock-wait until a ~100 ms
  sleep/timeout resolves it. Not yet root-caused; that's lead-owned product work.
- Flat (the 1M headline kind) profile at default settings (`perf_flat_report.txt`): **~58% of backend CPU is
  `crossbeam_epoch` pin/advance** (rayon's work-stealing bookkeeping across the 96-thread global pool), 9% is
  the actual VNNI scan kernel, ~4% CBOR decode. This is the CPU cost behind "≈ 16 cores per query" in
  Product findings, and why `RAYON_NUM_THREADS=1/4` cuts CPU-ms/q 8–20× at similar or better latency for
  small `search_k`.

**Consequence for the benchmark (lead decision):** the IVF cells as configured measure a pg_turbovec
scheduling bug, not its algorithm. Running ~6–20 h of IVF frontier cells at `scan_parallelism = auto` would
publish that bug as our 10M performance. Instead the main run is **paused before P1** while the bug is
fixed and released as 2.12.2. The plan's contestant config for pg_turbovec IVF stays "defaults", run on
the fixed release. P0's prism/HNSW/flat calibration stays valid, since those paths don't use the scan pool.

## Teardown (lead, 2026-10-14 ~20:45 UTC)
All artefacts pulled first. `pgvs_p0_teardown.sh` terminated i-0d2d9f4fc515b0ef3 (verified `terminated`); SG
sg-029a1bfb036d5524c, key pair pgvs-202610141901 and all tagged volumes are gone (verified by tag). Wall ≈ 1.7 h
on-demand ≈ $15.
