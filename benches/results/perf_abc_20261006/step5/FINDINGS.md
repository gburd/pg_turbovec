# Step 5: end-to-end A/B of Fix A + Fix B (2026-10-07)

**Result: Fix A (f64-lane kernels) + Fix B (decode the constant query once,
reuse its norm) cut 3.4–3.7 µs per rechecked candidate, giving 1.24× (default
storage) and 1.37× (`STORAGE MAIN`) at `search_k = 1024` on 1M-scale real
embeddings, recall unchanged. The remaining per-candidate cost is TOAST (~7.8
µs, default storage only) and the ONE remaining CBOR decode of each candidate
(~3.3 µs), which is the step-6 format work.**

## Setup

- Host: EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, Sapphire Rapids, 16 vCPU,
  AVX-512 VBMI+VNNI so turbovec's staged search is active), Debian 13 amd64.
- PostgreSQL 16.15 built `-O2 -g -fno-omit-frame-pointer`, no cassert;
  `shared_buffers = 8GB`, `jit = off`, `max_parallel_workers_per_gather = 0`.
  Postmaster restarted per arm with
  `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456`
  (Fix C's finding: without it some warm backends settle into a page-fault
  state that adds 2–3 ms/query at search_k=1024, bimodally per backend).
- Corpus: first 500,000 rows of CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  (en), 1024-d, L2-normalized; 200 held-out queries (the next 200 rows); exact
  top-10 GT by dot product (`tvp_corpus.py`).
- Two tables with identical rows: `docs_ext` (type default EXTENDED → vectors
  TOASTed: 29 MB heap + 2,790 MB TOAST) and `docs_main` (`SET STORAGE MAIN`
  before loading: 3,906 MB heap, 0 TOAST). Each has a flat 4-bit turbovec index
  (268 MB; index bytes are identical between arms — the index format is
  untouched).
- Arms (same PG install, `.so` swapped, postmaster restarted):
  - **old** = v2.11.0 source (built from `main` @ 8939b89, whose `src/`,
    `Cargo.toml`, `Cargo.lock` are byte-identical to the v2.11.0 tag),
    `old.so` md5 `507c3104…`;
  - **new** = `perf/integrate-ab` @ 59cee0d (Fix A 7377ed5 + Fix B 614a6d1 +
    the norm seam), `new.so` md5 `3e3d5d96…`.
- Method (`tvp_ab.py`, `tvp_arm.sh`, `tvp_ab_all.sh`): arms alternated
  old, new × 3 rounds; per (arm, table, search_k): one psql session, 20 warm-up
  queries, then 200 queries timed by top-level `EXPLAIN (ANALYZE, TIMING OFF)`
  Execution Time; recall@10 from the same queries' returned ids.
  `oversample = 1.0`, `hi_dim_rerank = off`. Max load average during runs 3.6.
- **Round 1 is excluded**: its first pass of each arm ran against a cold OS page
  cache (e.g. old docs_ext search_k=32 at 32.6 ms vs 1.9 ms in rounds 2–3).
  Rounds 2 and 3 agree within ~3% for every cell (`raw/results.jsonl`).

## Latency (p50 ms, median of rounds 2–3; recall@10 min over rounds)

| table | search_k | v2.11.0 | A+B | speedup | ms saved | µs saved / candidate | R@10 old/new |
|---|---|---|---|---|---|---|---|
| docs_ext | 32 | 1.93 | 1.82 | 1.06× | 0.11 | 3.45 | 1.000/1.000 |
| docs_ext | 100 | 3.51 | 3.17 | 1.11× | 0.34 | 3.44 | 1.000/1.000 |
| docs_ext | 256 | 6.22 | 5.27 | 1.18× | 0.95 | 3.71 | 1.000/1.000 |
| docs_ext | 1024 | 18.96 | 15.27 | **1.24×** | 3.69 | 3.60 | 1.000/1.000 |
| docs_main | 32 | 1.58 | 1.51 | 1.05× | 0.07 | 2.30 | 1.000/1.000 |
| docs_main | 100 | 2.77 | 2.49 | 1.11× | 0.28 | 2.80 | 1.000/1.000 |
| docs_main | 256 | 4.66 | 3.64 | 1.28× | 1.03 | 4.01 | 1.000/1.000 |
| docs_main | 1024 | 13.44 | 9.81 | **1.37×** | 3.63 | 3.55 | 1.000/1.000 |

The saving is ~3.5 µs per candidate in every cell, independent of storage —
consistent with Fix A + B removing one CBOR decode (~2.3 µs here) and most of
the distance kernel (~2.1 µs → ~0.0 µs in the profile below), and with the
remaining costs being per-candidate. At small `search_k` the fixed per-query
cost dominates and the relative gain is small. `MAIN` + A+B together take
search_k=1024 from 18.96 ms (v2.11.0, default storage) to **9.81 ms (1.93×)**.

## Where the per-candidate time goes now (backend on-CPU, µs/candidate)

`tvp_attr.py`: `perf -e task-clock -c 100000` on the backend thread only (each
sample = 100 µs of on-CPU time), stacks bucketed by first matching frame, 150
queries at `search_k = 1024` (`raw/attr.jsonl`).

| component | docs_ext old | docs_ext new | docs_main old | docs_main new | owner |
|---|---|---|---|---|---|
| TOAST fetch | 7.90 | 7.81 | 0 | 0 | ours (storage) |
| CBOR decode | 5.62 | **3.37** | 5.49 | **3.29** | ours (format) |
| distance kernel | 2.11 | **0.01** | 2.06 | **0.01** | ours |
| heap fetch | 0.93 | 0.82 | 3.26 | 3.14 | core |
| reorder queue | 0.04 | 0.04 | 0.17 | 0.14 | core |
| executor misc | 0.20 | 0.15 | 0.44 | 0.43 | core |
| turbovec scan, backend thread | 0.07 | 0.07 | 0.12 | 0.14 | ours |
| other (parse/plan/libpq/unattributed) | 1.56 | 2.12 | 1.37 | 1.96 | — |
| **total** | **18.42** | **14.39** | **12.92** | **9.09** | |

Notes:
- "distance kernel 0.01" means the kernel no longer appears as its own frame;
  the inlined f64-lane loops (`kernels::lane_sum`, `cosine_distance_with_qnorm`)
  are ~4.7% of backend samples (`perf report` on `attr_new_docs_main_1024`),
  ~0.4 µs/candidate, and are bucketed under "other".
- Heap fetch is larger under MAIN (3.1 vs 0.8 µs): each heap tuple is ~5 KB, so
  every candidate touches a different 8 KB page (and kernel page-cache work —
  `shmem_add_to_page_cache`, `do_user_addr_fault` show in the profile).
- The one remaining CBOR decode (~3.3 µs) is the candidate's own vector; the
  constant query is decoded once per expression (Fix B). Removing it needs the
  on-disk format change designed in `docs/design/RAW_VECTOR_VARLENA.md`
  (step 6).
- The query vectors here are client literals; a query vector fetched from a
  TOASTed table row (e.g. `(SELECT tv FROM t WHERE id = $1)`) still pays a
  TOAST fetch per call (Fix B follow-up, see `fix_b/RESULTS.md`).

## Not measured

- Graviton / aarch64 (the kernels vectorize to NEON per the Fix A review;
  not timed end-to-end).
- AVX2-only x86 end-to-end (the kernel speedup is portable SSE2-width; the
  turbovec scan share differs on hosts without the staged search).
- Concurrency / throughput (single connection only).
- IVF indexes, halfvec, ColBERT/MaxSim end-to-end (MaxSim uses the same
  kernels; not timed).
- 2-bit and 1-bit indexes.

## Raw data

`raw/results.jsonl` (48 latency rows: 2 arms × 3 rounds × 2 tables × 4
search_k), `raw/attr.jsonl` (8 attribution rows), `raw/env.txt` (md5s, PG
build flags, CPU, commits). Scripts in this directory.
