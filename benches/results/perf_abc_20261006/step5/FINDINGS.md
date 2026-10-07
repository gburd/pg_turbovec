# Step 5: end-to-end A/B of Fix A + Fix B (2026-10-07)

**Result: Fix A (f64-lane kernels) + Fix B (decode the constant query once,
reuse its norm) cut ~3.1–3.5 µs per rechecked candidate, giving 1.26× (default
storage) and 1.43× (`STORAGE MAIN`) at `search_k = 1024` on 500k × 1024-d real
embeddings with warm shared_buffers, recall@10 1.000 in both arms (cold
shared_buffers: 1.24× / 1.37×, same per-candidate saving). The remaining
per-candidate cost is TOAST (~5.9 µs warm, default storage only) and the ONE
remaining CBOR decode of each candidate (~3 µs), which is the step-6 format
work.**

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
  untouched): both arms read the same index relfiles, built once by
  `tvp_ab_setup.sh` under the old arm.
- Arms (same PG install, `.so` swapped, postmaster restarted):
  - **old** = v2.11.0 source (built from `main` @ 8939b89, whose `src/`,
    `Cargo.toml`, `Cargo.lock` are byte-identical to the v2.11.0 tag),
    `old.so` md5 `507c3104…`. The intended worktree `/work/wt-ab-old` failed to
    create (`raw/ab_setup.log` line 16), so the build ran in `/work/pg_turbovec`
    at `main`; that commit's `src/` is identical to `perf/recheck-abc`'s;
  - **new** = `perf/integrate-ab` @ 59cee0d (Fix A 7377ed5 + Fix B 614a6d1 +
    the norm seam), `new.so` md5 `3e3d5d96…`.
- Method (`tvp_ab.py`, `tvp_arm.sh`, `tvp_ab_all.sh`): arms alternated
  old, new × 3 rounds; per (arm, table, search_k): one psql session, 20 warm-up
  queries, then 200 queries timed by top-level `EXPLAIN (ANALYZE, TIMING OFF)`
  Execution Time; recall@10 from the same queries' returned ids.
  `oversample = 1.0`, `hi_dim_rerank = off`. Max load average during runs 3.6.
- **Round 1 is excluded.** The old arm ran first after setup and read from a
  cold OS page cache (old docs_ext search_k=32: 32.6 ms vs 1.93 ms in rounds
  2–3; p95 up to 470 ms). The new arm's round 1 ran after it and was already
  warm (within 6% of its rounds 2–3). Since only the old arm was cold,
  including round 1 would favour new; with it (median of 3) every speedup moves
  by ≤0.05× and no conclusion changes. Rounds 2 and 3 agree within 4% in every
  cell except old docs_main search_k=32 (1.52 vs 1.65 ms, 8.8%), where the old
  and new ranges overlap: that 1.05× is not resolved.
- **shared_buffers is cold in the timed runs.** The postmaster restarts per
  arm, and the `count(*)` warm-up only loads docs_ext's 29 MB heap into
  shared_buffers: docs_main's 3.9 GB heap exceeds shared_buffers/4, so its seq
  scan uses a ring buffer, and `count(*)` never reads TOAST. The search_k
  passes nest and each query's timed EXPLAIN runs before its untimed recall
  query, so most candidates are first touched inside the timed run (an OS-cache
  → shared_buffers copy plus a first-touch shmem fault). Both arms pay this
  identically, so the A/B deltas are unaffected and the speedup ratios are
  conservative. The absolute TOAST and heap-fetch costs, and the MAIN-vs-default
  comparison, are upper bounds for a server with warm shared_buffers. A re-run
  with `pg_prewarm` is in the "Warm shared_buffers" section below.
- Build: `cargo pgrx install --release` with rustup 1.98.0, no extra RUSTFLAGS
  (exact commands in `raw/build_cmd.txt`). The Rust `.so` therefore has no
  frame pointers (see † below). Code layout is not controlled (unlike
  `fix_b/build_ab64.sh`), so the ≤6% gains at small search_k may include layout
  noise.

## Warm shared_buffers (primary result)

Re-run of the same A/B with `shared_buffers = 12GB` and, after each per-arm
restart, `pg_prewarm` of both heaps, docs_ext's TOAST relation + its index, and
both turbovec indexes (7.26 GB total; `tvp_arm_warm.sh`, `raw/ab_warm.log`
shows the block counts). Same arms, same queries, 3 alternated rounds, all
rounds used (no cold first pass; round-to-round spread ≤ 2% except new docs_ext
search_k=100 at 7%). `raw/results_warm.jsonl`, `raw/attr_warm.jsonl`.

| table | search_k | v2.11.0 | A+B | speedup | ms saved | µs saved / candidate | R@10 old/new |
|---|---|---|---|---|---|---|---|
| docs_ext | 32 | 1.51 | 1.34 | 1.13× | 0.18 | 5.5 | 1.000/1.000 |
| docs_ext | 100 | 2.86 | 2.56 | 1.12× | 0.30 | 3.0 | 1.000/1.000 |
| docs_ext | 256 | 5.10 | 4.20 | 1.22× | 0.90 | 3.53 | 1.000/1.000 |
| docs_ext | 1024 | 15.66 | 12.45 | **1.26×** | 3.22 | 3.14 | 1.000/1.000 |
| docs_main | 32 | 1.29 | 1.19 | 1.08× | 0.10 | 3.1 | 1.000/1.000 |
| docs_main | 100 | 2.31 | 1.99 | 1.16× | 0.32 | 3.2 | 1.000/1.000 |
| docs_main | 256 | 3.76 | 2.90 | 1.30× | 0.86 | 3.36 | 1.000/1.000 |
| docs_main | 1024 | 11.14 | 7.79 | **1.43×** | 3.34 | 3.26 | 1.000/1.000 |

At search_k ≤ 100 the absolute saving is 0.1–0.3 ms, so its µs/candidate
figure is dominated by per-query noise; read the per-candidate saving from the
search_k ≥ 256 rows: **~3.1–3.5 µs**, the same as the cold run.

Per-candidate backend on-CPU at search_k = 1024, warm (`raw/attr_warm.jsonl`,
same method and † caveat as below):

| component | docs_ext old | docs_ext new | docs_main old | docs_main new | owner |
|---|---|---|---|---|---|
| TOAST fetch | 5.91 | 5.87 | 0 | 0 | ours (storage) |
| CBOR decode | 5.09 | **3.01** | 5.15 | **3.12** | ours (format) |
| distance kernel | 1.97 | **~0.5†** | 1.87 | **~0.5†** | ours |
| heap fetch | 0.70 | 0.75 | 1.62 | 1.59 | core |
| reorder queue | 0.04 | 0.04 | 0.14 | 0.14 | core |
| executor misc | 0.17 | 0.16 | 0.29 | 0.34 | core |
| turbovec scan, backend thread | 0.08 | 0.06 | 0.11 | 0.10 | ours |
| other (incl. inlined kernel, †) | 1.32 | 1.85 | 1.16 | 1.69 | — |
| **total** | **15.27** | **11.74** | **10.33** | **6.99** | |

With warm buffers PostgreSQL core is ~0.9 µs (default storage) to ~2.1 µs
(MAIN) per candidate; TOAST is 5.9 µs (not 7.8); MAIN's heap fetch is 1.6 µs
(not 3.1). What remains on the new arm is TOAST (default storage only) and the
one candidate-side CBOR decode (~3 µs).

`STORAGE MAIN` + A+B, warm: 15.66 → 7.79 ms at search_k = 1024 (2.01×; MAIN
alone 1.41×, A+B alone 1.43×; MAIN costs a 1.39× larger heap — see Fix C).

## Cold shared_buffers run (first pass)

The first A/B below ran without prewarm; its deltas agree with the warm run,
its absolute costs are upper bounds (see Method).

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

The saving is 3.4–3.7 µs per candidate for docs_ext at every search_k and
3.55–4.0 µs for docs_main at search_k ≥ 256. At docs_main search_k = 32/100 it
is 2.3/2.8 µs, but there the absolute differences (0.07/0.28 ms) are close to
round-to-round noise (old docs_main search_k=32 rounds differ by 0.13 ms). This
is consistent with removing one CBOR decode (−2.2 µs in the profile) and
shrinking the distance kernel from ~2.1 to ~0.5 µs (†). At small `search_k` the fixed per-query
cost dominates and the relative gain is small. Combining the code change with a schema change, `STORAGE MAIN` + A+B take
search_k=1024 from 18.96 ms (v2.11.0, default storage) to **9.81 ms (1.93×)**.
Of that, MAIN alone is 1.41× (18.96 → 13.44 ms, both on v2.11.0) and A+B alone
is 1.37×. MAIN is a DDL change with a cost (a 1.39× larger heap: 3,906 vs 2,819
MB), and both MAIN numbers are cold-shared_buffers (see Method). See the Fix C
guidance (`docs/PRODUCTION.md` § Column storage) before applying it.

## Where the per-candidate time goes now (backend on-CPU, µs/candidate)

`tvp_attr.py`: `perf -e task-clock -c 100000` on the backend thread only (each
sample = 100 µs of on-CPU time); each stack goes to the first bucket, in the
table's row order, that has a matching frame anywhere in it (TOAST and CBOR are
tested before the distance function because both run underneath it); 150
queries at `search_k = 1024` (`raw/attr.jsonl`).

| component | docs_ext old | docs_ext new | docs_main old | docs_main new | owner |
|---|---|---|---|---|---|
| TOAST fetch | 7.90 | 7.81 | 0 | 0 | ours (storage) |
| CBOR decode | 5.62 | **3.37** | 5.49 | **3.29** | ours (format) |
| distance kernel | 2.11 | **~0.5†** | 2.06 | **~0.5†** | ours |
| heap fetch | 0.93 | 0.82 | 3.26 | 3.14 | core |
| reorder queue | 0.04 | 0.04 | 0.17 | 0.14 | core |
| executor misc | 0.20 | 0.15 | 0.44 | 0.43 | core |
| turbovec scan, backend thread | 0.07 | 0.07 | 0.12 | 0.14 | ours |
| other (parse/plan/libpq/unattributed) | 1.56 | 2.12 | 1.37 | 1.96 | — |
| **total** | **18.42** | **14.39** | **12.92** | **9.09** | |

Notes:
- † The bucket reads 0.01: the Rust `.so` is built without frame pointers, so
  `--call-graph=fp` loses the parent frames of Rust leaves, and the kernel's
  samples (`kernels::lane_sum`, `cosine_distance_with_qnorm`, ~4.7% of backend
  samples on docs_main ≈ 0.4 µs) land in "other", which rose 0.56–0.59 µs. For
  the same reason the CBOR figure is a lower bound and "other" includes some
  Rust-side time. CBOR + distance + other together: 9.29 → 5.50 µs (docs_ext),
  8.92 → 5.26 µs (docs_main), i.e. −3.8 / −3.7 µs, against −3.60 / −3.55 µs
  from latency.
- Heap fetch is larger under MAIN (3.1 vs 0.8 µs): each ~5 KB tuple sits on its
  own 8 KB page, and docs_main's heap was not in shared_buffers in this run (see
  Method), so most candidates' page was copied in from the OS page cache during
  the timed query. `shmem_add_to_page_cache` / `do_user_addr_fault` in the
  profile are consistent with first-touch faults on shared_buffers after the
  per-arm restart. docs_ext's TOAST was not preloaded either, so its 7.8 µs
  carries the same overhead.
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
- `<->` (L2) and `<#>` (inner product): Fix A changes those kernels too, but
  only `<=>` (cosine, the only operator using Fix B's cached norm) was timed.
- Parameterised queries (`$1`, prepared statements / generic plans): the query
  is always a literal here.
- Dimensions other than 1024; corpora other than Cohere wiki-en; PostgreSQL
  versions other than 16.
- Code-layout control (see Method).
- Distance precision: recall@10 is 1.000 in every cell for both arms, so it
  cannot detect small ordering changes from the f64 lanes (covered by Fix A's
  unit tests, not here).

## Raw data

`raw/results.jsonl` (48 latency rows: 2 arms × 3 rounds × 2 tables × 4
search_k), `raw/attr.jsonl` (8 attribution rows), `raw/env.txt` (md5s, PG
build flags, CPU, commits). Scripts in this directory.
