# Z6 — IVF build peak memory: reproduced small, cause NOT yet identified

Host: c7i.4xlarge (16 vCPU, 32 GiB), us-east-2b, PostgreSQL 16.15,
pg_turbovec **v2.10.0** built with `debug = 1`, `lto = false` (resolvable
frames), `shared_buffers = 2GB`, `maintenance_work_mem = 2GB`,
`max_parallel_maintenance_workers = 0` (serial — parallelism was already
ruled out in `z5_ram_20260922`).

## Reproduced at 1/5 the scale, so this is now cheap to iterate on

**2M × 1024-d, `lists = 1414`: peak RSS 15.26 GiB** against an accounted
model of ~2.44 GiB — a **~6× gap**. The build *completes* (17:35) and RSS
falls back to ~1 GiB afterwards, so this is a **peak-memory** defect, not a
leak. At 10M × 1024-d the same growth OOM-kills a 61 GiB host
(`z5_ram_20260922`).

Progression (serial, 30 s samples of `/proc/<pid>/smaps` anonymous mappings):

| t | largest block | note |
|---|---:|---|
| 30 s | 1.98 GiB | plus a **constant 1.38 GiB** block |
| 150 s | 3.53 GiB | |
| 270 s | 5.09 GiB | **appears twice** (5.09 + 5.09) |
| 330 s | 5.86 GiB | total spiked to **13.11 GiB** |
| 390 s | 6.66 GiB | **appears twice** (6.66 + 6.66) |

Two things this establishes:

1. **The constant 1.38 GiB block is the k-means reservoir**, confirmed
   exactly: `lists × 256 × dim × 4` = 1414 × 256 × 1024 × 4 B = 1.38 GiB.
   It is correctly **capped**. This match validates the measurement method.
2. **The growing block is periodically present twice**, i.e. a realloc that
   holds old + new simultaneously — which is what produces the 13.11 GiB
   total spike. Its rate is ~**3576 B/row** at 2M rows.

## Four hypotheses formed and DISPROVEN

Recording these so they are not re-tried:

1. **`Vec` doubling triples peak RSS.** Wrong. A standalone harness growing a
   `Vec<u8>` to 4.77 GiB via `extend_from_slice` reached capacity 8.00 GiB but
   **peak RSS 4.77 GiB** — Linux `mremap` grows large blocks in place without
   copying.
2. **The k-means reservoir.** Wrong — capped, and separately accounted (it is
   the 1.38 GiB constant block).
3. **`Vec<Vec<u32>>` assignments.** Wrong. Measured directly: 10M rows of
   one-element inner `Vec`s = **0.52 GiB**, not ~12.
4. **`idx.prepare()` duplicating the codes.** Wrong, and this one was worth
   testing because the reasoning was sound: `prepare()` calls `pack::repack`,
   allocating a full second copy of `packed_codes`, while the IVF write
   consumes only `packed_codes` / `scales` / `slot_to_id` / the TQ+ pair —
   and `blocked_codes()` / `n_blocks()` are used **nowhere** in the codebase,
   so at build time the duplicate is allocated and thrown away.
   **A/B measured: peak 15.26 GiB with `prepare()`, 15.26 GiB without** —
   byte-identical peak, and 5 min *slower* without it. Not the cause.
   (Still worth removing on its own merit later: it is provably dead work at
   build time. It is NOT a memory fix.)

## What remains

The ~3576 B/row growing allocation is **unexplained**. It is neither our
`build.rs` buffers (all bounded: `chunk_flat` caps at 1 GiB via
`compute_chunk_rows`, `block_rows` caps at 64 K rows) nor turbovec's
accumulating fields (`packed_codes` is 512 B/row at 4-bit; `scales`,
`slot_to_id`, `id_to_slot` are all ≲ 0.06 GiB at 2M).

**Next step, and it must be an allocator profile — not arithmetic.** I spent
this session's budget disproving four plausible theories by measurement,
which is the right outcome but not an answer. `heaptrack --pid` failed to
attach (it needs `gdb` plus a uid-matched FIFO; `ptrace_scope=0` and `gdb`
installed were not sufficient under the `postgres` uid). Options for next
time, in order:

1. Build with `jemalloc` + `MALLOC_CONF=prof:true,prof_leak:false` and dump
   `prof.heap` at peak — no attach needed, in-process.
2. `heaptrack -- <cmd>` on a **standalone** harness that calls
   `ivf_build_and_write` outside PostgreSQL, avoiding the attach problem.
3. `LD_PRELOAD` a tiny `malloc` wrapper logging allocations > 256 MB with
   `backtrace()`.

## Cost / cleanup

c7i.4xlarge, ~2 h ≈ $1.50. Instance terminated, SG and key pair deleted,
verified 0 live across all three run tags from today. Other tenants'
untagged instances untouched.

---

# Session 2 — a quantitative model, and two more theories falsified

## Harness error found first: `ps` RSS includes `shared_buffers`

Every peak in Session 1 (and the `z5_ram` OOM figures) is inflated by
`shared_buffers`, because `ps` RSS counts the shared memory segment. With
`shared_buffers = 2GB`, the 15.26 GiB figure is **13.30 GiB private**.
Measured directly via `/proc/<pid>/smaps` `Private_Dirty`: a 1M × 1024-d
build peaks at **8.37 GiB private** against 10.45 GiB RSS — exactly the 2 GiB
difference. *Report private, not RSS.*

## Theory 5 (k-means `cross` matrix) — FALSIFIED by a pre-registered test

`train_kmeans` allocates `cross = vec![0.0f32; n_sample * lists]`
(`ivf.rs:~1195`) where `n_sample = lists * 256`, so it is **quadratic in
`lists`**: 1.91 GiB at `lists=1414`, **9.54 GiB at `lists=3162`** (the 10M
run). That looked decisive.

**Pre-registered prediction:** halving `lists` should cut peak ~4×.

| `lists` | predicted train terms | **measured peak** |
|---:|---:|---:|
| 1414 | 4.67 GiB | 15.30 GiB |
| 707 | 1.86 GiB | **14.06 GiB** |
| 354 | 0.81 GiB | **14.07 GiB** |

**Wrong.** Peak is essentially *independent of `lists`*. The quadratic term is
real but not dominant. (It will still matter at large `lists` — 9.54 GiB at
3162 — so it is worth bounding, just not the headline.)

## What the falsification bought: a fitted model

Because peak is `lists`-independent, I swept the other two axes:

| n | dim | peak RSS | peak private |
|---:|---:|---:|---:|
| 2M | 1024 | 15.30 GiB | 13.30 GiB |
| 2M | 512 | 9.63 GiB | 7.63 GiB |
| 1M | 1024 | 10.45 GiB | **8.37 GiB** (direct) |

Solving across the points:

```
peak_private ≈ 1.27 × (n × dim × 4)  +  ~3.6 GiB fixed
```

fitting all three within 0.82 GiB. Reading it:

- **The `1.27 × n × dim × 4` term is the defect.** One full f32 corpus is
  `n × dim × 4`; the coefficient being **> 1** means *more than a whole
  uncompressed corpus* is resident — in a build whose entire design is to
  keep the corpus on a spill file and stream it in `maintenance_work_mem`-
  bounded chunks. `maintenance_work_mem` does not bound it.
- The ~3.6 GiB fixed part is mostly accounted: reservoir + its
  `sample.clone()` at `build.rs:950` = 2.76 GiB at `lists=1414`. **That clone
  is a second full copy of the reservoir held while the original is live** —
  a cheap, certain win regardless of what else is found.

## Where the per-row term is NOT

Checked and ruled out: all four `spill.read_block` call sites (1008 and 1430
are chunk-bounded; **1286 and 1613 do read the whole spill into
`vec![0.0f32; n_vectors * dim]`, but they are the BQ and graph paths, not
IVF**); `chunk_flat` (capped at 1 GiB by `compute_chunk_rows`); `block_rows`
(capped at 64 K rows); `packed_codes` (512 B/row at 4-bit, = 0.95 GiB at 2M);
`assignments` (0.10 GiB at 2M). The spill *file* is not in RSS — `BufFile`
uses `read`/`write`, not `mmap`.

So ~7.7 GiB of per-row anonymous memory at 2M × 1024-d remains unattributed.

## Two concrete, defensible fixes available now

Neither depends on finding the last term:

1. **Drop `sample.clone()`** (`build.rs:950`). `rotate_corpus_into` reads
   `src` and writes `sample`; a double-buffer swap or an in-place rotation
   removes a full reservoir copy — **1.38 GiB at `lists=1414`, 3.09 GiB at
   `lists=3162`.**
2. **Bound the `cross` matrix** (`ivf.rs:~1195`). Chunk the Lloyd assignment
   over sample rows instead of materialising `n_sample × lists` at once:
   **9.54 GiB → a bounded working set at `lists=3162`.** This is what makes
   large-`lists` builds fail specifically.

`prepare()` remains dead build work (Session 1: `blocked_codes()` /
`n_blocks()` used nowhere; A/B peak identical) — worth removing for the ~5 min
it costs, not as a memory fix.

## Cost / cleanup

Two instances this session (c7i.4xlarge ×2), ~3 h ≈ $3. All four of today's
instances terminated; security groups and key pairs deleted; verified none in
a billable state. Other tenants' untagged instances untouched.

---

# Session 3 — one fix shipped, one optimisation correctly abandoned

Both candidate fixes from Session 2 were implemented with a byte-identity
guard written **before** the optimisation was trusted. That ordering paid for
itself immediately.

## SHIPPED: bound the Lloyd `cross` matrix

Chunking `gemm_lloyd_assign` over sample rows under a fixed ~256 MiB budget,
replacing an `n_sample × lists` allocation that is quadratic in `lists`
(**9.54 GiB at `lists = 3162`**). Guard
`kmeans_cross_chunking_is_bit_identical` compares the whole-sample call
against chunk sizes 1 / 7 / 64 / 199 / 500 (including sizes that do not
divide evenly) and requires identical `assign` output — **passes on all legs**.

This is the term that specifically breaks *large-`lists`* builds. It does not
address the `1.27 × n × dim × 4` per-row term.

## ABANDONED, with the reason tested: blocked reservoir rotation

The plan was to rotate the reservoir in row-blocks through a small scratch
buffer, removing the `sample.clone()` (3.09 GiB at `lists = 3162`).

**CI failed it at `rows_per = 1`.** `rotate_corpus_into` is a GEMM with
`Parallelism::Rayon(0)`, so its internal tiling — and therefore its
floating-point reduction order — **depends on the row count `m`**. Blocking
is mathematically equivalent but not bit-equal, so it would have silently
changed index bytes on disk.

The existing `rotate_corpus_bit_identical_across_pool_sizes` does **not**
cover this: it varies the *thread count* at a fixed shape. Shape-invariance
is a different invariant and it does not hold. Now pinned by
`rotate_corpus_is_not_row_block_shape_invariant` (same shape twice =
bit-identical; `m=1` vs `m=300` = numerically close only, max abs diff
< 1e-4) so the optimisation is not re-attempted.

`sample.clone()` is still removed in the weaker sense — the destination is
allocated once and swapped rather than cloned — but two reservoir-sized
buffers remain live at the peak. **The honest reservoir win must come from
`ivf_sample_cap` (currently `lists × 256`), not from reshaping the GEMM.**

## Where Z6 stands

| term | size at 10M × 1024-d, `lists=3162` | status |
|---|---:|---|
| Lloyd `cross` matrix | 9.54 GiB | **fixed** (bounded to ~256 MiB) |
| reservoir + rotation destination | 3.09 GiB × 2 | open — needs a smaller `ivf_sample_cap` |
| `1.27 × n × dim × 4` per-row term | ~48 GiB | **open, unattributed** |
| `prepare()` blocked cache | 4.77 GiB | dead build work; A/B showed no peak change |

Tests: **444 passed / 0 failed / 8 ignored**, uniform across pg13–19 + classic.

**Not claimed:** that a 10M × 1024-d build now fits in 61 GiB. The dominant
per-row term is untouched, so it almost certainly does not. Re-measuring on
EC2 is the next step, and it is now cheaper because the 2M repro is
sufficient to see the model.

---

# Session 4 — LOCALIZED: 85 % of peak is inside `train_kmeans`

The breakthrough was abandoning external profilers and instrumenting the
project's **own** `trace_stage!` hooks with a `/proc/self/smaps_rollup`
`Private_Dirty` read. Attribution then comes from our timeline, not guesswork.

## Stage timeline (2M × 1024-d, `lists = 1414`, `build_parallelism = 16`)

| stage | wall | private at end |
|---|---:|---:|
| `1_train_kmeans` | 794.6 s | **10.91 GiB** |
| `2_assign_sweep` | 146.9 s | 10.96 GiB |
| `3_build_permutation` | 0.2 s | 10.92 GiB |
| `4_quantize_encode` | 18.0 s | 12.21 GiB |
| `5_prepare_and_persist` | 24.5 s | 12.88 GiB |

**85 % of the peak is already resident when `train_kmeans` ends — before any
corpus streaming happens.** The stages that touch the 2M-row corpus add only
~2 GiB combined. Every earlier hypothesis looked in the wrong place: the
corpus-streaming drain was never the problem.

Within `train_kmeans`, `1a_kmeanspp_seeding` is ~400 s and `1b_lloyd_loop`
393 s (25 iterations at ~15.7 s).

## Thread scaling: 4.08 GiB is per-thread GEMM buffers

| `turbovec.build_parallelism` | peak private |
|---:|---:|
| 16 | **16.24 GiB** |
| 1 | **12.16 GiB** |

A real, measured **4.08 GiB** (≈ 0.27 GiB/thread) comes from thread-local
allocations inside the `gemm` calls (`Parallelism::Rayon(0)` packing buffers).
That is a genuine, actionable finding: **`build_parallelism` is a memory knob,
not only a speed knob**, and it is not documented as such.

But **12.16 GiB remains single-threaded**, so threads are not the main story.

## Accounting, with the remaining gap

At `lists = 1414`, `train_kmeans`'s explicit allocations are:

| term | size |
|---|---:|
| reservoir (`lists × 256 × dim × 4`) | 1.38 GiB |
| rotation destination (the swap that replaced `sample.clone()`) | 1.38 GiB |
| `cross` (now bounded, Session 3) | 0.25 GiB |
| **accounted** | **3.01 GiB** |
| measured at 16 threads | 10.91 GiB |
| unaccounted | 7.90 GiB → **3.82 GiB after subtracting the thread term** |

So ~3.8 GiB of single-threaded, `lists`-scaled allocation inside
`train_kmeans` is still unexplained — but the search space is now one function
and two sub-stages instead of the whole build.

## Tooling that did NOT work (recorded to save the next attempt)

- **`LD_PRELOAD` malloc interposer** — wrote one, it compiled and ran, but
  Debian's `pg_ctlcluster` wrapper **rejects `LD_PRELOAD`** in
  `/etc/postgresql/16/main/environment` ("invalid line") and strips it from the
  systemd unit. Also: glibc routes large allocations through `mmap`, so a
  `malloc`-only hook would have missed them anyway — `mmap` must be
  interposed too.
- **`heaptrack --pid`** — needs `gdb` plus a uid-matched FIFO; fails under the
  `postgres` uid even with `ptrace_scope=0` (Session 1).
- **Env vars via systemd** — the Debian wrapper sanitizes them.
  `TURBOVEC_BUILD_TRACE` only reached backends when the postmaster was started
  through `pg_ctl` directly with `env`.
- **A self-matching guard** — `pgrep -f "CREATE INDEX"` matches the *sweep
  script's own cmdline*, so the first parallelism sweep waited forever and
  burned ~2 h. Guard on `pg_stat_activity`, not `pgrep`.

## Next step

Instrument inside `train_kmeans` itself: add `Private_Dirty` reads around
`1a_kmeanspp_seeding` and each Lloyd iteration. The remaining 3.8 GiB is
`lists`-scaled and single-threaded, which points at a per-iteration buffer
that is not being reused — cheap to find now that the stage is known.

## Cost / cleanup

c7i.4xlarge, ~7 h (three full builds plus a wasted sweep) ≈ $6. Instance
terminated, SG and key pair deleted; all five of this run's instances verified
absent from every billable state.

---

# Session 5 — CORRECTION: `train_kmeans` is fully accounted; the attribution was wrong

## The error in Session 4's conclusion

`trace_stage!` reports **cumulative process private memory, not a per-stage
delta.** So the 10.91 GiB shown at `1_train_kmeans` includes everything
allocated since `ambuild` began — the entire heap scan that wrote the spill
and filled the reservoir. Session 4 read that as "`train_kmeans` allocated
10.91 GiB". It does not follow: the memory may have been resident *before*
training ran.

A marker (`0_scan_end(entry)`) is now emitted at drain entry so the scan phase
is measured separately. Future timelines will not repeat this misreading.

## `train_kmeans` reproduced LOCALLY, free, and it is fully accounted

`train_kmeans` has **zero** `pgrx`/`pg_sys` references, so its memory
behaviour can be reproduced outside PostgreSQL. A standalone crate
(`gemm` 0.18.2, same call shapes, same dimensions) at `lists = 1414`,
`dim = 1024`, `n_sample = 361 984`:

| step | private |
|---|---:|
| baseline | 0.00 GiB |
| + reservoir + rotation destination | 1.39 GiB |
| + rotate GEMM (`m = n_sample`) | 2.77 GiB |
| + cross GEMM (bounded `m = 47 460`) | **3.02 GiB** |

That matches the arithmetic exactly (reservoir 1.38 + rotation dest 1.38 +
`cross` 0.25 = 3.01 GiB). Scaling is clean and linear: `lists = 707` gives
**1.64 GiB**, `lists = 1414` gives **3.02 GiB** — 2× for 2×.

**There is no hidden allocation inside `train_kmeans`.** The `gemm` internal
packing buffers were checked directly in `gemm-common-0.18.2`
(`packed_lhs_stride = kc * MR`, so the lhs prepack is `kc × m × 4` ≈
0.35–0.69 GiB) and are not significant at these shapes.

**Threads add nothing here either:** 1 thread 3.02 GiB vs 16 threads 3.03 GiB.
So the 4.08 GiB thread-scaled term measured on EC2 (16 → 1 threads: 16.24 →
12.16 GiB) is real but comes from somewhere else in the build — most likely
the `2_assign_sweep`'s per-chunk rayon closures, which allocate
`norm` + `rot` scratch per task.

## Corrected state of Z6

| claim | status |
|---|---|
| `cross` matrix is quadratic in `lists` (9.54 GiB at 3162) | **fixed**, bit-identity guarded |
| `build_parallelism` is a memory knob (~0.27 GiB/thread) | **measured on EC2**, GUC text corrected |
| "85 % of peak is allocated by `train_kmeans`" | **RETRACTED** — cumulative reading, not a delta |
| `train_kmeans` itself | **fully accounted at 3.02 GiB**, no hidden term |
| the dominant per-row term | **still open**, and now known NOT to be in `train_kmeans` |

## What the next session should do first

The scan phase (`ambuild_callback` per heap row) is the remaining suspect and
has never been measured in isolation. The new `0_scan_end(entry)` marker
answers it in one traced build. The local harness pattern is also worth
reusing: `train_kmeans` was settled in **minutes at zero cost** after four
sessions of EC2 work, because it turned out to be pure code.

## Cost

Session 5 used **no EC2** — the decisive measurement ran locally.

## Session 5 addendum — the assign-sweep is also ruled out (by arithmetic, in-source)

Checked the remaining candidates in `2_assign_sweep`, read from source rather
than measured, so treat as indicative:

- per-chunk `norm` + `rot` scratch: `ivf_par_chunk_rows` targets ~4 MiB
  (`(4 MiB)/(2·dim·4)`, clamped 256..4096), so 512 rows at dim 1024 = **4.0
  MiB/task**, ≈ 0.06 GiB across 16 threads.
- `batched_assign_soft`'s per-call `cross = n_rows × lists × 4`: **2.8
  MiB/thread** at `lists = 1414`, 6.2 MiB at 3162 — ≈ 0.04–0.10 GiB across 16
  threads. (Note this one is *unbounded* in principle, unlike the Lloyd
  `cross` fixed in Session 3, but at the real chunk size it is small.)

Neither explains the 4.08 GiB thread-scaled term, so that term is also not in
the assign sweep. **By elimination the remaining memory is in the scan phase
(`ambuild_callback`), which has never been measured in isolation** — exactly
what the new `0_scan_end(entry)` marker exists to answer.

### Precise handoff for the next session

One traced build answers it:

```
sudo systemctl stop postgresql@16-main
sudo -u postgres env TURBOVEC_BUILD_TRACE=1 \
  /usr/lib/postgresql/16/bin/pg_ctl -D /var/lib/postgresql/16/main \
  -o "-c config_file=/etc/postgresql/16/main/postgresql.conf" \
  -l /tmp/pg_trace.log start          # env vars only survive via pg_ctl
```

then build 2M × 1024-d with `lists = 1414` and read
`sudo grep "build trace" /tmp/pg_trace.log`. The first line is now
`0_scan_end(entry)`:

- if it already shows ~10 GiB → the scan phase is the defect, and the spill's
  purpose is being defeated somewhere in `ambuild_callback`;
- if it shows ~3 GiB → the growth is between scan end and `1_train_kmeans`,
  which is a ~40-line window.

Either way it is one build (~17 min, ~$0.30), not a session.
