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
