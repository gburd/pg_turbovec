
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
