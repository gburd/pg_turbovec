# Z5 >RAM arm — BLOCKED by an IVF build-memory defect (the real finding)

Host: c7i.8xlarge (32 vCPU, AVX-512, **61 GiB**), us-east-2c, Ubuntu 24.04,
PostgreSQL 16.15, pg_turbovec **v2.10.0** (release tag, `f905181`),
`shared_buffers = 2GB` (deliberately small, to force a >RAM scan).

Goal: measure Z5's bounded-delta win where a full scan is **disk I/O** rather
than a RAM sweep — the regime `z5_delta_20260922` could not reach (139 MB
index, fully resident, so a full scan cost only 1.6× a 1-cell scan).

Corpus: **10M × 1024-d** (56 GB heap), clustered, loaded in 20 parallel
workers. 4-bit codes ≈ 5.1 GB vs a 2 GB buffer pool — the intended regime.

## The index could not be built. `CREATE INDEX` OOM-kills at 10M × 1024-d.

**Attempt 1** — `maintenance_work_mem = 8GB`, `max_parallel_maintenance_workers = 16`:

```
Out of memory: Killed process 16209 (postgres)
  total-vm:68527108kB  anon-rss:61311596kB  shmem-rss:2134156kB
  Failed process was running:
    CREATE INDEX c10_ivf ON c10 USING turbovec (emb vec_cosine_ops)
      WITH (bit_width=4, lists=3162);
```

61 GB of anonymous RSS against a declared 8 GB budget.

**Attempt 2** — serial (`max_parallel_maintenance_workers = 0`),
`maintenance_work_mem = 4GB`. **Parallelism was not the cause.** RSS grew
monotonically, ~0.8 GB per 45 s, with no plateau:

| elapsed | RSS |
|---|---:|
| 2 min | 4.5 GiB |
| 10 min | 11.8 GiB |
| 20 min | 19.5 GiB |
| 24 min | 22.6 GiB |
| 31 min | **27.6 GiB** (terminated before a second OOM) |

Measured composition at 25.8 GiB RSS (`/proc/<pid>/smaps`, anonymous
mappings > 500 MB):

- **20.7 GiB** — one contiguous Rust-side allocation, still growing
- **3.09 GiB** — exactly the k-means reservoir (`lists × 256 × dim × 4` =
  3162 × 256 × 1024 × 4 B = 3.09 GiB), correctly **capped**
- `pg_backend_memory_contexts` showed **nothing** over 100 MB, confirming the
  growth is Rust-side, not a PostgreSQL context

The spill is working: `pgsql_tmp` reached **16 GB**, so the *scan* phase is
bounded as designed. The growth is in the **drain** (`ivf_build_and_write`).

## What I could and could not establish

**Established:** an IVF build at 10M × 1024-d needs > 61 GB and its memory
use is *linear in rows*, not bounded by `maintenance_work_mem`. Documented
behaviour ("out-of-core end-to-end since v1.13.0", `maintenance_work_mem`-
bounded chunks) does **not** hold at this scale.

**Not established — I ran out of evidence before the answer.** I formed and
discarded three hypotheses by arithmetic (capped reservoir: wrong, it is
capped; `Vec<Vec<u32>>` assignments: wrong, ≲1 GB even with allocator
overhead; a full-corpus f32 array: the two that exist, `build.rs:1284` and
`:1611`, are the **BQ** and **graph** paths, not IVF). The 20.7 GiB
contiguous block is **unexplained**. Candidates for the next session, in
order: the `Vec<Vec<Vec<u32>>>` per-block assignment collect
(`build.rs:~1041`, three levels of nesting inside the bounded loop), and
`build_permutation_soft`'s output. This needs a heap profiler
(`jemalloc` stats or `heaptrack`), not more arithmetic.

## Consequences

1. **The Z5 >RAM measurement remains unmeasured.** The v2.10.0 claim stands
   as published: 11 % in-memory / 22 % OOC at 1M × 256-d, with the >RAM
   regime explicitly untested. No published number changes.
2. **A separate, arguably more serious defect exists:** IVF builds are not
   memory-bounded at 10M × 1024-d. This project targets ">1.7M in production"
   and "1-trillion scale via partitioning" — a 10M-row partition that cannot
   be indexed on a 61 GB host is a real ceiling. `docs/BQ_RECALL_BENCH`
   § 0.6e already noted a "20.3 GB OOM-killed build" with unbounded `mwm` at
   1M; this shows the same failure at 10M **with** `mwm` set.
3. The `shared_preload_libraries` lesson from the previous run was applied
   (20 GUCs verified registered before any measurement).

## Cost / cleanup

c7i.8xlarge, ~3.5 h ≈ $5. Instance terminated, security group and key pair
deleted, verified. Other tenants' untagged instances untouched.
