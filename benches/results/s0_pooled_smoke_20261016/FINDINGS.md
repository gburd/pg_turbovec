# S0 pooled-benchmark harness + v2.13.0 smoke baseline

> **Collapsed-single-host DIAGNOSTIC, not a headline.** Per
> `benches/METHODOLOGY.md`, the headline requires the three-tier distributed
> rig (DB on metal + pooler on its own instance + spot load fleet). This run
> is one 8-vCPU EC2 host doing DB + PgBouncer + driver together, on a small
> corpus with a deliberately small `shared_buffers` so the storage path isn't
> fully hidden. Its purpose: prove the harness produces the right metrics and
> get a provisional *before* number for Phase S1/S2. The S0 worker aborted
> before writing this file; it is reconstructed from the recovered host
> artefacts in `host/` and the result JSONs alongside.

## Setup (confirms Debian is a viable build/test platform)

- Host: EC2 `c7i.2xlarge`, Debian 12, 8 vCPU (Intel Xeon Platinum 8488C,
  AVX-512 VBMI/VNNI), 15 GiB RAM, 100 GB gp3.
- Toolchain: rust stable 1.99, **clang/LLVM 14** (much older than the
  project's usual 21/22), cargo-pgrx 0.19.3, PostgreSQL 18.6 (pgrx-managed).
- **pg_turbovec v2.13.0 built and installed cleanly on LLVM 14** — the
  turbovec fork's `build.rs` LLVM<22 inline-asm path (carry #5) handles the
  `vpdpbusd` intrinsic, so no AVX-512-VNNI build break. Debian + LLVM 14 works
  for build/test; recipe in `host/pgrx_install.log`.
- Corpus: 500,000 × 768-d, cosine, flat (lists=0, bit_width=4), k=10.
- PgBouncer transaction pooling in front of the DB; `shared_buffers=256MB`.

## Headline S0 datum: the per-backend memory wall (SCALE_PLAN bottleneck #1), reproduced

This is the most important result and it confirms the plan's central thesis
directly. `host/rss_curve.log` +
`smoke_flat_20261016T160430Z_rss_curve.json`, ramping persistent backends and
sampling each backend's `/proc/<pid>/status` VmRSS:

| backends | total RSS | mean RSS/backend | frac of 12.9 GB safe ceiling |
|--------:|----------:|-----------------:|-----:|
| 1  | 0.57 GB | 570 MB | 0.04 |
| 4  | 2.45 GB | 614 MB | 0.19 |
| 8  | 4.92 GB | 615 MB | 0.38 |
| 12 | 7.37 GB | 614 MB | 0.57 |
| 16 | 9.79 GB | 612 MB | 0.76 |
| 20 | 12.24 GB | 612 MB | 0.95 |
| 24 | 14.67 GB | 611 MB | **1.14 — over host RAM** |

**Each backend holds ~610 MB of PRIVATE decoded index cache.** Total RSS is
linear in backend count; **~24 backends crosses the 15 GB host and the
kernel OOM-killer takes a backend / crashes the cluster** (observed earlier at
64 clients before the ceiling was found; the single-client `sp=auto` sweep
cell recorded a `backend_crash_or_oom` for the same reason).

**The decisive comparison — `sp=1` does NOT fix the memory wall:**

| scan_parallelism | mean RSS/backend | OOM/ceiling at |
|---|---:|---:|
| auto (multi-thread default) | 614 MB | ~24 backends |
| 1 (one core/query) | 605 MB | ~24 backends |

`sp=1` cuts the per-backend RSS by only **~1.5%** and does not move the OOM
ceiling. **Threads are not the memory wall; the private decoded cache is.**
This confirms, with measurement, SCALE_PLAN's claim that **Phase S1 alone
cannot lift the memory wall — only Phase S2 (one shared decoded index per
relation) can.** It is also the concrete *before* number: 500k×768×4-bit →
~610 MB/backend; extrapolated to a TB corpus × hundreds of pooled backends,
this is the 0.5–5 TB duplication SCALE_PLAN names, and the reason a pooler
hits OOM long before QPS matters.

## Cold vs warm (per-backend cache build on first query)

Cold first query ≈ **327 ms** (builds the per-backend decoded cache from
`shared_buffers`) vs ≈ **63 ms** warm — consistent with the 250–500 ms
new-connection cost in the pg_vectorsearch comparison. For a pooler that
recycles backends (`server_lifetime`), every recycled backend pays this once.

## Throughput sweep (provisional S1 before-number; collapsed-host diagnostic)

Closed-loop, through PgBouncer, 500k×768 flat, this 8-vCPU box
(`host/agg_spauto.md`, `host/agg_sp1.md`). The box saturates by ~8–16 clients
(a 500k flat scan is CPU-heavy and 8 cores fill fast):

| scan_parallelism | peak sustained QPS | QPS/core | CPU-ms/q | p99 @ peak | p99.9 @ peak |
|---|---:|---:|---:|---:|---:|
| auto | 97 (@16 clients) | 12 | 81.8 | 246.6 ms | 271.3 ms |
| 1    | 100 (@16 clients) | 13 | 79.8 | 261.7 ms | 291.1 ms |

Open-loop fixed-arrival (tail under queueing) saturates at ~100 QPS for both;
beyond that the box is overloaded and p50 climbs into seconds (expected on a
collapsed 8-vCPU host — this is why the real rig needs the DB on its own
metal box). At this small scale `sp=auto` and `sp=1` are within noise on
throughput — the sp difference shows up at higher core counts (per the
v2.13.0 qualification: `sp=1` nearly doubled 10M arm c=96 TPS); an 8-core box
can't exhibit that, which is itself a reason the headline needs the real rig.

## The harness (the reusable deliverable)

`benches/scripts/pooled/` — general, parameterised, usable on the real
metal+pooler+fleet rig (NOT hardcoded to this box):

- `provision.md` — three-tier EC2 launch (DB metal on-demand, pooler on-demand,
  N spot load instances), SG scoped to operator /32 + intra-rig, tag `run=`,
  teardown-by-tag.
- `setup_db.sh` — PG tuning (huge-pages note, shared_buffers,
  effective_cache_size, max_connections, wal_level=replica), create extension,
  load corpus, build index.
- `setup_pooler.sh` — PgBouncer transaction pooling, parameterised pool size.
- `run_load.sh` — closed-loop (max QPS) + open-loop (`-R` fixed arrival)
  driver through the pooler; captures QPS, QPS/core, p50/p95/p99/p99.9,
  backend CPU; **crash-tolerant** (records a failure row on backend OOM instead
  of aborting the sweep).
- `rss_curve.sh` — the per-backend RSS-vs-backend-count curve + OOM ceiling
  (the datum above).
- `fresh_backend_cost.sh` — first-query cost on a recycled pooled backend.
- `aggregate.py` — folds per-driver JSON into the headline markdown table.
- `run_smoke.sh`, `common.sh`, `README.md` — orchestration + the LOUD
  collapsed-host-is-a-diagnostic warning.

## What the harness still needs for the real rig

- Multi-driver fan-out + clock-sync aggregation across the spot load fleet
  (`aggregate.py` merges, but the fleet-launch + staggered-start orchestration
  in `provision.md` is documented, not yet scripted end-to-end).
- ZFS + ZIL/SLOG + L2ARC storage setup on the metal DB host (methodology's
  target storage; `setup_db.sh` has the PG tuning but not the ZFS pool build).
- A matched-recall calibration pass for competitor comparisons (carried from
  `VS_BENCH_PLAN.md`; not needed for our own before/after).

## Caveats / provenance

- Collapsed single host; DB+pooler+driver share 8 cores — throughput numbers
  are a harness-validation diagnostic, NOT a published result.
- The `sp=auto` single-client QPS=0 cell is a recorded `backend_crash_or_oom`
  from the pre-ceiling instability, not a measurement of 0 QPS.
- Raw artefacts: `host/` (logs, aggregates, env, pgbouncer config) +
  `smoke_flat_*_local.json` (sweeps) + `*_rss_curve.json` (the RSS curve).
