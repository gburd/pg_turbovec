# Benchmark methodology — the workload we actually serve

> **This is the canonical methodology. Every published benchmark — ours or a
> competitor comparison — MUST lead with the target-market workload defined
> here. A single-client warm-p50 number is a *diagnostic*, never the
> headline.** Drift-check enforces that this file exists and that
> `docs/BENCHMARKS.md` links it.

## The target market, stated plainly

pg_turbovec's goal is to be the best vector-search extension at **TB–PB
scale**: a very large similarity-search index hit **constantly by thousands
of clients** through a **connection pooler** (PgBouncer/pgcat/Odyssey) that
keeps the **maximum number of backends** the host can sustain busy, on a host
**tuned for exactly that** — huge pages, large `shared_buffers`, and tuned
storage (e.g. ZFS with a separate ZIL + L2ARC on NVMe), and competitive at
every smaller scale too.

A benchmark that does not resemble that workload does not tell us whether we
are winning or losing for the people we are building this for. The
2026-10-14 pg_vectorsearch comparison measured single-client warm p50 as its
headline and only reached concurrency as one phase — and that is exactly the
axis on which we lost worst (15–100× fewer QPS at 96 clients). We will not
publish single-client latency as a headline again.

## Headline metrics (in priority order)

1. **Sustained QPS and QPS/core under a connection pooler**, at the backend
   count the host can actually sustain (not an arbitrary 16/96). Sweep client
   concurrency up to and past the saturation point; report the knee and the
   peak.
2. **Tail latency under that load** — p95 and **p99/p99.9**, not p50. At the
   target scale the SLO is a tail promise to thousands of clients, not a
   median.
3. **QPS/core** — the fair cross-system number when one system threads a query
   and another does not. Always reported beside raw QPS, with the per-query
   CPU-ms it implies.
4. **Memory per backend × backend count** — the aggregate RSS a pooler at
   full backend count actually consumes, since any per-backend cache is paid
   N times. A per-query latency that needs a 5 GB per-backend cache does not
   scale to 2,000 backends; say so.
5. **Larger-than-RAM behaviour** — the index must not fit in RAM at the
   headline scale. Measure with the working set deliberately exceeding
   `shared_buffers` + page cache, so the storage path (ZFS/ZIL/L2ARC/NVMe) is
   exercised, not hidden.

Single-client warm p50 stays in the report as a **diagnostic row** so we can
attribute per-query cost, but it is never the lead number.

## Benchmark rig (topology) — three tiers, not one host

The workload is distributed in production, so the rig is distributed. A
same-host `pgbench` hides the pooler, the network, and the real backend-count
ceiling — exactly the axes we lose on. Every headline run uses three tiers in
one VPC/subnet (low, measured inter-AZ or same-AZ latency; record it):

```
  [ load fleet ]            [ pooler ]              [ database ]
  N small spot instances → 1 pooler instance   → 1 metal instance
  (pgbench / load gen)     (PgBouncer/pgcat/      (PostgreSQL + pg_turbovec,
   thousands of logical     Odyssey, transaction   tuned: huge pages, big
   clients, open+closed     pooling)               shared_buffers, ZFS +
   loop)                                           ZIL/SLOG + L2ARC on NVMe)
```

1. **Database tier — one bare-metal instance.** PostgreSQL + pg_turbovec,
   tuned per *Host tuning* below. Metal (not a VM) so NUMA, huge pages, and
   local-NVMe ZFS are real, not virtualised. This is the system under test;
   nothing else runs on it.
2. **Pooler tier — its own instance.** The connection pooler (PgBouncer,
   pgcat, pgpool, or Odyssey — record which + version + config) in
   **transaction** pooling, between the load fleet and the database. Isolating
   it on its own host means its CPU and its connection multiplexing are
   measured, not stolen from either the DB or the drivers, and the DB sees
   exactly the pooler's `default_pool_size` backends. Record `pool_mode`,
   `default_pool_size`, `max_client_conn`, `server_lifetime`.
3. **Load tier — many small spot instances.** The driver fleet. Thousands of
   logical clients must come from enough network sources that no single
   driver host is CPU- or NIC-bound (a single pgbench box caps out and
   silently throttles the DB — then you measure the driver, not the server).
   Spot is fine here (the load gen is stateless and restartable); only the DB
   and pooler need on-demand stability. Aggregate the per-driver results into
   one QPS/latency series (synchronise clocks; stagger start; discard warmup).

Record the instance types, counts, AZ placement, and the
client↔pooler↔db RTTs. A result from a collapsed topology (driver on the DB
host, or no pooler) is a *diagnostic*, labelled as such — not a headline.

## Required workload shape

- **Pooler in front, transaction pooling, on its own tier.** Drive load
  through the pooler instance (see *Benchmark rig* above) in `transaction`
  pooling at a pool size matched to the host's sustainable backend count.
  Short-lived logical clients, long-lived backends — the real pattern. Report
  pool size, backend count, and client count separately.
- **Thousands of logical clients from the load fleet.** The fleet (pgbench
  `-c` across many small instances, or a closed/open-loop load gen) must
  sweep concurrency well past saturation and must not itself be the
  bottleneck — verify each driver host is below CPU/NIC saturation, or the
  number measured is the driver's, not the server's. Include an **open-loop** arrival run (`pgbench -R <rate>` or a Poisson
  driver) at a fixed target rate to expose tail latency under queueing, in
  addition to closed-loop max-QPS.
- **A fresh-backend cost row.** Because our per-backend cache is built on
  first query, measure the pooler-realistic case where a backend is recycled
  (`server_lifetime` / `max_db_connections` churn) and pays that cost. Report
  the first-query penalty and how often a pooled workload pays it.
- **Working set > RAM at the headline scale.** Size the corpus so the index +
  heap exceed `shared_buffers` + OS cache on the bench host. If you benchmark
  a RAM-resident corpus, label it a *small-scale* row, not the headline.

## Host tuning (record all of it; it is part of the result)

Tune the bench host the way an operator at this scale tunes it, and record
every knob in the report (a result is not reproducible otherwise):

- **Huge pages** backing `shared_buffers`; `shared_buffers` large (tens of GB
  to hundreds), `effective_cache_size`, `work_mem`, `max_connections` set for
  the backend count, `max_parallel_workers*` as configured.
- **Storage:** the real target is ZFS with a separate **ZIL (SLOG)** and
  **L2ARC** on NVMe; record `recordsize`, `compression`, `atime`, `logbias`,
  ARC size, and the ZIL/L2ARC devices. A plain-NVMe/XFS run is acceptable as
  a *substrate* row but must be labelled as not the target storage.
- **CPU:** governor, THP mode, NUMA layout, and SIMD level
  (AVX2 vs AVX-512 VBMI/VNNI vs SVE2) — SIMD level changes our numbers by
  ~1000×, so it is a headline caveat, never a footnote.
- **`turbovec.scan_parallelism`:** state it explicitly for every row. At
  pooled high-concurrency the right setting is `1` (one core/query, maximum
  QPS/core); the multi-thread default is a single-client-latency setting. Show
  both when they differ (the equal-CPU row).

## Fairness rules (unchanged, still mandatory)

Carried from `.agent/notes/VS_BENCH_PLAN.md`; a hostile competitor reviewer
must not be able to call a comparison unfair:

- Each system at its own documented recommended settings; matched-recall
  Pareto with operating points chosen in a **discarded calibration pass**.
- Equal query counts and an **equal-CPU row** (`scan_parallelism = 1` for us
  vs a 1-core competitor).
- Verify every cell uses its index (EXPLAIN) and recompute distances to check
  correctness; exact fp64 ground truth for recall.
- Declare round-exclusion in advance (round 0 discarded + ≥2 ABBA rounds).
- **Conflict of interest** stated in any competitor comparison; publish the
  harness, raw per-query data, and exact commits; invite the competitor to
  review their own configs before any wider claim.
- An explicit **"not measured"** list. Never present an estimate as a
  measurement.

## What this changes about our harness

`benches/scripts/concurrent.sh` and `benches/concurrent_knn.rs` measure
persistent-backend aggregate QPS / in-process mutex contention on a small
RAM-resident corpus on one host. They are kept as **regression guards**, not
as the target-market benchmark — their own headers say so. The headline run
needs the three-tier distributed rig above: a >RAM corpus, a pooler on its
own instance, a load fleet of many small instances, open-loop tail
measurement, and the tuned storage.

The headline harness lives under `benches/scripts/pooled/` and orchestrates
all three tiers (see the scale plan in `docs/SCALE_PLAN.md`). It must:

- **Provision:** 1 on-demand metal DB instance, 1 on-demand pooler instance,
  and N spot load instances — all in one subnet, tagged with a single
  `run=<id>`, security groups scoped to the operator /32 and to intra-rig
  traffic only (never `0.0.0.0/0`). On-demand for DB + pooler (a spot
  reclaim mid-run voids the result); spot for the stateless load fleet.
- **Orchestrate:** load corpus + build index on the DB tier; point the pooler
  at it; fan the driver out across the load fleet; collect per-driver results
  and aggregate (synchronised clocks, staggered start, discarded warmup).
- **Pull artefacts continuously** and **tear the whole rig down by tag** at
  the end (and on error) — the AWS burner rules in `AGENTS.md` (§ AWS burner
  accounts) apply: verify the profile first, scope SGs, tag everything,
  terminate-by-own-tag, write the teardown script at launch time.
