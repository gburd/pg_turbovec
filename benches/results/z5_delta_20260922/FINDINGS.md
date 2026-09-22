# Z5 Route A (bounded append delta) — EC2 validation

Host: c7i.4xlarge (16 vCPU, AVX-512, 30 GiB), us-east-2b, Ubuntu 24.04,
PostgreSQL 16.15, pg_turbovec 2.9.0 + Z5 (`83fa30f`), release build.
`shared_buffers = 8GB`, `maintenance_work_mem = 4GB`.

## Harness defects found and fixed BEFORE any number was trusted

Three in sequence, each of which produced a confidently wrong result:

1. **`shared_preload_libraries` was unset**, so `_PG_init` never ran and
   **zero** `turbovec.*` GUCs were registered. Every `SET turbovec.probes`
   silently did nothing. Caught because `SHOW turbovec.iterative_scan`
   errored with "unrecognized configuration parameter" — while
   `SET turbovec.probes=1` appeared to succeed.
   *We do not document this requirement anywhere; see ACTION below.*
2. **One `psql` per query = one backend per query.** turbovec loads the
   index into a per-backend cache on first scan, which costs ~460 ms at
   this size. Every "warm" measurement was actually that cold load, which
   is why probe pruning looked like it did nothing (463 ms at every probe
   setting). Fixed by running all queries in ONE session after a discarded
   warm-up. Same session: first query 463 ms, second 4.5 ms.
3. **`tail -1` on `actual time=`** grabbed the wrong plan node.

## Measured (1M × 256-d, bit_width=4, lists=1024, warm, one session)

| configuration | mean scan |
|---|---|
| probes=1 | 3.66 ms |
| probes=16 | 3.95 ms |
| probes=256 | 4.70 ms |
| probes=1024 (= full scan) | 5.97 ms |
| **degraded** (pre-Z5, delta disabled, 1000 inserts) | **4.39 ms** |
| **Z5 delta** (same 1000 inserts) | **4.20 ms** |

Index: 139 MB. Build: 43.8 s. Corpus resolvability probe (mandatory per
`docs/BQ_RECALL_BENCH` § 0.6d): nn1→nn100 spread ≈ 3×10⁵ %, far above the
tens-of-percent floor. `degraded` / `scan_fraction` / `est_slowdown`
reported exactly as designed (true/1/64 vs false/0.015625/1).

## Verdict at this scale: the predicted win does NOT appear

My model assumed scan cost dominates, so pruning 64× fewer rows would cut
latency ~64×. **It does not.** A *full* 1M × 256-d 4-bit scan is only
**1.6×** the cost of a 1-cell scan (5.97 vs 3.66 ms), because the 139 MB
index is entirely RAM-resident and per-query fixed costs (query encode,
heap recheck, executor) are the same order as the SIMD sweep itself.

Z5 captures roughly that available gap (4.20 vs 4.39 ms, ~4 %) and nothing
more, because there is nothing more to capture here.

**This is consistent with our own published numbers, which I should have
consulted before modelling:** `docs/BQ_RECALL_BENCH` § 0.6e has 1M bw4
**flat at 6.08 ms beating IVF at 16.04 ms** — v2.8.3 concluded "flat wins at
every target" for 4-bit at 1M. A regime where a full scan costs 6 ms is by
definition a regime where eliminating 98 % of it cannot win much.

## Where Z5 should matter (being tested next)

The win requires a regime where a full scan is expensive *relative to fixed
costs* — i.e. an index too large to be RAM-resident, where a full scan is
disk I/O rather than a RAM sweep. That is the >1.7M / 10M out-of-core case
this project targets. 1M × 256-d does not reach it.

## ACTION items independent of Z5

- **Document `shared_preload_libraries = 'pg_turbovec'`.** Without it every
  GUC silently vanishes and the extension appears to ignore all tuning.
  Not mentioned in README, PRODUCTION.md, or DEPLOYING_ON_MANAGED_POSTGRES.md.
- The ~460 ms first-scan per-backend load at 1M × 256-d is worth documenting
  as a cold-start cost; it dwarfs the warm scan by ~100×.

## Clean A/B (autovacuum disabled, fresh index per arm, health verified)

30 warm queries per arm, one session, `probes=16`, 1000 inserts, 1M × 256-d.
Each arm re-creates the index and re-inserts, so the arms are independent.
`deg|frac` is read AFTER the inserts, proving which contract was in force.

| arm | mode | median | p90 | min | degraded |
|---|---|---:|---:|---:|---|
| pre-Z5 (`pct=0`) | in-memory | 4.51 ms | 5.29 | 3.69 | **true** (frac 1.0) |
| **Z5 delta** (`pct=10`) | in-memory | **4.03 ms** | 4.37 | 3.40 | false (frac 0.0156) |
| pre-Z5 (`pct=0`) | out-of-core | 4.43 ms | 5.07 | 4.03 | **true** (frac 1.0) |
| **Z5 delta** (`pct=10`) | out-of-core | **3.46 ms** | 79.01 | 2.92 | false (frac 0.0156) |

**Z5 wins 11 % in memory and 22 % out-of-core, on the median.** The
functional contract works exactly as designed in every arm: the delta arm
keeps `degraded = false` and `scan_fraction = 0.0156` while the pre-Z5 arm
degrades to `1.0`. The OOC p90 of 79 ms is a per-query page-gather outlier
(the OOC path reads probed pages through the buffer manager on demand), not
a delta effect — the median and min both improve.

## Honest verdict

**The mechanism is correct and the win is real but small — roughly 10–20 %,
not the 64× I modelled.** I was wrong, and the error is instructive: I
modelled latency as proportional to rows scanned, but at 1M × 256-d a *full*
4-bit scan is only 1.6× a 1-cell scan (5.97 vs 3.66 ms), because the 139 MB
index is RAM-resident and per-query fixed costs are the same order as the
SIMD sweep. You cannot save 64× of something that is only 1.6× of the total.

This agrees with our own published result that I should have consulted
first: `docs/BQ_RECALL_BENCH` § 0.6e measured 1M bw4 **flat at 6.08 ms
beating IVF at 16.04 ms**, and v2.8.3 concluded "flat wins at every target"
for 4-bit at 1M.

### Should it ship?

Yes, but on its **functional** merit, not a latency claim:

- Before Z5, one INSERT turned an IVF index into a full-corpus scan until a
  REINDEX — an operational cliff with no cheap recovery.
- After Z5, the cell layout survives bounded appends, and past the bound the
  old degrade-and-report behaviour is preserved exactly.
- It fixes a **latent correctness bug** on the out-of-core path, where an
  unswept tail would have made appended rows unreachable (they exist on disk
  and the gather never reads them) — silent loss, not slowness.
- Cost: zero format change, `turbovec.ivf_max_delta_pct = 0` restores the
  previous behaviour byte-for-byte.

### Where a large win would live (NOT measured)

A regime where a full scan is expensive relative to fixed costs: an index
far larger than RAM, where a full scan is disk I/O. A 3M × 1024-d load was
started for this and did not finish inside the session budget — so the
large-scale claim is **explicitly unmeasured**, and the 10–20 % figures
above are what is actually supported.

## Sustained-load corruption validation (HARD MANDATE)

Required because Z5 touches the persist path, and because the v1.28.4 fix
shipped on reasoning alone and **re-corrupted in production**.

Workload: **6 concurrent writers** (25-row batched INSERTs + interleaved
DELETEs) + **4 concurrent readers** (ANN queries at `probes=16`), 5 minutes,
with **autovacuum enabled** and an explicit `VACUUM corpus` every 30 s —
VACUUM being the operation that historically exposed the tombstone/slot bugs
(v2.7.0 BQ tombstone resurrection, v1.29.1 deferred-flush lost update).

Result:

```
check: wire=8 kind=single n=1018215 slots=1018215 match=true
       dup=none CORRUPT=false reason=-
degradation: false | scan_fraction=0.015625
```

- **No corruption.** `is_corrupt = false`, no duplicate id, `n_vectors ==
  slot_count` exactly.
- **Wire format still v8** — no format change, as designed.
- **The cell layout SURVIVED** 5 minutes of concurrent write load plus
  repeated VACUUMs: still `degraded = false`, `scan_fraction = 0.0156`.
  Pre-Z5 this index would have been a full-corpus scan after the first
  commit.

Two apparent discrepancies, both checked rather than assumed:

- index `n_vectors` (1018215) exceeded heap rows (1017143) by 1072 —
  explained: 10072 dead tuples were awaiting vacuum, and VACUUM reclaims
  lazily. Not loss.
- a "reachable sample" query returned 254 of 1000 requested — explained:
  `turbovec.search_k = 32` caps the candidate set. Not loss.

## Cost / cleanup

Two instances (the first unreachable, see the SSH gotcha), ~2.5 h total on
c7i/m7i.4xlarge ≈ $2–3. Both terminated, security group and key pair
deleted, verified. Five untagged instances belonging to other users of this
burner account were present throughout and were **not touched**.
