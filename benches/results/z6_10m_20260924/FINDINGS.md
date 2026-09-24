# Z6 — 10M × 1024-d build MEASURED: the projection was exact

Closes the one number the v2.10.1 release notes carried as *projected, not
measured*.

## Setup — deliberately identical to the run that OOM-killed

| | value |
|---|---|
| host | **c7i.8xlarge**, 32 vCPU, **61 GiB** (same type as the OOM run) |
| corpus | **10M × 1024-d**, 56 GB heap, same generator as `z5_ram_20260922` |
| index | `bit_width = 4`, **`lists = 3162`** |
| config | `shared_buffers = 2GB`, **`maintenance_work_mem = 8GB`**, **`max_parallel_maintenance_workers = 16`** |
| build | `pg_turbovec` **v2.10.1** (released tag `df9b68e`) |

Every one of those settings matches the pre-fix attempt that died, so the only
variable is the per-tuple memory context.

## Result

| | before (≤ v2.10.0) | after (v2.10.1) |
|---|---|---|
| outcome | **OOM-killed** | **completed** |
| peak | `anon-rss` **58.47 GiB** (then killed) | peak private **11.20 GiB** |
| build time | — (never finished) | **69.8 min** |
| index size | — | 5340 MB |
| OOM events in `dmesg` | 1 | **0** |

**Projected in the v2.10.1 release notes: 11.2 GiB. Measured: 11.20 GiB.**

The projection was built term by term, and the breakdown holds:

| term | predicted |
|---|---:|
| `packed_codes` (4-bit) | 4.77 GiB |
| k-means reservoir (`lists × 256 × dim × 4`) | 3.09 GiB |
| rotation destination | 3.09 GiB |
| bounded Lloyd `cross` | 0.25 GiB |
| **sum** | **11.19 GiB** vs **11.20 measured** |

Memory was also *flat* for most of the build — sampled at 3.31 GiB across
~30 minutes of the k-means phase, rising only in the encode/persist stages.
That is the signature of a build that streams rather than accumulates.

## The index is correct, not merely built

A build that completes but produces a broken index proves nothing, so:

```
wire_version | kind   | n_vectors | slot_count | count_matches | is_corrupt
           8 | single |  10000000 |   10000000 | t             | f

degraded | lists | n_vectors | scan_fraction
       f |  3162 |  10000000 | 0.005060088551549652
```

`scan_fraction = 0.00506` is exactly 16/3162 — the cell pruning is live. The
index serves queries: **2.6 s cold** (first scan in a backend loads the
per-backend cache), **51 ms warm** at `probes = 16`.

## What this establishes

1. **10M × 1024-d IVF builds now fit comfortably on a 61 GiB host** — 11.20 of
   61 GiB, i.e. 18 %. Before the fix the same build could not be completed at
   all, which is what blocked the Z5 >RAM measurement.
2. **The root-cause model is validated quantitatively.** A projection derived
   purely from "the 5125 B/row CBOR retention is gone, these four terms remain"
   landed within 0.01 GiB. That is strong evidence the CBOR retention was the
   whole of the defect, not one contributor among several.
3. **`maintenance_work_mem = 8GB` with 16 parallel workers is no longer
   dangerous.** The original OOM used exactly this config; it now peaks at 11.2
   GiB, so the setting behaves as documented.

## Still not measured

The Z5 >RAM *latency* arm. This build makes it possible for the first time —
a 5.3 GB index on a host whose buffer pool can be constrained below it — but
latency was not measured here and the v2.10.0 claim (11 %/22 % at 1M × 256-d,
>RAM regime untested) stands unchanged.

## Cost

c7i.8xlarge, ~2.5 h (25 min load, 70 min build, rest provisioning) ≈ $5.
Instance terminated, security group and key pair deleted; all eight instances
across Z5/Z6 verified absent from every billable state.
