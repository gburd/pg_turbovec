# Cold-scan prefetch — MEASURED, REJECTED (naive PrefetchBuffer hurts on EBS)

Tested a fixed-window (64-page) PrefetchBuffer readahead in read_chain against
v2.10.3 baseline. Same c7i.4xlarge (16 vCPU), same 1M x 1024d 4-bit index
(534 MB), EBS gp3 (300 MB/s cap). 10 trials/regime. Instance terminated.

| regime | baseline (v2.10.3) | +prefetch | verdict |
|---|---|---|---|
| cold-backend / warm-OS (page cache hot) | 857.9 ms | 870.2 ms | no change (no real I/O to overlap) |
| true-cold disk (drop_caches per trial) | **6209 ms** | **10757 ms** | **prefetch 1.73x WORSE** |

## Why the naive prefetch backfires

On a throughput-capped device (EBS gp3, and managed-Postgres storage generally),
issuing 64 speculative PrefetchBuffer reads ahead of the synchronous
ReadBufferExtended floods the I/O queue with reads that CONTEND with the actual
sequential reads. read_chain already reads sequentially, so the OS/EBS layer
does its own readahead; adding explicit prefetch only adds queue pressure. On
uncapped local NVMe this might help, but pg_turbovec's target (managed PG) runs
on EBS-like storage where it demonstrably hurts.

## Decision: DO NOT merge the naive prefetch (feat/coldscan-prefetch abandoned).

The measurement is unambiguous and matches the mechanism. This is a measure-first
rejection — the same discipline that caught the "flat beats HNSW" artifact.

## The RIGHT path (not this): read_stream (PG17+), AIO-backed on PG18+.

PostgreSQL's read_stream API (read_stream_begin_relation + read_stream_next_buffer)
is designed exactly for this: it ADAPTS the prefetch distance to the device and
COALESCES reads, so it does not flood a capped queue the way a fixed window does.
On PG18+ it is backed by the AIO subsystem the user asked for. It is version-gated
(PG17+; pg_turbovec supports 13-19) so it needs a cfg-gated path with the
PrefetchBuffer-or-nothing fallback for 13-16. That is a larger, careful change
(a ReadStreamBlockNumberCB callback feeding read_chain's block sequence) and
should be its own measured effort — NOT shipped on the naive result. Filed as a
follow-up; the cold path stays as-is in v2.10.3 (repack already parallelized,
which was the CPU-side win; the residual is I/O that read_stream, not a fixed
prefetch window, is the tool for).

Artifacts: prefetch_ab_baseline.log, prefetch_ab_prefetch.log, pf_bench.sh.
