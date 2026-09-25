# Cold-scan latency — parallel repack A/B (2026-09-25)

Validates turbovec fork carry #3 (parallel pack::repack) end-to-end on real
AVX-512 hardware. Account 373102893032, c7i.4xlarge (16 vCPU, AVX-512),
1M × 1024-d 4-bit flat index (534 MB). SAME instance, SAME corpus, SAME index
across both arms — only the pg_turbovec binary differs. Instance terminated,
verified clean.

Regimes (item-2 methodology): R3 = warm-backend (all queries one psql session,
warm-ups discarded); R2 = cold-backend + warm-OS (fresh psql per query = fresh
backend = per-backend cache cold = repack repaid). R2 is the pooled-pool
reality (connection pools create/destroy backends).

| build | R3 warm p50 | **R2 cold-backend p50** |
|---|---|---|
| OLD (serial repack, turbovec f29a2f2) | 30.4 ms | **1765.9 ms** |
| NEW (parallel repack, turbovec 47a26a3) | 30.6 ms | **566.2 ms** |

**Cold-scan latency 1766 ms → 566 ms: a 3.1× reduction on 16 cores.** Warm is
unchanged (30.4 vs 30.6 ms, within noise) — correct, since a warm backend never
repays the repack. The remaining 566 ms is the `read_chain` relfile copy plus
the still-serial parts of cache install (future work; the repack was the
dominant, and now-parallel, term).

This is a SCAN/OPEN-TIME speed change with ZERO on-disk format impact (the
SIMD-blocked layout is recomputed per backend and never persisted; wire format
stays v8). No REINDEX. The byte-identity of the parallel repack vs serial is
pinned by turbovec's `parallel_repack_is_byte_identical_to_serial`
(bit-widths 2/3/4, sub/above threshold, tail-padding shapes).

Corroborating micro-bench (local, 8 cores, pure repack in isolation): 250k ×
1024-d × 4-bit ~6 s serial → ~250 ms parallel.

Artifacts: coldscan_new_parallel.log, coldscan_old_serial.log, cs_bench.sh.
