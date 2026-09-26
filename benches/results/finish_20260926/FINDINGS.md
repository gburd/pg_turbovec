# Finish-items EC2 work (2026-09-26)

Two experiments on one c7i.8xlarge (32 vCPU, AVX-512), account 373102893032,
pg_turbovec v2.10.3, 1M × 1024-d Cohere-wiki, exact top-10 GT. Instance
terminated, verified clean.

## 1. Cold-scan residual profile — WHERE the 566 ms goes after parallel repack

Instrumented the flat cold-open path (throwaway build) to time the two phases,
logged per fresh-backend query. 1M × 1024-d × 4-bit flat, cold per-backend cache:

| phase | median | share |
|---|---|---|
| `read_full_consistent` (buffer-manager copy of codes+scales+ids via `read_chain`) | **280.4 ms** | 99.3% |
| `prepare()` + repack (now PARALLEL, v2.10.3) | **1.95 ms** | 0.7% |

**Conclusion: the parallel-repack fix (v2.10.3) already captured essentially the
entire CPU-side cold win.** Repack is now 2 ms — negligible. The whole residual
cold cost is `read_chain`: ~65k serial `ReadBufferExtended` + memcpy calls to
pull the 534 MB codes chain through PostgreSQL's buffer manager.

**This is NOT a safe simple further win.** `read_chain` is buffer-manager I/O,
not CPU work. Parallelizing `ReadBufferExtended` from Rust threads inside a
backend is unsafe — PG's buffer manager and relation access are not thread-safe
off the backend's own thread. A real improvement would need a larger change
(buffer prefetch via `PrefetchBuffer`, a bulk multi-block read, or bringing back
a controlled mmap — which BUFFER_CACHE_ONLY_DESIGN.md deliberately forbids for
managed-PG). **Decision: do NOT code a cold-scan change now.** The measured fact
(repack was 99% of the CPU cold cost and is now fixed; the rest is I/O) is the
deliverable. Prefetch is a future L-effort item gated on its own measurement.

Artifacts: coldopen_profile.txt, fin_instrument.sh (the throwaway timing patch).

## 2. IVF ≥0.98 grid extension — "unreachable" was GRID-LIMITED, now pinned

The rebench sweep capped probes at 128 (of lists=1024) and reported IVF as
"unreachable at R@10≥0.98". Extended to probes ∈ {128, 256, 512}:

| arm | probes=128 | probes=256 | probes=512 |
|---|---|---|---|
| IVF-bw1 | 0.9650 / 37 ms | **0.9870 / 25.6 ms** | 1.0000 / 46 ms |
| IVF-bw4 | 0.9650 / 96 ms | **0.9900 / 174 ms** | 1.0000 / 364 ms |

(k picked per cell for best recall; representative p50.)

**Confirmed exactly as predicted — not a bug, not a quantization wall:**
- IVF-bw1 clears 0.98 at **probes=256** (0.987, 25.6 ms) and reaches 1.000 at
  probes=512.
- IVF-bw4 clears 0.98 at probes=256 (0.990, 174 ms), 1.000 at probes=512.
- **The pruning-stops-paying crossover is now measured:** at ≥0.98, IVF-bw1's
  25.6 ms (probes=256, ¼ of cells) barely beats flat-bw1's 29.8 ms full scan,
  and by R@10=1.000 IVF-bw1 (46 ms) is SLOWER than flat-bw1 (40 ms). IVF-bw4 at
  ≥0.98 (174 ms) is ~33× slower than flat-bw4 (5.2 ms). So once you need ≥0.98
  from these codes you must probe enough cells that IVF's pruning stops paying —
  use flat instead. This is the same conclusion the overall rebench reached,
  now quantified at the high-recall end.

Artifacts: ivf098_highprobe_sweep.log, fin_ivf098.sh.
