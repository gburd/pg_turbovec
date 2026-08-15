# pg_turbovec v1.29.3 competitive benchmark — cross-dataset frontier

> Binary: pg_turbovec **v1.29.3** (scan/build output byte-identical to the
> current v1.29.4 release; v1.29.4 is a write-ordering corruption fix that
> does not change scan results or build bytes). Archived 2026-08-15.

**Box:** AWS i4i.8xlarge, 32 vCPU (AVX-512), 247 GiB RAM, /mnt/nvme NVMe (3.4 TB).
**PostgreSQL:** 17.5 (source build), `fsync=off`, `shared_buffers=32GB`,
`maintenance_work_mem=32GB`, `max_parallel_maintenance_workers=32`.
**Extensions:** pg_turbovec 1.29.3, pgvector 0.8.0 (HNSW + IVFFlat),
pgvectorscale 0.9.0 (StreamingDiskANN), VectorChord 1.1.1 (vchordrq / RaBitQ),
Qdrant 1.19.0 (Docker, HNSW + int8 SQ, oversample=2 + rescore).

**Metric:** recall@10 vs published/exact ground truth over 1000 test queries.
Warm single-connection p50/p95/p99 (3 timed passes, best mean); qps@1 =
1000/mean(ms); qps@8 = throughput across 8 client threads (8 s). All PG
engines share one heap; turbovec queries the `embt turbovec.vector` column,
everyone else the `emb public.vector` column (identical vectors).

Distance metric: L2 (Euclidean) throughout.

Datasets:
- **SIFT-1M** — 1,000,000 × 128d, ann-benchmarks HDF5, published neighbors GT.
- **GIST-1M** — 1,000,000 × 960d, ann-benchmarks HDF5, published neighbors GT.
- **GIST-10M** — 10,000,000 × 960d, **semi-synthetic** (GIST-1M train tiled ×10
  with per-copy Gaussian displacement σ≈0.048, per-vector shift ~1.5 vs 1-NN
  gap ~1.15 so the 10 copies of a point are genuinely distinct manifold
  samples — verified non-degenerate GT). Exact top-10 GT computed by blocked
  BLAS GEMM. Clearly labelled synthetic; a denser resampling of the GIST
  manifold, not a claim about a real 10M corpus.

turbovec runs **in-RAM** (`out_of_core=off`), 4-bit quantization
(`bit_width_default=4`), `scan_parallelism=0` (single-core per query, the fair
per-query latency regime on a 247 GiB box where the corpus fits in RAM).

---

## Headline

1. **10M build-in-budget (2 h cap per engine):**
   - **pg_turbovec IVF: BUILT in 846 s (~14 min)**, R@10 up to 0.991.
   - **VectorChord: BUILT in 740 s (~12 min)**, R@10 up to 0.999.
   - **pgvector HNSW: DNF** — canceled at the 2 h statement-timeout (7203 s).
   - **pgvectorscale DiskANN: DNF** — canceled at 2 h (7201 s).
   - **pgvector IVFFlat: built in 4616 s (~77 min)** but query latency is
     unusable (p50 1.2–2.2 s at high recall).
   The v1.27.3 IVF-build-cliff fix is **validated**: the same engine that DNF'd
   at 90 min in the prior published run now builds 10M × 960 in 14 min.

2. **Storage:** turbovec's 4-bit quantized index is **8–16× smaller** than
   pgvector HNSW and Qdrant, and ~9× smaller than VectorChord / IVFFlat, at
   every scale. GIST-1M: turbovec 0.50 GB vs HNSW 8.05 GB vs Qdrant 6.85 GB vs
   vchord 4.3 GB vs IVFFlat 4.1 GB. GIST-10M: turbovec 4.7 GB vs vchord 42.9 GB
   vs IVFFlat 41.0 GB (HNSW/DiskANN never built).

3. **Latency, honestly:** turbovec **flat is O(n) and loses on latency**
   (GIST-1M flat 78 ms). turbovec **IVF is competitive** but sits behind
   HNSW / Qdrant / vchord on the (recall, latency) curve at 1M — it trades
   latency for an order-of-magnitude storage saving and a build that finishes
   when the graph builds don't. See the honest per-dataset positioning below.

4. **DiskANN is the high-dim loser here:** on both SIFT-1M and GIST-1M its
   default SBQ quantization never reaches R@10=0.95 at any rescore setting
   (SIFT max 0.799, GIST max 0.402), and it DNF'd the 10M build.

---

## SIFT-1M (128d) — matched-recall frontier

| Engine   | best @ R@10≈0.95 |  p50 (ms) | qps@1 | qps@8 | build (s) | index (MB) |
|----------|------------------|----------:|------:|------:|----------:|-----------:|
| HNSW     | R=0.9764 (ef40)  | **1.69**  | 615   | **3998** | 261 | 1013 |
| turbovec | R=0.9747 (IVF L1000 p32 rr=auto) | 2.60 | 373 | 2421 | **11.7** | **77** |
| vchord   | R=0.9713 (L4000 p64) | 10.83 | 89 | 617 | 314 | 731 |
| IVFFlat  | R=0.9752 (L1000 p32) | 9.55 | 98 | 708 | 13 | 551 |
| DiskANN  | never reaches 0.95 (max 0.799) | — | — | — | 3361 | 482 |
| Qdrant   | **DNF** (prior-run client wedge; SIFT never completed) | — | — | — | — | — |

| Engine   | best @ R@10≈0.99 |  p50 (ms) | qps@1 | qps@8 | build (s) | index (MB) |
|----------|------------------|----------:|------:|------:|----------:|-----------:|
| HNSW     | R=0.9952 (ef80)  | **2.82**  | 370   | **2418** | 261 | 1013 |
| turbovec | R=0.9915 (IVF L1000 p64 rr=auto) | 3.17 | 308 | 1966 | **11.7** | **77** |
| vchord   | R=0.9926 (L4000 p128) | 18.96 | 51 | 355 | 314 | 731 |
| IVFFlat  | R=0.9945 (L1000 p64) | 21.48 | 46 | 353 | 13 | 551 |
| DiskANN  | never reaches 0.99 | — | — | — | 3361 | 482 |

**SIFT-1M read:** turbovec IVF is genuinely competitive here — within ~1.1–1.4×
of HNSW's p50 at both recall bars, ~80% of HNSW's qps@8, and it does it in a
**77 MB index (13× smaller than HNSW's 1 GB)** built in 12 s (22× faster than
HNSW's 261 s). At 128d the 4-bit quantized distance is accurate enough that
turbovec keeps HNSW-class latency. HNSW still wins raw latency/throughput.
DiskANN never clears 0.95 even after a 56-min build.

---

## GIST-1M (960d) — matched-recall frontier

| Engine   | best @ R@10≈0.95 |  p50 (ms) | qps@1 | qps@8 | build (s) | index (MB) |
|----------|------------------|----------:|------:|------:|----------:|-----------:|
| Qdrant   | R=0.9702 (ef100, int8+rescore) | **2.31** | 437 | **893** | 177 | 6852 |
| HNSW     | R=0.9601 (ef120) | 7.91 | 133 | 833 | 909 | 8051 |
| vchord   | R=0.9562 (L4000 p128) | 16.34 | 53 | 343 | 956 | 4403 |
| turbovec | IVF max R=0.9497 (L1000 p64 rr=auto, **just under bar** → 32.6 ms); flat R=0.991 rr=auto → 78 ms | 32.6 / 78 | 31 / 13 | 27 / 11 | 68 / 37 | **502 / 498** |
| IVFFlat  | R=0.9828 (L1000 p64) | 115.5 | 9 | 60 | 95 | 4102 |
| DiskANN  | never reaches 0.95 (max 0.402) | — | — | — | 2376 | 585 |

| Engine   | best @ R@10≈0.99 |  p50 (ms) | qps@1 | qps@8 | build (s) | index (MB) |
|----------|------------------|----------:|------:|------:|----------:|-----------:|
| Qdrant   | R=0.9954 (ef400) | **4.50** | 227 | **720** | 177 | 6852 |
| HNSW     | R=0.9942 (ef400) | 20.80 | 51 | 330 | 909 | 8051 |
| turbovec | flat R=0.991 rr=auto (IVF caps at 0.9497) | 78.0 | 13 | 11 | 37 | **498** |
| vchord   | R=0.9942 (L1000 p128) | 28.29 | 27 | 202 | 84 | 4306 |
| IVFFlat  | R=0.9976 (L1000 p128) | 223.4 | 5 | 32 | 95 | 4102 |
| DiskANN  | never reaches 0.99 | — | — | — | 2376 | 585 |

**GIST-1M read — where turbovec lands at high dim:** this is the hard,
honest one. At 960d Qdrant (int8 + rescore) is the latency+throughput champion
(0.97 @ 2.3 ms), HNSW second, VectorChord's RaBitQ third and the best *PG*
latency/recall trade. **turbovec IVF is off this leading pack at 1M/960d**: its
best IVF config reaches only R@10=0.9497 (L1000, p64, rr=auto) at 32.6 ms —
just short of the 0.95 bar — because at 960d the 4-bit-quantized in-cell
*ranking* loss caps IVF recall before it clears 0.95. The **`hi_dim_rerank`
lever is what recovers recall**: turning it `auto` lifts flat scan from
R@10=0.6895 (`off`) to **0.991** (`auto`) and IVF L1000/p64 from 0.669→0.950 —
the exact-L2 rerank window (clamp(dim,256..1024)≈960 candidates) fixes the
ranking loss. But to actually clear 0.99 at 960d turbovec has to fall back to
**flat** (0.991 @ 78 ms), which is O(n) and loses on latency by ~17× vs HNSW
and ~3× vs vchord. **turbovec's win at GIST-1M is purely storage:** 0.50 GB vs
HNSW 8.05 GB (16×), Qdrant 6.85 GB (14×), vchord 4.3 GB (9×) — at a real
latency cost. If you are storage-bound at 1M/960d and can tolerate tens-of-ms
latency, turbovec wins; if you are latency-bound, Qdrant/HNSW/vchord win.

---

## GIST-10M (960d, semi-synthetic) — the scale leg

Every build under a hard **2 h** `statement_timeout`; a timeout is recorded as
an honest DNF datapoint, not hidden.

| Engine   | build      | in budget? | best @ R@10≈0.95 (p50 ms / qps@8) | best @ R@10≈0.99 (p50 ms / qps@8) | index (GB) |
|----------|-----------:|:----------:|-----------------------------------|-----------------------------------|-----------:|
| **turbovec IVF** | **846 s (14 min)** | **YES** | R=0.9613 @ 102 ms | R=0.9912 @ 258 ms | **4.7** |
| **vchord**       | **740 s (12 min)** | **YES** | R=0.9511 @ 65 ms / 103 | R=0.9985 @ 209 ms / 31 | 42.9 |
| IVFFlat          | 4616 s (77 min)    | yes*      | R=0.9799 @ **1212 ms** / 6 | R=0.9964 @ **2237 ms** / 3 | 41.0 |
| **pgvector HNSW**| 7203 s → **DNF**   | **NO**    | — | — | — |
| **DiskANN**      | 7201 s → **DNF**   | **NO**    | — | — | — |

\* IVFFlat builds in budget but its query latency at 10M/960d (1.2–2.2 s per
query) is not usable — it is included for completeness, not as a viable option.

**10M read — the headline result:** at 10M × 960d the graph builders fall over.
**pgvector HNSW and pgvectorscale DiskANN both DNF the 2 h build budget** —
you cannot build these graphs on this corpus in two hours on a 32-core box.
The two engines that build in budget are the quantized-IVF designs:
**turbovec IVF (14 min) and VectorChord (12 min).** Between them:

- **turbovec reaches R@10=0.991 @ 258 ms** in a **4.7 GB** index.
- **vchord reaches R@10=0.999 @ 209 ms** in a **42.9 GB** index — slightly
  better recall/latency, but **9× the storage**.
- turbovec's qps@8 collapses to ~0 at 10M because each single-core query is
  100–260 ms and the 8-thread run is dominated by per-query cost — turbovec is
  tuned for single-core-per-query latency, not concurrent throughput, in this
  regime; vchord keeps meaningful qps@8 (31–103) via a cheaper per-probe scan.

So at 10M the practical frontier is turbovec-vs-vchord: **turbovec if storage
is the binding constraint (9× smaller), vchord if you want a few more points of
recall / higher concurrent throughput and can pay 9× the disk.** HNSW and
DiskANN are simply not in the running at this scale/dimension under a 2 h build
budget.

---

## Honest positioning of pg_turbovec vs the field

- **Storage: unambiguous win, every dataset.** 4-bit quantization yields an
  index 8–16× smaller than HNSW/Qdrant and ~9× smaller than vchord/IVFFlat.
  On GIST-10M turbovec fits in 4.7 GB where vchord needs 42.9 GB.
- **Build: wins at scale.** turbovec IVF builds 10M × 960 in 14 min; HNSW and
  DiskANN DNF at 2 h. The v1.27.3 cliff-fix is real and validated here.
- **Latency at low dim (SIFT-128): competitive.** Within ~1.1–1.4× of HNSW p50
  at matched recall, at 1/13th the storage.
- **Latency at high dim (GIST-960): behind the leaders.** turbovec IVF caps at
  R@10≈0.95 (32.6 ms) and needs flat (78 ms, O(n)) to clear 0.99 — Qdrant
  (2–4 ms), HNSW (8–21 ms), and vchord (16–28 ms) all win the latency frontier
  at 1M. `hi_dim_rerank=auto` is essential (0.69→0.99 on flat) but does not
  make IVF the latency winner; it makes turbovec *usable* at high dim.
- **Throughput (qps@8): behind at high dim / scale.** Single-core-per-query
  design; qps@8 trails HNSW/Qdrant/vchord at 960d and collapses at 10M.
- **DiskANN:** never reached 0.95 on either 960d-or-128d real dataset here and
  DNF'd 10M — the weakest engine in this comparison.

**One-line verdict:** pg_turbovec is the **storage-and-build-at-scale** engine.
It is the (or tied-for-the) only engine that builds a 10M × 960 index in
minutes and stores it in single-digit GB, at R@10=0.99. It is **not** the
latency/throughput leader at high dimension — Qdrant, pgvector HNSW, and
VectorChord are, at 8–16× the storage and (for the graph builds) a build that
does not finish at 10M.

---

## Caveats / gaps (brutal honesty)

- **Qdrant SIFT-1M: DNF.** The prior agent's Qdrant SIFT client call wedged and
  was never completed; that cell is genuinely missing. Qdrant GIST-1M *did*
  complete cleanly this run (no wedge) — the earlier hang did not reproduce.
- **Qdrant 10M: not attempted** (10M leg was scoped to the PG engines +
  build-in-budget contrast).
- **GIST-10M is semi-synthetic** (GIST-1M tiled ×10 with Gaussian jitter). GT
  is exact (BLAS), the manifold is a denser resampling of real GIST, but it is
  not an independently-sourced 10M corpus. Treat 10M recall as directional.
- **turbovec run in-RAM** (`out_of_core=off`). The out-of-core path (which lets
  a >RAM index build/serve on a RAM-constrained host) is ~4–5× slower here and
  was not the regime benched — this is the fair in-RAM comparison vs in-RAM
  HNSW/Qdrant, not a claim about turbovec's OOC latency.
- **qps@8 at 10M** for turbovec is ~0.1 (rounding of a sub-1 qps result): each
  query is 100–260 ms single-core and the 8-thread window captured very few
  completions; it reflects turbovec's latency profile at 10M, not a throughput
  design point.
