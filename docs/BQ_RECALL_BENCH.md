# 1-bit sign-BQ recall / storage / latency frontier — methodology + RESULTS

**Status: MEASURED 2026-09-08 on `arnold` (AVX2). See § 0 for the results.**
The methodology below is what the run followed; § 7's predictions were
recorded *before* it and are scored in § 0.4.

---

## 0. Results — Cohere-wiki 250k × 1024-d, `arnold`, 2026-09-08

Artefact: `benches/results/bq_frontier_20260908/bq_frontier_arnold_20260908.json`
(24 configs, 3 indexes, plus the sweep log).

**Provenance.** `arnold`, i9-12900H, `kernel_tier = avx2`,
`latency_publishable = true` (per § 2, latency is only valid on an AVX2
host).

> ### ⚠ Correction (2026-09-09), RESOLVED (2026-09-10): the original rows were contended; a clean re-run confirms the ratios
>
> The v2.7.3 release presented the p50s below without this caveat. That was
> wrong, and this is the correction. **All 24 rows carry
> `latency.contention.contended_flag = true`** — 1-minute loadavg 3.16–4.64
> against the harness's gate of 1.5. The harness flagged it correctly in the
> artefact; the write-up did not surface it.
>
> The gate is **unreachable on `arnold`** and not because of this benchmark.
> Two pre-existing processes pin the idle floor near 2.0: a stuck
> `systemd --user` spinning at 77–86 % CPU for 31 days, and an unrelated
> long-running agent process at ~100 %. Idle loadavg is 2.0–2.4 before any
> bench starts, so *any* run on this host is flagged.
>
> **How much does it matter?** Less than the flag alone implies, and this is
> measured rather than assumed: `cpu_busy_pct` across the 24 rows is only
> **16.3–26.0 %**. The load is runnable-elsewhere processes, not saturation of
> the pinned cores 2-5 — the noise processes have affinity 0-19, so the kernel
> migrates them off the bench cores. The absolute milliseconds are therefore
> *indicative but noisy*, and the **ratios and the shape of the
> window-vs-recall curve are the defensible result** — which is the stance
> § 0.5 already takes and the reason the headline is expressed as ratios.
>
> Nothing about recall, storage, bytes/vector or build time is affected;
> those are CPU-independent or honestly host-specific.
>
> **RESOLVED 2026-09-10.** The stuck `systemd --user` was fixed on the host, its
> idle load floor dropped from ~2.0 to **0.46**, and the whole sweep was re-run
> on the same corpus. Artefact: `benches/results/bq_retime_20260910/`.
>
> - **18 of 24 rows now come back clean** (was 0 of 24). The 6 still flagged
>   drifted over the gate as the sweep itself warmed the box — self-inflicted
>   load, not foreign.
> - **Recall reproduced EXACTLY** at every window for all three bit widths —
>   the third independent confirmation of § 0's recall numbers.
> - **The contended p50s were uniformly 14–16 % pessimistic** (e.g. bw1 at
>   window 800: 36.75 → 30.93 ms). Uniform, not erratic, which is exactly what
>   "noise, not saturation" predicts and is why the ratios survived.
> - **The published ratios hold to two decimal places**: 1-bit vs 2-bit at
>   matched recall was 2.70× / 6.13× contended and is **2.75× / 6.09×** clean.
>   The headline **2.7–6.1×** range is unchanged.
>
> So the § 0 tables below are correct as ratios and ~15 % pessimistic in
> absolute milliseconds. The absolute numbers are left as originally published
> rather than silently restated, with this correction pointing at the clean
> artefact — the honest record is more useful than a tidy one. Still check
> `latency.contention.contended_flag` in any artefact before quoting a p50. PostgreSQL 17.9, pg_turbovec 2.7.2, `shared_buffers = 2GB`,
postmaster and driver pinned to P-cores 2-5. 250 000 × 1024-d Cohere-wiki
vectors stored as a native `turbovec.vector` column; **100 held-out
queries** (verified zero overlap with the indexed corpus); ground truth is
an exact top-100 in-DB seqscan using the *same* operator the index serves
(860 s). Every one of the 24 configs was confirmed to run a real
`Index Scan`, not a masked seqscan.

### 0.1 Storage — the reason to consider 1-bit at all

| `bit_width` | index size | bytes/vector | vs 4-bit | build |
|---|---:|---:|---:|---:|
| **1** | 33.9 MiB | **142.3** | **3.98× smaller** | 7.4 s |
| 2 | 66.9 MiB | 280.5 | 2.02× smaller | 6.8 s |
| 4 | 134.9 MiB | 565.7 | 1.00× | 7.8 s |

### 0.2 The full curve — R@10 and R@100 vs the exact-rerank window

Read this **as a function of the window**, not of the mode name (§ 3).
`p50` is server-side `Execution Time`.

| window | bw1 R@10 | bw1 R@100 | bw1 p50 | bw2 R@10 | bw2 R@100 | bw2 p50 | bw4 R@10 | bw4 p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.744 | 0.244 | 5.5 ms | 0.993 | 0.319 | 6.0 ms | 1.000 | 9.2 ms |
| 100 | 0.899 | 0.531 | 10.4 ms | 1.000 | 0.834 | 11.1 ms | 1.000 | 14.2 ms |
| 256 | 0.967 | 0.762 | 16.2 ms | 1.000 | 0.995 | 17.2 ms | 1.000 | 20.6 ms |
| 400 | 0.981 | 0.845 | 21.8 ms | 1.000 | 0.999 | 23.2 ms | 1.000 | 26.7 ms |
| 800 | 0.994 | 0.929 | 36.7 ms | 1.000 | 1.000 | 40.9 ms | 1.000 | 44.2 ms |
| 1024 | 0.994 | 0.948 | 44.8 ms | 1.000 | 1.000 | 51.3 ms | 1.000 | 55.0 ms |
| 2000 | 1.000 | 0.981 | 81.5 ms | 1.000 | 1.000 | 103.6 ms | 1.000 | 106.2 ms |
| **`auto`** (1024) | 0.994 | 0.948 | 45.4 ms | 1.000 | 1.000 | 51.7 ms | 1.000 | 54.9 ms |

### 0.3 Iso-recall — the headline

Cheapest config per `bit_width` that clears each target:

| target | bw1 | bw2 | bw4 |
|---|---|---|---|
| R@10 ≥ 0.90 | w=256, 16.2 ms, 142 B/vec | w=32, **6.0 ms**, 281 B/vec | w=32, 9.2 ms, 566 B/vec |
| R@10 ≥ 0.95 | w=256, 16.2 ms, 142 B/vec | w=32, **6.0 ms**, 281 B/vec | w=32, 9.2 ms, 566 B/vec |
| R@10 ≥ 0.99 | w=800, 36.7 ms, 142 B/vec | w=32, **6.0 ms**, 281 B/vec | w=32, 9.2 ms, 566 B/vec |

**1-bit buys 2× the storage saving of 2-bit and pays 2.7–6.1× the
latency for it.** At R@10 ≥ 0.99 it needs a **25× wider** rerank window
than 2-bit (800 vs 32) to get there. So the honest positioning is: 1-bit is
for workloads where **storage is the binding constraint and latency has
slack** — not a general-purpose default, which is exactly how the reloption
is documented.

The R@100 column sharpens this. 1-bit never reaches R@100 = 1.0 within the
swept range (0.981 at w=2000), while 2-bit hits 1.000 by w=800 and 4-bit by
w=256. **1-bit degrades faster at depth than at k=10** — if you re-rank or
paginate beyond the top-10, budget a wider window than the R@10 table
suggests.

### 0.4 The recorded predictions, scored

§ 7 was written before the run. All four held:

| | prediction | outcome |
|---|---|---|
| **P1** | 1-bit far below 2-bit at a narrow window; rerank closes it | **CONFIRMED.** w=32: 0.744 vs 0.993. w=2000: both 1.000. |
| **P2** | 1-bit needs a materially wider window for equal recall | **CONFIRMED.** R@10 ≥ 0.99 first cleared at w=800 (bw1) vs w=32 (bw2) — 25×. |
| **P3** | storage ≈ exactly 2× / 4× | **CONFIRMED.** 2.02× and 3.98×. |
| **P4** | 1-bit will not win latency at matched recall | **CONFIRMED.** 36.7 ms vs 6.0 ms at R@10 ≥ 0.99 (6.1× slower). |

P2's falsification condition was "1-bit crosses at a comparable window,
which would mean the `hi_dim_rerank` 1-bit special-case is unnecessary".
It did not: the special-case is **justified** by this data.

### 0.5 Caveats — what this run does NOT license

- **250k × 1024-d, one corpus, one host.** Not 1M, not a dim sweep. The
  1M-row table on the same host was tried first and abandoned: exact GT over
  13 GB of pgvector-typed data with a per-row cast cost ~53 s **per query**
  (~3 h for GT alone) on a 31 GB box. That is a harness-cost finding, not a
  turbovec result — see § 0.6.
- **`arnold` is a shared desktop** (load ≈ 1.7–2.2 at start). Cores 2-5
  were pinned, but absolute p50s carry that noise. The *ratios* are the
  result; treat the absolute milliseconds as indicative.
- **No IVF arm.** `bit_width = 1` with `lists > 0` composes as of 2.7.0 but
  was not swept here; this is the flat-BQ frontier only.
- **R@10 = 1.000 for 2-bit and 4-bit at almost every window** means this
  corpus/query set is *easy* at those widths — it cannot separate 2-bit from
  4-bit on recall. It separates 1-bit from both, which is what it was for.

### 0.6a IVF + 1-bit BQ (2026-09-09) — it works, and it does NOT pay at 250k

Artefact: `benches/results/bq_ivf_20260909/`. Same corpus, same 100 held-out
queries, isolated `bq_ivf` database, `lists = 512`, probes swept 8→128.
**Same contention caveat as § 0** — recall and storage stand, the p50s carry
the flag.

`WITH (lists = N, bit_width = 1)` builds and scans correctly. Storage
overhead over flat BQ is small, and is a fixed ~8.5 B/vector of coarse
centroids + cell directory regardless of `bit_width` — so it is proportionally
worst for the smallest codes:

| arm | bytes/vec | vs flat, same bw | build |
|---|---:|---:|---:|
| bw1 flat | 142.3 | — | 7.3 s |
| bw1 + lists=512 | 150.8 | **+6.0 %** | 76.3 s |
| bw2 + lists=512 | 289.0 | +3.0 % | 89.0 s |
| bw4 + lists=512 | 574.2 | +1.5 % | 50.6 s |

**The finding: IVF imposes a recall CEILING that a wider rerank window cannot
break.** Recall at the `auto` window (1024), where a default user lands:

| | bw1 R@10 | bw1 R@100 |
|---|---:|---:|
| flat | **0.994** | **0.948** |
| ivf512 probes=8 | 0.846 | 0.784 |
| ivf512 probes=16 | 0.906 | 0.852 |
| ivf512 probes=32 | 0.954 | 0.905 |
| ivf512 probes=64 | 0.978 | 0.933 |
| ivf512 probes=128 | 0.984 | 0.945 |

At `probes = 8`, bw1 saturates at R@10 = 0.846 and stays there from window 256
all the way to 2000: the true neighbours are not in the probed cells, and no
amount of exact re-ranking invents them. Each probe count has its own hard
ceiling (0.846 / 0.906 / 0.955 / 0.981 / 0.989 for 8/16/32/64/128).

**This is the mirror image of Gap-B (v1.25.0), and the distinction is the
useful part.** There, high-dim recall loss was *not* retrieval-bound — cell
recall was 0.98–0.996 and a wider exact window fixed it. Here it *is*
retrieval-bound: the window is already wide and the cells bind. Same symptom,
opposite cause. Diagnose which one you have before reaching for a knob — the
probes count and the rerank window are not interchangeable.

**Guidance: at 250k, flat BQ dominates IVF+BQ.** Flat reaches 0.994 with no
probe tuning at all; IVF needs `probes = 128` to reach 0.984 and never
catches up. IVF's whole value is bounding scan cost as `n` grows, so this is a
**scale-dependent** answer and 250k is below the crossover. It is *not* a
verdict that IVF+BQ is useless — that is precisely the error the graph kind's
early iso-beam numbers invited. The honest deliverable is a documented
boundary: **below ~1M, prefer flat BQ**; above it, unmeasured.

Incidental but operationally sharp: `bit_width = 4` + `lists = 512` at 1024-d
**OOM-killed the backend** at 20.3 GB anon-RSS on a 31 GB host with
`maintenance_work_mem = 3GB` (kernel `Out of memory: Killed process ...
(postgres)`, signal 9 — not a crash in our code). It completed in 50.6 s with
`maintenance_work_mem = 1GB` and `max_parallel_maintenance_workers = 2`. High-dim
× many-lists k-means wants a bounded `maintenance_work_mem`; the default-ish
3 GB times parallel workers is enough to get a backend killed, which presents
to an operator as an unexplained termination.

### 0.6c Dimension sweep (2026-09-09) — the 1-bit penalty shrinks sharply with dim

Artefacts: `benches/results/bq_dimsweep_20260909/` (256-d, 512-d, 1024-d, plus
a 256-d wide-window extension to w=32000). 250 000 rows and 100 held-out
queries per dim, matching § 0. Same contention caveat as § 0 — recall and
storage stand, absolute p50s are indicative.

**Validation first.** The 1024-d arm was re-measured from a fresh database and
a re-sliced corpus, and reproduced § 0's recall **bit-identically at all seven
windows** (0.744 / 0.899 / 0.967 / 0.981 / 0.994 / 0.994 / 1.000; p50s within
1–3 %). Independently re-verified by the lead against the published artefact.
That validates both the published numbers and the rebuilt, isolated harness.

**The hypothesis held, monotonically.** 1-bit R@10 at a *fixed* rerank window,
across dim:

| window | 256-d | 512-d | 1024-d |
|---:|---:|---:|---:|
| 32 | 0.394 | 0.581 | 0.744 |
| 100 | 0.577 | 0.771 | 0.899 |
| 256 | 0.729 | 0.888 | 0.967 |
| 400 | 0.782 | 0.925 | 0.981 |
| 800 | 0.866 | 0.966 | 0.994 |
| 1024 | 0.890 | 0.977 | 0.994 |
| 2000 | 0.933 | 0.990 | 1.000 |

Rises at **all seven** windows, no exception. More dimensions means more sign
bits, which means better 1-bit retrieval at equal rerank effort.

**Iso-recall: the window penalty collapses as dim rises.** Window needed to
clear R@10 ≥ 0.95, and the ratio against 2-bit:

| dim | 1-bit window | 2-bit window | penalty |
|---:|---:|---:|---:|
| 256 | **4000** | 100 | **125×** |
| 512 | 800 | 32 | 25× |
| 1024 | 256 | 32 | **8×** |

At 256-d, reaching R@10 ≥ 0.99 needs a window of **16 000** — reranking 6.4 %
of the entire 250k corpus. **1-bit is effectively unusable at 256-d and below.**
At 1024-d the penalty is 8×, which is a real trade rather than a
disqualification. This is the sharpest practical guidance the BQ work has
produced: **1-bit is a high-dimension technique.**

**Storage advantage also grows with dim** — the opposite of the sweep's own
prediction:

| dim | 1-bit B/vec | vs 2-bit | vs 4-bit | `dim/8` ideal | overhead |
|---:|---:|---:|---:|---:|---:|
| 256 | 41.65 | 1.902× | 3.513× | 32.0 | 30 % |
| 512 | 75.20 | 1.946× | 3.730× | 64.0 | 18 % |
| 1024 | 142.31 | 1.971× | 3.975× | 128.0 | 11 % |

1-bit carries fixed per-index overhead (the corpus mean, ids, meta) that
dilutes its `dim/8` edge at small dim and amortises away as dim grows. So both
axes — recall *and* storage — favour 1-bit more strongly at higher dimension.

**Depth degrades worse at low dim.** R@100 at the `auto` default: 0.455
(256-d), 0.737 (512-d), 0.948 (1024-d). At 256-d the default loses **more than
half** the true top-100.

> **⚠ Caveat that bounds all of the above: the 256-d and 512-d corpora are
> PREFIX SLICES of the 1024-d Cohere-wiki embedding, not natively-trained
> embeddings of those dimensions.** A native 256-d model concentrates its
> information into 256 coordinates; truncating a 1024-d vector keeps only the
> first quarter of a representation that was spread across all of them. That
> almost certainly makes the sliced low dims look **worse** than a native model
> would. So the measured trend is an **upper bound** on the penalty's
> dim-sensitivity — directionally sound, magnitude not transferable to native
> low-dim models. Nothing was padded to fabricate a higher dim.

Prediction scoring, recorded before the run: the hypothesis (P-D2) was
confirmed and understated, but **three of four predictions were wrong in some
respect** — the storage direction was backwards (P-D1), the `hi_dim_rerank`
mechanism was misnamed (P-D3, which surfaced the § 3 documentation error), and
the latency-vs-dim shape was non-linear rather than linear (P-D4). Pre-registration
earned its keep here precisely by being wrong in public.

### 0.6e 1M REAL corpus (2026-09-10) — the IVF+BQ crossover EXISTS, and the *benefit* is 1-bit-only

> **Read this heading precisely — it is about which `bit_width` BENEFITS, not
> which is SUPPORTED.** A production user read the original phrasing
> ("the crossover is 1-bit-only") as meaning IVF cannot be combined with
> `bit_width = 4`, and asked whether support could be added. It already exists.
>
> **`WITH (lists = N)` works with every `bit_width`, and always has.** 4-bit IVF
> is the *original* IVF path — out-of-core end-to-end since v1.13.0 — and 2-bit
> and 3-bit work too. The only `bit_width`/kind combination the code rejects is
> `bit_width = 1` with `graph = true` (`options.rs`). Nothing about IVF is
> gated on `bit_width`.
>
> What is 1-bit-only is the **measured latency win** below. See § 0.6g for what
> that means for a 4-bit user, and for the honest limits of what was measured.

Artefacts: `benches/results/bq_1m_20260910/`. Corpus:
**CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (en), 1 000 000 × 1024-d**
— a real embedding corpus, not synthetic. 100 held-out queries. AWS
`r7i.8xlarge` (32 vCPU, 247 GB, **AVX-512**), so latency is publishable.

**Resolvability gate passed** (§ 0.6d): nn1→nn100 spread **154–203 %** across
5 queries, versus 6.6–10.4 % for the synthetic corpus that had to be discarded.
This corpus is rankable and the recall numbers mean something.

> **⚠ Cross-scale caveat — this is NOT the same corpus as the 250k runs.** The
> 250k arms (§ 0, § 0.6a, § 0.6c) used the pre-existing `bench_wiki` table on
> `arnold`, an older Cohere wiki snapshot. `Cohere/wikipedia-22-12-en-embeddings`
> is now **gated (HTTP 401)** and unreachable without credentials, so the 1M run
> used `CohereLabs/wikipedia-2023-11-embed-multilingual-v3` — same publisher and
> dimensionality, but a **different model** (`embed-multilingual-v3`) and
> snapshot.
>
> So any 1M-versus-250k delta below **conflates scale with a corpus change** and
> must not be read as a pure scale effect. The **within-run flat-versus-IVF
> comparisons are unaffected**, because both arms of each run share one corpus —
> and those are the comparisons the conclusions rest on. Where a cross-scale
> number appears (e.g. IVF's storage overhead halving, or the per-probe ceilings
> landing close to the 250k figures) treat it as suggestive, not measured.

#### Storage: IVF's overhead halves at 1M

| arm | bytes/vec | index | build |
|---|---:|---:|---:|
| bw1 flat | 140.5 | 141 MB | 17.6 s |
| bw1 + `lists=1024` | 144.8 | 145 MB | 94.2 s |
| bw2 flat | 278.9 | 279 MB | 14.2 s |
| bw2 + `lists=1024` | 283.2 | 283 MB | 100.2 s |

**+3.1 %** at 1M, against +6.0 % at 250k — the fixed coarse-centroid and
cell-directory cost amortises over 4× the rows.

#### The per-probe recall ceiling is structural, not a small-corpus artefact

max R@10 for bw1: flat **1.000**; `lists=1024` at probes 8/16/32/64/128 =
0.846 / 0.904 / 0.944 / 0.971 / **0.986**. Essentially identical for bw2. These
are within a hair of the 250k ceilings (0.846/0.906/0.955/0.981/0.989), so
cell-restricted search caps recall by construction at both scales.

#### THE ANSWER: yes at 1M, but only for 1-bit and only up to R@10 ≈ 0.95

| target | bw1 flat | bw1 IVF (`lists=1024`) | verdict |
|---|---|---|---|
| R@10 ≥ 0.90 | 25.6 ms (w=100) | **13.7 ms** (p=64, w=100) | **IVF wins 47 %** |
| R@10 ≥ 0.95 | 33.2 ms (w=256) | **20.7 ms** (p=128, w=256) | **IVF wins 38 %** |
| R@10 ≥ 0.98 | **39.3 ms** (w=400) | 44.2 ms (p=128, w=800) | IVF loses 12 % |
| R@10 ≥ 0.99 | **56.8 ms** (w=800) | unreachable | flat only |

**Verified robust to contention:** re-computing using only rows *without*
`contended_flag` gives the identical 47 % and 38 % wins, so this is not a load
artefact. 46 of 96 rows are flagged, and the cause is benign and identified:
`cpu_busy_pct` on flagged rows is only **3.1–3.2 %** with `cpu_steal ≤ 0.01`, so
it is not saturation. Two sources — loadavg *decaying* after each parallel
k-means build (visible as a monotonic decline across consecutive rows:
3.96 → 3.72 → 3.50 → 3.30 → 2.95 → 2.65 → 2.26), and an RSS sampler that forked
a Python interpreter every 2 s and pinned loadavg at exactly 1.00 for the whole
of arm 1.

> **⚠ A trap in that verification method, worth knowing before you reuse it.**
> "Filter to unflagged rows and re-check" is only sound if the **baseline
> survives the filter**. On the `lists = 4096` arm it does not: the bw1 *flat*
> rows all ran first, while load was still decaying from the build, so **all 8
> of them are flagged and 0 survive** — an unflagged-only comparison there would
> silently drop the baseline entirely and "prove" IVF wins against nothing.
>
> The check *was* valid for the headline above: on the `lists = 1024` arm all 8
> bw1 flat rows are unflagged (loadavg 1.0, busy 3.2 %), so both sides of the
> comparison survived. That was partly luck of execution order. **Always assert
> the filtered baseline is non-empty before trusting a filtered comparison** —
> the same lesson as asserting the control arm is non-trivial in the WAL test
> (`docs/TESTING.md`).

**For 2-bit the answer is a clean no.** Flat is 4.8 ms at *every* target and IVF
never gets closer than 16 ms. 2-bit's coarse ranking is accurate enough to need
only `w=32`, so its full scan is already cheap and cell-restriction is pure
overhead.

#### Why, mechanically

At 250k, flat dominated IVF+BQ at every target (§ 0.6a). The crossover appears
at 1M only where **both** conditions hold: flat's `O(n)` scan has grown
expensive, **and** the quantizer is lossy enough to need a wide rerank window.
1-bit at 1M needs w=100–256 evaluated over 1M rows; restricting that to 64–128
cells of ~1000 rows each is a real saving. 2-bit needs only w=32, so there is
nothing to save.

So the guidance is **two-axis**, not one:

- `bit_width = 1`, n ≳ 1M, recall target ≲ 0.95 → **use `lists = N`**
- `bit_width = 1`, recall target ≳ 0.98 → **flat** (IVF cannot reach it at all)
- `bit_width ≥ 2` → **flat**, at least to 1M

#### Build memory at 1M — treat these as lower bounds

`maintenance_work_mem = 4GB` with 4 parallel maintenance workers held every
build comfortably inside 247 GB with **no OOM** (contrast the 250k run, where an
unbounded `3GB` setting got a backend OOM-killed at 20.3 GB on a 31 GB box — see
`docs/PRODUCTION.md`). Sampled peak anon-RSS:

| arm | peak anon-RSS |
|---|---:|
| bw1 flat | ≥ 6.6 GiB |
| bw2 + `lists = 1024` | ≥ 41.7 GiB |
| bw1 + `lists = 4096` | **not measured** |

Stated as **lower bounds on purpose**: the sampler polled every 2 s, so a
shorter spike could have been missed entirely, and it was stopped before the
second arm — so the 4096 build has no measurement at all. Do not quote these as
peaks.

#### `lists = 4096` is WORSE than `lists = 1024` — more lists is not better

The second arm landed, and it is a useful negative:

| | `lists = 1024` | `lists = 4096` |
|---|---:|---:|
| bytes/vector | 144.8 | **157.4** |
| build | 94.2 s | **1066.7 s** (11×) |
| p50 at R@10 ≥ 0.90 | **13.7 ms** (p=64) | 21.5 ms (p=256) |
| p50 at R@10 ≥ 0.95 | **20.7 ms** (p=128) | 27.9 ms (p=256) |
| p50 at R@10 ≥ 0.98 | 44.2 ms | **103.6 ms** (flat is 38.3) |

Quadrupling the list count made every axis worse: 8.7 % more storage, **11×**
the build time, and ~50 % higher latency at matched recall. The reason is the
per-probe ceiling again — at 4096 lists each cell holds ~4× fewer rows, so
reaching a given recall needs ~4× the probes (`p=256` where 1024 lists needed
`p=64`), and probing 256 cells costs more than probing 64 larger ones. The
`lists ≈ sqrt(n)` rule of thumb (≈1000 at 1M) is doing real work here; going
above it is a pure loss for BQ.

Caveats: one corpus, one dimension, two list counts. The 250k boundary in
§ 0.6a stands — this refines it rather than replacing it.

### 0.6b Answered: real 1M scale (see § 0.6e)

The "does the trade hold at 1M?" question is **closed** — § 0.6e measured it on
a real 1M × 1024-d Cohere corpus. Retained here only for the negative result
that preceded it:

- A **synthetic** 1M attempt was discarded, not published, because the generated
  corpus was statistically unrankable (nn1→nn100 spread 6.6–10.4 % versus
  37–268 % on a real corpus). Post-mortem, and the resolvability probe that now
  gates every synthetic corpus, in
  `benches/results/bq_scale_20260909/DISCARDED.md`.

  **Storage and build from that arm ARE valid** (they do not depend on corpus
  geometry) and were the salvage:

  | 1M × 768-d | build | bytes/vec | peak build RssAnon |
  |---|---:|---:|---:|
  | bw1 | 167.3 s | 104.87 | 8.98 GiB |
  | bw2 | 152.0 s | 209.72 | 5.95 GiB |
  | bw4 | 161.0 s | 404.76 | 6.18 GiB |

  `bw2/bw1 = 2.000` **exactly**, confirming the `dim/8` sign-code stride at 1M.
  Note bw1 has the **highest** peak build RSS despite the smallest output: BQ
  reads the corpus back resident to compute the corpus mean before it can take
  signs, so build memory tracks `n · dim · 4` regardless of bit width.

Still genuinely open: a **dimension sweep at 1M** (§ 0.6c covered 250k), and
`lists = 4096` at 1M.

### 0.6f Ground-truth build: two plan defects, fixed and validated

The harness's exact-GT query had two compounding parallelism defects. Both are
fixed; ground truth is **byte-for-byte unchanged** at each step, so no published
recall figure moves.

**Defect 1 — the window function forced a leader-side sort.** The original used
a `CROSS JOIN LATERAL` correlated on `q.qvec` with
`row_number() OVER (ORDER BY <dist>)` inside, which forces
`WindowAgg → Sort → Gather`: workers ship **every row** to the leader and the
leader sorts the whole corpus single-threaded. Verified with `EXPLAIN ANALYZE`
on real 1024-d data, 250k × 20 queries: `Gather (actual rows=250000, loops=20)`
— 5 M rows crossing the Gather — with a **13.9 MB leader quicksort** per query,
**148.2 s**. Replaced with one statement per query whose vector is a scalar
subquery, making it an **InitPlan constant** so each worker top-N sorts its own
share (`Gather Merge` → per-worker top-N heapsort, 33 kB): **76.8 s, 1.93×**.

**Defect 2 — `INSERT` silently threw that away again.** My first version of the
fix wrapped the corrected SELECT in `INSERT INTO ... SELECT`. PostgreSQL
generates **no parallel plan for any statement that writes data**; the only
exemptions are `CREATE TABLE AS`, `SELECT INTO` and `CREATE MATERIALIZED VIEW`
(`doc/src/sgml/parallel.sgml`, "The query writes any data"). Measured, same
inner query:

| form | plan | time |
|---|---|---:|
| `INSERT INTO ... SELECT` | **serial** `Seq Scan` | **7.49 s** |
| `CREATE TABLE AS` | `Gather Merge`, 6 workers, 33 kB top-N | **3.99 s** |

The harness now CTASes each query's top-`gt_depth` into a temp table, then does
a cheap `INSERT ... SELECT` of those few rows.

**Validated at 1M scale by accident, which is the best kind.** The two arms of
the 1M run straddled the fix: arm 1 built GT with the pre-fix `INSERT` path in
**3755.7 s**, arm 2 with CTAS in **275.4 s** — **13.6×**, matching the 12×
predicted from the plan shape. The **16 flat configs the two arms share
reproduce bit-identically** across the two GT implementations (R@10 and R@100
both, zero mismatches) — third-party confirmation the fix changes no measured
number.

**Correctness gate, applied at both steps.** GT is what every recall number is
measured against, so equality was the requirement rather than a nicety: over
10 queries × top-100, `old EXCEPT new` and `new EXCEPT old` on
`(qid, hit_id, rk)` are **both 0**, and membership ignoring rank is 0 — checked
for the InitPlan version *and* again for the CTAS version.

Two deliberate properties: it orders by `(distance, id)` rather than distance
alone, so tie order is **deterministic** instead of plan-dependent (distances
tie often at 1-bit, and an arbitrary tie order would let recall depend on
planner choice); and `rk` comes from `row_number() OVER ()` over the ordered
subquery rather than a window function inside the scan, which is what lets the
sort push into the workers.

**The lesson, recorded because it was mine:** I measured a bare `SELECT`, got
1.93×, then shipped an `INSERT` — a different statement with a different plan —
and never re-timed. The 1M run caught it from `pg_stat_activity` (one backend at
99.9 % CPU, zero parallel workers). **Benchmark the statement you are going to
ship, not a proxy for it.**

### 0.6g "I'm on `bit_width = 4` and I want IVF" — you already have it

Prompted by a production report. Three separate questions get conflated here,
so they are answered separately.

**1. Is it supported?** **Yes, and it always has been.** `WITH (lists = N)`
composes with every `bit_width`. 4-bit IVF is the *original* IVF path
(out-of-core end-to-end since v1.13.0); 2-bit and 3-bit work as well. The only
`bit_width`/kind combination the code rejects is `bit_width = 1` with
`graph = true`. There is nothing to enable and nothing to wait for.

**2. Will it make your queries faster?** **No — now measured, not predicted.**
`bit_width = 4` + `lists = 1024` was swept at 1M on 2026-09-11
(`benches/results/bq_1m_bw4ivf_20260911/`, 48 configs, AVX-512, all confirmed
`Index Scan`). Flat wins at every target:

| target | bw4 flat | bw4 `lists = 1024` | |
|---|---|---|---|
| R@10 ≥ 0.90 | **6.08 ms** (w=32) | 16.04 ms (p=64) | flat by 62 % |
| R@10 ≥ 0.95 | **6.08 ms** (w=32) | 16.13 ms (p=128) | flat by 62 % |
| R@10 ≥ 0.98 | **6.08 ms** (w=32) | **unreachable** | flat only |
| R@10 ≥ 0.99 | **6.08 ms** (w=32) | **unreachable** | flat only |

**Lead with the ceiling, not the milliseconds — the conclusion does not depend
on any timing at all.** IVF cannot reach R@10 ≥ 0.98 at any setting, and at
`probes = 128` widening the rerank window from 32 to 2000 leaves recall at
**exactly 0.959 across all 8 windows** — it does not move by a single query,
because the true neighbours are not in the probed cells. Recall is
CPU-independent, so **discard every latency number and bw4 IVF still loses at
the top two targets.** The 62 % latency gap is corroboration, not the argument.

Why flat is unbeatable here: its *cheapest* config is also its *most accurate*
(**R@10 = 1.000 at window 32, 6.08 ms**), so IVF never gets an opening. That is
the mechanism § 0.6g originally predicted — the crossover needs both an
expensive `O(n)` scan **and** a quantizer lossy enough to demand a wide window,
and 4-bit needs only 32 where 1-bit needs 800. 2-bit measured the same way at
1M, so all three bit widths now agree.

For completeness, the single sub-flat p50 in 48 rows: at window 2000, IVF is
116.77 ms versus flat's 122.19 ms (0.96×). It is **not** a win — 0.959 recall
against flat's 1.000, so it fails a matched-recall comparison — but it is
published here rather than left for a reader to find. The tightest honest
statement of "never wins": IVF's fastest configuration *anywhere* is 15.76 ms at
R@10 = 0.875, against flat's 6.08 ms at R@10 = 1.000.

**3. What could it cost you?** Two things worth weighing before enabling it on
a production index:

- **IVF caps recall, and no knob recovers it.** Cell-restricted search has a
  hard per-probe ceiling — 0.986 at `probes = 128`, with R@10 ≥ 0.99
  **unreachable at any setting** (§ 0.6e). Widening the rerank window does not
  help, because the true neighbours are not in the probed cells at all. If you
  chose 4-bit you are probably recall-sensitive, so IVF may *cost* you accuracy
  rather than buy you latency.
- **Build cost, now measured at 1M — and the OOM worry is retired for this
  configuration.** With `maintenance_work_mem = '4GB'` and
  `max_parallel_maintenance_workers = 4` at 1M × 1024-d:

  | arm | bytes/vec | index | build | peak build anon-RSS |
  |---|---:|---:|---:|---:|
  | bw4 flat | 559.95 | 534 MB | **14.89 s** | **2.641 GiB** |
  | bw4 `lists = 1024` | 564.17 | 538 MB | **101.22 s** (6.8×) | **2.154 GiB** |

  These are **real peaks, not lower bounds** — 0.25 s sampling, 466 in-build
  samples, from a pure-bash sampler that left idle loadavg at 0.00–0.01. Note
  IVF's peak is *below* flat's, and 2.6 GiB is nowhere near the **20.3 GB** an
  *unbounded* `3GB` setting reached on the 250k `bw4 + lists = 512` build (which
  was OOM-killed on a 31 GB box). **So the memory hazard is about leaving
  `maintenance_work_mem` unbounded, not about 4-bit IVF itself** — bound it and
  this configuration is cheap. The real cost is the **6.8× build time**, which
  matters more if you rebuild often. See `docs/PRODUCTION.md`.

  Storage overhead here is **+0.75 %**. Do **not** pair that with § 0.6e's
  +3.1 % as though it were a bit-width effect — those are different runs and the
  resolvability caveat below applies.

> **Caveat on comparing this arm to § 0.6e.** Its resolvability spread was
> **18.6–69.1 %**, well below the **154–203 %** § 0.6e measured on nominally the
> same corpus and shards. Both clear the ~10 % unusable floor, so each run's
> *internal* flat-versus-IVF comparison stands — but the discrepancy is
> **unexplained**, so do not treat this arm and § 0.6e as the same corpus.
> Cross-run deltas between them (storage overhead, ceilings) are suggestive
> only. This is the second corpus-identity surprise in this document; the first
> forced the v2.8.1 corrections.
>
> Two further notes for anyone re-running: all 48 latency rows carry
> `contended_flag = true` from loadavg decaying after the 438 s parallel GT build
> (`cpu_busy` 3.8–40.2 %, `cpu_steal` ≤ 0.03 %), and the unflagged-row filter is
> **unavailable** here — 0 of 8 flat and 0 of 40 IVF rows survive it, exactly the
> trap § 0.6e warns about. The bias runs *against* IVF (flat ran at 22.0 % mean
> `cpu_busy` versus IVF's 6.1 %, a 3.6× difference in CPU busy, or 1.87× in mean
> loadavg) and flat still won by 62 %, so a quiet re-time could only widen flat's
> margin. And GT took **438.4 s**, not the ~275 s § 0.6f reports — it was on the
> parallel-CTAS path, so this is not a harness regression, but the 1.6×
> discrepancy is unexplained and is flagged so a future diff does not misread it.

**Recommended course:** measure it on your own data rather than trusting either
the measurement above or a published curve. Build `WITH (lists = N)` on a copy
with `maintenance_work_mem` bounded, and compare against your existing flat
index **at your recall target** — not at whatever target a benchmark chose. That
is decisive in a way this document cannot be, especially since (per § 0.6e's
cross-scale caveat) the published 1M numbers come from a different corpus than
the 250k ones. Your corpus is the only authority for your workload.

**What this arm does and does not settle.** If your recall target is **≥ 0.98**
you are answered at *any* scale: the per-probe ceiling is structural, not a size
effect, so IVF cannot reach it for 4-bit however large `n` grows. If your target
is **≲ 0.90 and your corpus is well above 1M**, this arm does **not** answer you
— flat's `O(n)` scan keeps growing while IVF's probed-cell cost does not, so a
crossover could appear further out. Nothing here measures 4-bit IVF beyond 1M.

This arm also extends the "`bit_width ≥ 2` → flat" half of § 0.6e's two-axis
rule from 2-bit to 4-bit at 1M; the 1-bit half is unchanged.

### 0.6d MANDATORY pre-flight for any synthetic corpus

Learned the expensive way (a 6.4-hour ground-truth build whose recall numbers
were then unusable). **Before trusting a recall number from generated data, run
this and require the spread to be comparable to a real corpus:**

```sql
-- Resolvability probe: is the top-100 actually rankable, or is it a tie?
WITH q AS (SELECT qid, qvec FROM <query_table> ORDER BY qid LIMIT 5)
SELECT q.qid,
       round(min(d)::numeric, 5) AS nn1,
       round(max(d)::numeric, 5) AS nn100,
       round((100.0 * (max(d) - min(d)) / greatest(min(d), 1e-9))::numeric, 2)
         AS spread_pct
FROM q, LATERAL (SELECT (x.<vec> OPERATOR(<op>) q.qvec) AS d
                 FROM <corpus> x ORDER BY 1 LIMIT 100) k
GROUP BY q.qid ORDER BY q.qid;
```

Reference values, both measured:

| corpus | nn1→nn100 spread | verdict |
|---|---:|---|
| real Cohere-wiki, 1024-d | **37–268 %** | rankable |
| synthetic 200-cluster Gaussian, 768-d | **6.6–10.4 %** | **unusable** |

Under ~10 % the top-100 is a statistical tie, no quantizer can order it, and
"recall" measures tie-break order rather than retrieval quality.

**Two checks, not one.** `is_degenerate()` (in `onebit.rs`) asks whether every
sign code is identical — a corpus can pass that and still be unrankable. Only
this probe predicts whether a recall number will mean anything. And note that
cluster *separation* is not sufficient either: the discarded corpus had
well-separated clusters (0.108 within vs 0.990 across) and still failed,
because 5000 iid points *inside* each cluster concentrated at d = 768.

### 0.6 Harness finding worth keeping

The driver's `--vec-expr` accepts an arbitrary SQL expression, and a cast
like `(emb::real[]::turbovec.vector)` is evaluated **per row per query** —
200 M casts for a 1M × 200-query GT build. Materialising a native
`turbovec.vector` column first took one query from **53 s → 2.7 s (20×)**.
If you run this on a pgvector-typed corpus, materialise first; do not pass a
cast as the vec-expr.

---

v2.6.0 shipped `WITH (bit_width = 1)` (sign binary quantization: `dim/8`
bytes per vector, no per-vector scale, Hamming coarse ranking + exact
re-rank). Its correctness, storage ratio and end-to-end scan behaviour are
covered by `#[pg_test]`s.

**Its frontier is now measured — see § 0 above, which supersedes the
"unmeasured" framing this section was originally written with.** The sections
from § 1 onward are the *runbook*: the methodology a run must follow, retained
because it is what § 0's numbers were produced by and what any re-run should
repeat. Read § 0 for results, § 1–6 for how to get them, and § 0.5 / § 0.6d for
what they do and do not license.
Every number-shaped thing below is either an input you set, a formula, or a
labelled *prediction* recorded in advance (§7) so a real run confirms or
falsifies something written down rather than being interpreted after the fact.

Deliverables of an actual run: a JSON artefact under `benches/results/` in
the shape `benches/scripts/bq/bq_frontier.py --print-schema` emits, and a
results section appended to `docs/RECALL.md` next to the existing
`bit_width` comparison tables.

---

## 1. The harness

| file | role |
|---|---|
| `benches/scripts/bq/bq_frontier.py` | the sweep driver. Builds one index per `bit_width`, measures R@k vs exact GT, `pg_relation_size`, bytes/vector, build wall-clock, and (AVX2 only) warm p50/p95/p99 + 1-conn QPS. |
| `benches/scripts/bq/run_bq_frontier.sh` | phase runner: `preflight` (the SIMD gate, out loud) → `queryset` → `sweep`. |
| `benches/scripts/bq/smoke_stub.sql` | plumbing-only smoke fixture. **Measures nothing** — a SQL-cosine stub with no turbovec index, used to prove the driver executes. |

Self-checks that run anywhere, with no database and no numpy:

```bash
python3 benches/scripts/bq/bq_frontier.py --self-check      # pure logic
python3 benches/scripts/bq/bq_frontier.py --print-schema    # all-null output shape
python3 benches/scripts/bq/bq_frontier.py --dry-run-sql \
    --dsn x --out /tmp/x.json --dim 1536 --query-provenance held_out
```

`--self-check` asserts the SIMD gate refuses latency without AVX2, and that
this driver's re-rank-window mirror agrees with every case in
`src/guc.rs::hi_dim_rerank_tests`. If `guc.rs` ever changes, the self-check
failing is the bug report.

---

## 2. WHICH HOST — the constraint that decides whether the numbers are worth anything

Per `AGENTS.md` § "Bench hosts":

| axis | valid on `meh` (pre-AVX2) | valid on `arnold` (AVX2) |
|---|---|---|
| R@10 / R@100 | **yes** (CPU-independent) | yes |
| index storage, bytes/vector | **yes** | yes |
| build wall-clock | yes | yes (different absolute numbers) |
| peak build RSS | yes | yes |
| **warm p50 / p95 / p99, QPS** | **NO** | **yes** |

turbovec dispatches AVX-512 > AVX2 > scalar at runtime. The scalar fallback
is *correct* (since v1.7.3) but roughly **1000× slower** for a full-corpus
scan. A "warm p50" measured on `meh` would be a property of the scalar
fallback, not of the shipped product, and publishing one has burned this
project before (§ "Caveats" in `docs/BENCHMARKS.md`).

**The harness enforces this structurally rather than by convention:**

1. `run_bq_frontier.sh preflight` reads `/proc/cpuinfo`, prints the kernel
   tier, and on a scalar host prints a loud block saying latency is not
   publishable from it.
2. `bq_frontier.py` classifies the host itself and, on a scalar host,
   **does not time queries at all**. Each config row carries
   `latency: {"measured": false, "reason": "..."}` — never a null that could
   read as "fast".
3. `--force-scalar-latency` exists for recording the scalar floor
   deliberately. It files the timings under the key
   `latency_scalar_fallback_NOT_PUBLISHABLE`, sets
   `publishable: false` inside it, leaves `latency.measured = false`, and
   omits the flat `p50_ms` key the derived tables read. You cannot quote it
   as a result by accident.
4. `meta.latency_publishable` is a top-level boolean in every artefact.

**Recommended split:** run the recall + storage + build legs anywhere
convenient (including `meh`, which has the RAM for a big corpus); run the
latency leg on **`arnold`** only. Both artefacts archive; the `meta.simd`
block says which is which.

---

## 3. The re-rank window is a swept variable, not a hidden default

`src/guc.rs::hi_dim_rerank_candidate_count` treats a 1-bit index as high-dim
at **any** `dim` — though note the *effect* is confined to `dim < 256`
(see below):

```rust
let effective_dim = if bit_width == 1 { dim.max(HI_DIM_RERANK_MIN_DIM) } else { dim };
...
let floor = effective_dim.clamp(HI_DIM_RERANK_MIN_DIM /*256*/, 1024);
user_count.max(floor)
```

So `hi_dim_rerank = auto` (the shipped default) widens BQ's exact re-rank
window to `clamp(max(dim,256), 256..=1024)` candidates — 256 even at
SIFT-128 — whereas a 2-bit index at `dim < 256` gets no widening at all.

**Corrected 2026-09-09, and the correction changes how to read § 0.** That
asymmetry exists **only below 256-d**. At `dim >= 256` the 1-bit and 2-bit
`auto` windows are *identical* (`clamp(dim, 256..=1024)` either way), so the
1-bit special case is a **no-op** there. Consequences:

- A 1-bit-vs-2-bit comparison at `dim >= 256` and default settings compares
  **equal** windows — the § 0 (1024-d) and § 0.6c (512/1024-d) results are
  therefore quantizer-vs-quantizer, not knob-vs-knob. That makes them
  *stronger* than originally claimed, not weaker.
- Below 256-d the windows genuinely differ and any such comparison must
  control for it explicitly.

This was found by re-deriving the clamp against the driver's mirror of
`guc.rs` during the dim sweep. The sweep's own prediction P-D3 named the
wrong mechanism, and chasing that down is what surfaced the doc error.

This is exactly the v2.2.0 failure mode: the graph kind's beam was
`(candidate_count * 4).max(64)`, so `hi_dim_rerank = auto` bought it a
3840-wide beam at 960-d that no documented knob named, cost 194 ms p50, and
nobody knew (`docs/GRAPH_EF_BENCH.md` § 1).

The harness therefore:

- sweeps `hi_dim_rerank = off` with an **explicit** `search_k` in
  `{32, 100, 256, 400, 800, 1024, 2000}` — the window is exactly what was set;
- adds one `hi_dim_rerank = auto` row per index with `search_k` **left at its
  default**, because an explicit `search_k` override defeats the floor
  (`user_count.max(floor)` — the user's number wins when larger). This row is
  the default a real user gets;
- records `rerank_window_predicted` on **every** row, computed by this
  driver's mirror of the Rust clamp, so no row's cost can be explained after
  the fact by a window nobody wrote down;
- records `search_k`, `search_k_effective`, `oversample`, `probes`,
  `hi_dim_rerank` mode and the observed GUC defaults in `meta`.

Read the frontier **as a function of `rerank_window_predicted`**, not as a
function of the mode name. Two rows with the same window and different modes
should land in the same place; if they don't, that is a finding.

---

## 4. Comparing 1-bit against 2-bit and 4-bit

The interesting question is not "is 1-bit recall high" — 1-bit is expected to
be lossy. It is **what the recall/storage/latency curve looks like versus
`bit_width = 2` and `4`**. Three comparisons, all three required:

### 4a. Iso-knob (the full curve)

Every `(bit_width, hi_dim_rerank, search_k, probes)` row is emitted. This is
the raw material; it is *not* the headline. An iso-knob table can flatter a
lossy scheme by comparing it at a knob setting that happens to give it a
wider effective window.

### 4b. Iso-recall (the headline)

`derived.iso_recall_flat` picks, per `bit_width`, the **cheapest config that
clears** each of R@10 ≥ 0.90 / 0.95 / 0.99, and reports the storage and p50
of that config. `null` means that `bit_width` **never cleared the target on
this corpus** — that absence is the result and must be reported as such, not
softened.

Cost basis is warm p50 when latency was publishable, else
`rerank_window_predicted` (the CPU-independent proxy for scan + recheck
work); each entry records which via `cost_basis`.

This ordering is deliberate. Iso-*beam* comparisons made the graph kind look
competitive while the iso-*recall* comparison showed flat beating it on both
axes on both corpora, which is what got the kind deprecated in v2.5.0
(`docs/GRAPH_EF_BENCH.md` § 5.2). **Do not repeat that. Publish the iso-recall
table as the headline and the full curve as the appendix.**

### 4c. Iso-storage (a second run, not a second column)

1-bit stores `dim/8` B/vec; 2-bit stores `dim/4 + 4` B/vec. At an equal byte
budget, 1-bit holds roughly **twice the rows**. So "1-bit at n rows vs 2-bit
at n rows" understates 1-bit's real offer, which is *more corpus per byte*.

Run the driver twice against two tables and compare at equal `idx_bytes`:

```bash
# arm A: 2n rows at bit_width = 1
BQ_TABLE=docs_2n BQ_EXTRA="--bit-widths 1" bash run_bq_frontier.sh sweep
# arm B: n rows at bit_width = 2
BQ_TABLE=docs_n  BQ_EXTRA="--bit-widths 2" bash run_bq_frontier.sh sweep
```

Ground truth is per-table, so the two arms answer different retrieval
questions — say so when reporting. The honest framing is "at X GB of index,
1-bit serves 2n rows at R@10 = a and 2-bit serves n rows at R@10 = b", not a
single ratio.

---

## 5. What the corpus and query set must be

- **Real embeddings, not synthetic random.** Sign-BQ's whole premise is that
  a coordinate's sign carries signal. Synthetic uniform-random vectors have
  no cluster structure and produce a curve that generalises to nothing. Use
  a public corpus already in this repo's rotation:
  `dbpedia-entities-openai-1M` (1536-d, cosine, unit-norm;
  `benches/scripts/download_dbpedia.py` + `load_dbpedia_1M.py`), and/or
  GIST-960 and Cohere-wiki-1024 for a dim sweep.
- **Include at least one dense-positive corpus** (GIST features are
  non-negative). That is where the centering footgun lives: raw
  `sign(coord)` sets every bit and measures R@10 = 0.0
  (`benches/results/bq_20260710/`), while mean-centering — which v2.6.0 does
  — is what makes it work. A run that only uses zero-centered text
  embeddings never exercises the load-bearing part of the feature.
- **Held-out queries, strongly preferred.** With in-corpus queries rank 1 is
  trivially the query itself, so R@10 carries a 1/10 floor for *any* index
  and the 1-bit-vs-2-bit gap is compressed toward zero — the caveat
  `docs/RECALL.md` § 2.2 already flags. `run_bq_frontier.sh queryset` refuses
  `held_out` without a held-out source table and records
  `query_set.provenance` + its caveat in the artefact either way.
- **Ground truth is computed in-DB, by exact seqscan, with the SAME operator
  the index serves.** No BLAS, no `.npy`, no h5py. This is deliberate: in
  `docs/GRAPH_EF_BENCH.md` scoring cosine results against a published *L2*
  ground truth produced a recall deficit that had nothing to do with the
  feature under test. Using one operator for both sides makes that class of
  error impossible.
- **≥ 200 queries.** The existing 50-query dbpedia run is thin for resolving
  the few-percentage-point recall differences this comparison hinges on.

`bit_width = 1` currently **rejects** `lists > 0` (IVF) and `graph = true`.
The driver's optional `--ivf-lists` arm records that rejection as a finding
(`indexes[].build_error`) rather than crashing, so the artefact documents the
current boundary.

---

## 6. How to run it

### 6.1 Recall + storage + build (any host)

```bash
export BQ_DSN="host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_dbpedia"
export BQ_TABLE=docs
export BQ_VEC_EXPR='(emb::real[]::turbovec.vector)'   # or a turbovec.vector column
export BQ_DIM=1536
export BQ_OPCLASS=vec_cosine_ops
export BQ_OPERATOR='turbovec.<=>'
export BQ_HELDOUT_TABLE=docs_heldout                  # NOT in BQ_TABLE
export BQ_QUERY_PROVENANCE=held_out
export BQ_N_QUERIES=200
export BQ_OUT=/scratch/bq/bq_frontier_$(hostname)_$(date -u +%Y%m%d).json

bash benches/scripts/bq/run_bq_frontier.sh preflight    # READ THIS OUTPUT
bash benches/scripts/bq/run_bq_frontier.sh queryset
```

The sweep runs for many minutes (a 1M × 1536-d build alone is minutes per
`bit_width`), so wrap it per `.pi/skills/long-running-bench/SKILL.md`:

```bash
nohup bash benches/scripts/lib/with-heartbeat.sh /scratch/bq/sweep.log \
    bash benches/scripts/bq/run_bq_frontier.sh sweep > /dev/null 2>&1 &

# poll (never `tail -f`, never a pager — it wedges sub-agents)
bash benches/scripts/poll-heartbeat.sh /scratch/bq/sweep.log 60
```

The driver rewrites its JSON after **every** config, so a dropped SSH loses
at most one row.

### 6.2 Latency (`arnold` only)

Same commands, plus the isolation protocol the v1.9.1 / Phase A-2 runs used
(`docs/BENCHMARKS.md` § "Isolation method"), because `arnold` is a busy
shared desktop:

- dedicated bench postmaster started under `taskset -c 2-5` (P-cores, off
  the kernel/IRQ cores 0-1 and the E-cores); all backends inherit the mask;
- pin the driver to the same cores (`taskset -c 2-5 bash ... sweep`);
- latency basis is server-side `Execution Time` from `EXPLAIN (ANALYZE)` —
  the driver already does this; client RTT excluded;
- per-batch contention (`loadavg`, CPU busy/iowait/steal, free RAM) is
  sampled before and after and stored per row with a `contended_flag` against
  `--load-gate` (default 1.5). **Discard or re-run any batch flagged
  contended;** do not publish it.

### 6.3 Archiving

Copy the artefact to `benches/results/bq_frontier_<host>_<YYYYMMDD>.json` (or
a `bq_frontier_<date>/` directory with a `scripts/` copy, the convention
`benches/results/q1_real_20260711/` and `competitive_gist_2.0.0_20260827/`
use), then append a results section to `docs/RECALL.md` and update the
`bit_width = 1` row of the README's "Choose your `bit_width`" table. (Both were
done for the 2026-09-08/09 runs; a future re-run should update them in place
rather than appending a second set of numbers.)

### 6.4 Plumbing smoke test (proves the driver runs; measures nothing)

Never against the shared pgrx cluster — two sibling agents use it and
`cargo pgrx test` binds a fixed port.

```bash
initdb -D /tmp/bqsmoke -U bq -A trust
pg_ctl -D /tmp/bqsmoke -o "-p 54329 -k /tmp -c listen_addresses=''" start
psql -h /tmp -p 54329 -U bq -d postgres -f benches/scripts/bq/smoke_stub.sql
python3 benches/scripts/bq/bq_frontier.py \
    --dsn "host=/tmp port=54329 user=bq dbname=postgres" \
    --table docs --vec-expr emb --dim 16 --query-provenance held_out \
    --bit-widths 1 --search-k-sweep 10,50 --k 10 --k-deep 10 --gt-depth 10 \
    --n-warm 1 --skip-build --out /tmp/bq_smoke.json
pg_ctl -D /tmp/bqsmoke stop -m fast     # NEVER kill -9 a postmaster
```

The stub has no turbovec index, so the driver's plan check **warns on every
row** that the plan is a seq scan, not an Index Scan. That warning firing is
part of what the smoke test proves. Add `--pretend-scalar` to exercise the
latency-refusal path on an AVX2 box.

---

## 7. Expected shape — written down BEFORE the run

Recorded in advance so a real run can falsify it. These are **predictions,
not measurements**, and they carry no numbers that could be mistaken for
results. Their basis is the offline FAISS feasibility study in
`benches/results/bq_20260710/` (a FAISS analog, not pg_turbovec) and the
Gap-B re-rank mechanism `docs/RECALL.md` § 2.1 / v1.25.0 established.

**P1 — Raw Hamming ranking alone will be far below 2-bit; the exact re-rank
is what closes it.** The coarse rank over `dim/8` bytes has only `dim + 1`
distinct Hamming values, so ties dominate (`src/index/onebit.rs` says so).
Expect the R@10-vs-window curve for 1-bit to start well below 2-bit at a
narrow window and climb steeply as the window widens, while 2-bit and 4-bit
start high and are nearly flat.

**P2 — 1-bit needs a materially wider re-rank window than 2-bit for the same
recall.** This is the whole reason `hi_dim_rerank` treats 1-bit as high-dim
at any dim. Concretely: the window at which 1-bit crosses a given R@10 should
be well above the window at which 2-bit crosses it. *Falsified if* 1-bit
crosses at a comparable window — which would mean the special-case in
`hi_dim_rerank_candidate_count` is unnecessary and should be reconsidered.

**P3 — Storage will land near exactly 2× smaller than 2-bit and 4× smaller
than 4-bit, slightly better than the ratio at the O(1) terms.** `dim/8`
vs `dim/4 + 4` per vector, and 1-bit additionally has no scales chain, no
codebook and no rotation matrix (`docs/ONEBIT_BQ.md` § 3, unit-asserted by
`codes_stride`). *Falsified if* the measured `bytes_per_vector` ratio is not
close to 2 — that would mean the per-index O(1) overheads dominate at the
corpus size tested, or something is being persisted that shouldn't be.

**P4 — At matched recall, 1-bit will probably NOT win on latency.** The scan
kernel gets cheaper (fewer bytes swept, integer popcount vs a f32 LUT) but
the exact re-rank gets more expensive (P2: a wider window means more heap
fetches and more exact distance recomputations), and prior runs found the
re-rank, not the kernel, dominating end-to-end SQL latency
(`docs/RECALL.md` § 2.1.3). Expect 1-bit's honest pitch to be **storage at a
recall cost**, not speed. *Falsified if* 1-bit's p50 at matched recall beats
2-bit's — which would be a genuinely good result and should be checked hard
against P2 before being believed.

**P5 — The Hamming kernel is deliberately scalar** (`u8::count_ones` →
`POPCNT`; SIMD popcount is unwired, `docs/ONEBIT_BQ.md` § 6 note 7). So the
`arnold` latency leg measures a *scalar-Hamming* coarse pass plus a
SIMD-independent exact re-rank. If 1-bit's p50 is disappointing, "SIMD
popcount is not implemented yet" is a candidate explanation to check before
concluding anything about the scheme.

**P6 — A dense-positive corpus will be fine after centering and catastrophic
without it.** The v2.6.0 build always centers, so the prediction is that
GIST-960 at 1-bit works. The measurement worth making is how much of the
FAISS study's raw-sign collapse the centering recovers.

**P7 — IVF and graph arms will be rejected at build**, with a clear ERROR
(`bit_width = 1` with `lists > 0`; `bit_width = 1` with `graph = true`). The
artefact should record those rejections. *Falsified if* either builds — that
would mean the guard regressed.

---

## 8. What a completed run would and would NOT license you to claim

**Would license:**

- "On `<corpus>` at `<n>` × `<dim>`, `bit_width = 1` reaches R@10 = `<x>` at
  `<y>` bytes/vector, versus `bit_width = 2` at R@10 = `<a>` / `<b>`
  bytes/vector" — with the re-rank window stated for each.
- "At matched R@10 ≥ 0.95, the cheapest 1-bit config is `<...>` and the
  cheapest 2-bit config is `<...>`" — or "1-bit does not reach R@10 = 0.95 on
  this corpus at any window we swept", which is equally publishable.
- Storage and build-time ratios, from any host.
- Warm p50/p95/p99 and 1-conn QPS **only** from the AVX2 artefact, only for
  batches whose `contended_flag` is false.

**Would NOT license:**

- Any latency claim from a scalar-fallback host. Not "approximately", not
  "as a floor". The artefact structurally refuses to hand you one.
- A general "1-bit is good/bad" verdict. One corpus at one dim is one point.
  The dim sweep (GIST-960 / Cohere-1024 / dbpedia-1536) and at least one
  dense-positive corpus are needed before generalising.
- A comparison against pgvector's `binary_quantize()` + `bit_hamming_ops`.
  That is a different measurement (different scheme, no exact re-rank) and
  needs pgvector in the same cluster. The README's existing claim that
  `bit_width = 2` is the better replacement for `bit_hamming_ops` rests on
  the earlier dbpedia run, not on this harness.
- Multi-connection QPS. The driver measures single-connection only; a
  `qps@8`-style number needs the concurrent path (see
  `benches/results/vecbench2_20260703/scripts/bench_lib.py::measure_qps`).
- Anything about `bit_width = 1` + IVF, or + the graph kind. Both are
  rejected at build.
- Insert or VACUUM throughput on a BQ index. `aminsert` is an O(n)
  whole-relfile rewrite and VACUUM is tombstone-only
  (`docs/ONEBIT_BQ.md` § 6); neither is exercised here.

---

## 9. Output shape

`bq_frontier.py --print-schema` prints the all-null skeleton. The real
artefact matches `benches/results/`'s existing conventions: a `meta` block
(host, `simd`, PG/extension versions, corpus, query-set provenance, GT
method, protocol) plus flat per-config rows, the same shape as
`latency_frontier_arnold_cohere_1m_v1_9_0_2026_06_15.json` and
`q1_real_20260711/*.json`.

Per-config keys that exist specifically to keep this benchmark honest:

| key | why it exists |
|---|---|
| `meta.simd.kernel_tier` / `meta.latency_publishable` | the AVX2 gate, machine-readable |
| `rerank_window_predicted` | the exact-rerank window, never invisible (§ 3) |
| `search_k` vs `search_k_effective` | `null` search_k means "left at default so `auto`'s floor engages" |
| `latency.measured` + `latency.reason` | absence of latency is explicit, not a null |
| `latency_scalar_fallback_NOT_PUBLISHABLE` | scalar timings, un-quotable by accident |
| `plan.index_scan` + `plan.warning` | proves the row measured the turbovec index and not a seq scan |
| `indexes[].build_error` | a rejected combination is recorded as a finding |
| `derived.iso_recall_flat[...] = null` | "never cleared this target" is a result |
| `latency.contention.contended_flag` | a batch you must discard, flagged in place |

---

## 10. Cross-references

- `docs/ONEBIT_BQ.md` — the design, and § 6 "Still open" (this gap).
- `docs/RECALL.md` — the recall methodology this is consistent with; where
  results land.
- `docs/BENCHMARKS.md` — the host caveats and the `arnold` isolation protocol.
- `docs/GRAPH_EF_BENCH.md` — the iso-beam-vs-iso-recall lesson (§ 5.2) and
  the hidden-window lesson (§ 1).
- `benches/results/bq_20260710/` — the offline FAISS BQ feasibility study
  that motivated the feature (a FAISS analog; **not** pg_turbovec numbers).
- `AGENTS.md` § "Bench hosts" — the AVX2 rule.
- `.pi/skills/long-running-bench/SKILL.md` — the heartbeat protocol.
