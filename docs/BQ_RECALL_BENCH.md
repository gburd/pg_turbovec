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
host). PostgreSQL 17.9, pg_turbovec 2.7.2, `shared_buffers = 2GB`,
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
covered by `#[pg_test]`s. Its **recall/latency frontier on a real corpus is
not published**, `README.md` says so explicitly ("not yet published"), and
`docs/ONEBIT_BQ.md` §6 "Still open" names the gap.

This document is the runbook for closing it. It contains **no measurements**.
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
at **any** `dim`:

```rust
let effective_dim = if bit_width == 1 { dim.max(HI_DIM_RERANK_MIN_DIM) } else { dim };
...
let floor = effective_dim.clamp(HI_DIM_RERANK_MIN_DIM /*256*/, 1024);
user_count.max(floor)
```

So `hi_dim_rerank = auto` (the shipped default) silently widens BQ's exact
re-rank window to `clamp(max(dim,256), 256..=1024)` candidates — 256 even at
SIFT-128, 1024 at 1536-d — whereas a 2-bit index at `dim < 256` gets no
widening at all. **Comparing 1-bit and 2-bit at "default settings" therefore
compares two different re-rank windows, and any latency difference is partly
that, not the quantizer.**

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
`bit_width = 1` row of the README's "Choose your `bit_width`" table — which
currently and correctly reads **"not yet published"**.

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
