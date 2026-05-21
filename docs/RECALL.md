# Recall and Performance — Methodology

`pg_turbovec` is a Postgres front-end for `turbovec`, which itself
implements the [TurboQuant](https://arxiv.org/abs/2504.19874)
algorithm. Quantitative recall and latency come from `turbovec`'s
own benchmark suite and the bench harness in this repo.

This document describes the methodology — actual numbers will land
in `bench/results/` once the v0.3.0 IndexAm is in place and a
matched-bit-budget run against `pgvector` is feasible.

## 1. Pure-Rust kernel benchmarks (Phase 3, **shipped**)

```bash
cargo bench --bench distance --no-default-features
```

Measures the f64-accumulator distance kernels (`dot`, `l2_sq`,
`l1_abs`, `cosine_distance`) at dims 128, 384, 768, 1536, 3072.
These are the kernels that back the `<->` / `<#>` / `<=>` / `<+>`
operators on small (in-page) result sets and the `vector_norm` /
`tvector_normalize` helpers.

## 2. End-to-end ANN benchmarks (Phase 4, planned)

`benches/ann_recall.rs` (not yet implemented) will:

1. Spin up a `cargo pgrx` cluster.
2. Load `glove-200`, `openai-1536`, `openai-3072` into Postgres
   tables under both `pg_turbovec.tvector` and `pgvector.vector`.
3. For each of (k = 1, 10, 100):
   - run 1 000 random queries through `turbovec.knn(...)` at
     `bit_width ∈ {2, 3, 4}`,
   - run the same queries through pgvector's brute-force
     `ORDER BY emb <=> $1 LIMIT k`,
   - run them through pgvector's `hnsw` index at recall-matched
     `ef_search`,
4. Record p50 / p95 / p99 latency and Recall@k vs an exact
   brute-force baseline.

Output: a JSON file in `bench/results/` and a Markdown summary
table appended to this document.

## 3. Methodology choices

- **Matched bit budget.** When comparing TurboQuant and pgvector
  PQ, we size pgvector's PQ subquantizers so the per-vector byte
  count matches: `m = d / 4` at 2-bit, `m = d / 2` at 4-bit. This
  is the same convention the upstream `turbovec` paper uses.
- **Recall at k.** Defined as `|retrieved ∩ ground_truth| / k`,
  averaged over 1 000 queries. Ground truth is exact L2 / IP
  on FP32 from the source dataset's released `groundtruth.npy`.
- **Latency, not throughput.** We report per-query p50/p95/p99
  with `max_parallel_workers_per_gather = 0`, so reproducibility
  doesn't depend on the planner choosing the same parallel plan
  every run.
- **Cold vs warm.** Phase 4 will report both — v0.2's
  build-on-every-call latency is exactly the cold case; v0.3's
  cached AM is exactly the warm case.

## 4. Hardware

Phase 4 results will be reported on:

- **ARM**: Apple M3 Max (8P + 4E cores)
- **x86**: GCP `c3-standard-8` (Sapphire Rapids, 8 vCPU)

These match the upstream `turbovec` paper's hardware.

## 5. What we are NOT measuring

- Build-time cost of the v0.2 `turbovec.knn()` — it rebuilds on
  every call by design.
- Multi-tenant / RLS overhead — the side-table strategy in v0.3
  inherits Postgres RLS semantics, but it is the user's
  responsibility to secure `turbovec.am_storage`.
- Cross-node performance — we are a single-node extension.

## 6. Reproducing locally

```bash
# Pure-Rust kernel benches (no Postgres required).
cargo bench --bench distance --no-default-features

# End-to-end ANN bench (Phase 4, when implemented):
# cargo pgrx run pg17 --release
# Inside psql:
# \i bench/load_glove.sql
# \i bench/run_ann_bench.sql
```

## 7. Citing

If you publish results derived from this benchmark suite, please
cite the upstream paper as well:

> Codrai, R. *TurboQuant: Online Vector Quantization with Near-
> optimal Distortion Rate.* arXiv:2504.19874, 2025.
