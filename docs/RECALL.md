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
`vec_normalize` helpers.

## 2. Pure-Rust recall benchmark (Phase 14, **shipped**)

```bash
cargo bench --bench recall --no-default-features
```

Generates `n` deterministic random unit-norm vectors per
`(dim, bit_width)` configuration, builds a `turbovec::IdMapIndex`,
runs 50 random queries, and reports R@k vs a brute-force ground
truth. Output is one JSON line per criterion sample (run with
`--quick` for a single sample per config).

### 2.1 Latest results (2026-05-21, 1 000 corpus rows, 50 queries)

| dim | bit_width | R@1  | R@10 | R@100 |
|----:|---------:|-----:|-----:|------:|
| 128 |        2 | 0.40 | 0.65 |  0.76 |
| 128 |        4 | 0.80 | 0.89 |  0.93 |
| 384 |        2 | 0.34 | 0.62 |  0.76 |
| 384 |        4 | 0.78 | 0.89 |  0.93 |
| 768 |        2 | 0.50 | 0.62 |  0.76 |
| 768 |        4 | 0.82 | 0.88 |  0.92 |

Observations:

- **4-bit hits R@1 ≈ 0.80 across all tested dims**, with R@100
  approaching 0.93. This is the recommended setting for general
  workloads.
- **2-bit costs ~40 R@1 points** — use only when memory pressure
  dominates and the application can absorb the recall hit (e.g.
  pre-rerank candidate generation).
- Recall on this corpus is lower than the upstream paper's
  numbers (which use pre-trained embeddings from real datasets
  like GloVe / OpenAI). Random vectors are a *harder* recall
  test — they have no clustering structure for the quantiser
  to exploit. Real embeddings will recall better.

Full machine-readable history under [`benches/results/`](../benches/results/).

## 3. End-to-end ANN benchmarks (Phase 15+, planned)

`benches/ann_recall.rs` (not yet implemented) will:

1. Spin up a `cargo pgrx` cluster.
2. Load `glove-200`, `openai-1536`, `openai-3072` into Postgres
   tables under both `pg_turbovec.vector` and `pgvector.vector`.
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

# Pure-Rust recall bench (no Postgres required).
cargo bench --bench recall --no-default-features

# End-to-end ANN bench (Phase 4, when implemented):
# cargo pgrx run pg17 --release
# Inside psql:
# \i bench/load_glove.sql
# \i bench/run_ann_bench.sql
```

## 6.1 Real-world fixtures (optional)

The synthetic random-vector recall numbers in § 2.1 are
deliberately *pessimistic* — random points have no clustering
structure for the quantiser to exploit. To run the bench against
real embeddings (GloVe, OpenAI, sentence-transformers, ...), set
the `TURBOVEC_FIXTURE_PATH` environment variable:

```bash
TURBOVEC_FIXTURE_PATH=$(pwd)/fixtures/glove-200.bin \
    cargo bench --bench recall --no-default-features
```

### Fixture file format

Binary, little-endian:

```
offset  bytes  meaning
  0       4    dim   (u32, dimensionality of each vector)
  4       4    n     (u32, number of vectors)
  8     4*n*d  data  (n contiguous rows of dim f32 values)
```

If the env var is unset, the file is missing, or the fixture's
`dim` doesn't match the bench configuration, the bench falls back
to synthetic random vectors and prints a notice on stderr. **No
failure** — this is by design so CI doesn't break when fixtures
aren't checked in.

### Converting common public fixtures

We deliberately don't ship pre-built fixtures (they're large and
license-encumbered). Conversion scripts:

* **GloVe-200**
  ([nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/)):

  ```python
  # convert_glove.py: text -> turbovec fixture
  import struct, sys
  rows = []
  with open(sys.argv[1]) as f:
      for line in f:
          parts = line.split()
          rows.append([float(x) for x in parts[1:]])
  dim = len(rows[0])
  n   = len(rows)
  with open(sys.argv[2], 'wb') as f:
      f.write(struct.pack('<II', dim, n))
      for r in rows:
          f.write(struct.pack(f'<{dim}f', *r))
  ```

* **OpenAI embeddings** (any of the `text-embedding-*` models):
  the OpenAI API returns FP32 lists; concatenate them into the
  same `<II>` + `<dim>f` layout.

* **HuggingFace `sentence-transformers/*-MiniLM-*`**: load the
  model, encode a corpus, and `.tofile()` after writing the
  8-byte header.

## 7. Citing

If you publish results derived from this benchmark suite, please
cite the upstream paper as well:

> Codrai, R. *TurboQuant: Online Vector Quantization with Near-
> optimal Distortion Rate.* arXiv:2504.19874, 2025.
