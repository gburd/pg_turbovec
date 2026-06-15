# Testing coverage and known gaps

This document describes what the `pg_turbovec` test suite covers, what
it deliberately does **not** cover, and why. It exists because a
silent wrong-results regression (the "pre-AVX2 bug", below) shipped
once already through a gap that looked, at a glance, like good
coverage. Read this before assuming a class of bug is tested.

## How to run

```bash
cargo pgrx test pg16          # full suite on PG 16 (local loop)
bash scripts/drift-check.sh   # project-level invariants (docs vs code)
```

CI runs `cargo pgrx test pg<N>` for N in 13..=18 (see `docs/CI.md`).

## The bug this coverage exists to catch

A previous turbovec revision returned **silently wrong** ANN results
on CPUs **without AVX2**: instead of the top-N distinct neighbours it
returned the *same* TID N times. It shipped undetected because:

1. **No automated test exercised a corpus larger than ~2000 rows.**
   The bug only manifested at scale and non-trivial dimensionality on
   a specific CPU class. Every `#[pg_test]` used tiny corpora (often
   64 rows, never more than 2000).
2. **No recall-regression test compared against a real ground truth
   at meaningful scale.** A 64-row "recall" test is far below where
   quantiser behaviour is observable.
3. **CI ran only unit tests on AVX2 hardware**, so the scalar fallback
   path was never exercised by `pg_turbovec`'s own CI.

The single cheapest guard against the whole "wrong-ranking" class is a
**distinct-ids assertion** on every ANN result: the duplicate-id bug
fails it instantly, regardless of which SIMD path runs or how high
recall happens to be.

## What the suite covers

- **Unit-scale ANN correctness** — `#[pg_test]`s that build a turbovec
  index on small corpora (8–384 dims, up to 2000 rows) and assert
  nearest-neighbour ordering, self-recall, and operator behaviour
  (`<->`, `<#>`, `<=>`, L1/L2/cosine opclasses).
- **Distinct-ids invariant** — every ANN-scan `#[pg_test]` that pulls
  back more than one id runs the result through `assert_distinct_ids`.
  This is the regression guard for the pre-AVX2 bug class. The helper
  and `fetch_ids` live in the `tests` module in `src/lib.rs`.
- **Medium-scale recall floor** — `index_am_recall_floor_{2,3,4}bit`
  build a 20 000 × 128 corpus of distinct deterministic random
  vectors, compute brute-force exact top-10 for 20 held-out queries
  (forced seqscan, exact `<=>`), and assert the index's recall@10
  clears a per-bit-width floor **and** that every returned id is
  distinct. This is the test the pre-AVX2 bug would have failed: it is
  ~10× larger than the historical ceiling and asserts distinct ids.
  - **Observed recall@10 = 1.000 at every bit width** on this
    synthetic uniform-random corpus (the vectors are near-orthogonal
    in 128-d, so even 2-bit TurboQuant separates them cleanly). The
    floors (4-bit ≥ 0.95, 3-bit ≥ 0.90, 2-bit ≥ 0.80) are therefore
    **catastrophic-collapse guards** — they fire well before the
    ~0.1 recall the duplicate-id bug produced — not fine-grained
    quality gates. Fine-grained per-bit-width quality is measured by
    VectorDBBench on real embeddings, not by this unit test.
  - In-harness corpus + query + index build is well under the ~30 s
    budget per test (the whole 3-test set, including compile, runs in
    ~60 s).
- **SQL surface** — operator/opclass registration, type round-trips
  (`vector`, `halfvec`, `sparsevec`, `bitvec`), `knn()` table function,
  filtered allowlist, reloption validation (`bit_width` 2..=4).
- **Iterative scan** — refill correctness, `max_scan_tuples` ceiling,
  `iterative_scan = off` single-batch behaviour, and **no duplicate
  TIDs across refill batches**.
- **Index lifecycle** — REINDEX, CREATE INDEX CONCURRENTLY, VACUUM /
  `ambulkdelete` dead-tuple removal, parallel vs serial build
  equivalence (same ranked top-k, same relfile size).
- **Wire-format stability** — `wire_format_version_is_stable`
  (`EXPECTED_WIRE_FORMAT_VERSION = 3`) and the legacy-meta
  `ambeginscan` error paths (`is_legacy_v{1,2}` → clear `ERROR` with a
  `REINDEX` hint).
- **Upgrade matrix** — `migration_files_cover_documented_versions`
  cross-checks `migrations/` against the documented release history.

## What the suite does NOT cover (and why)

### (a) Large-scale behaviour (1M+ rows)

Unit tests cap at the ~20 000-row recall-floor corpus. Building 1M+
rows in the in-process pgrx harness is too slow for a unit test and
the harness holds the whole corpus in one transaction. **The
large-scale evidence is the VectorDBBench run** (see `benches/` and
`docs/RECALL.md`), not the unit suite. A regression that only appears
above ~20 000 rows would not be caught by `cargo pgrx test`; it is the
benchmark's job.

### (b) The pre-AVX2 scalar fallback path

**CI runs on GitHub `ubuntu-latest`, which is AVX2-capable.** turbovec
selects its SIMD kernel with a *runtime* `is_x86_feature_detected!`
check, so on AVX2 hardware the scalar path never runs — and you cannot
force it from `pg_turbovec` (turbovec's `FORCE_SCALAR_FALLBACK` is
`pub(crate)`). Compile-time `-C target-feature=-avx2` does **not** help:
it changes what the compiler emits, not what the runtime feature-detect
selects on an AVX2 machine, so a CI job built that way would still run
the AVX2 path and prove nothing. We deliberately do **not** add such a
job — it would be coverage theatre.

The real mitigations for a pre-AVX2 regression are:

1. **turbovec's own upstream test** —
   `x86_scalar_fallback_tests::scalar_fallback_matches_simd_topk`
   flips `FORCE_SCALAR_FALLBACK` and asserts the scalar top-k matches
   the SIMD top-k. That test owns this path.
2. **Validate turbovec bumps on a pre-AVX2 host (or QEMU).** When
   bumping the `turbovec` git rev in `Cargo.toml`, run the suite on a
   non-AVX2 target before tagging. The `rv` (riscv64) bench host
   exercises a non-x86 path; an x86 pre-AVX2 host or
   `qemu-x86_64 -cpu Nehalem` covers the scalar x86 kernel.

A future turbovec bump that reintroduced the scalar-path bug would
**not** be caught by `pg_turbovec` CI. Treat the turbovec rev bump as
the trigger for the host/QEMU validation above.

### (c) Cross-PG-version wire compatibility

The matrix tests each PG version independently (build-on-N,
read-on-N). It does **not** test build-on-16 / read-on-17. The
on-disk format is PG-version-independent and wire-format-stable
(`MetaPageData::version = 3` since v1.4.0), so this is low risk, but it
is not directly asserted.

### (d) Concurrency / races

Beyond the existing CREATE INDEX CONCURRENTLY test and the
parallel-vs-serial build equivalence test, the suite does not stress
concurrent insert/scan/vacuum interleavings. The pgrx harness runs
each test in a single backend, so true multi-backend race coverage
would need an external harness (see `benches/sql/`).

## The "ignored" tests are not skipped tests

`cargo pgrx test` reports a number of **ignored** entries. These are
` ```ignore ` doctests — SQL usage examples embedded in `///` doc
comments that are not valid standalone Rust and so are marked
`ignore`. They are documentation, not disabled tests. Nothing in the
real test suite is being silently skipped; the `ignored` count is
purely these doc-comment SQL snippets.
