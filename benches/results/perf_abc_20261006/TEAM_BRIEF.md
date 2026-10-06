# Team brief: per-candidate recheck cost work (2026-10-06)

Read this whole file before acting. It is the shared, authoritative fact base
for every agent on this task. Read `AGENTS.md` at the repo root too; its HARD
MANDATE and versioning policy are binding.

## Repo / branch / host

- Repo: `~/ws/pg_turbovec` (local), mirrored on GitHub `gburd/pg_turbovec`
  (remote `github`) and Codeberg (`origin`). `main` is at `8939b89`
  (v2.11.0 + recheck analysis). Work on a branch `perf/recheck-abc` off main;
  never push `main`, never tag, never force-push.
- EC2 dev/test/bench host (Debian 13 amd64, c7i.4xlarge: 16 vCPU Sapphire
  Rapids with AVX-512 VBMI+VNNI so turbovec's staged search engages; 30 GiB):
  `ssh -i ~/.ssh/tvperf-20261006-135223.pem -o IdentitiesOnly=yes -o IdentityAgent=none admin@18.219.111.242`
  - repo clone: `/work/pg_turbovec` (fetch your branch from `github`)
  - rustup 1.98.0 + cargo-pgrx 0.19.1 in `~/.cargo/bin`; pgrx PG16 (cassert)
    for `cargo pgrx test pg16`
  - NON-assert PG 16 for profiling/benchmarks: `/work/pg16rel/bin` (`-O2 -g
    -fno-omit-frame-pointer`, pageinspect installed). Install the extension
    into it with `cargo pgrx install --release --pg-config /work/pg16rel/bin/pg_config`.
  - pgvector source: `/work/pgvector`; python venv `/work/venv` (numpy,
    pyarrow, psycopg, huggingface_hub).
  - Do NOT run AWS commands; the lead owns the instance lifecycle.
- **Only one `cargo pgrx test` at a time on the box** (fixed port; parallel
  runs kill each other). If a run is in progress (`pgrep -f 'cargo pgrx test'`),
  wait. Never `kill -9` a postmaster; use `pg_ctl stop -m fast`.
- Long commands (>60 s): wrap in `~/with-heartbeat.sh <log> <cmd...>` with
  `nohup ... &` and poll with `~/poll-heartbeat.sh <log> 120`. Never pipe to a
  pager.

## The measured facts (benches/results/recheck_20261006/FINDINGS.md)

Real-query backend attribution, PG 16.15 non-assert, 200k × 1024-d real Cohere
embeddings, 4-bit flat index, `search_k = 1024` (floki, AVX2 laptop):

| component | µs/candidate (default storage) | with `SET STORAGE PLAIN` | owner |
|---|---|---|---|
| CBOR decode of BOTH vector args (pgrx serde) | 8.1 | 7.8 | ours |
| TOAST fetch of candidate vector | 5.0 | 0 | ours |
| exact distance kernel | 2.9 | 2.9 | ours |
| heap fetch | 0.75 | 0.98 | core |
| reorder queue | 0.09 | 1.0 | core |
| executor misc | 0.12 | 0.04 | core |
| total backend on-CPU | ~18.0 | ~13.7 | |

Facts that are settled; do not re-litigate, do not contradict:
- This is NOT a PostgreSQL core performance problem: core is ~1-2 µs/candidate.
- pgvector does NOT heap-recheck: `hnswscan.c:330` and `ivfscan.c:415` set
  `xs_recheckorderby = false`; its index stores full-precision vectors and
  returns exact distances. pg_turbovec rechecks because its index stores only
  4-bit codes. (Any doc saying pgvector "re-ranks vs heap" is wrong.)
- The v2.11.0 claim "~48 µs per candidate is the heap fetch + exact recheck"
  was a curve-fit slope of end-to-end latency vs search_k, NOT a measurement,
  and is wrong. It appears in `CHANGELOG.md` (~line 47),
  `benches/results/tv111_arm_20261005/FINDINGS.md` (~line 94),
  `docs/BENCHMARKS.md` (~line 40), and README's v2.11.0 callout says "the
  per-candidate heap recheck is now most of a query's cost". The correct
  statement: on Graviton4 after the 1.1 kernel the scan is ~1 ms, so the
  per-candidate recheck (~17-18 µs at 1024-d on x86; not measured on Graviton)
  dominates at large search_k; that recheck is ~1-2 µs core and ~15 µs ours
  (CBOR decode ×2, TOAST, scalar distance kernel). On AVX2 hosts the scan is
  still roughly half of a query.
- The serde-CBOR `Vector` is pgrx `#[derive(PostgresType)]` in `src/vec.rs`.
  Distance fns in `src/distance.rs` take `(a: Vector, b: Vector)` by value, so
  the constant query is CBOR-decoded again for EVERY candidate.
- `src/kernels.rs`: `dot`, `norm2`, `l2_sq`, `l1` accumulate in serial f64
  (one dependent add chain — LLVM won't vectorize). `cosine_distance` computes
  `norm2(a)`, `norm2(b)`, `dot(a,b)` per call. Bench (`recheck_20261006/
  cosine_bench.rs`): 1.36 µs ours vs 0.079 µs 16-lane f32 w/ cached query norm,
  |Δ| 2.8e-10.
- Stored vectors are unit length ONLY when `turbovec.normalize_on_insert`
  (GUC, default on) was on at insert; it does not apply to `<->` (L2) or `<#>`
  (IP) semantics. Do not assume unit length without a cheap check.
- A tighter advertised lower bound cannot skip the heap fetch
  (`IndexNextWithReorder` fetches + rechecks before consulting the bound); see
  the "Tier-1 #1b" comment in `src/index/scan.rs`. Out of scope.
- `ALTER TYPE ... SET (STORAGE = main)` affects only columns created later;
  existing columns need `ALTER TABLE ... ALTER COLUMN ... SET STORAGE MAIN`
  and existing rows stay out of line until rewritten. Changing the type's
  storage is an SQL change needing an upgrade script → minor release.

## The plan (order is binding)

1. Correct the published "~48 µs" claim + the pgvector re-ranking row (docs).
2. Fix A: vectorized f32 distance kernels; query norm computed once; skip a
   row norm only when verifiably unit length. No format change.
3. Fix B: decode the constant query once per scan/expression (fn_extra
   caching), not per candidate. No format change.
4. Fix C: storage guidance docs (when `SET STORAGE MAIN` helps, what it costs).
   Do NOT change the type's default storage in this release.
5. Measure A+B+C end-to-end, A/B alternated, on the EC2 host (and note what
   was not measured).
6. Design doc ONLY (no code) for replacing serde-CBOR with a raw
   `dim + float4[]` varlena read zero-copy, under the HARD MANDATE (read old
   CBOR forever, version tag, corruption/round-trip tests).
7. Optional comments-only core patch for `IndexNextWithReorder` (Andres Freund,
   pgsql-hackers 2019-04-19 "Comments for lossy ORDER BY are lacking",
   message-id 20190419003020.6u5uob4yhltrp6t2@alap3.anarazel.de).

Steps 1-4 target a patch release 2.11.1 (no wire change, no SQL change).

## Conventions

- Commit messages: imperative summary line, body explains why; no AI
  attribution trailers. Run `bash scripts/drift-check.sh` and `cargo fmt
  --check` before committing.
- Every non-trivial logic change ships with a test that fails before and
  passes after (TDD). Persist/format code is safety-critical (HARD MANDATE).
- Prose: plain, measured, no marketing; never state an unmeasured number as
  measured. Label estimates as estimates.
