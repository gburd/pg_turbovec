# Agent notes — `pg_turbovec`

This file captures the rules and conventions any AI coding agent should
read before making changes to this repo. It's the canonical source for
versioning policy, build environment, and project-specific gotchas.

If you're a human and you're updating something here, also propagate
the change to `docs/UPGRADING.md` (versioning policy) and the
`.pi/skills/drift-check/SKILL.md` (enforcement rules).

---

## Versioning policy — **READ THIS BEFORE BUMPING ANY VERSION**

`pg_turbovec` follows a **wire-format-aware** SemVer policy. The version
number tells users what they have to do at upgrade time:

### Patch releases (X.Y.Z → X.Y.Z+1)

**Wire format is FROZEN across patch releases.** The on-disk index
format must be byte-identical to the prior patch in the same minor
line. Patch releases may:

- Change build-time behaviour (e.g. memory profile, build wall-clock).
- Fix scan-side bugs that don't change the on-disk format.
- Add bench results, docs, or non-functional improvements.
- Bundle bench-results-only commits.

Patch releases must NOT:

- Change `MetaPageData::version` (currently 4).
- Change page layout, chain ordering, meta-page field layout.
- Change the SQL surface (operators, type names, function signatures).
- Require any user action to upgrade. `ALTER EXTENSION ... UPDATE` must
  be sufficient and cannot fail on existing indexes.

This is enforced mechanically by:
- `scripts/drift-check.sh` § 7 (forbids `VERSION` constant change in a
  patch bump).
- `wire_format_version_is_stable` `#[pg_test]` in `src/lib.rs`
  (`EXPECTED_WIRE_FORMAT_VERSION = 4` constant).

### Minor releases (X.Y.Z → X.Y+1.0)

**Must provide a non-destructive, online, efficient upgrade path from
ANY prior minor in the same major line.** Concretely:

- Existing on-disk indexes from any earlier minor must remain readable
  and writable, OR
- A clear `REINDEX INDEX <name>;` migration is documented in
  `docs/UPGRADING.md` AND the binary detects pre-format indexes via
  an `is_legacy_v{N}()` predicate AND emits a clear `ERROR` with a
  `HINT: REINDEX INDEX ...` from `ambeginscan` at the first scan,
  not silent corruption at runtime.
- The `migrations/NNN_pg_turbovec_vX.Y.0.sql` file is checked in,
  even if empty (so `ALTER EXTENSION pg_turbovec UPDATE TO 'X.Y.0';`
  resolves cleanly).
- The migration matrix in `docs/UPGRADING.md` gets a new row spelling
  out the upgrade action.

Minor bumps may change the wire format, the SQL surface (additively),
or runtime behaviour. They may NOT remove SQL objects without a
two-release deprecation window (one minor adds the deprecation
warning; the next minor removes it).

### Major releases (X.Y.Z → X+1.0.0)

**May break backwards compatibility, but it is preferable to provide
an online upgrade path from earlier versions if at all possible.**
The bar:

- A `REINDEX INDEX` migration is acceptable; a full `pg_dump | pg_restore`
  is acceptable as a last resort.
- If the upgrade is destructive (e.g. SQL surface removed), the
  migration matrix must explicitly say so AND `docs/UPGRADING.md` must
  describe the cleanup steps.
- Pre-major indexes that can't be migrated must `ERROR` clearly at
  `ambeginscan`, not silently misbehave.
- A release note in `docs/UPGRADING.md` summarises why the break was
  necessary.

The current major (1.x) line held `MetaPageData::version = 3` from
v1.4.0 through v1.9.x; v1.10.0 bumped it to **4** for the IVF layer,
backward-compatibly (a v4 binary reads v3 indexes as flat, no
REINDEX). Future majors should attempt to remain online-upgradable
from the 1.x line unless the cost of doing so is prohibitive.

### Current (as of v1.25.1, 2026-07-09)

| From               | To       | Action            |
|--------------------|----------|-------------------|
| 1.0.x / 1.1.x      | 1.25.1   | `REINDEX INDEX` once |
| 1.2.x              | 1.25.1   | `REINDEX INDEX` once |
| 1.3.x              | 1.25.1   | `REINDEX INDEX` once (rotation matrix migration) |
| 1.4.x → 1.25.0     | 1.25.1   | `ALTER EXTENSION pg_turbovec UPDATE` only |

**v1.25.1 is a release-tooling + docs/benchmark patch** — no shippable
code change (binary byte-identical to v1.25.0; no wire/SQL change, no
REINDEX). It adds the tag-triggered PGXN + pgsql-announce publish
pipeline (first release to exercise it) and the Qdrant/ANN-Benchmarks
competitive benchmark that validated v1.25.0's `hi_dim_rerank` at
scale (GIST-960-1M recall 0.876→0.953; vs Qdrant we lose latency
3–18×, win storage 5–8×).

**v1.25.0 adds `turbovec.hi_dim_rerank`** (enum off/auto/on, default
auto) — the Gap-B fix. An offline investigation
(an internal design note) established the
high-dim recall gap (GIST-1M/960d ~0.86) is NOT retrieval-bound (the
true NNs DO land in the probed cells — cell recall 0.98-0.996 at
probes 64-128) but an in-cell quantized-RANKING loss, curable
scan-side by a wider exact-L2 rerank window (measured: an SQ4 analog
goes R@10 0.666→0.978 at 960d by reranking ~800 vs ~64 candidates).
`auto` applies a `clamp(dim, 256..=1024)` candidate floor only for
`dim >= 256` (SIFT-128 untouched; explicit `search_k`/`oversample`
override wins), so it's a smarter default, not a new mechanism
— identical result set to setting the candidate count by hand. One
new GUC, additive; no wire change (v6), no REINDEX. This also
CORRECTS the earlier "retrieval-recall ceiling" root-cause claim.

**v1.24.0 adds VACUUM + incremental INSERT for the graph kind**
(Phase G-2b). Both previously raised a clear `ERROR` (v1.23.0 was
build+scan only); they now work. **No wire change** — wire format
stays v6, byte-identical to v1.23.0, no REINDEX. VACUUM reuses the
generic per-slot tombstone bitmap IVF already uses; `aminsert` is a
deliberate O(n)-per-row whole-relfile rewrite (build-then-serve
model; heavy churn should still REINDEX). Two real bugs fixed en
route: a tombstone-chain/graph-adjacency-chain block-offset collision
that corrupted a graph index on insert-after-VACUUM (`write_
tombstones_and_meta` omitted `+ graph_count`, a no-op for every
non-graph kind), and a VACUUM entry-point fallback that missed the
"entry point survives but all its out-neighbors got tombstoned"
dead-end. Also caught + fixed a **test-harness** data-generation bug
(not shipped code): an uncorrelated `random()` subquery PostgreSQL
hoisted, making every graph-test row identical (`n_distinct=1`) and
producing spurious "recall collapse" failures; correlating the inner
`generate_series` to the outer row fixed it, and confirmed the
insert/vacuum/quantization paths were correct all along. Still
deferred: G-2c (SIMD traversal + build parallelism), G-2d (the
5M-scale AVX2 HNSW-latency gate).

**v1.23.0 adds `WITH (graph = true)`** — Phase G-2a, a new opt-in
Vamana-style navigable-graph index kind, the first step toward
matching HNSW's query latency while keeping TurboQuant's storage
compression (an internal design note). Wire format v6,
ADDITIVE per kind: existing v4/v5 indexes decode byte-identical, no
REINDEX. Determinism is relaxed for this kind ONLY (fixed-seed/one-
machine, not byte-identical cross-machine — an explicit, documented
trade-off, not an oversight). **Correctness-first scope**: real
Vamana build (greedy search + RobustPrune) + real beam-search scan,
verified recall against exact linear scan, but VACUUM/`aminsert`
against a graph index raise a clear `ERROR` (not yet supported) and
the real 5M-scale HNSW-latency gate has NOT been measured — no
latency/recall-vs-HNSW claim is made by this release. See
an internal design note for the sub-phase breakdown
(G-2b VACUUM, G-2c SIMD/parallelism, G-2d the gate measurement, all
follow-up work).

**v1.22.2 raises `turbovec.probes`'s default from 8 to 16** — the
old default capped out-of-the-box recall at R@10=0.796 (SIFT-1M) /
R@10=0.407 (GIST-1M), measured during the v1.22.1 a cloud VM competitive
re-benchmark. `probes=16` reaches R@10=0.918 / 0.557 for ~1.5-1.6×
the latency — the better point on the curve for a default most users
never tune. Scan-side default only, no wire change, no SQL surface
change, no REINDEX.

**v1.22.1 closes a real fraction of the IVF build-cliff gap** —
`gemm_lloyd_assign`'s Lloyd-loop cross-term GEMM (the dominant
k-means training cost at high `lists`, ~26-112× the FLOPs of the
row-blocked stages v1.20.0/v1.21.0 already parallelized) now runs
`Parallelism::Rayon(0)` instead of `Parallelism::None`. Bit-identical
output confirmed empirically (`gemm`'s own tiling never reduces
across threads for a given output element). **Measured on real
GIST-1M-scale k-means training (16-core AVX-512 a cloud VM): 2686.6s →
768.4s, a 3.50× speedup.** Scan/build-path only, no wire change, no
SQL surface change, no REINDEX. See CHANGELOG.md for the full
investigation writeup, including two dead-end findings (a test-
harness stride bug that looked like a `gemm` crate bug; a test-
harness thread-pool-scoping bug that overstated the problem) caught
and retracted before being reported.

**v1.22.0 is a repo-cleanup release, no functional change.**

`turbovec.mmap_static_blocked` (deprecated no-op since v1.19.0) is
removed after a three-minor deprecation window — `SET
turbovec.mmap_static_blocked = ...` now errors like any unknown GUC.
Also: `cargo fmt`'d the whole tree (244 pre-existing violations;
`fmt-check` was never wired into the real CI, only into an
already-dead `.woodpecker/ci.yaml`, now also removed and replaced
with a `fmt-check` job in `.github/workflows/test.yml` +
`.githooks/pre-push`), fixed literal `\uXXXX` escape-sequence
artifacts in several doc files, deleted a test made meaningless by
the mmap removal, fixed a stale dead-code warning. No wire change,
no REINDEX.

**v1.21.0 (Phase G-1) adds an in-memory centroid graph** for
sublinear IVF coarse-cell selection (`lists >= 4096`, gated by the
new `turbovec.coarse_graph` GUC, default `auto`). Computed in-memory
at index-open from the already-persisted coarse centroids — no wire
change, no REINDEX. **Correction to the v1.20.0 CHANGELOG entry**:
that release's "sublinear two-level coarse quantizer" claim was
aspirational and was never actually implemented; v1.20.0 shipped
only parallel k-means seeding/build and `turbovec.scan_parallelism`.
`coarse_probe` stayed the plain O(lists·dim) linear scan through
v1.20.1. v1.21.0 is the first release to ship real sublinear
coarse-cell selection. See an internal design note and
`CHANGELOG.md`.

**v1.20.1 is a critical perf fix, not a wire/feature change** —
`turbovec.iterative_scan` default flipped `relaxed_order` → `off`
(PostgreSQL's reorder queue can never pop a tuple early when we
advertise `NEG_INFINITY`, so the old default drove the AM's full
iterative-refill schedule to completion on every `ORDER BY ...
LIMIT` query regardless of `LIMIT` size — measured ~450x tax,
SIFT-1M/128d ~2ms vs ~900ms). Upgrade via `ALTER EXTENSION
pg_turbovec UPDATE`, no REINDEX. See `CHANGELOG.md` and
`docs/UPGRADING.md`.

`MetaPageData::version` is **6** as of v1.23.0 (was **5** for
v1.17.0–v1.22.x, **4** for v1.10.0–v1.16.x), but every bump is
**strictly additive per index kind**: a single-vector index
(`vec_*_ops` over a `vector` column) still emits wire **version 4**
with a zeroed `kind` byte (page offset 30), **byte-identical to
v1.16.0**; a ColBERT index (`vec_colbert_ops` over a `vector[]`
column, Phase F-2) is v5 (`kind = KIND_COLBERT`); a graph index
(`WITH (graph = true)`, Phase G-2a) is v6 (`kind = KIND_GRAPH`). A
v4 meta decodes as `KIND_SINGLE` and a v5 meta decodes unaffected
under the v6 binary, so `is_legacy_v4()` never trips and v1.4.x–
v1.22.x single-vector/ColBERT indexes need **no REINDEX**. IVF is
opt-in via `WITH (lists = N)`; as of v1.13.0 IVF is out-of-core
end-to-end (build AND query), so a >RAM IVF index can be built and
served on a RAM-constrained host. The graph kind is NOT out-of-core
(RAM-resident by design, per an internal design note's explicit
trade-off) and does not yet support VACUUM or `aminsert` (v1.23.0,
see that release's CHANGELOG entry).

**v1.7.3+ is the recommended floor for all x86_64 users** — it
fixes a kernel bug where pre-AVX2 CPUs returned wrong ANN results.
v1.8.0 added iterative scans, parallel build, a cold-scan latency
cut, and `||`/halfvec arithmetic. v1.9.0 added `turbovec.oversample`
(tunable recall) + the first published benchmark.

---

## Build environment (NixOS local worktree)

```bash
export LIBCLANG_PATH=/nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /nix/store/x8lqlydsxbrwvf6p7v18gws8kn1xl37f-glibc-2.38-23-dev/include -isystem /nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib/clang/17/include"
# Live openblas store path (the older wavv74... path was nix-GC'd 2026-06).
# Re-derive if this one is GC'd too: `ls -d /nix/store/*openblas-0.3.30`
export RUSTFLAGS="-L /nix/store/qbq20d6v6qf87cnlv5k55i0hnpzy00hq-openblas-0.3.30/lib -C link-arg=-fuse-ld=bfd"
```

**Toolchain:** turbovec >= 0.9.0 uses `avx512f`/`avx512bw`
`target_feature`s that require **Rust >= 1.89**. The default `stable`
toolchain (1.95) works. The old 1.85 pin cannot compile turbovec
v0.9.0+. The `-C link-arg=-fuse-ld=bfd` flag is needed because the
rustup `stable` toolchain's bundled `gcc-ld/ld.lld` wrapper
references a GC'd rustup store path on this box; bfd is the system
fallback.

Pre-test cleanup:
```bash
pkill -9 -f "test-pgdata"; sleep 2
test -d target/test-pgdata && mv target/test-pgdata /tmp/orphan-$$
```

Then `cargo pgrx test pg16` is the full local test loop.
`cargo build --release` compiles the production binary.
`bash scripts/drift-check.sh` enforces project-level invariants.

### Bench hosts

| Host    | Arch     | SIMD | Cores | RAM     | Disk      | Notes |
|---------|----------|------|------:|--------:|----------:|-------|
| `meh`   | x86_64   | **AVX only, NO AVX2** | 24 | 125 GiB | 779 GiB | NixOS; RAM-rich; pgrx 17.9 in `/scratch/pg_turbovec-bench/`. Ivy Bridge Xeon E5-2697 v2 — pre-AVX2. turbovec takes the SCALAR fallback here (~1000x slower than its AVX2/AVX-512 kernels: a 1M x 1024-d warm scan is ~68 s, not ms). **Use meh for correctness / recall / storage / build / memory ONLY — NEVER for latency or QPS.** Any tens-of-ms "meh warm p50" in old docs predates the v1.7.3 pre-AVX2 fix and was the fast-but-WRONG path. |
| `arnold`| x86_64   | **AVX2** | 20 | 31 GiB  | 1.9 TiB   | Fedora 44; the physical "NUC"; RAM-constrained (exposes buffer-manager bottlenecks). i9-12900H, has AVX2 — **this is the host for turbovec LATENCY / QPS benchmarks** (the SIMD kernels actually run). |
| `rv`    | riscv64  | scalar (no RVV) | 8 | 7.7 GiB | 165 GiB | Ubuntu 24.04; arch-correctness only; needs `LD_PRELOAD=libopenblas.so.0`. Scalar-path-slow like meh — correctness only, not latency. |

**SIMD matters more than RAM for turbovec latency.** The kernel
dispatches at runtime via `is_x86_feature_detected!`: AVX-512 > AVX2 >
scalar fallback. The scalar fallback is correct (since v1.7.3 /
turbovec v0.9.0) but ~1000x slower for the full-corpus scan. **Latency
and QPS benchmarks REQUIRE an AVX2+ host (arnold); meh and rv only
measure correctness, recall, storage, build time, and memory.** This
is why the Phase A1 "regression" looked like a bug (meh was on the
buggy fast path) and why the published latency frontier must come from
arnold, not meh.

`nuc` is NOT a separate host — it's an old name for `arnold` per session
history. Don't assume `nuc` resolves; it's not on tailscale.

The pgrx test cluster on `meh` listens on
`/scratch/pg_turbovec-bench/.s.PGSQL.28815`, NOT `/tmp/.s.PGSQL.*`.
Connect with `psql -h /scratch/pg_turbovec-bench -p 28815`.

---

## Heartbeat protocol for long-running benches

Read `.pi/skills/long-running-bench/SKILL.md`. Wrap any command longer
than ~60s with `benches/scripts/lib/with-heartbeat.sh`. Poll with
`benches/scripts/poll-heartbeat.sh`. Don't pipe through pagers
(`less`, `tail -f`, `nvim` etc.) — they wedge sub-agents.

---

## Operational gotchas

- **Never `kill -9` a running postmaster.** Crash recovery truncates
  `UNLOGGED` tables. Always `pg_ctl stop -m fast` or `-m smart`. The
  Phase W-2 validation cost a 31-minute corpus reload because of this.
- **Codeberg HTTPS endpoint is flaky.** Returns 504 intermittently;
  `git fetch origin` may fail. SSH endpoint banner exchange also
  occasionally times out. Just retry; the GitHub mirror is the
  fallback for cargo pulls.
- **Sub-agent worktree changes can leak into parent's main worktree.**
  Check `git status -sb` before commit. Use `git rm --cached
  vendor/turbovec/target/` if build artifacts leak (covered by
  `.gitignore` now).
- **Parallel sub-agents share one pgrx test cluster.** `cargo pgrx
  test` binds a fixed port (`32200 + major`, e.g. 32216 for pg16) and
  uses `~/.pgrx/data-16` — there is no per-worktree override in pgrx
  0.17. Two agents running `cargo pgrx test pg16` in different
  worktrees will collide: one's run kills the other's cluster
  ("terminating connection due to administrator command"). Serialize
  test runs across parallel worktree agents, or have each agent
  `pg_ctl stop -m fast` and retry on collision. **Never `kill -9`** the
  shared postmaster (truncates UNLOGGED tables). The pre-test cleanup
  `pkill` pattern must be scoped to `/target/test-pgdata` (not bare
  `test-pgdata`, which matches a worktree dir name and kills the
  agent's own postmaster).
- **Stale task notifications for completed agents are common.** Safe to
  acknowledge as "no action needed" if the work is already merged.

---

## Releases policy reminder

Every tagged release must:

1. Have an entry in `CHANGELOG.md` with the date and a Migration
   section describing the upgrade action.
2. Have a corresponding migration file in `migrations/`, even if empty.
3. Pass `cargo pgrx test pg16` cleanly (current count: 145/145).
4. Pass `bash scripts/drift-check.sh`.
5. Be tagged AND pushed to BOTH `origin` (Codeberg) and `github`
   (mirror). Use `git push origin vX.Y.Z` and `git push github vX.Y.Z`.
6. Have its CHANGELOG date match the tag's commit date.

Bench-results-only releases (no source code change) are still patch
bumps. They go in CHANGELOG with a "Bench-results-only release. Wire
format unchanged from X.Y.Z; no REINDEX needed." preamble.

---

## Where to find things

- Index AM core: `src/index/{mod,build,insert,scan,vacuum,cost,validate,options,page,relfile,mmap_static}.rs`
- Cache + xact: `src/cache.rs`, `src/xact.rs`, `src/guc.rs`
- Type surface: `src/{vec,halfvec,sparsevec,bitvec}.rs` (one file per concrete type)
- Distance kernels: `src/{distance,kernels}.rs`
- Phase progress notes: `docs/PHASE_*.md`
- Versioning policy detail: `docs/UPGRADING.md`
- Pgvector parity: `docs/PARITY_GAPS.md`, `docs/MIGRATING_FROM_PGVECTOR.md`
- CI: `docs/CI.md`, `.github/workflows/`, `.forgejo/workflows/`
- Bench results archive: `benches/results/`
- Drift checker: `scripts/drift-check.sh`, `.pi/skills/drift-check/SKILL.md`
- Heartbeat protocol: `.pi/skills/long-running-bench/SKILL.md`

See .agent-steering-domains.md for domain-specific steering (local).
