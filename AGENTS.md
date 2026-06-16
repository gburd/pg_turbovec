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

### Current (as of v1.10.0, 2026-06-16)

| From               | To       | Action            |
|--------------------|----------|-------------------|
| 1.0.x / 1.1.x      | 1.10.0   | `REINDEX INDEX` once |
| 1.2.x              | 1.10.0   | `REINDEX INDEX` once |
| 1.3.x              | 1.10.0   | `REINDEX INDEX` once (rotation matrix migration) |
| 1.4.x → 1.10.x     | 1.10.0   | `ALTER EXTENSION pg_turbovec UPDATE` only |

`MetaPageData::version` is **4** as of v1.10.0 (was 3 for v1.4.0–
v1.9.x). The bump is backward-compatible: a v1.10.0 binary reads a
v3 index as a flat (`lists = 0`) index, so v1.4.x–v1.9.x indexes
need **no REINDEX**. IVF is opt-in per index via `WITH (lists = N)`.

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
