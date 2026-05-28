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

- Change `MetaPageData::version` (currently 3).
- Change page layout, chain ordering, meta-page field layout.
- Change the SQL surface (operators, type names, function signatures).
- Require any user action to upgrade. `ALTER EXTENSION ... UPDATE` must
  be sufficient and cannot fail on existing indexes.

This is enforced mechanically by:
- `scripts/drift-check.sh` § 7 (forbids `VERSION` constant change in a
  patch bump).
- `wire_format_version_is_stable` `#[pg_test]` in `src/lib.rs`
  (`EXPECTED_WIRE_FORMAT_VERSION = 3` constant).

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

The current major (1.x) line has been wire-format-stable since v1.4.0
(`MetaPageData::version = 3`, persisted rotation matrix). Future
majors should attempt to remain online-upgradable from the 1.x line
unless the cost of doing so is prohibitive.

### Current (as of v1.7.2, 2026-05-27)

| From               | To       | Action            |
|--------------------|----------|-------------------|
| 1.0.x / 1.1.x      | 1.7.2    | `REINDEX INDEX` once |
| 1.2.x              | 1.7.2    | `REINDEX INDEX` once |
| 1.3.x              | 1.7.2    | `REINDEX INDEX` once (rotation matrix migration) |
| 1.4.x → 1.7.x      | 1.7.2    | `ALTER EXTENSION pg_turbovec UPDATE` only |

`MetaPageData::version = 3` has held across **v1.4.0 → v1.7.2**.

---

## Build environment (NixOS local worktree)

```bash
export LIBCLANG_PATH=/nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /nix/store/x8lqlydsxbrwvf6p7v18gws8kn1xl37f-glibc-2.38-23-dev/include -isystem /nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib/clang/17/include"
export RUSTFLAGS="-L /nix/store/wavv74sn7l8l21pvdpnwshjfz4nz0fqz-openblas-0.3.30/lib"
```

Pre-test cleanup:
```bash
pkill -9 -f "test-pgdata"; sleep 2
test -d target/test-pgdata && mv target/test-pgdata /tmp/orphan-$$
```

Then `cargo pgrx test pg16` is the full local test loop.
`cargo build --release` compiles the production binary.
`bash scripts/drift-check.sh` enforces project-level invariants.

### Bench hosts

| Host    | Arch     | Cores | RAM     | Disk      | Notes |
|---------|----------|------:|--------:|----------:|-------|
| `meh`   | x86_64   | 24    | 125 GiB | 779 GiB   | NixOS; RAM-rich; pgrx 17.9 in `/scratch/pg_turbovec-bench/` |
| `arnold`| x86_64   | 20    | 31 GiB  | 1.9 TiB   | Fedora 44; the physical "NUC"; RAM-constrained, exposes buffer-manager bottlenecks |
| `rv`    | riscv64  | 8     | 7.7 GiB | 165 GiB   | Ubuntu 24.04; arch-correctness only; needs `LD_PRELOAD=libopenblas.so.0` |

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
- **Stale task notifications for completed agents are common.** Safe to
  acknowledge as "no action needed" if the work is already merged.

---

## Releases policy reminder

Every tagged release must:

1. Have an entry in `CHANGELOG.md` with the date and a Migration
   section describing the upgrade action.
2. Have a corresponding migration file in `migrations/`, even if empty.
3. Pass `cargo pgrx test pg16` cleanly (current count: 118/118).
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
