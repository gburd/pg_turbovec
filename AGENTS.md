# Agent notes — `pg_turbovec`

This file captures the rules and conventions any AI coding agent should
read before making changes to this repo. It's the canonical source for
versioning policy, build environment, and project-specific gotchas.

If you're a human and you're updating something here, also propagate
the change to `docs/UPGRADING.md` (versioning policy) and the
`.pi/skills/drift-check/SKILL.md` (enforcement rules).

---

## Versioning policy — **READ THIS BEFORE BUMPING ANY VERSION**

> ### HARD MANDATE (2026-08-11, non-negotiable)
>
> 1. **Corruption is NOT acceptable, ever.** Any code path that writes,
>    reads, reconstructs, or reindexes the `.tvim` relfile / `slot_to_id`
>    id table is safety-critical. A change to format/persist/scan code
>    ships ONLY with: a reproduction test that fails before and passes
>    after, AND validation that it does not RE-CORRUPT under sustained
>    insert load (the v1.28.4 fix shipped on code-reasoning alone and
>    FAILED in production — never again). When in doubt, do not ship.
> 2. **Every non-major upgrade MUST be either (a) zero format change, or
>    (b) online-upgradable in place.** A minor may NOT require a
>    `REINDEX` as its migration for the flat/IVF kinds — rebuilding from
>    an extremely large corpus is not an option and MUST be avoided. If
>    the wire format must change, the new binary MUST read the OLD
>    format transparently (additive/back-compatible decode), so existing
>    indexes keep working with no rebuild.
> 3. **Even MAJOR upgrades must offer an OFFLINE migration TOOL when a
>    format break is unavoidable** — an in-place converter that rewrites
>    the relfile pages without re-embedding/re-quantizing from the heap.
>    A full `REINDEX`/rebuild-from-corpus is a LAST resort, allowed only
>    when an in-place converter is genuinely impossible, and must be
>    justified in `docs/UPGRADING.md`.
>
> These rules override the softer language below where they conflict.

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

**Must provide a non-destructive, online, efficient, IN-PLACE upgrade
path from ANY prior minor in the same major line.** Per the HARD
MANDATE above, a `REINDEX` is NOT an acceptable minor-version migration
for the flat/IVF kinds (rebuild-from-corpus is banned). Concretely:

- Existing on-disk indexes from any earlier minor MUST remain readable
  and writable with no rebuild — if the wire format changes, the new
  binary reads the old format transparently (additive/back-compatible
  decode), the way v4→v5→v6 were strictly additive per kind.
- A `REINDEX`-only migration is a policy VIOLATION for a minor unless
  the format is genuinely unreadable AND an in-place page converter is
  proven impossible (document why in `docs/UPGRADING.md`) — and even
  then prefer shipping an offline converter tool over forcing a
  corpus rebuild.
- If a pre-format index truly cannot be read, the binary detects it via
  an `is_legacy_v{N}()` predicate AND emits a clear `ERROR` with a
  `HINT` from `ambeginscan` at first scan — NEVER silent corruption.
- The `migrations/NNN_pg_turbovec_vX.Y.0.sql` file is checked in, AND a
  runnable `sql/pg_turbovec--<from>--<to>.sql` upgrade script is
  generated (`cargo pgrx schema`) and committed so `ALTER EXTENSION
  pg_turbovec UPDATE` actually creates/alters new SQL objects in place
  (the v1.28.4 release FAILED to ship this — `turbovec_check()` was
  never created on in-place upgrade; see the 2026-08-11 report).
- The migration matrix in `docs/UPGRADING.md` gets a new row spelling
  out the (in-place, no-rebuild) upgrade action.

Minor bumps may change the wire format additively (old indexes still
decode), the SQL surface (additively), or runtime behaviour. They may
NOT remove SQL objects without a two-release deprecation window.

### Major releases (X.Y.Z → X+1.0.0)

**May break backwards compatibility ONLY as a last resort, and MUST
provide an OFFLINE in-place migration tool when a format break is
unavoidable** (per the HARD MANDATE). The bar:

- Prefer a transparent back-compatible decode (no migration at all).
- If the on-disk format truly must change incompatibly, ship an
  **offline converter that rewrites the relfile pages in place**
  (meta + chains) WITHOUT re-embedding/re-quantizing from the heap.
  Rebuilding from an extremely large corpus is banned except where an
  in-place converter is provably impossible.
- A full `REINDEX` / `pg_dump | pg_restore` is the ABSOLUTE last
  resort, justified explicitly in `docs/UPGRADING.md`.
- Pre-major indexes that can't be migrated must `ERROR` clearly at
  `ambeginscan`, not silently misbehave.

The current major (1.x) line held `MetaPageData::version = 3` from
v1.4.0 through v1.9.x; v1.10.0 bumped it to **4** for the IVF layer,
backward-compatibly (a v4 binary reads v3 indexes as flat, no
REINDEX). Future majors should attempt to remain online-upgradable
from the 1.x line unless the cost of doing so is prohibitive.

### Current (as of v2.7.3, 2026-09-08)

`docs/UPGRADING.md` holds the authoritative, per-release migration matrix —
it is updated every release and drift-check enforces that. The summary:

| From        | To     | Action |
|-------------|--------|--------|
| any 1.x     | 2.7.3  | `ALTER EXTENSION` **then `REINDEX INDEX`** (wire v7→v8 in v2.0.0 is NOT additive; migration is REINDEX-from-heap, and an in-place converter was measured too lossy at −20.7 pp R@10) |
| 2.0.0–2.7.2 | 2.7.3  | `ALTER EXTENSION` only — no REINDEX. Wire format has been **v8** since v2.0.0 and every 2.x bump has been additive-or-code-only. |

One exception worth knowing: a **`bit_width = 1`** index created before
v2.7.0 that took inserts after a VACUUM should be `REINDEX`ed — v2.7.0 fixed
a tombstone-resurrection bug in the BQ insert path. 2/3/4-bit was never
affected.

### Where the project actually is (v2.7.3)

Per-release detail lives in `CHANGELOG.md`; this section is only what an
agent needs to orient. Do not add release blurbs here — they go stale and
CHANGELOG is authoritative.

**Index kinds.** `flat` (default) · `IVF` (`WITH (lists = N)`, out-of-core
end-to-end since v1.13.0) · `ColBERT` (multivector) · `1-bit sign-BQ`
(`WITH (bit_width = 1)`, v2.6.0, composable with IVF since v2.7.0) ·
**`graph` — DEPRECATED in v2.5.0**, build-path removal scheduled. The graph
kind emits a deprecation WARNING; measured at *matched recall* it lost on
every user-visible axis at every scale reached, so do not invest in it. Its
techniques were kept: `turbovec.graph_ef`, `pack::repack`, and the
`coarse_graph` centroid navigation IVF relies on.

**Which to recommend:** flat below ~1M vectors, `WITH (lists = N)` at scale.
It is IVF, not the graph, that beats flat's O(n) wall. For 1-bit: only when
storage is the binding constraint and latency has slack (measured 3.98×
smaller than 4-bit but 2.7–6.1× slower at matched recall, and at 250k flat BQ
dominates IVF+BQ).

**Corruption history — read before touching persist/scan code.** Five
distinct root causes have been found and fixed (counter-drift, VACUUM
lost-update, interrupted-flush torn write, graph unlocked RMW, BQ tombstone
resurrection). The recurring one is a **chain-offset running sum that omits a
chain** — it has bitten four times (`graph_count` in v1.24.0, three sites
missing `bq_mean_count` in v2.6.0, two more in v2.7.0). If you add a chain,
grep every running sum in `page.rs` and `relfile.rs` and add it to all of
them. Meta page is always written **LAST**, after every chain.

**Known upstream bug.** `SELECT ctid ... ORDER BY <vec-op>` projects
`(4294967295,0)` — a PostgreSQL core defect in
`ExecForceStoreHeapTuple` (reproduces with core GiST and no turbovec
loaded), reported upstream 2026-09-08 with a verified one-line fix. The
`knn_scan_ctid_projection_upstream_limitation` test asserts today's broken
behaviour on purpose and will FAIL when a fixed PG lands — that failure is
the signal to flip it, not a regression. See
`docs/upstream/bug6-pgsql-hackers-FILED.md`.

`MetaPageData::version` is **8** as of v2.0.0 (was **7** for
v1.27.0–v1.29.x, **6** for v1.23.0–v1.26.x, **5** for v1.17.0–v1.22.x,
**4** for v1.10.0–v1.16.x). v8 adopted turbovec 1.0.0's TQ+ calibration and
block-Hadamard rotation; like v7 it is NOT additive, so every pre-v8 index
needs a REINDEX, detected by `MetaPageData::is_legacy_v7()` (`version < 8`)
with `ambeginscan` raising a clear `HINT: REINDEX INDEX <name>;`.

New index KINDS since then have been additive via the `kind` byte rather
than a version bump — `KIND_BQ = 3` (1-bit sign-BQ, v2.6.0) is the current
example: existing indexes keep their kind and decode byte-identically. That
is the pattern to follow; prefer a new `kind` over a wire bump.

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

## AWS burner accounts — check expiry BEFORE launching

Bench work on EC2 runs in short-lived burner accounts that **expire without
warning**, and an expiring account strands whatever it was running.

**Current burner: `lava` → account `769093516156`, region `us-east-2`**
(configured in `~/.aws/config`).

Expired/dead, do not use: `bene` (292759875395, expired **2026-09-11 12:03
UTC**), and before it `chiuso`, `lala`, `fred`, `egret`, `numa`.

Rules, each of which has been paid for at least once:

- **Verify the profile before you launch:** `aws sts get-caller-identity
  --profile lava`. A launch on a nearly-expired account is money you cannot
  reclaim, because you lose the ability to terminate.
- **`InvalidClientTokenId` on a profile that worked minutes ago means the
  account expired, not that you broke something.** Do not spend turns retrying
  — check `get-caller-identity`, then escalate. On 2026-09-11 an agent burned
  ~20 retries over 15 minutes on exactly this.
- **NEVER open a security group to `0.0.0.0/0`** — it killed an earlier burner.
  Scope SSH to the current egress `/32` (`curl -s https://checkip.amazonaws.com`;
  it churns between Starlink and Comcast ranges), or use SSM with no inbound
  port at all.
- **Tag every instance `run=<something>`** and touch only your own tag. Other
  people's untagged instances share these accounts — there is an untagged
  `i4i.metal` in `lava` right now that is not ours.
- **Pull artefacts down as you go, not at the end.** The 2026-09-11 run survived
  a mid-run account expiry with zero data loss purely because the agent had
  already copied everything locally.
- **Terminate first, report second** if you are low on turns. A forgotten
  `i4i.8xlarge` once ran for days.

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
3. Pass `cargo pgrx test pg16` cleanly (current count: 427 passed,
   8 ignored, uniform across every CI leg pg13-19).
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
- BQ (1-bit) design + measured frontier: `docs/ONEBIT_BQ.md`,
  `docs/BQ_RECALL_BENCH.md` (§ 0 = results, § 0.5 = what they do NOT license)
- Upstream bug reports we filed: `docs/upstream/`
- Drift checker: `scripts/drift-check.sh`, `.pi/skills/drift-check/SKILL.md`
- Heartbeat protocol: `.pi/skills/long-running-bench/SKILL.md`

See .agent-steering-domains.md for domain-specific steering (local).
