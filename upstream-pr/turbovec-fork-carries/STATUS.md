# turbovec fork-carry upstreaming — DRAFTS, NOT YET SUBMITTED

Three additive deltas on gburd/turbovec@pgtv-2.0.0-port over upstream 1.0.0
(ccab9f3) that we want to retire the fork by landing upstream:
  #1 pub pack::repack (537a289)
  #2 IdMapIndex parts API (f29a2f2)
  #3 parallel pack::repack (47a26a3) — byte-identical, shipped in pg_turbovec v2.10.3

Strategy (00-strategy.md): issue-FIRST, not cold PRs (turbovec is invite-only
for PRs per its CONTRIBUTING). Lead with #3+#1 as a linked pair on the existing
#70 thread the maintainer invited reopening; #2 separately after (widest surface,
maintainer already flagged it as maybe a fork concern).

BEFORE SUBMITTING:
- Re-verify the #3 micro-bench (250k x 1024d x 4-bit ~6s->~250ms on 8 cores) on
  an AVX2 host (arnold) before quoting it publicly. The end-to-end cold-scan
  1766->566 ms is already independently confirmed (benches/results/rebench_20260925/,
  finish_20260926/coldopen_profile.txt shows repack is now 2ms / 0.7% of cold).
- Verify current live state of issue #70 and upstream issue numbers on github.

## 2026-10-05: rebased onto 1.1.1 + carry #4

Carries #1-#3 cherry-picked unchanged onto upstream v1.1.1 (branch
`pgtv-2.11.0-port`). New carry #4 (`04-parallel-planes-repack.md`): 1.1's
planes cache build is serial and bypassed carry #3 on staged-search hosts
(279 ms / 1.09 s at 1M x 1024-d on Graviton4 -> 18-32 ms). Not yet filed;
natural follow-on to #545.

## 2026-10-14: merged upstream 1.1.2 + carry #5 (now on fork `main`)

Fork `main` = `pgtv-2.12.1-port` = `29de559`: upstream `c3b043b` (1.1.2:
staged-search recall fix on bunched corpora, #562/#563) merged with carries
#1-#4 unchanged, plus NEW carry #5 (`29de559`): `turbovec/build.rs` detects
LLVM < 22 and a `dpbusd!` macro emits `vpdpbusd` via inline asm instead of
`_mm512_dpbusd_epi32`, so the crate builds with nixpkgs' rustc (LLVM 21;
stdarch binds the intrinsic with LLVM 22's operand types). Pinned by
`dpbusd_matches_the_scalar_definition`. pg_turbovec 2.12.1 pins it.

Carry #5 is a good upstream candidate on its own: any Nix user of turbovec
hits the same error. It is not yet filed. File it as an issue first, per
`00-strategy.md`; nothing is posted without the user.

## 2026-10-15: carry #6 (fork `main` = `7873157`, pinned by pg_turbovec 2.13.0)

`codebook()`'s `memo_lock()` retries `try_lock` for up to 10 ms
(`MEMO_PATIENCE`) instead of re-solving the Lloyd-Max codebook on a lost
race. A contended warm codebook used to be re-solved (25-100 ms), which was
the out-of-core IVF ~100 ms tail. Pinned by
`warm_codebook_is_never_resolved_under_contention` and
`warm_lookup_waits_for_a_briefly_held_lock`. Upstream candidate: any caller
that builds throwaway indexes concurrently hits it.

## 2026-10-16: carry #7, Phase S2a (fork `main` = `1202a9e`, pinned by pg_turbovec main after 2.14.0)

Additive `prepared` module: `TurboQuantIndex::serialize_prepared() -> PiImage`
(pointer-free, `(offset,len)`-sectioned blob of the decoded search
structures), `PreparedView::new(&[u8])` (zero-copy borrowed view; validates
header, section bounds/alignment, and that section lengths match the
geometry, since `search::search` reads them with unchecked SIMD loads), and
`search_from_prepared`. Results are bit-identical to the owned `search`,
including over a relocated copy, in both classic and planes layouts
(`turbovec/tests/prepared_view.rs`, 6 tests). Prerequisite for pg_turbovec
Phase S2b (one decoded index per relation in shared memory). Upstream
candidate only if upstream wants foreign-memory search; otherwise a
long-lived carry.
