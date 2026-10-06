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
