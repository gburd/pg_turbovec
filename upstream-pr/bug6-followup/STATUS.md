# BUG#6 follow-up — v5 patchset VALIDATED, reply ready for Greg to send

State (2026-09-26): The active pgsql-hackers thread ("[PATCH] Corruption Issue:
Fix missing tts_tid in ExecForceStoreHeapTuple") iterated to a v4 3-patch
series. Nikolay Samokhvalov's review (2026-09-21) raised two CONFIRMED-real
defects; his review has been unanswered since. Full reconstruction in
thread-state-2026-09-26.md.

## What was done this session

Authored + VALIDATED a v5 patchset against PostgreSQL master (HEAD f25c50f) on
a c7i.2xlarge (terminated):

- **v5-0001** (v5-validated/v5-0001-fix-and-test.diff): the execTuples.c fix
  (unchanged) + Defect 1 fix — the regression test resynced to the temp-table +
  NOT EXISTS form the v4 cover letter described (`invalid_ctids` + `rows_not_found`,
  both 0-when-correct), replacing the stale `ctid_matches` join.
- **v5-0002** (v5-0002-reorder-assert.diff): the reorder-path assertion,
  unchanged from v4-0002.
- **v5-0003** (v5-0003-heapside-guard.diff): Defect 2 fix, option (a) — a
  heap-side `ItemPointerGetBlockNumberNoCheck(tid) == InvalidBlockNumber` guard
  in heap_lock_tuple() immediately before ReadBuffer(P_NEW), no table-AM
  invariant claimed. Verified the moved-partitions marker never arrives as an
  input tid here (it's a lock/update RESULT), so no false positive.

## Validation (real, not projected)

- Fail-before on stock master: reproducer `invalid_ctids=4, rows_not_found=4`.
- Pass-after with the series: `0, 0` (and reverts to 4/4 on removing 0001).
- gist regression test: green (the shipped test file matches the patched output).
- Full `make check`: **239/239 passed** with all three patches (0003's new elog
  breaks nothing). Log: v5-validated/regress_239_passed.log.

## Action still owed (by Greg, not the agent)

Send reply-to-nikolay-DRAFT.md (now truthful — the green-build claim is real),
attaching the three v5 patches reformatted to git-format-patch headers. The
agent does not send email. Option (b) — drop 0003 — is offered to committers in
the reply; 0001 is the actual fix and wants backpatching regardless.
