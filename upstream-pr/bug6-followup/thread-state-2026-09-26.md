# BUG#6 upstream thread — reconstructed state (research, 2026 sweep)

## TL;DR

The repo doc `docs/upstream/bug6-pgsql-hackers-FILED.md` is **stale**. It says
"Responses as of last check: none yet." That was true only for the first day.
Since then the discussion **migrated to an earlier thread by the original
reporter (Virender Singla)**, drew in two committers (Andres Freund, Michael
Paquier) plus Dilip Kumar and Nikolay Samokhvalov, iterated through v2→v4 of a
3-patch series, and now sits with **one unanswered review from Nikolay
Samokhvalov dated 2026-09-21** raising two concrete, valid technical points.

**Action is owed by us**: a reply to Nikolay on the Virender thread, plus a v5
patch that (a) resyncs the 0001 regression test with what the v4 cover letter
claimed, and (b) fixes the too-weak invalid-TID guard in 0003. Draft reply in
`/tmp/bug6_reply_draft.md`.

## Two threads exist

### Thread A — the one the task cited (our filing)
- msgid root: `0498c10f-839b-4f68-9994-c29b454e55a4@app.fastmail.com`
- Subject: "ExecForceStoreHeapTuple() loses tts_tid, so ORDER BY-op index scans project an invalid ctid"
- 7 messages, 2026-09-08 → 2026-09-14.
- Greg [msg 8, 09-14 13:26] discovered the bug was **already reported** and
  said "I suggest we continue on that thread," pointing to Thread B
  (`CAM6Zo8wZOLnCWRO_tuuXVX9J4N4JN6GsEnk8WJtT0=_0zy-1dw@mail.gmail.com`).
- Andres' last note here [09-14 15:22]: "Just had replied there..." → confirms
  the live conversation moved to Thread B.
- **Thread A is effectively closed / superseded. No action owed here.**

### Thread B — the ACTIVE thread (original reporter Virender Singla)
- msgid root: `CAM6Zo8wZOLnCWRO_tuuXVX9J4N4JN6GsEnk8WJtT0=_0zy-1dw@mail.gmail.com`
- Subject: "[PATCH] Corruption Issue: Fix missing tts_tid in ExecForceStoreHeapTuple"
- 10 messages, 2026-09-01 → 2026-09-21.

Message order (Thread B):

| # | date (UTC)         | from                | gist |
|---|--------------------|---------------------|------|
| 1 | 2026-09-01 06:18   | Virender Singla     | original report + patch; notes FOR UPDATE extends the relation on disk |
| 2 | 2026-09-14 12:10   | Virender Singla     | (bump / follow-up) |
| 3 | 2026-09-14 12:16   | Burd, Greg          | joins the thread |
| 4 | 2026-09-14 12:31   | Dilip Kumar         | review |
| 5 | 2026-09-14 13:19   | Greg Burd           | v2 (folds in Virender's FOR UPDATE + LIMIT-5 fixture) |
| 6 | 2026-09-14 15:21   | Andres Freund       | review: add invalid-TID error path; assert in reorder path only; use format() in test; put result in temp table; use NOT EXISTS |
| 7 | 2026-09-14 19:19   | Greg Burd           | **v4** cover letter (0001 fix+test, 0002 assert, 0003 error path) |
| 8 | 2026-09-15 09:42   | Virender Singla     | applied v4, verifies it fixes + passes |
| 9 | 2026-09-15 11:12   | Greg Burd           | thanks |
|10 | 2026-09-21 04:24   | Nikolay Samokhvalov | **REVIEW — currently unanswered** |

## Nikolay's 09-21 review (the open item), verbatim gist

> My AI harness noticed that v4-0001 still has the old ctid_matches join
> returning 5, in both gist.sql and gist.out. It looks like the email and
> attachment got out of sync.

> This catches the reported (InvalidBlockNumber, 0), but ItemPointerIsValid()
> only checks for a non-NULL pointer and ip_posid != 0. For example,
> (InvalidBlockNumber, 1) still reaches the AM. If the intended protection is
> specifically against passing P_NEW to heap's ReadBuffer(), should this check
> be heap-side? A stronger generic check would need a clearly stated table-AM
> invariant; the moved-partitions marker is also an InvalidBlockNumber encoding
> with a nonzero offset.

Both points **verified against the actual artifacts in /tmp**:

1. **Out-of-sync test — CONFIRMED.** `/tmp/reply-tts-tid-v4.txt` (the cover
   letter, msg 7) says the checks were reworked to a temp table + a
   `rows_not_found` `NOT EXISTS`. But `/tmp/v4-0001-Restore-tts_tid-in-ExecForceStoreHeapTuple.patch`
   still ships the older `ctid_matches ... join ... = 5` form in both
   `gist.sql` and `gist.out`. Prose and attachment disagree; Nikolay is right.

2. **Weak 0003 guard — CONFIRMED.** `/tmp/v4-0003-Reject-an-invalid-TID-in-table_tuple_lock.patch`
   guards with `if (unlikely(!ItemPointerIsValid(tid)))`. In PG,
   `ItemPointerIsValid(p)` ≡ `PointerIsValid(p) && (p)->ip_posid != 0` — it
   only rejects offset 0, not `InvalidBlockNumber`. So `(InvalidBlockNumber, 1)`
   passes the guard and still reaches `ReadBuffer(P_NEW)`. And a naive
   "reject InvalidBlockNumber block" fix would wrongly reject the legitimate
   moved-partitions tuple marker (block = InvalidBlockNumber, offset =
   `MovedPartitionsOffsetNumber`, which is nonzero). Nikolay is right again.

## Is action owed?

**Yes.** The last message on the active thread (Thread B) is a substantive,
correct review that has stood unanswered since 2026-09-21. Two things are owed:
- a reply acknowledging both points, and
- a v5 patchset: 0001 test resynced; 0003 either narrowed to the P_NEW/heap
  case or given a stated table-AM invariant that also permits the
  moved-partitions marker.

Note: Nikolay is a prominent community member, not a committer. The committers
in the thread are **Andres Freund** and **Michael Paquier**; Andres drove the
review. Paquier's only question ("what kind of testing did you do to spot
that?") was already answered honestly in Thread A msg 6 and Thread B — found
via the pg_turbovec AM, not a pre-existing test.

## Confidence & gaps

- **High confidence** on thread structure, message order, senders, dates, and
  the two open technical points — these came straight off the live
  postgresql.org flat-thread HTML (fetched this session) and were
  cross-checked against the /tmp draft artifacts.
- **Could not determine**: whether Greg has *already* replied to Nikolay from
  an email client in a way not yet reflected in the archive snapshot I fetched
  (the archive showed nothing after 09-21, but archive lag is possible). Also
  cannot tell whether a v5 is in progress offline — no v5 artifact exists in
  /tmp or the repo.
- **Not verified this session**: the actual PG source definition of
  `ItemPointerIsValid` / `MovedPartitionsOffsetNumber` (no PG checkout in this
  worktree). The claims rest on well-known PG semantics and Nikolay's review;
  they should be re-confirmed against the target tree before sending.
