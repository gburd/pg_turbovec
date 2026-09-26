# DRAFT — validated, ready for Greg to send (not auto-sent from the agent).
# v5 was actually built and tested against PostgreSQL master (HEAD f25c50f) on
# a c7i.2xlarge: reproducer flips 4/4 (invalid_ctids/rows_not_found) on stock
# to 0/0 with the patchset; `make check` = 239/239 green with all three
# patches; the gist regress test with the corrected v5-0001 test is green.
# Validated patches: upstream-pr/bug6-followup/v5-validated/*.diff, regress log
# alongside. The "builds clean, full suite green" claim below is now TRUE.
# Greg: attach the three v5 patches (reformat to git-format-patch headers) and
# send. Nikolay's review has been unanswered since 2026-09-21.

To: Nikolay Samokhvalov
Cc: Virender Singla, Andres Freund, Michael Paquier, Dilip Kumar,
    PostgreSQL Hackers
Subject: Re: [PATCH] Corruption Issue: Fix missing tts_tid in ExecForceStoreHeapTuple
In-Reply-To: <3403658477f27e85edc2a0233e696bdf@...>   [Nik's 2026-09-21 message]

Hi Nik,

Thanks for reading it this closely — both points are right, and v5 is
attached.

> My AI harness noticed that v4-0001 still has the old ctid_matches join
> returning 5, in both gist.sql and gist.out. It looks like the email and
> attachment got out of sync.

Correct, and my fault: the cover letter described the temp-table +
NOT EXISTS form I meant to send, but the attachment carried the older
`ctid_matches` join. v5-0001 is the version the prose described — the
ordered result goes into a temp table first, then the two checks read zero
when correct:

  select count(*) as invalid_ctids from gist_knn_ctid_res
  where c = '(4294967295,0)'::tid;

  select count(*) as rows_not_found
  from gist_knn_ctid_res r
  where not exists (select 1 from gist_knn_ctid t
                    where t.ctid = r.c and t.id = r.id);

On the testing behind it, so it is on the record rather than implied. I
did not find this with a pre-existing test; nothing in core projects ctid
off an ORDER BY-op index scan, which is why it went unnoticed. I found it
through an out-of-tree AM that sets xs_recheckorderby, then reproduced it
with core GiST alone. The verification is a plain A/B: one machine, stock
tree vs. the same tree with only the execTuples.c hunk applied, same
script both ways. On that A/B the regression test's checks flip with the
one line — reverting 0001 alone (assertion out of the way) takes
invalid_ctids 0→4 and rows_not_found 0→4 and aborts the FOR UPDATE; with
it, both read 0 and the lock succeeds. Four of five rather than five is
the was_exact fast path returning the first tuple without queueing, which
is why the test looks past LIMIT 1. I also checked there is no query-level
escape — WITH ... AS MATERIALIZED, a text cast inside a subquery, and
extra subquery nesting all still return the sentinel, because it is
already in the slot before any of them run — so the fix is a genuine
necessity, not a preference over a workaround. I have not run this beyond
the one A/B host, and the FOR UPDATE relation-extension consequence I am
taking from Virender's original report rather than having reproduced it in
a non-assert production build myself; the 0003 commit message says as much.

> This catches the reported (InvalidBlockNumber, 0), but ItemPointerIsValid()
> only checks for a non-NULL pointer and ip_posid != 0. For example,
> (InvalidBlockNumber, 1) still reaches the AM. If the intended protection is
> specifically against passing P_NEW to heap's ReadBuffer(), should this check
> be heap-side? A stronger generic check would need a clearly stated table-AM
> invariant; the moved-partitions marker is also an InvalidBlockNumber
> encoding with a nonzero offset.

Also right, and it is the sharper of the two. ItemPointerIsValid() only
rejects a zero offset, so it catches the exact sentinel this bug produces,
(InvalidBlockNumber, 0), but not (InvalidBlockNumber, 1); and as you note
the moved-partitions marker is a legitimate InvalidBlockNumber-with-
nonzero-offset, so a blanket "reject InvalidBlockNumber" at the table-AM
boundary would reject a valid encoding. The generic invariant I would have
to assert to justify the check living in table_tuple_lock() is not one I
can state cleanly, which is exactly your objection.

So for v5 I think the honest options are:

  (a) Move the guard heap-side, right before the ReadBuffer(P_NEW) that
      does the damage — a targeted "block == InvalidBlockNumber" check in
      heap_lock_tuple(), which is where the actual harm (relation
      extension) happens and where InvalidBlockNumber unambiguously means
      P_NEW. That makes no claim about other AMs. The moved-partitions
      marker cannot trip it: that marker is only ever a lock/update RESULT
      in tmfd->ctid, interpreted by callers (execReplication.c,
      nodeModifyTable.c) after the fact, never an input tid to
      heap_lock_tuple().

  (b) Drop 0003 entirely and rely on 0001 (the real fix) plus 0002 (the
      reorder-path assertion). With tts_tid restored, the sentinel never
      reaches a lock in the first place; 0003 is defence-in-depth against
      a future caller, not part of the fix.

v5-0003 attached takes (a): a `ItemPointerGetBlockNumberNoCheck(tid) ==
InvalidBlockNumber` guard immediately before the ReadBuffer in
heap_lock_tuple(), with an elog(ERROR). I confirmed on master (HEAD
f25c50f) that the moved-partitions marker never arrives here as an input
tid, so the guard has no false positive; `make check` is 239/239 green
with it in. I still defer to Andres / Michael on whether the
defence-in-depth is wanted at all — dropping 0003 (option b) is fine, since
0001 is the actual fix and neither 0001 nor 0002 depends on it.

Attached v5: 0001 (fix + the corrected regression test), 0002 (the
reorder-path assertion, unchanged), 0003 (the heap-side guard, option a).
Builds clean on master (HEAD f25c50f); the reproducer's invalid_ctids and
rows_not_found both go 4→0 with the series (and back to 4 on reverting
0001), the gist regression test is green, and the full `make check` is
239/239 under cassert.

Thanks again for the careful read.

best,
-greg
