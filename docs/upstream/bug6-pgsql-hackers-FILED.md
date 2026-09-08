# BUG#6 upstream submission — **FILED 2026-09-08**

**Status: SENT.** Filed on pgsql-hackers by Greg Burd, 2026-09-08 17:28 UTC.

- **Thread:** https://www.postgresql.org/message-id/0498c10f-839b-4f68-9994-c29b454e55a4%40app.fastmail.com
- **Subject:** `ExecForceStoreHeapTuple() loses tts_tid, so ORDER BY-op index scans project an invalid ctid`
- **Attachment as filed:** `v1-0001-ExecForceStoreHeapTuple-loses-the-tuple-s-item-po.patch`
- **Responses as of last check:** none yet.

## What was filed vs what this repo verified

The filed patch's `execTuples.c` hunk is **identical** to the one verified
here by an A/B build (`slot->tts_tid = tuple->t_self;` plus its comment);
only a blank line differs. The filed version additionally adds a **core
regression test** to `src/test/regress/{sql,expected}/gist.sql|out` — 20
thin diagonal triangles and a ctid self-join over the top-5.

That added test was itself checked against both builds here:

| | unpatched 18.4 | patched 18.3 |
|---|---:|---:|
| upstream test's `ctid_matches` (expects 5) | **1** | **5** |

So the regression test genuinely gates the fix rather than passing
vacuously — worth knowing, because a test that passes either way is worse
than no test.

## Tracking on our side

`knn_scan_ctid_projection_upstream_limitation` in `src/lib.rs` is the
tripwire: it asserts the CURRENT (broken) behaviour, so **it will fail
loudly once a fixed PostgreSQL reaches CI**. That failure is the signal to:

1. flip the tripwire to assert correct ctids, gated on the PG version that
   ships the fix;
2. relax the `docs/FILTERING.md` "do not harvest ctid from a kNN scan"
   warning to name the fixed versions;
3. note the fix in `CHANGELOG.md`.

Until then the workaround guidance stands unchanged, and it is a proven
necessity rather than a preference: `MATERIALIZED`, text casts and subquery
nesting were all tested and all still yield the sentinel.

---

## Message body as filed (archived for reference)


Hi,

`ExecForceStoreHeapTuple()` does not set `slot->tts_tid` when the target
slot is a `TTS_IS_BUFFERTUPLE` slot. Any plan that re-stores a heap tuple
through it and then projects `ctid` therefore gets `(4294967295,0)`
instead of the row's real heap TID.

The affected branch (`src/backend/executor/execTuples.c`):

```c
    else if (TTS_IS_BUFFERTUPLE(slot))
    {
        MemoryContext oldContext;
        BufferHeapTupleTableSlot *bslot = (BufferHeapTupleTableSlot *) slot;

        ExecClearTuple(slot);                       /* invalidates tts_tid */
        slot->tts_flags &= ~TTS_FLAG_EMPTY;
        oldContext = MemoryContextSwitchTo(slot->tts_mcxt);
        bslot->base.tuple = heap_copytuple(tuple);
        slot->tts_flags |= TTS_FLAG_SHOULDFREE;
        MemoryContextSwitchTo(oldContext);
        /* tts_tid is never restored from tuple->t_self */

        if (shouldFree)
            pfree(tuple);
    }
```

`ExecClearTuple()` reaches `tts_buffer_heap_clear()`, which does
`ItemPointerSetInvalid(&slot->tts_tid)`. The tuple is then copied in, but
`tts_tid` is left invalid. The sibling path — `ExecStoreHeapTuple()` ->
`tts_heap_store_tuple()` — *does* `slot->tts_tid = tuple->t_self`, so this
reads as a plain asymmetry rather than an intentional choice.

It is user-visible because `slot_getsysattr()` answers
`SelfItemPointerAttributeNumber` directly out of `slot->tts_tid`
(`src/include/executor/tuptable.h`).

`nodeIndexscan.c` reaches it on a normal code path:
`reorderqueue_pop()` hands its palloc'd copy to
`ExecForceStoreHeapTuple()`. So for any index AM that sets
`xs_recheckorderby = true`, every tuple routed through the reorder queue
projects the invalid-TID sentinel — even though the AM set `xs_heaptid`
correctly, which is why the row data is right and only `ctid` is wrong.

### Reproducer — core GiST only, no extensions

Thin diagonal triangles, so the bounding-box distance strictly
under-estimates the true polygon distance: `gist_poly_consistent` sets
recheck, `was_exact` comes out false, and the tuples are pushed to the
reorder queue.

```sql
CREATE TABLE tri (id int, p polygon);
INSERT INTO tri
  SELECT i, ('((' || i*10 || ',0),(' || (i*10+9) || ',9),('
                  || (i*10+9) || ',0))')::polygon
  FROM generate_series(1,3000) i;
CREATE INDEX tri_idx ON tri USING gist (p);
ANALYZE tri;
SET enable_seqscan = off;

SELECT ctid, id FROM tri ORDER BY p <-> point(15000,4) LIMIT 5;
```

On 18.4:

```
      ctid      |  id
----------------+------
 (23,4)         | 1499     <- returned directly, ctid correct
 (4294967295,0) | 1500     <- came off the reorder queue
 (4294967295,0) | 1501
 (4294967295,0) | 1498
 (4294967295,0) | 1502
```

The one row `IndexNextWithReorder()` returned without queueing keeps its
real ctid, which pins the fault to the requeue path.

### Consequences

```sql
-- ctid self-join: finds 1 row, not 5
WITH k AS (SELECT ctid AS c FROM tri ORDER BY p <-> point(15000,4) LIMIT 5)
SELECT count(*) FROM tri t JOIN k ON t.ctid = k.c;

-- and this quietly updates ONE row instead of five, with no error
WITH k AS (SELECT ctid AS c FROM tri ORDER BY p <-> point(15000,4) LIMIT 5)
UPDATE tri SET ... WHERE ctid IN (SELECT c FROM k);
```

The `UPDATE` is the case I would highlight: it does not fail, it just
affects the wrong number of rows.

### Verification

Built both ways on one machine and ran one script — stock 18.4 versus an
18.3 tree with only the attached hunk applied:

```
                                    unpatched   patched
    ctid self-join, expect 5              1         5
    UPDATE ... WHERE ctid, expect 5       1         5
    sentinel ctids at LIMIT 50        49/50      0/50
```

I also checked that there is no query-level workaround: `WITH ... AS
MATERIALIZED`, casting to `text` inside a subquery, and extra subquery
nesting all still return the sentinel, since it is already in the slot
before any of them run. Forcing a seqscan returns correct ctids but
abandons the index.

49 of 50 rather than 50 is the `was_exact` fast path again: a tuple whose
index-returned ORDER BY value compares equal to the recomputed one is
returned without queueing. An AM that cannot usefully bound its ORDER BY
value and advertises `-inf` has 100% of its tuples queued.

### Patch

One line plus a comment, restoring `tts_tid` in that branch, mirroring
what `tts_heap_store_tuple()` already does:

```c
    slot->tts_tid = tuple->t_self;
```

`ExecForceStoreHeapTuple()`'s body is byte-identical in REL_13_STABLE,
REL_14_STABLE, REL_15_STABLE, REL_16_STABLE, REL_17_STABLE and
REL_18_STABLE (I hashed the function in each tree), so it applies
unchanged to all of them. I'll leave the back-patch decision to you.

I found this via an out-of-tree index AM that sets
`xs_recheckorderby = true` to re-rank approximate distances exactly, but
as above it needs nothing outside core to reproduce.

Thanks,
Greg Burd
