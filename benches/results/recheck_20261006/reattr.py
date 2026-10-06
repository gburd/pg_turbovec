import subprocess, collections, sys
data, nq, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
anywhere = [  # priority order; first bucket with a matching frame anywhere in the stack wins
  ("TOAST fetch", ("toast_fetch_datum", "heap_fetch_toast_slice", "detoast_attr", "toast_open_indexes", "toast_close_indexes")),
  ("CBOR decode (pgrx serde)", ("serde_cbor",)),
  ("reorder queue: tuple copy + heap (core)", ("reorderqueue_push", "reorderqueue_pop", "pairingheap", "ExecCopySlotHeapTuple", "ExecForceStoreHeapTuple", "datumCopy")),
  ("heap fetch of candidate (core)", ("index_fetch_heap", "heapam_index_fetch_tuple")),
  ("turbovec scan, backend thread", ("turbovec::search", "rayon", "ReadOnlyIndex", "amgettuple", "amrescan", "ambeginscan")),
  ("exact distance kernel (ours)", ("cosine_distance_wrapper",)),
  ("other executor (core)", ("IndexNextWithReorder", "ExecInterpExpr", "ExecScan")),
]
out = subprocess.run(["perf", "script", "-i", data, "-F", "tid,ip,sym"], capture_output=True, text=True).stdout
cnt = collections.Counter()
for blk in out.split("\n\n"):
    lines = [l.strip() for l in blk.strip().splitlines()]
    if len(lines) < 2: continue
    stack = [l.split(None, 1)[1] if " " in l else l for l in lines[1:]]
    for name, keys in anywhere:
        if any(any(x in fr for x in keys) for fr in stack): cnt[name] += 1; break
    else: cnt["parse/plan/libpq/other"] += 1
tot = sum(cnt.values())
print(f"{data.split('/')[-1]}: backend on-CPU {tot*100/nq/1e3:.2f} ms/query")
for name in [b[0] for b in anywhere] + ["parse/plan/libpq/other"]:
    print(f"   {name:42s} {cnt[name]*100/nq/k:6.2f} us/candidate")
