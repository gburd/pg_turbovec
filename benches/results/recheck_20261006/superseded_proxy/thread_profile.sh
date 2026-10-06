# usage: thread_profile.sh <perf.data> <backend_tid>
D=$1; T=$2
echo "== backend thread $T, self time by symbol"
perf report -i $D --tid=$T --no-children --sort symbol --stdio 2>/dev/null | grep -E '^\s+[0-9.]+%' | head -25 | cut -c1-140
echo "== backend thread $T, inclusive (children) for key frames"
perf report -i $D --tid=$T --children --sort symbol --stdio 2>/dev/null | grep -E '^\s+[0-9.]+%\s+[0-9.]+%' | grep -E 'IndexNextWithReorder|index_getnext_slot|amgettuple|index_fetch_heap|heapam_index_fetch|EvalOrderByExpressions|ExecInterpExpr|cosine_distance|cbor|detoast|toast_|heap_fetch|reorderqueue|ExecCopySlot|search_multi|search::|rayon|tts_buffer|heap_hot_search|ReadBuffer|PinBuffer|pairingheap|ExecScan|ExecProject|slot_deform|amrescan|ambeginscan|flat_search|ReadOnlyIndex|populate_batch|pg_detoast|heap_tuple_untoast|toast_fetch|table_relation_fetch_toast|index_beginscan|ExecutorRun|standard_ExecutorRun|PortalRun|exec_simple_query' | head -40 | cut -c1-150
