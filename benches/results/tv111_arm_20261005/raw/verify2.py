#!/usr/bin/env python3
"""Correct oracle for the soak: compare soak_idx against soak_ref (fresh
CREATE INDEX on the same heap) -- both use HOT-chain ROOT TIDs, unlike
`SELECT ctid` (which returns the heap-only tuple's own TID).
  missing  = in ref, not in soak   -> a live row the index lost   (MUST be 0)
  mismatch = in both, bytes differ -> wrong persisted codes        (MUST be 0)
  extra    = in soak, not in ref   -> classify each TID's heap line pointer:
             LP_DEAD / dead tuple awaiting index vacuum = benign (PG14+ may
             bypass index vacuuming); LP_UNUSED or a LIVE root = BAD (a stale
             entry on a reusable/reused TID)."""
import struct, collections, psycopg
c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True)
c.execute("SET search_path=turbovec,public")
def chain(rel, first, stride, rpp, n):
    out = bytearray(); left = n * stride; blk = first
    while left > 0:
        pg = c.execute("SELECT get_raw_page(%s, %s)", (rel, blk)).fetchone()[0]
        take = min(rpp * stride, left); out += pg[24:24 + take]; left -= take; blk += 1
    return bytes(out)
def image(rel):
    m = c.execute("SELECT get_raw_page(%s, 0)", (rel,)).fetchone()[0][24:]
    n, = struct.unpack_from("<Q", m, 12)
    cf, _, sf, _, idf, _, rpc, rps, rpi, stride = struct.unpack_from("<IIIIIIIIII", m, 20)
    codes = chain(rel, cf, stride, rpc, n); scales = chain(rel, sf, 4, rps, n)
    ids = struct.unpack(f"<{n}Q", chain(rel, idf, 8, rpi, n))
    return {i: (codes[s*stride:(s+1)*stride], scales[s*4:(s+1)*4]) for s, i in enumerate(ids)}, n
soak, ns = image("soak_idx"); ref, nr = image("soak_ref")
missing = [t for t in ref if t not in soak]
mismatch = [t for t in ref if t in soak and soak[t] != ref[t]]
extra = [t for t in soak if t not in ref]
print(f"soak_idx rows={ns} soak_ref rows={nr} | missing={len(missing)} mismatch={len(mismatch)} extra={len(extra)}")
# classify extra TIDs by heap line-pointer state
byblk = collections.defaultdict(list)
for t in extra: byblk[t >> 32].append(t & 0xFFFF)
cls = collections.Counter()
for blk, offs in byblk.items():
    rows = c.execute("SELECT lp, lp_flags, t_xmax, (t_infomask2 & 32768) <> 0 AS heap_only FROM heap_page_items(get_raw_page('soak', %s))", (blk,)).fetchall()
    lp = {r[0]: r for r in rows}
    for o in offs:
        r = lp.get(o)
        if r is None: cls["beyond_page_end"] += 1
        elif r[1] == 0: cls["LP_UNUSED"] += 1
        elif r[1] == 3: cls["LP_DEAD"] += 1
        elif r[1] == 2: cls["LP_REDIRECT"] += 1
        else: cls["LP_NORMAL_heaponly" if r[3] else "LP_NORMAL_root"] += 1
print("extra TID heap state:", dict(cls))
# any extra on a NORMAL root must be a dead (deleted/updated) tuple, never visible
if cls.get("LP_NORMAL_root"):
    vis = 0
    for t in extra:
        b, o = t >> 32, t & 0xFFFF
        r = c.execute("SELECT count(*) FROM soak WHERE ctid = %s::tid", (f"({b},{o})",)).fetchone()[0]
        vis += r
    print("extra TIDs that are VISIBLE heap tuples:", vis)
print("first mismatches:", mismatch[:5])
