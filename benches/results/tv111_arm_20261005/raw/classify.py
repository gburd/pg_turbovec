#!/usr/bin/env python3
"""Classify soak_idx vs soak_ref discrepancies by CONTENT.
mismatch at TID T: does soak's (codes) at T equal ref's codes at some OTHER
TID (i.e. a stale copy of a live row)? how many code bytes differ from ref[T]?
extra at T: does soak's codes at T equal any live row's codes (stale old
version of an UPDATEd row) or none (an aborted insert)?"""
import struct, collections, hashlib, psycopg
c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True); c.execute("SET search_path=turbovec,public")
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
    return {i: (codes[s*stride:(s+1)*stride], scales[s*4:(s+1)*4], s) for s, i in enumerate(ids)}
soak = image("soak_idx"); ref = image("soak_ref")
by_codes = {}
for t, (cd, sc, _) in ref.items(): by_codes.setdefault(hashlib.blake2b(cd + sc, digest_size=12).digest(), t)
def tid(t): return f"({t>>32},{t&0xFFFF})"
mism = [t for t in ref if t in soak and soak[t][:2] != ref[t][:2]]
k = collections.Counter(); diffb = []
for t in mism:
    h = hashlib.blake2b(soak[t][0] + soak[t][1], digest_size=12).digest()
    other = by_codes.get(h)
    k["soak_bytes_equal_another_live_row" if other is not None else "soak_bytes_match_no_live_row"] += 1
    diffb.append(sum(a != b for a, b in zip(soak[t][0], ref[t][0])))
print("MISMATCH", len(mism), dict(k), "code-bytes-differing (of 512): min/median/max",
      min(diffb) if diffb else None, sorted(diffb)[len(diffb)//2] if diffb else None, max(diffb) if diffb else None)
extra = [t for t in soak if t not in ref]
k2 = collections.Counter()
for t in extra:
    h = hashlib.blake2b(soak[t][0] + soak[t][1], digest_size=12).digest()
    k2["equals_a_live_row(stale old version)" if h in by_codes else "matches_no_live_row(aborted insert?)"] += 1
print("EXTRA", len(extra), dict(k2))
# slot positions of bad entries in soak (near the tail = recent appends?)
n = len(soak)
pos = sorted(soak[t][2] for t in mism + extra)
print("bad-entry slot positions: first", pos[:3], "median", pos[len(pos)//2], "last", pos[-3:], "of", n)
print("sample mismatch TIDs:", [tid(t) for t in mism[:5]], "sample extra:", [tid(t) for t in extra[:5]])
