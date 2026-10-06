#!/usr/bin/env python3
"""For each mismatched TID T, find the live TID U whose codes soak holds at T.
Then: is T a HOT chain member of U's row? (same heap block, same `id` column?)
Read the heap row at T (if visible) and at U and compare the `id` column."""
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
    return {i: (codes[s*stride:(s+1)*stride] + scales[s*4:(s+1)*4], s) for s, i in enumerate(ids)}
soak = image("soak_idx"); ref = image("soak_ref")
by = {hashlib.blake2b(v[0], digest_size=12).digest(): t for t, v in ref.items()}
def tid(t): return f"({t>>32},{t&0xFFFF})"
def row_id(t):
    r = c.execute("SELECT id FROM soak WHERE ctid = %s::tid", (tid(t),)).fetchone()
    return r[0] if r else None
mism = [t for t in ref if t in soak and soak[t][0] != ref[t][0]]
k = collections.Counter(); samples = []
for t in mism:
    u = by[hashlib.blake2b(soak[t][0], digest_size=12).digest()]
    same_blk = (t >> 32) == (u >> 32)
    it, iu = row_id(t), row_id(u)
    k[f"same_block={same_blk} same_id_col={it == iu}"] += 1
    if len(samples) < 6: samples.append((tid(t), it, tid(u), iu, soak[t][1], soak[u][1] if u in soak else None))
print(dict(k))
for s in samples: print("  soak@T", s[0], "id", s[1], "holds codes of live U", s[2], "id", s[3], "| slots T,U:", s[4], s[5])
# Are the mismatched ids' heap rows ones that were UPDATEd (UPDATE soak SET tv=tv) ?
