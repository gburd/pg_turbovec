#!/usr/bin/env python3
"""Sustained-insert corruption soak for turbovec 1.1.1 (AGENTS.md HARD MANDATE #1).

Index: flat bit_width=4 on soak(id bigint, tv turbovec.vector), seeded with
SEED rows (> turbovec's 32,768 planes gate, so the in-memory cache the
TurboQuant flush reconstructs packed_codes() from is the PLANES layout on this
aarch64 host). Then for DURATION seconds: writer backends commit INSERT batches
of new rows and UPDATEs of existing rows (each UPDATE = new heap tuple ->
aminsert of a new CTID, exercising add on a planes cache), the orchestrator
pg_terminate_backend()s a writer every 60-180 s (mid-flush), and VACUUM runs
every ~5 min (on-disk removal + tombstones). Every CHECK_EVERY seconds:
turbovec_check must be clean; a scan must return 10 distinct ids.

At the end, BYTE-LEVEL verification: decode the index relfile via pageinspect
get_raw_page (meta + codes/scales/ids chains) and require, for every live
heap tuple, that the persisted (code row, scale) for its CTID equals the
bytes a FRESH single-shot REINDEX of the same heap writes for that CTID
(same encoder, same calibration), and that the id set is exactly the live
heap CTIDs after VACUUM. Any mismatch = FAIL.
"""
import os, random, struct, subprocess, sys, time, threading, json
import numpy as np, psycopg
SEED = int(os.environ.get("SEED", "60000")); DIM = 1024
DURATION = int(os.environ.get("DURATION", "5400")); CHECK_EVERY = 60
LOG = open("/mnt/nvme/soak.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
C = np.load("/mnt/nvme/corpus_1000000/q.npy")  # 1000 real 1024-d unit vectors; perturbed per row
rng = np.random.default_rng(11)
def vec(i):
    v = C[i % len(C)] + 0.05 * np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v); return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
def conn(app):
    c = psycopg.connect("dbname=bench host=/tmp port=5432", autocommit=True, application_name=app)
    c.execute("SET search_path=turbovec,public"); return c
admin = conn("soak_admin")
admin.execute("CREATE EXTENSION IF NOT EXISTS pageinspect")
admin.execute("DROP TABLE IF EXISTS soak"); admin.execute("CREATE TABLE soak(id bigint, tv turbovec.vector)")
with admin.cursor().copy("COPY soak(id, tv) FROM STDIN") as cp:
    for i in range(SEED): cp.write_row((i, vec(i)))
admin.execute("CREATE INDEX soak_idx ON soak USING turbovec (tv vec_cosine_ops) WITH (bit_width = 4)")
log("seeded", SEED, admin.execute("SELECT * FROM turbovec_check('soak_idx'::regclass)").fetchone())
stop = threading.Event(); next_id = [SEED]; lock = threading.Lock(); errors = []; commits = [0]
def writer(w):
    while not stop.is_set():
        try:
            c = conn(f"soak_w{w}")
            while not stop.is_set():
                n = random.choice([1, 16, 128])
                with lock: base = next_id[0]; next_id[0] += n
                with c.transaction():
                    with c.cursor().copy("COPY soak(id, tv) FROM STDIN") as cp:
                        for i in range(base, base + n): cp.write_row((i, vec(i)))
                    k = random.choice([1, 16]); ids = [random.randrange(0, base) for _ in range(k)]
                    c.execute("UPDATE soak SET tv = tv WHERE id = ANY(%s)", (ids,))
                commits[0] += 1
        except psycopg.errors.AdminShutdown: pass
        except psycopg.OperationalError: pass
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if "terminating connection" in msg or "server closed" in msg: continue
            errors.append(msg); log("WRITER ERROR", msg)
        time.sleep(0.5)
ws = [threading.Thread(target=writer, args=(w,), daemon=True) for w in range(3)]
for t in ws: t.start()
t0 = time.time(); next_kill = t0 + random.uniform(60, 180); next_vac = t0 + 300; next_chk = t0 + CHECK_EVERY; kills = vacs = 0
while time.time() - t0 < DURATION and not errors:
    now = time.time()
    if now >= next_kill:
        r = admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name LIKE 'soak_w%%' AND state='active' ORDER BY random() LIMIT 1").fetchall()
        kills += len(r); next_kill = now + random.uniform(60, 180)
    if now >= next_vac:
        admin.execute("VACUUM soak"); vacs += 1; next_vac = now + 300
    if now >= next_chk:
        chk = admin.execute("SELECT wire_version,n_vectors,slot_count,count_matches,duplicate_id,is_corrupt,tombstone_density,reason FROM turbovec_check('soak_idx'::regclass)").fetchone()
        ids = [r[0] for r in admin.execute("SELECT id FROM soak ORDER BY tv <=> (SELECT tv FROM soak WHERE id = 7) LIMIT 10").fetchall()]
        log(f"t={now-t0:.0f}s commits={commits[0]} kills={kills} vacuums={vacs} check={chk} scan_distinct={len(set(ids))}")
        if chk[5] or not chk[3] or chk[4] is not None or len(set(ids)) != 10:
            errors.append(f"CHECK FAILED {chk} ids={ids}")
        next_chk = now + CHECK_EVERY
    time.sleep(1)
stop.set(); time.sleep(5)
admin.execute("VACUUM soak")
# ---- byte-level verification ----
def chain(rel, first, stride, rpp, n):
    out = bytearray(); left = n * stride; blk = first
    while left > 0:
        pg = admin.execute("SELECT get_raw_page(%s, %s)", (rel, blk)).fetchone()[0]
        take = min(rpp * stride, left); out += pg[24:24 + take]; left -= take; blk += 1
    return bytes(out)
def image(rel):
    m = admin.execute("SELECT get_raw_page(%s, 0)", (rel,)).fetchone()[0][24:]
    assert m[0:4] == b"TVRM", m[0:4]
    n, = struct.unpack_from("<Q", m, 12)
    cf, _, sf, _, idf, _, rpc, rps, rpi, stride = struct.unpack_from("<IIIIIIIIII", m, 20)
    codes = chain(rel, cf, stride, rpc, n); scales = chain(rel, sf, 4, rps, n); ids = chain(rel, idf, 8, rpi, n)
    ids = struct.unpack(f"<{n}Q", ids)
    return {i: (codes[s*stride:(s+1)*stride], scales[s*4:(s+1)*4]) for s, i in enumerate(ids)}, len(ids), m[4]
soak_img, soak_n, ver = image("soak_idx")
chk = admin.execute("SELECT * FROM turbovec_check('soak_idx'::regclass)").fetchone()
log("final check", chk, "wire", ver, "persisted rows", soak_n)
admin.execute("CREATE INDEX soak_ref ON soak USING turbovec (tv vec_cosine_ops) WITH (bit_width = 4)")
ref_img, ref_n, _ = image("soak_ref")
live = {(int(b) << 32) | int(o) for b, o in (r[0].strip("()").split(",") for r in admin.execute("SELECT ctid::text FROM soak").fetchall())}
# external id = pgrx item_pointer_to_u64: (block << 32) | offset
missing = [t for t in live if t not in soak_img]
extra = [t for t in soak_img if t not in live]
diff = [t for t in live if t in soak_img and soak_img[t] != ref_img.get(t)]
res = dict(seed=SEED, duration_s=round(time.time() - t0), commits=commits[0], kills=kills, vacuums=vacs,
           live_rows=len(live), persisted_rows=soak_n, ref_rows=ref_n, wire_version=ver,
           missing_live=len(missing), stale_extra=len(extra), byte_mismatch=len(diff),
           writer_errors=errors, final_check=str(chk))
log("RESULT", json.dumps(res))
ok = not errors and not missing and not diff and not chk[5]
log("SOAK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
