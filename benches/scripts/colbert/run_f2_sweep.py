"""
Phase F-2 confirmation: load NFCorpus ColBERT embeddings into pg16, build the
PERSISTENT vec_colbert_ops index, and run the sweep that confirms (or refutes)
the SciFact recall gain on a second, out-of-domain corpus.

Schema cb2 (vs F-1's cb). Three arms, top-k=10:
  A (F-2 PERSISTENT): turbovec.colbert_search(...) reading the persistent
                      vec_colbert_ops index (auto-routed once the index
                      exists; same SQL surface as F-1).
  B (Phase D baseline): pooled-index stage1 (ORDER BY pooled <=> q LIMIT cand)
                        + turbovec.max_sim rerank to top-k.
  Ceiling:   brute-force exact max_sim over ALL docs (no quantization), python.

NFCorpus _ids are strings; embed_nfcorpus.py mapped docs to synthetic int64
ids (cb2.docs.id) and kept the real query-id strings (cb2.queries.qid text).
qrels.json already has synthetic int doc-ids as keys.

Persistent-index confirmation: we (1) report build time + on-disk size via
pg_relation_size, (2) prove Arm A reads the persistent index rather than
rebuilding a per-call backend cache by measuring backend RSS stability and
the absence of the F-1 ~28 MB/call climb across a no-reconnect probe, and (3)
check Arm A == exact ceiling at high (per_token_k, candidate_n).

Memory discipline identical to F-1: floki has ZERO swap. The sweep client
reconnects every RECONNECT_EVERY queries as a belt-and-suspenders cap (the
F-2 persistent path is expected NOT to leak, but we keep the guard), and a
watchdog aborts (writing partial results) below 6 GiB available.
"""
import os, sys, json, time, argparse, subprocess
import numpy as np
import psycopg

OUT = os.environ.get("OUTDIR", "/tmp/colbert-nfcorpus")
DSN = os.environ.get("DSN", "host=/home/gburd/.pgrx port=28816 dbname=postgres")
DIM = 128


def vec_literal(arr):
    return "[" + ",".join(f"{x:.6f}" for x in arr) + "]"


def vecarr_literal(mat):
    parts = []
    for row in mat:
        inner = "[" + ",".join(f"{x:.6f}" for x in row) + "]"
        parts.append('"' + inner + '"')
    return "{" + ",".join(parts) + "}"


def load_shard(prefix):
    tokens = np.load(os.path.join(OUT, f"{prefix}_tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(OUT, f"{prefix}_offsets.npy"))
    pooled = np.load(os.path.join(OUT, f"{prefix}_pooled.npy"))
    ids = np.load(os.path.join(OUT, f"{prefix}_ids.npy"))
    return tokens, offsets, pooled, ids


def setup_schema(conn):
    # turbovec is installed as a bare schema + AM (no pg_extension row), so we
    # do NOT run CREATE EXTENSION; the turbovec.* objects already exist.
    with conn.cursor() as cur:
        cur.execute("drop schema if exists cb2 cascade;")
        cur.execute("create schema cb2;")
        cur.execute(
            "create table cb2.docs(id bigint primary key, "
            "tokens turbovec.vector[], pooled turbovec.vector);")
        cur.execute(
            "create table cb2.queries(qid text primary key, "
            "tokens turbovec.vector[], pooled turbovec.vector);")
        cur.execute(
            "create table cb2.qrels(qid text, doc_id bigint, rel int);")
    conn.commit()


def load_docs(conn):
    tokens, offsets, pooled, ids = load_shard("docs")
    n = len(ids)
    t0 = time.time()
    with conn.cursor() as cur:
        for i in range(n):
            tok = np.asarray(tokens[offsets[i]:offsets[i + 1]])
            cur.execute(
                "insert into cb2.docs(id, tokens, pooled) values (%s, %s::turbovec.vector[], %s::turbovec.vector)",
                (int(ids[i]), vecarr_literal(tok), vec_literal(pooled[i])))
            if i % 500 == 499:
                conn.commit()
                print(f"  docs {i+1}/{n}", flush=True)
        conn.commit()
    print(f"loaded {n} docs in {time.time()-t0:.1f}s", flush=True)
    return tokens, offsets, pooled, ids


def load_queries(conn):
    # query tokens parallel to queries_ids.npy (synthetic 0..n); the real
    # string ids come from queries_qids.json (cb2.queries.qid is the string).
    tokens, offsets, pooled, ids = load_shard("queries")
    qstrings = json.load(open(os.path.join(OUT, "queries_qids.json")))
    assert len(qstrings) == len(ids), "queries_qids.json length mismatch"
    n = len(ids)
    with conn.cursor() as cur:
        for i in range(n):
            tok = np.asarray(tokens[offsets[i]:offsets[i + 1]])
            cur.execute(
                "insert into cb2.queries(qid, tokens, pooled) values (%s, %s::turbovec.vector[], %s::turbovec.vector)",
                (qstrings[i], vecarr_literal(tok), vec_literal(pooled[i])))
        conn.commit()
    print(f"loaded {n} queries", flush=True)
    return tokens, offsets, pooled, ids, qstrings


def load_qrels(conn):
    qmap = json.load(open(os.path.join(OUT, "qrels.json")))
    with conn.cursor() as cur:
        for qid, docs in qmap.items():
            for did, rel in docs.items():
                cur.execute("insert into cb2.qrels values (%s,%s,%s)",
                            (qid, int(did), int(rel)))
        conn.commit()
    print(f"loaded qrels for {len(qmap)} queries", flush=True)
    return qmap


def build_pooled_index(conn, bit_width):
    with conn.cursor() as cur:
        cur.execute("drop index if exists cb2.docs_pooled_idx;")
        t0 = time.time()
        cur.execute(
            f"create index docs_pooled_idx on cb2.docs using turbovec "
            f"(pooled vec_cosine_ops) with (bit_width={bit_width});")
        conn.commit()
        dt = time.time() - t0
    print(f"  built pooled index bit_width={bit_width} in {dt:.1f}s", flush=True)
    return dt


def build_colbert_index(conn, bit_width, lists=None):
    """Build the PERSISTENT F-2 vec_colbert_ops token index. Returns
    (build_seconds, on_disk_bytes)."""
    opts = [f"bit_width={bit_width}"]
    if lists:
        opts.append(f"lists={lists}")
    with_clause = " with (" + ", ".join(opts) + ")"
    with conn.cursor() as cur:
        cur.execute("drop index if exists cb2.cb2_colbert;")
        conn.commit()
        t0 = time.time()
        cur.execute(
            f"create index cb2_colbert on cb2.docs using turbovec "
            f"(tokens vec_colbert_ops){with_clause};")
        conn.commit()
        dt = time.time() - t0
        cur.execute("select pg_relation_size('cb2.cb2_colbert');")
        size = int(cur.fetchone()[0])
    print(f"  built PERSISTENT colbert index bit_width={bit_width} lists={lists} "
          f"in {dt:.1f}s, on-disk {size/1e6:.1f} MB", flush=True)
    return dt, size


# ---------- metrics ----------
def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_doc_ids, qrel, k=10):
    rels = [qrel.get(str(d), 0) for d in ranked_doc_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids, qrel, k=10):
    rel_docs = {int(d) for d, r in qrel.items() if r > 0}
    if not rel_docs:
        return None
    got = set(int(d) for d in ranked_doc_ids[:k])
    return len(got & rel_docs) / len(rel_docs)


# ---------- exact ceiling (python, no quantization) ----------
def exact_maxsim_ceiling(qtoks, qoff, qstrings, dtoks, doff, did_arr, qmap, k=10):
    ndcgs, recalls = [], []
    dlist = [np.asarray(dtoks[doff[j]:doff[j + 1]]) for j in range(len(did_arr))]
    rankings = {}
    for qi in range(len(qstrings)):
        qid = qstrings[qi]
        if qid not in qmap:
            continue
        q = np.asarray(qtoks[qoff[qi]:qoff[qi + 1]])
        scores = np.empty(len(dlist), dtype=np.float32)
        for j, d in enumerate(dlist):
            sim = q @ d.T
            scores[j] = sim.max(axis=1).sum()
        top = np.argsort(-scores)[:k]
        ranked = [int(did_arr[j]) for j in top]
        rankings[qid] = ranked
        ndcgs.append(ndcg_at_k(ranked, qmap[qid]))
        r = recall_at_k(ranked, qmap[qid])
        if r is not None:
            recalls.append(r)
    return float(np.mean(ndcgs)), float(np.mean(recalls)), rankings


# ---------- reconnect-bounded backend (belt + suspenders) ----------
RECONNECT_EVERY = int(os.environ.get("RECONNECT_EVERY", "40"))


def _fresh_conn():
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("set search_path=cb2,turbovec,public;")
    return c


# ---------- arm A: colbert_search (auto-routes to persistent index) ----------
def run_arm_a(conn, qmap, qtoks, qoff, qstrings, per_token_k, candidate_n, bit_width,
             k=10, time_it=False, return_rankings=False):
    ndcgs, recalls, lats = [], [], []
    rankings = {}
    work = _fresh_conn()
    done = 0
    try:
        cur = work.cursor()
        for qi in range(len(qstrings)):
            qid = qstrings[qi]
            if qid not in qmap:
                continue
            if done and done % RECONNECT_EVERY == 0:
                cur.close(); work.close()
                work = _fresh_conn(); cur = work.cursor()
            q = np.asarray(qtoks[qoff[qi]:qoff[qi + 1]])
            qlit = vecarr_literal(q)
            t0 = time.perf_counter()
            cur.execute(
                "select id, score from turbovec.colbert_search("
                "'cb2.docs'::regclass,'id','tokens', %s::turbovec.vector[], %s, %s, %s, %s)",
                (qlit, k, per_token_k, candidate_n, bit_width))
            rows = cur.fetchall()
            dt = (time.perf_counter() - t0) * 1000
            if time_it:
                lats.append(dt)
            ranked = [int(r[0]) for r in rows]
            if return_rankings:
                rankings[qid] = ranked
            ndcgs.append(ndcg_at_k(ranked, qmap[qid]))
            rr = recall_at_k(ranked, qmap[qid])
            if rr is not None:
                recalls.append(rr)
            done += 1
    finally:
        work.close()
    return float(np.mean(ndcgs)), float(np.mean(recalls)), lats, rankings


# ---------- arm B: pooled index stage1 + max_sim rerank ----------
def run_arm_b(conn, qmap, qtoks, qoff, qpooled, qstrings, candidate_n, k=10, time_it=False):
    ndcgs, recalls, lats = [], [], []
    work = _fresh_conn()
    done = 0
    try:
        cur = work.cursor()
        for qi in range(len(qstrings)):
            qid = qstrings[qi]
            if qid not in qmap:
                continue
            if done and done % RECONNECT_EVERY == 0:
                cur.close(); work.close()
                work = _fresh_conn(); cur = work.cursor()
            q = np.asarray(qtoks[qoff[qi]:qoff[qi + 1]])
            qlit = vecarr_literal(q)
            plit = vec_literal(qpooled[qi])
            t0 = time.perf_counter()
            cur.execute(
                "with cand as ("
                "  select id, tokens from cb2.docs "
                "  order by pooled <=> %s::turbovec.vector limit %s"
                ") select id, turbovec.max_sim(%s::turbovec.vector[], tokens) as score "
                "from cand order by score desc limit %s",
                (plit, candidate_n, qlit, k))
            rows = cur.fetchall()
            dt = (time.perf_counter() - t0) * 1000
            if time_it:
                lats.append(dt)
            ranked = [int(r[0]) for r in rows]
            ndcgs.append(ndcg_at_k(ranked, qmap[qid]))
            rr = recall_at_k(ranked, qmap[qid])
            if rr is not None:
                recalls.append(rr)
            done += 1
    finally:
        work.close()
    return float(np.mean(ndcgs)), float(np.mean(recalls)), lats


def pctl(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 999.0


def backend_rss_probe(qmap, qtoks, qoff, qstrings, per_token_k, candidate_n, bit_width,
                      n_calls=60):
    """Confirm Arm A reads the PERSISTENT index (no per-call rebuild / leak):
    issue n_calls colbert_search calls on a SINGLE backend (NO reconnect) and
    track that backend's RSS. The F-1 backend-cache path climbed ~28 MB/call;
    the persistent path should stay flat after the first warm call. Returns
    {rss_first_mb, rss_last_mb, growth_per_call_kb, pid}."""
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("set search_path=cb2,turbovec,public;")
    pid = c.execute("select pg_backend_pid();").fetchone()[0]

    def rss_kb():
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except FileNotFoundError:
            return None
        return None

    cur = c.cursor()
    rss_samples = []
    issued = 0
    qi = 0
    while issued < n_calls and qi < len(qstrings):
        qid = qstrings[qi]; qi += 1
        if qid not in qmap:
            continue
        q = np.asarray(qtoks[qoff[qi - 1]:qoff[qi]])
        qlit = vecarr_literal(q)
        cur.execute(
            "select id, score from turbovec.colbert_search("
            "'cb2.docs'::regclass,'id','tokens', %s::turbovec.vector[], 10, %s, %s, %s)",
            (qlit, per_token_k, candidate_n, bit_width))
        cur.fetchall()
        issued += 1
        if issued in (1, 5, 10, 20, 40, n_calls):
            rss_samples.append((issued, rss_kb()))
    c.close()
    # growth per call measured from call 5 (warm) to last, ignoring cold ramp.
    warm = [(n, r) for n, r in rss_samples if n >= 5 and r is not None]
    if len(warm) >= 2:
        dn = warm[-1][0] - warm[0][0]
        dr = warm[-1][1] - warm[0][1]
        growth_kb = dr / dn if dn else 0.0
    else:
        growth_kb = None
    first = next((r for n, r in rss_samples if r is not None), None)
    last = rss_samples[-1][1] if rss_samples else None
    return {
        "pid": pid, "calls": issued,
        "rss_first_mb": round(first / 1024, 1) if first else None,
        "rss_last_mb": round(last / 1024, 1) if last else None,
        "rss_growth_per_call_kb": round(growth_kb, 1) if growth_kb is not None else None,
        "samples": [{"call": n, "rss_mb": round(r / 1024, 1) if r else None}
                    for n, r in rss_samples],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="(re)load data")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--colbert-lists", type=int, default=None,
                    help="lists reloption for the persistent colbert index (default: AM auto)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conn = psycopg.connect(DSN, autocommit=False)
    conn.execute("set search_path=cb2,turbovec,public;")

    qtoks, qoff, qpooled, qids = load_shard("queries")
    dtoks, doff, dpooled, dids = load_shard("docs")
    qmap = json.load(open(os.path.join(OUT, "qrels.json")))
    qstrings = json.load(open(os.path.join(OUT, "queries_qids.json")))

    if args.load:
        setup_schema(conn)
        load_docs(conn)
        load_queries(conn)
        load_qrels(conn)

    results = {"corpus": "nfcorpus", "n_docs": int(len(dids)),
               "n_queries_with_qrels": sum(1 for q in qstrings if q in qmap),
               "n_judgments": sum(len(v) for v in qmap.values()),
               "avg_tok_per_doc": float((doff[-1]) / len(dids)),
               "avg_tok_per_query": float((qoff[-1]) / len(qids)),
               "n_tokens_total": int(doff[-1]),
               "configs": []}

    if args.ceiling:
        print("computing exact ceiling (brute-force max_sim, all docs)...", flush=True)
        t0 = time.time()
        cn, cr, _ = exact_maxsim_ceiling(qtoks, qoff, qstrings, dtoks, doff, dids, qmap)
        results["exact_ceiling"] = {"ndcg@10": round(cn, 4), "recall@10": round(cr, 4),
                                    "compute_s": round(time.time() - t0, 1)}
        print(f"  ceiling nDCG@10={cn:.4f} Recall@10={cr:.4f} "
              f"({results['exact_ceiling']['compute_s']}s)", flush=True)

    if args.sweep:
        # focused grid mirroring the SciFact run (task: ~6-8 configs)
        configs = [
            (2, 64, 256), (4, 64, 256),      # bit erosion @ value point
            (4, 64, 128), (4, 64, 512),      # candidate_n effect (low budget = the F-2 thesis)
            (4, 128, 256), (4, 128, 512),    # per_token_k effect / best-recall corner
            (2, 128, 512),                   # 2-bit best-effort
        ]
        built_pooled = {}
        built_colbert = {}
        TIMED_N = int(os.environ.get("TIMED_N", "100"))

        # --- build persistent colbert indexes per bit_width up front + record size ---
        results["persistent_colbert_index"] = {}
        for bw in sorted({c[0] for c in configs}):
            avail = avail_gib()
            print(f"=== build persistent colbert index bw={bw} (avail={avail:.1f} GiB) ===", flush=True)
            if avail < 6.0:
                print(f"  ABORT before build: avail {avail:.1f} GiB < 6 GiB.", flush=True)
                results["aborted_low_memory_at"] = {"phase": "colbert_build", "bit_width": bw,
                                                    "avail_gib": round(avail, 1)}
                break
            dt, size = build_colbert_index(conn, bw, lists=args.colbert_lists)
            built_colbert[bw] = (dt, size)
            results["persistent_colbert_index"][str(bw)] = {
                "build_s": round(dt, 1), "on_disk_bytes": size,
                "on_disk_mb": round(size / 1e6, 1), "lists": args.colbert_lists}

        # --- confirm Arm A reads the persistent index (no per-call rebuild/leak) ---
        if built_colbert:
            probe_bw = sorted(built_colbert.keys())[-1]
            # rebuild that bw's persistent index so the probe runs against it
            build_colbert_index(conn, probe_bw, lists=args.colbert_lists)
            print(f"=== persistent-read probe (bw={probe_bw}, single backend, no reconnect) ===", flush=True)
            probe = backend_rss_probe(qmap, qtoks, qoff, qstrings, 64, 256, probe_bw, n_calls=60)
            results["persistent_read_probe"] = probe
            print(f"  backend rss: first={probe['rss_first_mb']}MB last={probe['rss_last_mb']}MB "
                  f"growth/call={probe['rss_growth_per_call_kb']}KB (F-1 was ~28000 KB/call)", flush=True)

        for (bw, ptk, cand) in configs:
            avail = avail_gib()
            print(f"=== config bw={bw} ptk={ptk} cand={cand} (avail={avail:.1f} GiB) ===", flush=True)
            if avail < 6.0:
                print(f"  ABORT: available RAM {avail:.1f} GiB < 6 GiB floor.", flush=True)
                results["aborted_low_memory_at"] = {"bit_width": bw, "per_token_k": ptk,
                                                    "candidate_n": cand, "avail_gib": round(avail, 1)}
                break
            if bw not in built_pooled:
                built_pooled[bw] = build_pooled_index(conn, bw)
            # rebuild the persistent colbert index for this bw (one resident at a
            # time keeps memory bounded; build is fast).
            if bw not in built_colbert:
                dt, size = build_colbert_index(conn, bw, lists=args.colbert_lists)
                built_colbert[bw] = (dt, size)
            else:
                build_colbert_index(conn, bw, lists=args.colbert_lists)
            cb_build_s, cb_size = built_colbert[bw]

            an, ar, _, _ = run_arm_a(conn, qmap, qtoks, qoff, qstrings, ptk, cand, bw)
            timed = qstrings[:TIMED_N]
            _, _, alat, _ = run_arm_a(conn, qmap, qtoks, qoff, timed, ptk, cand, bw, time_it=True)

            bn, br, _ = run_arm_b(conn, qmap, qtoks, qoff, qpooled, qstrings, cand)
            _, _, blat = run_arm_b(conn, qmap, qtoks, qoff, qpooled, timed, cand, time_it=True)

            entry = {
                "bit_width": bw, "per_token_k": ptk, "candidate_n": cand,
                "pooled_index_build_s": round(built_pooled[bw], 2),
                "colbert_index_build_s": round(cb_build_s, 2),
                "colbert_index_mb": round(cb_size / 1e6, 1),
                "arm_a_colbert_persistent": {
                    "ndcg@10": round(an, 4), "recall@10": round(ar, 4),
                    "p50_ms": round(pctl(alat, 50), 2), "p95_ms": round(pctl(alat, 95), 2)},
                "arm_b_pooled_rerank": {
                    "ndcg@10": round(bn, 4), "recall@10": round(br, 4),
                    "p50_ms": round(pctl(blat, 50), 2), "p95_ms": round(pctl(blat, 95), 2)},
                "delta_ndcg@10": round(an - bn, 4),
                "delta_recall@10": round(ar - br, 4),
            }
            results["configs"].append(entry)
            print(f"  A(persist): nDCG={an:.4f} R={ar:.4f} p50={pctl(alat,50):.1f}ms | "
                  f"B: nDCG={bn:.4f} R={br:.4f} p50={pctl(blat,50):.1f}ms | "
                  f"dNDCG={an-bn:+.4f} dR={ar-br:+.4f}", flush=True)
            if args.out:
                json.dump(results, open(args.out, "w"), indent=2)

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
