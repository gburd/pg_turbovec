"""
Phase F-1 gate: load SciFact ColBERT embeddings into pg16 + run the sweep.

Two arms, top-k=10:
  A (F-1):   turbovec.colbert_search(rel,id,token_col,query_tokens,k,per_token_k,candidate_n,bit_width)
  B (Phase D baseline): pooled-index stage1 (ORDER BY pooled <=> qpooled LIMIT cand)
                        + turbovec.max_sim rerank to top-k
  Ceiling:   brute-force exact max_sim over ALL docs (no quantization), in Python.

Metrics per config: nDCG@10, Recall@10 vs qrels; warm p50/p95 latency.

Memory-light: streams vectors to PG; never holds all token tensors beyond
what numpy mmap gives. Run normally (no torch here) but still modest RAM.
"""
import os, sys, json, time, argparse
import numpy as np
import psycopg

OUT = os.environ.get("OUTDIR", "/tmp/colbert-scifact")
DSN = os.environ.get("DSN", "host=/home/gburd/.pgrx port=28816 dbname=postgres")
DIM = 128


def vec_literal(arr):
    """1-D float32 array -> turbovec.vector text literal '[a,b,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in arr) + "]"


def vecarr_literal(mat):
    """2-D (n,128) -> postgres array-of-vector literal for cast to vector[]."""
    # Build as ARRAY['..'::vector, ...] via parameterized text -> we instead
    # produce the array literal string: {"[..]","[..]"}
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
    # NB: turbovec on this cluster is installed as a bare schema + AM (no
    # pg_extension row / control file), so we do NOT run CREATE EXTENSION;
    # the turbovec.* objects already exist. ponytail: skip extension create.
    with conn.cursor() as cur:
        cur.execute("drop schema if exists cb cascade;")
        cur.execute("create schema cb;")
        cur.execute(
            "create table cb.docs(id bigint primary key, "
            "tokens turbovec.vector[], pooled turbovec.vector);")
        cur.execute(
            "create table cb.queries(qid text primary key, "
            "tokens turbovec.vector[], pooled turbovec.vector);")
        cur.execute(
            "create table cb.qrels(qid text, doc_id bigint, rel int);")
    conn.commit()


def load_docs(conn):
    tokens, offsets, pooled, ids = load_shard("docs")
    n = len(ids)
    t0 = time.time()
    with conn.cursor() as cur:
        for i in range(n):
            tok = np.asarray(tokens[offsets[i]:offsets[i + 1]])
            cur.execute(
                "insert into cb.docs(id, tokens, pooled) values (%s, %s::turbovec.vector[], %s::turbovec.vector)",
                (int(ids[i]), vecarr_literal(tok), vec_literal(pooled[i])))
            if i % 500 == 499:
                conn.commit()
                print(f"  docs {i+1}/{n}", flush=True)
        conn.commit()
    print(f"loaded {n} docs in {time.time()-t0:.1f}s", flush=True)
    return tokens, offsets, pooled, ids


def load_queries(conn):
    tokens, offsets, pooled, ids = load_shard("queries")
    n = len(ids)
    with conn.cursor() as cur:
        for i in range(n):
            tok = np.asarray(tokens[offsets[i]:offsets[i + 1]])
            cur.execute(
                "insert into cb.queries(qid, tokens, pooled) values (%s, %s::turbovec.vector[], %s::turbovec.vector)",
                (str(ids[i]), vecarr_literal(tok), vec_literal(pooled[i])))
        conn.commit()
    print(f"loaded {n} queries", flush=True)
    return tokens, offsets, pooled, ids


def load_qrels(conn):
    qmap = json.load(open(os.path.join(OUT, "qrels.json")))
    with conn.cursor() as cur:
        for qid, docs in qmap.items():
            for did, rel in docs.items():
                cur.execute("insert into cb.qrels values (%s,%s,%s)",
                            (qid, int(did), int(rel)))
        conn.commit()
    print(f"loaded qrels for {len(qmap)} queries", flush=True)
    return qmap


def build_pooled_index(conn, bit_width):
    with conn.cursor() as cur:
        cur.execute("drop index if exists cb.docs_pooled_idx;")
        t0 = time.time()
        cur.execute(
            f"create index docs_pooled_idx on cb.docs using turbovec "
            f"(pooled vec_cosine_ops) with (bit_width={bit_width});")
        conn.commit()
        dt = time.time() - t0
    print(f"  built pooled index bit_width={bit_width} in {dt:.1f}s", flush=True)
    return dt


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
def exact_maxsim_ceiling(qtoks, qoff, qids, dtoks, doff, did_arr, qmap, k=10):
    """Brute force MaxSim over all docs for each query -> top-k ranking."""
    ndcgs, recalls = [], []
    # preslice doc token arrays once
    dlist = [np.asarray(dtoks[doff[j]:doff[j + 1]]) for j in range(len(did_arr))]
    for qi in range(len(qids)):
        qid = str(qids[qi])
        if qid not in qmap:
            continue
        q = np.asarray(qtoks[qoff[qi]:qoff[qi + 1]])  # (nq,128)
        scores = np.empty(len(dlist), dtype=np.float32)
        for j, d in enumerate(dlist):
            sim = q @ d.T  # (nq, nd)
            scores[j] = sim.max(axis=1).sum()
        top = np.argsort(-scores)[:k]
        ranked = [int(did_arr[j]) for j in top]
        ndcgs.append(ndcg_at_k(ranked, qmap[qid]))
        r = recall_at_k(ranked, qmap[qid])
        if r is not None:
            recalls.append(r)
    return float(np.mean(ndcgs)), float(np.mean(recalls))


# ---------- arm A: colbert_search ----------
# NB: colbert_search leaks ~28 MB of backend RSS per call (the backend-cached
# token index workspace is not fully freed between calls within a session).
# ponytail: reconnect every RECONNECT_EVERY queries to bound peak backend RSS
# to ~2-3 GiB instead of letting it climb to 18 GiB and OOM the postmaster.
# floki has ZERO swap; this is a hard safety requirement, not an optimisation.
RECONNECT_EVERY = int(os.environ.get("RECONNECT_EVERY", "40"))


def _fresh_conn():
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("set search_path=cb,turbovec,public;")
    return c


def run_arm_a(conn, qmap, qtoks, qoff, qids, per_token_k, candidate_n, bit_width, k=10, time_it=False):
    ndcgs, recalls, lats = [], [], []
    work = _fresh_conn()
    done = 0
    try:
        cur = work.cursor()
        for qi in range(len(qids)):
            qid = str(qids[qi])
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
                "'cb.docs'::regclass,'id','tokens', %s::turbovec.vector[], %s, %s, %s, %s)",
                (qlit, k, per_token_k, candidate_n, bit_width))
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


# ---------- arm B: pooled index stage1 + max_sim rerank ----------
def run_arm_b(conn, qmap, qtoks, qoff, qpooled, qids, candidate_n, k=10, time_it=False):
    ndcgs, recalls, lats = [], [], []
    work = _fresh_conn()
    done = 0
    try:
        cur = work.cursor()
        for qi in range(len(qids)):
            qid = str(qids[qi])
            if qid not in qmap:
                continue
            if done and done % RECONNECT_EVERY == 0:
                cur.close(); work.close()
                work = _fresh_conn(); cur = work.cursor()
            q = np.asarray(qtoks[qoff[qi]:qoff[qi + 1]])
            qlit = vecarr_literal(q)
            plit = vec_literal(qpooled[qi])
            t0 = time.perf_counter()
            # stage 1: pooled ANN candidates; stage 2: max_sim rerank to k
            cur.execute(
                "with cand as ("
                "  select id, tokens from cb.docs "
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
    """Available RAM in GiB from /proc/meminfo (MemAvailable)."""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 999.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="(re)load data + build index")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conn = psycopg.connect(DSN, autocommit=False)
    conn.execute("set search_path=cb,turbovec,public;")

    qtoks, qoff, qpooled, qids = load_shard("queries")
    dtoks, doff, dpooled, dids = load_shard("docs")
    qmap = json.load(open(os.path.join(OUT, "qrels.json")))

    if args.load:
        setup_schema(conn)
        load_docs(conn)
        load_queries(conn)
        load_qrels(conn)

    results = {"corpus": "scifact", "n_docs": int(len(dids)),
               "n_queries_with_qrels": sum(1 for q in qids if str(q) in qmap),
               "avg_tok_per_doc": float((doff[-1]) / len(dids)),
               "avg_tok_per_query": float((qoff[-1]) / len(qids)),
               "configs": []}

    if args.ceiling:
        print("computing exact ceiling (brute-force max_sim, all docs)...", flush=True)
        t0 = time.time()
        cn, cr = exact_maxsim_ceiling(qtoks, qoff, qids, dtoks, doff, dids, qmap)
        results["exact_ceiling"] = {"ndcg@10": cn, "recall@10": cr,
                                    "compute_s": round(time.time() - t0, 1)}
        print(f"  ceiling nDCG@10={cn:.4f} Recall@10={cr:.4f} "
              f"({results['exact_ceiling']['compute_s']}s)", flush=True)

    if args.sweep:
        # sweep configs (sensible subset)
        configs = []
        for bw in [2, 3, 4]:
            for ptk in [32, 64, 128]:
                for cand in [128, 256, 512]:
                    configs.append((bw, ptk, cand))
        # trim combinatorial explosion: keep a focused grid (~12)
        configs = [
            (2, 64, 256), (3, 64, 256), (4, 64, 256),    # bit erosion @ mid
            (4, 32, 256), (4, 128, 256),                 # per_token_k effect
            (4, 64, 128), (4, 64, 512),                  # candidate_n effect
            (3, 128, 512), (4, 128, 512),                # best-recall corners
            (2, 128, 512),                               # 2-bit best-effort
        ]
        built_bw = {}
        TIMED_N = int(os.environ.get("TIMED_N", "100"))  # subset for warm-latency
        for (bw, ptk, cand) in configs:
            avail = avail_gib()
            print(f"=== config bw={bw} ptk={ptk} cand={cand} (avail={avail:.1f} GiB) ===", flush=True)
            # memory watchdog: floki has ZERO swap; never risk the postmaster.
            if avail < 6.0:
                print(f"  ABORT: available RAM {avail:.1f} GiB < 6 GiB floor; "
                      f"stopping sweep to protect the postmaster.", flush=True)
                results["aborted_low_memory_at"] = {"bit_width": bw, "per_token_k": ptk,
                                                    "candidate_n": cand, "avail_gib": round(avail, 1)}
                break
            if bw not in built_bw:
                built_bw[bw] = build_pooled_index(conn, bw)
            idx_build_s = built_bw[bw]

            # Arm A: full recall pass (all queries) + a timed warm pass on a subset
            an, ar, _ = run_arm_a(conn, qmap, qtoks, qoff, qids, ptk, cand, bw)
            timed_ids = qids[:TIMED_N]
            _, _, alat = run_arm_a(conn, qmap, qtoks, qoff, timed_ids, ptk, cand, bw, time_it=True)

            # Arm B uses the same pooled index (bw) + candidate_n
            bn, br, _ = run_arm_b(conn, qmap, qtoks, qoff, qpooled, qids, cand)
            _, _, blat = run_arm_b(conn, qmap, qtoks, qoff, qpooled, timed_ids, cand, time_it=True)

            entry = {
                "bit_width": bw, "per_token_k": ptk, "candidate_n": cand,
                "pooled_index_build_s": round(idx_build_s, 1),
                "arm_a_colbert_search": {
                    "ndcg@10": round(an, 4), "recall@10": round(ar, 4),
                    "p50_ms": round(pctl(alat, 50), 2), "p95_ms": round(pctl(alat, 95), 2)},
                "arm_b_pooled_rerank": {
                    "ndcg@10": round(bn, 4), "recall@10": round(br, 4),
                    "p50_ms": round(pctl(blat, 50), 2), "p95_ms": round(pctl(blat, 95), 2)},
                "delta_ndcg@10": round(an - bn, 4),
                "delta_recall@10": round(ar - br, 4),
            }
            results["configs"].append(entry)
            print(f"  A: nDCG={an:.4f} R={ar:.4f} p50={pctl(alat,50):.1f}ms | "
                  f"B: nDCG={bn:.4f} R={br:.4f} p50={pctl(blat,50):.1f}ms | "
                  f"dNDCG={an-bn:+.4f} dR={ar-br:+.4f}", flush=True)
            # incremental write so partial results survive an abort
            if args.out:
                json.dump(results, open(args.out, "w"), indent=2)

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
