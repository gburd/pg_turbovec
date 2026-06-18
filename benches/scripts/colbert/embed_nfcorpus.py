"""
Phase F-2 confirmation: embed NFCorpus corpus + test queries with ColBERTv2.

Second corpus for the F-2 gate: NFCorpus (BeIR/nfcorpus) is small (~3,633
docs / 323 test queries), medical/nutrition domain (out-of-domain vs
SciFact's scientific text) and entity-heavier, which is exactly the
character the qualified GO asks to confirm on.

Differs from embed_scifact.py in ONE substantive way: NFCorpus _ids are
STRINGS (docs "MED-2427", queries "PLAIN-2"), not integers. We assign each
doc a synthetic int64 id (0..n_docs) and write an id_map so the loader and
qrels can translate. Queries keep their string id (cb.queries.qid is text);
qrels.json maps query-id-string -> {doc-string-id: rel}, then we rewrite the
doc keys to synthetic ints via the id_map.

Memory-bounded: batched, CPU torch, incremental writes to .npy shards.
Run under: systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0

Outputs to OUTDIR (default /tmp/colbert-nfcorpus):
  {docs,queries}_tokens.npy   concatenated token arrays, float32
  {docs,queries}_offsets.npy  (n+1,) row offsets into the concat
  {docs,queries}_pooled.npy   (n, 128) pooled vectors
  docs_ids.npy                (n_docs,) synthetic int64 corpus ids
  queries_ids.npy             (n_queries,) ... here also synthetic int64,
                              parallel to queries_qids.json (the real strings)
  queries_qids.json           list[str] real query ids (parallel to ids)
  doc_id_map.json             {doc_string_id: synthetic_int_id}
  qrels.json                  {query_string_id: {synthetic_int_doc_id: rel}}
"""
import os, sys, json, gc
import numpy as np
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(__file__))
from colbert_encoder import ColBERT

OUT = os.environ.get("OUTDIR", "/tmp/colbert-nfcorpus")
BATCH = int(os.environ.get("BATCH", "32"))
os.makedirs(OUT, exist_ok=True)


def embed_set(cb, items, prefix, is_query):
    """items: list of (synthetic_int_id, text). Writes concat tokens + offsets
    + pooled + ids. (synthetic ids; the real string ids are tracked separately)"""
    ids, offsets, pooled = [], [0], []
    shards = []
    n = len(items)
    for s in range(0, n, BATCH):
        batch = items[s:s + BATCH]
        texts = [t for _, t in batch]
        if is_query:
            embs = cb.encode_queries(texts)
        else:
            embs = cb.encode_docs(texts)
        for (idv, _), e in zip(batch, embs):
            if e.shape[0] == 0:  # safety: never-empty
                e = np.zeros((1, 128), dtype=np.float32)
            shards.append(e)
            ids.append(int(idv))
            offsets.append(offsets[-1] + e.shape[0])
            pooled.append(cb.pooled(e))
        if (s // BATCH) % 10 == 0:
            print(f"  {prefix}: {min(s+BATCH,n)}/{n}", flush=True)
        gc.collect()
    concat = np.concatenate(shards, axis=0).astype(np.float32)
    np.save(os.path.join(OUT, f"{prefix}_tokens.npy"), concat)
    np.save(os.path.join(OUT, f"{prefix}_offsets.npy"), np.array(offsets, dtype=np.int64))
    np.save(os.path.join(OUT, f"{prefix}_pooled.npy"), np.stack(pooled).astype(np.float32))
    np.save(os.path.join(OUT, f"{prefix}_ids.npy"), np.array(ids, dtype=np.int64))
    avg = (offsets[-1] / n)
    print(f"{prefix}: n={n} total_tokens={offsets[-1]} avg_tok/item={avg:.1f} "
          f"concat={concat.nbytes/1e6:.1f}MB", flush=True)
    del shards, concat
    gc.collect()


def main():
    cb = ColBERT(q_maxlen=32, d_maxlen=180)
    print("model loaded", flush=True)

    corpus = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    queries = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels = load_dataset("BeIR/nfcorpus-qrels")["test"]

    # --- synthetic int doc ids: NFCorpus _ids are strings ("MED-2427") ---
    doc_id_map = {}        # string id -> synthetic int
    docs = []              # (synthetic_int_id, text)
    for i in range(len(corpus)):
        sid = str(corpus[i]["_id"])
        syn = len(doc_id_map)
        doc_id_map[sid] = syn
        text = (corpus[i]["title"] + " " + corpus[i]["text"]).strip()
        docs.append((syn, text))
    json.dump(doc_id_map, open(os.path.join(OUT, "doc_id_map.json"), "w"))

    # --- qrels -> only test queries; rewrite doc keys to synthetic ints ---
    # A handful of qrel corpus-ids may not be in the corpus split; drop those.
    qmap = {}
    dropped = 0
    for r in qrels:
        qid = str(r["query-id"])
        did_str = str(r["corpus-id"])
        if did_str not in doc_id_map:
            dropped += 1
            continue
        qmap.setdefault(qid, {})[str(doc_id_map[did_str])] = int(r["score"])
    test_qids = set(qmap.keys())
    json.dump(qmap, open(os.path.join(OUT, "qrels.json"), "w"))
    print(f"qrels: {len(qmap)} test queries, "
          f"{sum(len(v) for v in qmap.values())} judgments "
          f"({dropped} judgments dropped: corpus-id not in corpus split)", flush=True)

    embed_set(cb, docs, "docs", is_query=False)
    del docs; gc.collect()

    # queries: only the test ones with qrels. Keep real string id <-> synthetic.
    q_strings = []
    qitems = []
    for i in range(len(queries)):
        sid = str(queries[i]["_id"])
        if sid not in test_qids:
            continue
        syn = len(q_strings)
        q_strings.append(sid)
        qitems.append((syn, queries[i]["text"]))
    json.dump(q_strings, open(os.path.join(OUT, "queries_qids.json"), "w"))
    embed_set(cb, qitems, "queries", is_query=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
