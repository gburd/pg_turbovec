"""
Phase F-1 gate: embed SciFact corpus + test queries with ColBERTv2.

Memory-bounded: batched, CPU torch, incremental writes to .npy shards.
Run under: systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0

Outputs to OUTDIR (default /tmp/colbert-scifact):
  docs.jsonl         one {id, ntok, file, row} per doc (manifest)
  docs_tokens/NNN.npy   token shards (concatenated tokens, float32)
  docs_offsets.npy   (n_docs+1,) row offsets into the concat for slicing
  docs_pooled.npy    (n_docs, 128) pooled vectors
  doc_ids.npy        (n_docs,) int64 corpus ids
  queries_*.{npy}    same shape for the 300 test queries
  qrels.json         {qid: {doc_id: rel}}
"""
import os, sys, json, gc
import numpy as np
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(__file__))
from colbert_encoder import ColBERT

OUT = os.environ.get("OUTDIR", "/tmp/colbert-scifact")
BATCH = int(os.environ.get("BATCH", "32"))
os.makedirs(OUT, exist_ok=True)


def embed_set(cb, items, prefix, is_query):
    """items: list of (id_str, text). Writes concat tokens + offsets + pooled + ids."""
    ids, offsets, pooled = [], [0], []
    concat_path = os.path.join(OUT, f"{prefix}_tokens.npy")
    # accumulate token arrays in a list of shards, concat at end (bounded:
    # SciFact docs <=180 tok * 128 * 4B = ~92KB/doc; 5183 docs ~ 480MB total)
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
    np.save(concat_path, concat)
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

    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels = load_dataset("BeIR/scifact-qrels")["test"]

    # qrels -> only test queries
    qmap = {}
    for r in qrels:
        qid = str(r["query-id"])
        qmap.setdefault(qid, {})[str(r["corpus-id"])] = int(r["score"])
    test_qids = set(qmap.keys())
    json.dump(qmap, open(os.path.join(OUT, "qrels.json"), "w"))
    print(f"qrels: {len(qmap)} test queries, "
          f"{sum(len(v) for v in qmap.values())} judgments", flush=True)

    # docs: title + text (BEIR convention)
    docs = [(str(corpus[i]["_id"]),
             (corpus[i]["title"] + " " + corpus[i]["text"]).strip())
            for i in range(len(corpus))]
    embed_set(cb, docs, "docs", is_query=False)
    del docs; gc.collect()

    # queries: only the test ones with qrels
    qitems = [(str(queries[i]["_id"]), queries[i]["text"])
              for i in range(len(queries)) if str(queries[i]["_id"]) in test_qids]
    embed_set(cb, qitems, "queries", is_query=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
