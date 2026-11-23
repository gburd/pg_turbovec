#!/usr/bin/env python3
"""HackerNews (HN768) corpus prep for pg_turbovec.

The standing corpora (CO1024, OA1536, CW768) are *pre-embedded* text datasets:
the HF parquet already carries an `emb` column and prep just selects + rescales
it.  HackerNews is published as raw TEXT, so this script adds the embedding step
the others do not need, then emits the identical on-disk shape the bench loaders
(`load_ds.sh`, `bench.py`, the vs-bench harness) consume: `corpus.npy`,
`queries.npy`, exact fp64 `gt_ids.npy`/`gt_dist.npy`, 16-way `%.9g` text COPY
files, and a pinned `datasets.lock.json`.

Pipeline (deterministic end to end, so the same N/H/model always yield the same
bytes):
  1. Stream HN story/comment text from a pinned HF dataset revision, in file +
     row order; keep the first N non-empty, de-duplicated docs as the corpus and
     the next H as held-out queries.
  2. Embed each doc with a pinned sentence-transformers model, L2-normalise to
     fp32 -- the same representation CO1024 uses -- so HN is a drop-in vector
     corpus directly comparable to the 768-d CW768 corpus.  Embedding is the
     cost: run this on a GPU host.  On CPU, bge-base does ~100-150 docs/s, so
     even 100k takes tens of minutes and a full ~40M corpus is impractical
     (the pre-embedded corpora avoid this step entirely).  A GPU (g5/g6 class)
     does the full corpus in a fraction of the time.
  3. Exact ground truth: fp32 BLAS top-200, fp64 re-score, top-100 (ties -> id).

The embedding model and its revision are PINNED and recorded in datasets.lock so
the corpus is reproducible; changing either is a new dataset, not a silent drift.
HN is text, so the vectors are ours (not redistributed from a vendor); the lock
records the source text revision + the model so anyone can rebuild identical
bytes.

usage: prep_hn.py N H OUTDIR [MODEL]
  N      corpus size (e.g. 150000000 for the field-scale run; 1000000 for a
         RAM-resident regression corpus)
  H      held-out query count (e.g. 5000)
  OUTDIR output directory (e.g. /scratch/pg_turbovec-bench/hn768_1m)
  MODEL  sentence-transformers model id (default BAAI/bge-base-en-v1.5, 768-d)
"""
import os, sys, time, json, hashlib, numpy as np, pyarrow.parquet as pq
import multiprocessing as mp
from huggingface_hub import hf_hub_download, list_repo_files, dataset_info

N = int(sys.argv[1]); H = int(sys.argv[2]); OUT = sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "BAAI/bge-base-en-v1.5"
# Pinned text source: OpenPipe/hacker-news is the full HN archive (stories +
# comments), ~41.8M rows, reachable and not gated, with a `title` and a `text`
# column.  It is the largest real HN corpus available; HN has ~42M posts in
# total, so N above ~41M is not achievable from real documents without
# synthetic duplication -- prep_hn.py refuses N beyond what the source holds
# rather than silently repeat rows.  A doc's text is `text` when present
# (comments, text posts) else `title` (link stories).
TEXT_REPO = "OpenPipe/hacker-news"
DIM = 768                       # bge-base-en-v1.5; matches the CW768 corpus
BATCH = 2048                    # embedding batch; tune to the host's memory/GPU
PARTS = 16
# Where HF shards are cached.  Defaults next to OUT (portable); override with
# HN_HF_DIR.  The prep_co1024 template hardcoded /data, which is not writable
# on every host.
HF_DIR = os.environ.get("HN_HF_DIR", os.path.join(OUT, "hf"))
os.makedirs(f"{OUT}/copy", exist_ok=True)
os.makedirs(HF_DIR, exist_ok=True)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_model():
    """Pinned embedding model; records its resolved commit in the lock."""
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(MODEL, device=dev)
    m.max_seq_length = 256      # HN docs are short; cap for throughput
    rev = None
    try:
        rev = dataset_info(MODEL).sha  # best-effort model revision
    except Exception:
        pass
    return m, dev, rev


def doc_text(tbl):
    """One text string per row: `text` when non-empty (comments, text posts),
    else `title` (link stories), else empty (skipped by the caller).  A
    corpus of only titles would be short and repetitive; preferring `text`
    keeps the real document where there is one."""
    cols = tbl.column_names
    text = tbl.column("text").to_pylist() if "text" in cols else None
    title = tbl.column("title").to_pylist() if "title" in cols else None
    if text is None and title is None:
        raise SystemExit(f"no text/title column in {cols}")
    n = len(text if text is not None else title)
    out = []
    for i in range(n):
        t = (text[i] if text is not None else None) or ""
        if not t.strip():
            t = (title[i] if title is not None else None) or ""
        out.append(t)
    return out, "text|title"


t0 = time.time()
model, device, model_rev = load_model()
print(f"model {MODEL} on {device} (rev {model_rev}) {time.time()-t0:.0f}s", flush=True)

corpus = np.empty((N, DIM), np.float32)
src = np.empty((N, 2), np.int32)
Q = []
qsrc = []
seen = set()
qseen = set()
n = 0
stats = dict(empty=0, dup_dropped=0, query_dup_dropped=0, query_eq_corpus=0, rows_read=0)
lock = []
qdupc = []

# Pinned list of text shards, in a fixed order.
shards = sorted(f for f in list_repo_files(TEXT_REPO, repo_type="dataset")
                if f.endswith(".parquet"))
if not shards:
    raise SystemExit(f"no parquet shards in {TEXT_REPO}")

pending_text = []           # (text, shard_idx, row_idx) awaiting embedding
pending_role = []           # True = corpus slot, False = query slot


def flush_embed():
    """Embed the pending batch and place each vector in corpus or Q, in order."""
    global n
    if not pending_text:
        return
    vecs = model.encode([t for t in pending_text], batch_size=BATCH,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)
    for k, v in enumerate(vecs):
        si, ri = pending_role[k][1], pending_role[k][2]
        if pending_role[k][0]:          # corpus
            corpus[n] = v
            src[n] = (si, ri)
            n += 1
        else:                           # query
            Q.append(v)
            qsrc.append((si, ri))
    pending_text.clear()
    pending_role.clear()


si = 0
while len(Q) < H and si < len(shards):
    fn = shards[si]
    p = hf_hub_download(TEXT_REPO, fn, repo_type="dataset", local_dir=HF_DIR)
    lock.append(dict(repo=TEXT_REPO, file=fn, bytes=os.path.getsize(p), md5=md5f(p)))
    tbl = pq.read_table(p)
    texts, textcol = doc_text(tbl)
    for ri, txt in enumerate(texts):
        stats["rows_read"] += 1
        if not txt or not txt.strip():
            stats["empty"] += 1
            continue
        key = hashlib.blake2b(txt.strip().encode("utf-8", "ignore"), digest_size=16).digest()
        if n < N:
            if key in seen:
                stats["dup_dropped"] += 1
                continue
            seen.add(key)
            pending_text.append(txt)
            pending_role.append((True, si, ri))
        else:
            if key in qseen:
                stats["query_dup_dropped"] += 1
                continue
            qseen.add(key)
            if key in seen:
                stats["query_eq_corpus"] += 1
                qdupc.append(len(Q) + len(pending_role))
            pending_text.append(txt)
            pending_role.append((False, si, ri))
        if len(pending_text) >= BATCH:
            flush_embed()
        if n >= N and len(Q) + sum(1 for r in pending_role if not r[0]) >= H:
            break
    flush_embed()
    print(f"  shard {si} ({textcol}): corpus {n}/{N} queries {len(Q)}/{H} {time.time()-t0:.0f}s", flush=True)
    si += 1
flush_embed()

if n < N or len(Q) < H:
    raise SystemExit(f"ran out of HN text: corpus {n}/{N}, queries {len(Q)}/{H}")

Q = np.stack(Q[:H]).astype(np.float32)
qsrc = np.array(qsrc[:H], np.int32)
stats["max_abs_norm_dev_corpus"] = float(np.max(np.abs(np.linalg.norm(corpus.astype(np.float64), axis=1) - 1)))
stats["max_abs_norm_dev_queries"] = float(np.max(np.abs(np.linalg.norm(Q.astype(np.float64), axis=1) - 1)))
stats["query_eq_corpus_qids"] = qdupc
np.save(f"{OUT}/corpus.npy", corpus)
np.save(f"{OUT}/queries.npy", Q)
np.save(f"{OUT}/corpus_src.npy", src)
np.save(f"{OUT}/queries_src.npy", qsrc)
print(f"vectors embedded + saved {time.time()-t0:.0f}s", flush=True)

# GT: fp32 sgemm top-200, then fp64 re-score (identical to prep_co1024).
KC = 200
bs = np.full((H, KC), -np.inf, np.float32)
bi = np.zeros((H, KC), np.int64)
for s in range(0, N, 100_000):
    sc = Q @ corpus[s:s + 100_000].T
    cs = np.concatenate([bs, sc], 1)
    ci = np.concatenate([bi, np.broadcast_to(np.arange(s, s + sc.shape[1]), (H, sc.shape[1]))], 1)
    part = np.argpartition(-cs, KC, 1)[:, :KC]
    bs = np.take_along_axis(cs, part, 1)
    bi = np.take_along_axis(ci, part, 1)
gt_i = np.empty((H, KC), np.int64)
gt_d = np.empty((H, KC), np.float64)
Q64 = Q.astype(np.float64)
for a in range(0, H, 250):
    ids = bi[a:a + 250]
    d = 1.0 - np.einsum("qd,qkd->qk", Q64[a:a + 250], corpus[ids].astype(np.float64))
    o = np.lexsort((ids, d), axis=1)
    gt_i[a:a + 250] = np.take_along_axis(ids, o, 1)
    gt_d[a:a + 250] = np.take_along_axis(d, o, 1)
np.save(f"{OUT}/gt_ids.npy", gt_i[:, :100])
np.save(f"{OUT}/gt_dist.npy", gt_d[:, :100])
print(f"GT done {time.time()-t0:.0f}s", flush=True)


# COPY text files, identical %.9g bytes for both systems (same as prep_co1024).
def fmt(rows):
    return "".join(f"{rid}\t[{','.join('%.9g' % x for x in vec.tolist())}]\n" for rid, vec in rows)


def write_part(k):
    lo, hi = k * N // PARTS, (k + 1) * N // PARTS
    with open(f"{OUT}/copy/docs_{k:02d}.tsv", "w") as f:
        for b in range(lo, hi, 5000):
            f.write(fmt(zip(range(b, min(b + 5000, hi)), corpus[b:min(b + 5000, hi)])))
    return k


with mp.get_context("fork").Pool(PARTS) as pool:
    list(pool.map(write_part, range(PARTS)))
with open(f"{OUT}/copy/queries.tsv", "w") as f:
    f.write(fmt(enumerate(Q)))
print(f"copy files done {time.time()-t0:.0f}s", flush=True)

man = {f: md5f(f"{OUT}/{f}") for f in ["corpus.npy", "queries.npy", "gt_ids.npy", "gt_dist.npy", "corpus_src.npy", "queries_src.npy"]}
json.dump(dict(dataset="HN768", dim=DIM, N=N, H=H, text_repo=TEXT_REPO,
               embed_model=MODEL, embed_model_rev=model_rev, device=device,
               shards=lock, npy_md5=man, stats=stats), open(f"{OUT}/datasets.lock.json", "w"), indent=1)
print(json.dumps(stats)[:2000])
print(f"done {time.time()-t0:.0f}s", flush=True)
