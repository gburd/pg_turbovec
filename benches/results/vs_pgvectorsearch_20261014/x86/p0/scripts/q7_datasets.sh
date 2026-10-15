#!/usr/bin/env bash
# Q7: resolve dataset URLs, pin md5 of what we can fetch now, count test queries, check exact dups at 1M (cohere_medium_1m).
set -x
R=~/results/p0/q7; mkdir -p $R /data/vdb; cd /data/vdb
B=https://assets.zilliz.com/benchmark
for u in http://ann-benchmarks.com/cohere-wikipedia-22-12-1M-angular.hdf5 https://ann-benchmarks.com/cohere-wikipedia-22-12-1M-angular.hdf5 \
         http://ann-benchmarks.com/glove-100-angular.hdf5 https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings \
         https://huggingface.co/api/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3; do
  echo "$u $(curl -sL -o /dev/null --max-time 30 -w '%{http_code}' $u)"; done > $R/url_status.txt
for d in cohere_medium_1m cohere_large_10m openai_large_5m; do
  mkdir -p $d
  for f in test.parquet neighbors.parquet; do curl -s -C - -o $d/$f $B/$d/$f; done
  for i in $(seq 0 9); do f=train-0$i-of-10.parquet; echo "$d/$f $(curl -sI $B/$d/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{l=$2} tolower($1)=="etag:"{e=$2} END{print l, e}')"; done >> $R/vdb_train_heads.txt
  echo "$d/train.parquet $(curl -sI $B/$d/train.parquet | tr -d '\r' | awk 'tolower($1)=="content-length:"{l=$2} tolower($1)=="etag:"{e=$2} END{print l, e}')" >> $R/vdb_train_heads.txt
done
curl -s -o cohere_medium_1m/train.parquet $B/cohere_medium_1m/train.parquet
md5sum */*.parquet > $R/vdb_md5.txt
python3 - <<'PY' > $R/vdb_counts.txt 2>&1
import pyarrow.parquet as pq, numpy as np, hashlib, glob
for f in sorted(glob.glob("/data/vdb/*/*.parquet")):
    m = pq.ParquetFile(f).metadata; print(f, "rows", m.num_rows, "schema", pq.ParquetFile(f).schema_arrow)
def load(f):
    t = pq.read_table(f); col = [c for c in t.column_names if c in ("emb","vector","emb_vector")][0]
    ids = t.column("id").to_numpy(); a = t.column(col).combine_chunks().flatten().to_numpy().astype(np.float32)
    return ids, a.reshape(len(ids), -1)
qi, Q = load("/data/vdb/cohere_medium_1m/test.parquet"); ci, C = load("/data/vdb/cohere_medium_1m/train.parquet")
print("cohere_medium_1m: test", Q.shape, "train", C.shape)
nq = np.linalg.norm(Q, axis=1); nc = np.linalg.norm(C, axis=1)
print("norms test min/max", nq.min(), nq.max(), "train min/max", nc.min(), nc.max())
h = lambda v: hashlib.blake2b(v.tobytes(), digest_size=16).digest()
Qn = (Q / nq[:, None]).astype(np.float32); Cn = (C / nc[:, None]).astype(np.float32)
cs = {}; dupc = 0
for i, v in enumerate(Cn):
    k = h(v)
    if k in cs: dupc += 1
    else: cs[k] = i
qd = sum(1 for v in Qn if h(v) in cs)
print("corpus exact-dup rows (after normalise):", dupc, " test queries exactly equal to a corpus row:", qd)
best = np.max(Qn @ Cn.T, axis=1)
print("test queries with max cos >= 1-1e-6:", int((best >= 1 - 1e-6).sum()), " >= 0.9999:", int((best >= 0.9999).sum()))
PY
cat $R/*.txt
echo Q7_DONE
