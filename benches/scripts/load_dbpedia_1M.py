#!/usr/bin/env python3
"""Load dbpedia-entities-openai-1M into Postgres `bench_dbpedia.docs`.

- Reads parquet shards from /scratch/pg_turbovec-bench/dbpedia/
- COPY-streams (id, ext_id, emb) tuples in vector(1536) text format.
- Single sequential pass; no temp CSV.
"""
import glob, io, os, sys, time
import pyarrow.parquet as pq
import psycopg

DBPEDIA = "/scratch/pg_turbovec-bench/dbpedia"
DSN = os.environ.get(
    "DBPEDIA_DSN",
    "host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_dbpedia",
)
BATCH = 2000  # rows per parquet read batch

shards = sorted(glob.glob(os.path.join(DBPEDIA, "train-*.parquet")))
print(f"{len(shards)} shards", flush=True)

def fmt_vec(arr) -> str:
    # arr is a list/array of float32 length 1536 - format as pgvector text.
    # str(float) is ~7 sig figs; that's fine for ada-002 normalised vectors.
    return "[" + ",".join(f"{v:.7g}" for v in arr) + "]"

def rows():
    rid = 0
    for shard in shards:
        t0 = time.time()
        pf = pq.ParquetFile(shard)
        nr = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=["_id", "openai"]):
            ids = batch.column("_id").to_pylist()
            embs = batch.column("openai").to_pylist()
            for ext_id, emb in zip(ids, embs):
                rid += 1
                yield rid, ext_id, fmt_vec(emb)
            nr += batch.num_rows
        print(f"  {os.path.basename(shard)}: {nr} rows in {time.time()-t0:.1f}s "
              f"(total {rid})", flush=True)

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = off")
        cur.execute("SET maintenance_work_mem = '4GB'")
        cur.execute("TRUNCATE docs")
        t0 = time.time()
        with cur.copy("COPY docs (id, ext_id, emb) FROM STDIN") as cp:
            for rid, ext_id, emb_str in rows():
                # tab-separated; ext_id is safe (uri-style strings, no \t/\n)
                cp.write_row((rid, ext_id, emb_str))
        conn.commit()
        cur.execute("SELECT count(*), pg_size_pretty(pg_relation_size('docs')) FROM docs")
        n, sz = cur.fetchone()
        print(f"loaded {n} rows in {time.time()-t0:.1f}s, heap = {sz}", flush=True)
