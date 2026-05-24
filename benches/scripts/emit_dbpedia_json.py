#!/usr/bin/env python3
"""Emit Phase J JSON from results.tsv + storage.tsv + build_times.txt."""
import json, os, subprocess, sys
from datetime import date

OUT = "/scratch/pg_turbovec-bench/dbpedia_sweep"

def parse_results():
    rows = []
    for line in open(os.path.join(OUT, "results.tsv")):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 8: continue
        label, n, mn, p50, p95, mx, mean, r10 = parts
        rows.append({
            "label": label, "n": int(n),
            "min_ms": round(float(mn), 2),
            "p50_ms": round(float(p50), 2),
            "p95_ms": round(float(p95), 2),
            "max_ms": round(float(mx), 2),
            "mean_ms": round(float(mean), 2),
            "r_at_10": round(float(r10), 4),
        })
    return rows

def parse_storage():
    sizes = {}
    for line in open(os.path.join(OUT, "storage.tsv")):
        line = line.rstrip("\n")
        if not line: continue
        parts = line.split("\t")
        if len(parts) == 2:
            sizes[parts[0]] = int(parts[1])
    return sizes

def parse_builds():
    bt = {}
    p = os.path.join(OUT, "build_times.txt")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                bt[k] = int(v)
    return bt

def query_meta():
    import psycopg
    DSN = "host=/scratch/pg_turbovec-bench port=28815 user=gburd dbname=bench_dbpedia"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SHOW server_version"); pg_v = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'"); pgv = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'"); tv = cur.fetchone()[0]
        cur.execute("""
            SELECT bit_width, n_vectors, dim
            FROM turbovec.am_storage s JOIN pg_class c ON c.oid = s.indexrelid
            ORDER BY c.relname
        """)
        tv_meta = cur.fetchall()
    return pg_v, pgv, tv, tv_meta

def main():
    rows = parse_results()
    sizes = parse_storage()
    bt = parse_builds()
    pg_v, pgv_v, tv_v, tv_meta = query_meta()
    head = subprocess.check_output(
        ["git", "-C", "/scratch/pg_turbovec-bench/pg_turbovec", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

    indexes = [{
        "name": "docs_pgv_hnsw",
        "type": "pgvector hnsw",
        "size_bytes": sizes.get("docs_pgv_hnsw"),
        "size_mb": round(sizes.get("docs_pgv_hnsw", 0) / 1e6, 1),
        "build_time_s": bt.get("pgv_hnsw_build_s"),
        "params": {"m": 16, "ef_construction": 64},
    }]
    # tv_meta lists by relname order (4bit then 2bit alpha-sort: 2bit < 4bit, so 2bit first)
    for (bw, n_vec, dim) in tv_meta:
        relname = f"docs_tv_{bw}bit"
        indexes.append({
            "name": relname,
            "type": f"pg_turbovec {bw}-bit",
            "payload_size_bytes": sizes.get(relname),
            "payload_size_mb": round(sizes.get(relname, 0) / 1e6, 1),
            "bit_width": bw, "n_vectors": n_vec, "dim": dim,
            "build_time_s": bt.get(f"tv_{bw}bit_build_s"),
        })

    # canonical order
    label_order = ["hnsw_ef40", "hnsw_ef200",
                   "tv_4bit_k100", "tv_4bit_k500",
                   "tv_2bit_k100", "tv_2bit_k500"]
    rows.sort(key=lambda r: label_order.index(r["label"]) if r["label"] in label_order else 99)

    by = {r["label"]: r for r in rows}
    pgv_idx = next(i for i in indexes if i["name"] == "docs_pgv_hnsw")
    tv4_idx = next(i for i in indexes if i["name"] == "docs_tv_4bit")
    tv2_idx = next(i for i in indexes if i["name"] == "docs_tv_2bit")

    summary_lines = []
    if "hnsw_ef200" in by and "tv_4bit_k500" in by:
        summary_lines.append(
            f"On 1M x 1536-d OpenAI ada-002 embeddings, pg_turbovec 4-bit (search_k=500) "
            f"reaches R@10={by['tv_4bit_k500']['r_at_10']:.3f} at "
            f"{tv4_idx['payload_size_mb']:.0f} MB / p50 {by['tv_4bit_k500']['p50_ms']:.1f} ms; "
            f"pgvector HNSW (ef_search=200) gets R@10={by['hnsw_ef200']['r_at_10']:.3f} at "
            f"{pgv_idx['size_mb']:.0f} MB / p50 {by['hnsw_ef200']['p50_ms']:.1f} ms."
        )
    if "tv_2bit_k500" in by:
        summary_lines.append(
            f"2-bit comes in at {tv2_idx['payload_size_mb']:.0f} MB / "
            f"p50 {by['tv_2bit_k500']['p50_ms']:.1f} ms / R@10={by['tv_2bit_k500']['r_at_10']:.3f}."
        )

    out = {
        "head": head,
        "build_profile": "release",
        "host": "arnold (Intel i9-12900H, 32 GiB)",
        "pg_version": pg_v,
        "pgvector_version": pgv_v,
        "pg_turbovec_version": tv_v,
        "date": str(date.today()),
        "n_queries": 50,
        "corpus": {
            "name": "dbpedia-entities-openai-1M",
            "source": "huggingface://KShivendu/dbpedia-entities-openai-1M",
            "n": 1_000_000,
            "dim": 1536,
            "embedding_model": "text-embedding-ada-002 (OpenAI)",
            "distance": "cosine",
            "normalized": True,
        },
        "indexes": indexes,
        "configs": rows,
        "summary": " ".join(summary_lines),
        "method_notes": (
            "Per-query Execution Time captured via plpgsql clock_timestamp() "
            "around `ORDER BY emb <=> q LIMIT 10`. Two warmup queries per "
            "config before the timed pass. Query set = first 50 docs in the "
            "corpus, so top-1 is trivially the query itself; R@10 is dominated "
            "by ranks 2..10. Brute-force cosine ground truth from a parallel "
            "seqscan with indexscan/bitmapscan disabled. Indexes were "
            "renamed in/out of the way (no rebuild) to force the planner to "
            "pick a single AM per phase."
        ),
    }
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
