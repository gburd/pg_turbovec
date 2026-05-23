#!/usr/bin/env python3
"""End-to-end recall + latency comparison for pg_turbovec vs pgvector.

Drives a single Postgres cluster that has BOTH `pg_turbovec` and
`vector` (pgvector) extensions loaded. Builds three indexes on the
same corpus table:

  1. pgvector HNSW (`vector_cosine_ops`) at default m / ef_construction
  2. pg_turbovec at `bit_width = 4`
  3. pg_turbovec at `bit_width = 2`

then runs the same query workload against each and records
recall@k and per-query latency (p50/p95/p99 at LIMIT=K) versus an
exact brute-force ground truth (loaded from `ground_truth.bin`).

Usage:

    nix-shell -p python3Packages.numpy python3Packages.psycopg2 --run "
        python3 benches/scripts/run_recall_vs_pgvector.py \\
            fixtures/glove-100 \\
            benches/results/recall_vs_pgvector_<DATE>.json \\
            --pg-bin /home/gburd/.pgrx/install-pg16/bin \\
            --pg-data /home/gburd/.pgrx/data-16 \\
            --port 28816 --socket-dir /tmp/pgturbosock"

The script DOES NOT run `initdb` or destroy data — it expects a
working PG cluster (the pgrx-managed one is fine; we only CREATE
EXTENSION and a TEMP-like schema). The cluster must already have
`pg_turbovec.so` and pgvector's `vector.so` installed.

Output: a JSON file with the schema documented in
`docs/RECALL.md` and a Markdown summary on stdout.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psycopg2
import psycopg2.extras


# --------------------------------------------------------------------- IO
def read_f32_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        dim, n = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(dim * n * 4), dtype="<f4").reshape(n, dim)
    return data


def read_u32_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        k, n = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(k * n * 4), dtype="<u4").reshape(n, k)
    return data


# --------------------------------------------------------------------- PG lifecycle
def ensure_cluster(pg_bin: str, pg_data: str, port: int, sock: str) -> None:
    """Start the cluster if it isn't already up. Always restart with
    `TMPDIR=/tmp` so pg_turbovec's persist layer (which writes
    serialised IdMapIndex files via `std::env::temp_dir()`) doesn't
    inherit a now-defunct `nix-shell` TMPDIR."""
    pgctl = os.path.join(pg_bin, "pg_ctl")
    # Make sure the socket dir exists.
    os.makedirs(sock, exist_ok=True)
    # A clean environment for the postmaster: keep PATH, USER, HOME,
    # but force TMPDIR=/tmp because postgres backends inherit it and
    # pg_turbovec uses std::env::temp_dir() for staging serialised
    # index payloads.
    pg_env = {
        k: v for k, v in os.environ.items()
        if k in ("PATH", "USER", "HOME", "LANG", "LC_ALL", "PG_CONFIG")
    }
    pg_env["TMPDIR"] = "/tmp"
    pg_env["TMP"] = "/tmp"
    pg_env["TEMP"] = "/tmp"
    pg_env["TEMPDIR"] = "/tmp"

    # Always restart so the env is clean.
    pid_file = os.path.join(pg_data, "postmaster.pid")
    if os.path.exists(pid_file):
        subprocess.run(
            [pgctl, "-D", pg_data, "stop", "-m", "fast"],
            timeout=15, check=False,
            capture_output=True,
        )
        # Give the OS a moment to release the socket.
        time.sleep(0.5)

    log = "/tmp/pgturbovec_recall.log"
    r = subprocess.run(
        [
            pgctl, "-D", pg_data, "-l", log,
            "-o", f"-p {port} -k {sock}",
            "start",
        ],
        timeout=30, capture_output=True, text=True,
        env=pg_env,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
    # Poll for readiness via libpq (psycopg2 against the unix socket).
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(
                host=sock, port=port, dbname="postgres",
                user=os.environ.get("USER", "postgres"),
            )
            conn.close()
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"could not connect to PG on port {port} via {sock}")


def connect(sock: str, port: int, dbname: str = "postgres"):
    return psycopg2.connect(
        host=sock, port=port, dbname=dbname,
        user=os.environ.get("USER", "postgres"),
    )


# --------------------------------------------------------------------- queries
def vec_literal(v: np.ndarray, pad_to: int = 0) -> str:
    body = ",".join(f"{x:.7g}" for x in v)
    if pad_to > len(v):
        body += ",0" * (pad_to - len(v))
    return "[" + body + "]"


def percentile(samples: List[float], p: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(samples, p * 100))


def recall_at_k(brute: np.ndarray, indexed: List[int], k: int) -> float:
    if not indexed or len(brute) == 0:
        return 0.0
    take = min(k, len(brute), len(indexed))
    bset = set(int(x) for x in brute[:take])
    hits = sum(1 for x in indexed[:take] if int(x) in bset)
    return hits / take


# --------------------------------------------------------------------- runners
def setup_corpus(cur, schema: str, dim: int, train: np.ndarray) -> int:
    """(Re)create the corpus table with both vector types: a `vector(dim)`
    column for pgvector and a `turbovec.vector` column for pg_turbovec.
    Both columns store the same data (modulo zero-padding required
    for pg_turbovec — see below).

    pg_turbovec's index AM requires `dim % 8 == 0`. The fixture
    dim is whatever the source dataset uses (e.g. 100 for GloVe-100).
    We zero-pad the pg_turbovec column to the next multiple of 8;
    on unit-norm input this is exactly identity-preserving for
    cosine similarity.

    Returns the *padded* dim (= what `emb_tv` declares).
    """
    padded_dim = ((dim + 7) // 8) * 8
    cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(
        f"""
        CREATE TABLE {schema}.corpus (
            id    bigint PRIMARY KEY,
            emb_pgv public.vector({dim}),
            emb_tv  turbovec.vector
        )
        """
    )
    n = train.shape[0]
    print(f"  inserting {n} rows (dim={dim}, padded for pg_turbovec={padded_dim}) ...",
          flush=True)
    t0 = time.time()
    import io
    buf = io.StringIO()
    pad = ",0" * (padded_dim - dim)
    for i in range(n):
        v = train[i]
        body = ",".join(f"{x:.7g}" for x in v)
        lit_pgv = "[" + body + "]"
        lit_tv = "[" + body + pad + "]"
        buf.write(f"{i}\t{lit_pgv}\t{lit_tv}\n")
    buf.seek(0)
    cur.copy_expert(
        f"COPY {schema}.corpus(id, emb_pgv, emb_tv) FROM STDIN", buf
    )
    print(f"  insert took {time.time() - t0:.2f}s", flush=True)
    return padded_dim


def measure_index(
    conn, schema: str, label: str,
    create_sql: str, drop_sql: str, query_template: str,
    queries: np.ndarray, ground_truth: np.ndarray,
    limit_k: int, gt_k: int,
    pad_to: int = 0,
) -> dict:
    """Build `create_sql`, run `query_template % vec_literal` for every
    row in `queries`, compute recall@1/10/100 and p50/p95/p99 latency."""
    cur = conn.cursor()
    print(f"\n--- {label} ---", flush=True)
    # Drop any prior copy.
    cur.execute(drop_sql)
    conn.commit()

    print(f"  CREATE INDEX: {create_sql}", flush=True)
    t0 = time.time()
    cur.execute(create_sql)
    conn.commit()
    build_secs = time.time() - t0
    print(f"  built in {build_secs:.2f}s", flush=True)

    # Index size on disk (pg_relation_size).
    cur.execute(f"""
        SELECT pg_relation_size(c.oid)
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = 'idx'
    """, (schema,))
    index_bytes = cur.fetchone()[0]
    print(f"  pg_relation_size = {index_bytes} bytes", flush=True)

    # Force planner to use the index by disabling sequential scans.
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET max_parallel_workers_per_gather = 0")

    # Warm.
    for i in range(min(20, queries.shape[0])):
        q = queries[i]
        cur.execute(query_template, (vec_literal(q, pad_to=pad_to), 100))
        cur.fetchall()

    n_queries = queries.shape[0]
    lats = []
    sum_r1 = sum_r10 = sum_r100 = 0.0
    k_max = min(100, gt_k)
    for i in range(n_queries):
        q = queries[i]
        ql = vec_literal(q, pad_to=pad_to)
        # Recall pass: ask for k_max so we can score @1, @10, @100.
        cur.execute(query_template, (ql, k_max))
        ids_r = [int(r[0]) for r in cur.fetchall()]
        gt_row = ground_truth[i]
        sum_r1 += recall_at_k(gt_row, ids_r, 1)
        sum_r10 += recall_at_k(gt_row, ids_r, 10)
        sum_r100 += recall_at_k(gt_row, ids_r, k_max)

        # Latency pass: ask for `limit_k` (the fast path).
        t0 = time.time()
        cur.execute(query_template, (ql, limit_k))
        cur.fetchall()
        lats.append((time.time() - t0) * 1e6)  # μs

    nf = float(n_queries)
    out = {
        "label": label,
        "create_sql": create_sql,
        "build_secs": build_secs,
        "index_bytes": int(index_bytes),
        "n_queries": int(n_queries),
        "limit_k": limit_k,
        "r_at_1": sum_r1 / nf,
        "r_at_10": sum_r10 / nf,
        "r_at_100": sum_r100 / nf,
        "p50_us": percentile(lats, 0.50),
        "p95_us": percentile(lats, 0.95),
        "p99_us": percentile(lats, 0.99),
        "mean_us": float(np.mean(lats)),
    }
    print(f"  R@1={out['r_at_1']:.3f}  R@10={out['r_at_10']:.3f}  "
          f"R@100={out['r_at_100']:.3f}", flush=True)
    print(f"  p50={out['p50_us']:.0f}us  p95={out['p95_us']:.0f}us  "
          f"p99={out['p99_us']:.0f}us", flush=True)
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture_dir", type=Path,
                   help="dir containing corpus.bin, queries.bin, ground_truth.bin")
    p.add_argument("results_path", type=Path,
                   help="output JSON file")
    p.add_argument("--pg-bin", default="/home/gburd/.pgrx/install-pg16/bin")
    p.add_argument("--pg-data", default="/home/gburd/.pgrx/data-16")
    p.add_argument("--port", type=int, default=28816)
    p.add_argument("--socket-dir", default="/home/gburd/.pgrx")
    p.add_argument("--limit-k", type=int, default=10)
    p.add_argument("--schema", default="bench_recall")
    p.add_argument("--ef-search", type=str, default="40",
                   help="pgvector HNSW ef_search; comma-separated list to sweep "
                        "(default 40)")
    p.add_argument("--hnsw-m", type=int, default=16)
    p.add_argument("--hnsw-efc", type=int, default=64)
    args = p.parse_args()

    print(f"loading fixture from {args.fixture_dir}")
    train = read_f32_matrix(args.fixture_dir / "corpus.bin")
    test = read_f32_matrix(args.fixture_dir / "queries.bin")
    gt = read_u32_matrix(args.fixture_dir / "ground_truth.bin")
    n_corpus, dim = train.shape
    n_queries, _ = test.shape
    n_gt, gt_k = gt.shape
    assert n_gt == n_queries
    print(f"  corpus = {n_corpus} x {dim}, queries = {n_queries}, gt_k = {gt_k}")

    print("ensuring PG cluster is running")
    ensure_cluster(args.pg_bin, args.pg_data, args.port, args.socket_dir)

    conn = connect(args.socket_dir, args.port)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_turbovec")
    conn.commit()
    cur.execute("SHOW server_version")
    pg_version = cur.fetchone()[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
    pgv_version = cur.fetchone()[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'")
    pgtv_version = cur.fetchone()[0]
    print(f"  PostgreSQL {pg_version}, pgvector {pgv_version}, "
          f"pg_turbovec {pgtv_version}")

    setup_corpus_padded_dim = setup_corpus(cur, args.schema, dim, train)
    conn.commit()

    # Set the search path so unqualified identifiers in our query
    # templates resolve to both extensions' operators (we
    # disambiguate via column type and explicit operator class).
    cur.execute(f"SET search_path = {args.schema}, turbovec, public")

    schema = args.schema
    blocks = []

    # ---- pgvector HNSW (cosine) on emb_pgv — sweep ef_search.
    pgv_drop = f"DROP INDEX IF EXISTS {schema}.idx"
    pgv_create = (
        f"CREATE INDEX idx ON {schema}.corpus "
        f"USING hnsw (emb_pgv vector_cosine_ops) "
        f"WITH (m = {args.hnsw_m}, ef_construction = {args.hnsw_efc})"
    )
    pgv_query = (
        f"SELECT id FROM {schema}.corpus "
        f"ORDER BY emb_pgv OPERATOR(public.<=>) %s::public.vector LIMIT %s"
    )
    ef_search_list = [int(x.strip()) for x in args.ef_search.split(",") if x.strip()]
    # Build the index ONCE; iterate ef_search per session.
    cur.execute(pgv_drop)
    conn.commit()
    print(f"\n--- pgvector HNSW build (m={args.hnsw_m}, efc={args.hnsw_efc}) ---",
          flush=True)
    t0 = time.time()
    cur.execute(pgv_create)
    conn.commit()
    pgv_build_secs = time.time() - t0
    cur.execute("SELECT pg_relation_size(c.oid) FROM pg_class c "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=%s AND c.relname='idx'", (schema,))
    pgv_idx_bytes = cur.fetchone()[0]
    print(f"  built in {pgv_build_secs:.2f}s, size {pgv_idx_bytes} bytes",
          flush=True)
    for efs in ef_search_list:
        cur.execute(f"SET hnsw.ef_search = {efs}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET max_parallel_workers_per_gather = 0")
        # Warm.
        for i in range(min(20, test.shape[0])):
            cur.execute(pgv_query, (vec_literal(test[i]), 100))
            cur.fetchall()
        lats = []
        sum_r1 = sum_r10 = sum_r100 = 0.0
        k_max = min(100, gt_k)
        for i in range(test.shape[0]):
            ql = vec_literal(test[i])
            cur.execute(pgv_query, (ql, k_max))
            ids_r = [int(r[0]) for r in cur.fetchall()]
            sum_r1 += recall_at_k(gt[i], ids_r, 1)
            sum_r10 += recall_at_k(gt[i], ids_r, 10)
            sum_r100 += recall_at_k(gt[i], ids_r, k_max)
            t0 = time.time()
            cur.execute(pgv_query, (ql, args.limit_k))
            cur.fetchall()
            lats.append((time.time() - t0) * 1e6)
        nf = float(test.shape[0])
        out = {
            "label": f"pgvector HNSW (m={args.hnsw_m}, efc={args.hnsw_efc}, ef_search={efs})",
            "create_sql": pgv_create,
            "build_secs": pgv_build_secs,
            "index_bytes": int(pgv_idx_bytes),
            "n_queries": int(test.shape[0]),
            "limit_k": args.limit_k,
            "r_at_1": sum_r1 / nf,
            "r_at_10": sum_r10 / nf,
            "r_at_100": sum_r100 / nf,
            "p50_us": percentile(lats, 0.50),
            "p95_us": percentile(lats, 0.95),
            "p99_us": percentile(lats, 0.99),
            "mean_us": float(np.mean(lats)),
        }
        print(f"  ef_search={efs}: R@1={out['r_at_1']:.3f} R@10={out['r_at_10']:.3f} "
              f"R@100={out['r_at_100']:.3f}  p50={out['p50_us']:.0f}us", flush=True)
        blocks.append(out)
    cur.execute(pgv_drop)
    conn.commit()

    # ---- pg_turbovec @ bit_width = 4 on emb_tv.
    tv4_drop = f"DROP INDEX IF EXISTS {schema}.idx"
    tv4_create = (
        f"CREATE INDEX idx ON {schema}.corpus "
        f"USING turbovec (emb_tv vec_cosine_ops) "
        f"WITH (bit_width = 4)"
    )
    tv4_query = (
        f"SELECT id FROM {schema}.corpus "
        f"ORDER BY emb_tv OPERATOR(turbovec.<=>) %s::turbovec.vector LIMIT %s"
    )
    blocks.append(measure_index(
        conn, schema, "pg_turbovec (bit_width=4)",
        tv4_create, tv4_drop, tv4_query,
        test, gt, args.limit_k, gt_k,
        pad_to=setup_corpus_padded_dim,
    ))

    # ---- pg_turbovec @ bit_width = 2 on emb_tv.
    tv2_drop = f"DROP INDEX IF EXISTS {schema}.idx"
    tv2_create = (
        f"CREATE INDEX idx ON {schema}.corpus "
        f"USING turbovec (emb_tv vec_cosine_ops) "
        f"WITH (bit_width = 2)"
    )
    tv2_query = tv4_query
    blocks.append(measure_index(
        conn, schema, "pg_turbovec (bit_width=2)",
        tv2_create, tv2_drop, tv2_query,
        test, gt, args.limit_k, gt_k,
        pad_to=setup_corpus_padded_dim,
    ))

    # ---- pgvector exact / brute-force baseline on emb_pgv (no index).
    cur.execute(f"DROP INDEX IF EXISTS {schema}.idx")
    cur.execute("SET enable_seqscan = on")
    pgv_brute = (
        f"SELECT id FROM {schema}.corpus "
        f"ORDER BY emb_pgv OPERATOR(public.<=>) %s::public.vector LIMIT %s"
    )
    print("\n--- exact brute-force (pgvector seq scan) ---")
    lats = []
    for i in range(n_queries):
        q = test[i]
        t0 = time.time()
        cur.execute(pgv_brute, (vec_literal(q), args.limit_k))
        cur.fetchall()
        lats.append((time.time() - t0) * 1e6)
    brute = {
        "label": "pgvector seq-scan exact (no index)",
        "n_queries": int(n_queries),
        "limit_k": args.limit_k,
        "r_at_1": 1.0,
        "r_at_10": 1.0,
        "r_at_100": 1.0,
        "p50_us": percentile(lats, 0.50),
        "p95_us": percentile(lats, 0.95),
        "p99_us": percentile(lats, 0.99),
        "mean_us": float(np.mean(lats)),
    }
    print(f"  p50={brute['p50_us']:.0f}us  p95={brute['p95_us']:.0f}us  "
          f"p99={brute['p99_us']:.0f}us", flush=True)

    cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()

    report = {
        "fixture_dir": str(args.fixture_dir),
        "fixture_dim": int(dim),
        "fixture_corpus_n": int(n_corpus),
        "fixture_queries_n": int(n_queries),
        "ground_truth_k": int(gt_k),
        "limit_k": args.limit_k,
        "host": os.environ.get("HOSTNAME", "unknown"),
        "postgres_version": pg_version,
        "pgvector_version": pgv_version,
        "pg_turbovec_version": pgtv_version,
        "hnsw_m": args.hnsw_m,
        "hnsw_efc": args.hnsw_efc,
        "hnsw_ef_search": args.ef_search,
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "indexes": blocks,
        "exact_brute_force_seqscan": brute,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.results_path}")
    # Markdown summary
    print("\n## Summary table\n")
    print("| Index | R@1 | R@10 | R@100 | p50 µs | p95 µs | p99 µs | bytes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in blocks:
        print(
            f"| {b['label']} | {b['r_at_1']:.3f} | {b['r_at_10']:.3f} "
            f"| {b['r_at_100']:.3f} | {b['p50_us']:.0f} | "
            f"{b['p95_us']:.0f} | {b['p99_us']:.0f} | "
            f"{b['index_bytes']} |"
        )
    print(
        f"| {brute['label']} | 1.000 | 1.000 | 1.000 | "
        f"{brute['p50_us']:.0f} | {brute['p95_us']:.0f} | "
        f"{brute['p99_us']:.0f} | (heap) |"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
