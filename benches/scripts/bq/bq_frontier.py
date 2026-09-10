#!/usr/bin/env python3
"""1-bit sign-BQ (`WITH (bit_width = 1)`) recall / storage / latency frontier.

Sweeps `bit_width` in {1, 2, 4} over ONE already-loaded corpus and reports,
per configuration: R@k vs exact ground truth, index storage
(`pg_relation_size`), bytes/vector, build wall-clock, and -- ONLY on an AVX2+
host -- warm p50/p95/p99 + single-connection QPS.

Methodology contract (see docs/BQ_RECALL_BENCH.md):

  * LATENCY IS GATED ON AVX2. Per AGENTS.md, turbovec takes a ~1000x slower
    SCALAR fallback on a pre-AVX2 CPU (`meh`), so a "warm p50" from such a
    host is meaningless. This driver detects AVX2 from /proc/cpuinfo and
    REFUSES to emit latency at all without it. `--force-scalar-latency`
    still works but files the numbers under the key
    `latency_scalar_fallback_NOT_PUBLISHABLE` and sets
    meta.latency_publishable = false. Recall, storage and build time are
    CPU-independent and are always measured.

  * THE RE-RANK WINDOW IS A SWEPT VARIABLE, NOT A HIDDEN DEFAULT.
    `guc::hi_dim_rerank_candidate_count` treats a 1-bit index as high-dim at
    ANY dim, so `hi_dim_rerank = auto` silently widens BQ's exact-rerank
    window. Every row records the mode, the explicit `search_k`, the
    `oversample`, and `rerank_window_predicted` (this driver's mirror of the
    Rust clamp logic) so no row's latency can be explained after the fact by
    a window nobody wrote down. (v2.2.0's lesson: an accidental beam of 3840
    cost 194 ms and nobody knew.)

  * COMPARE AT MATCHED RECALL, not just at matched knob. The `derived`
    section picks, per bit_width, the lowest-cost config that CLEARS each
    recall target, and reports `null` when a bit_width never clears it. An
    iso-knob table that looks good while the iso-recall table looks bad is
    how the graph kind got deprecated; both are emitted here.

Requires: a corpus table with a `turbovec.vector`-typed expression, a query
set table, and psycopg (v3). No numpy / h5py / BLAS: ground truth is computed
in-DB by exact seqscan with the SAME operator the index serves, so the metric
can never drift from the index's metric (the GRAPH_EF_BENCH L2-vs-cosine trap).

Usage (see docs/BQ_RECALL_BENCH.md for the full runbook):

  bq_frontier.py --dsn "$DSN" --table docs --vec-expr embt --dim 1536 \\
      --query-table bq_query_set --query-provenance held_out \\
      --out /scratch/bq/bq_frontier_arnold_$(date -u +%Y%m%d).json

  bq_frontier.py --self-check     # pure-logic self-test, no DB, no numpy
"""

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time

# Every DB-touching path needs psycopg; --self-check and --print-schema must
# not. Import lazily so the self-check runs on a bare interpreter.
try:
    import psycopg
except ImportError:  # pragma: no cover - exercised by --self-check on a bare box
    psycopg = None


# --------------------------------------------------------------------------
# Host / SIMD gate. The single most important correctness property of this
# harness: latency may only be published from an AVX2+ host.
# --------------------------------------------------------------------------

def cpu_flags(path="/proc/cpuinfo"):
    """The `flags` set of the first core, or an empty set if unreadable."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_model(path="/proc/cpuinfo"):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def simd_profile(flags):
    """Classify a flags set into the dispatch tier turbovec will actually take.

    turbovec dispatches AVX-512 > AVX2 > scalar via `is_x86_feature_detected!`.
    The scalar fallback is CORRECT but ~1000x slower for a full-corpus scan,
    which is why latency from such a host is not a publishable number.
    """
    avx512 = "avx512f" in flags
    avx2 = "avx2" in flags
    tier = "avx512" if avx512 else ("avx2" if avx2 else "scalar")
    return {
        "avx": "avx" in flags,
        "avx2": avx2,
        "avx512f": avx512,
        "kernel_tier": tier,
        # The gate. Anything but `scalar` may publish latency.
        "latency_publishable": tier != "scalar",
    }


# --------------------------------------------------------------------------
# Mirror of src/guc.rs::hi_dim_rerank_candidate_count. Kept here so every
# emitted row records the exact-rerank window it ran with, instead of leaving
# it as an invisible default. Asserted against the Rust unit-test cases in
# --self-check; if it ever disagrees with guc.rs, the self-check is the bug
# report.
# --------------------------------------------------------------------------

HI_DIM_RERANK_MIN_DIM = 256
HI_DIM_RERANK_MAX_FLOOR = 1024


def rerank_window_predicted(mode, dim, bit_width, search_k, oversample=1.0):
    """Candidate (exact-rerank) window the scan will use, per guc.rs."""
    import math

    user_count = int(math.ceil(search_k * oversample))
    # 1-bit sign-BQ is lossy at ANY dim, so it is treated as high-dim.
    effective_dim = max(dim, HI_DIM_RERANK_MIN_DIM) if bit_width == 1 else dim
    if mode == "off":
        apply = False
    elif mode == "on":
        apply = True
    else:  # auto
        apply = effective_dim >= HI_DIM_RERANK_MIN_DIM
    if not apply:
        return user_count
    floor = min(max(effective_dim, HI_DIM_RERANK_MIN_DIM), HI_DIM_RERANK_MAX_FLOOR)
    return max(user_count, floor)


# --------------------------------------------------------------------------
# stats (same shapes as benches/scripts/vectordbbench/sweep_latency_isolated.py)
# --------------------------------------------------------------------------

def pctl(s, p):
    if not s:
        return None
    s = sorted(s)
    k = (len(s) - 1) * p
    lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo + 1] - s[lo])


def trimmed_mean(s, frac=0.05):
    if not s:
        return None
    s = sorted(s)
    n = len(s)
    k = int(n * frac)
    core = s[k:n - k] if n - 2 * k > 0 else s
    return statistics.mean(core)


def recall_at(pred_ids, truth_ids, k):
    """|top-k(pred) ∩ top-k(truth)| / k, or None if the truth is empty."""
    t = set(truth_ids[:k])
    if not t:
        return None
    return len(t & set(pred_ids[:k])) / float(len(t))


def sample_contention():
    """loadavg + CPU busy/iowait/steal + free RAM, so a contended batch is
    visible in the artefact instead of being silently trusted."""
    out = {}
    try:
        la = open("/proc/loadavg").read().split()
        out["loadavg_1m"] = float(la[0])
    except OSError:
        out["loadavg_1m"] = None
    try:
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        user, nice, system, idle, iowait, irq, softirq, steal = (v + [0] * 8)[:8]
        out["_busy"] = user + nice + system + irq + softirq
        out["_idle"] = idle
        out["_iowait"] = iowait
        out["_steal"] = steal
    except OSError:
        out.update({"_busy": 0, "_idle": 0, "_iowait": 0, "_steal": 0})
    try:
        for line in open("/proc/meminfo"):
            key, *rest = line.split()
            if key.rstrip(":") == "MemAvailable":
                out["mem_avail_mib"] = int(rest[0]) // 1024
                break
    except OSError:
        out["mem_avail_mib"] = None
    return out


def delta_cpu(before, after):
    db = after["_busy"] - before["_busy"]
    di = after["_idle"] - before["_idle"]
    dio = after["_iowait"] - before["_iowait"]
    dst = after["_steal"] - before["_steal"]
    tot = db + di + dio + dst
    if tot <= 0:
        return {"cpu_busy_pct": None, "cpu_iowait_pct": None, "cpu_steal_pct": None}
    return {"cpu_busy_pct": round(100.0 * db / tot, 2),
            "cpu_iowait_pct": round(100.0 * dio / tot, 2),
            "cpu_steal_pct": round(100.0 * dst / tot, 2)}


# --------------------------------------------------------------------------
# derived: iso-recall + iso-knob views
# --------------------------------------------------------------------------

def iso_recall(rows, targets, cost_key="p50_ms"):
    """Per bit_width, the CHEAPEST config that clears each recall target.

    `None` when a bit_width never clears the target -- that absence is the
    result, not a gap to paper over. Cost is warm p50 when latency was
    measured, else the exact-rerank window (the CPU-independent proxy for
    scan+recheck work).
    """
    out = {}
    for target in targets:
        per_bw = {}
        for bw in sorted({r["bit_width"] for r in rows}):
            ok = [r for r in rows
                  if r["bit_width"] == bw
                  and r.get("recall_at_k") is not None
                  and r["recall_at_k"] >= target]
            if not ok:
                per_bw[str(bw)] = None
                continue
            def cost(r):
                v = r.get(cost_key)
                return v if v is not None else r["rerank_window_predicted"]
            best = min(ok, key=cost)
            per_bw[str(bw)] = {
                "rerank_mode": best["rerank_mode"],
                "search_k": best["search_k"],
                "probes": best["probes"],
                "rerank_window_predicted": best["rerank_window_predicted"],
                "recall_at_k": best["recall_at_k"],
                "p50_ms": best.get("p50_ms"),
                "idx_bytes": best.get("idx_bytes"),
                "bytes_per_vector": best.get("bytes_per_vector"),
                "cost_basis": cost_key if best.get(cost_key) is not None
                              else "rerank_window_predicted",
            }
        out[f"recall_at_k>={target}"] = per_bw
    return out


# --------------------------------------------------------------------------
# SQL plumbing
# --------------------------------------------------------------------------

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$]*$")


def ident(name, what):
    """Reject anything that isn't a bare (optionally schema-qualified) name.

    Bench scripts interpolate identifiers; this keeps a typo from becoming a
    DDL accident.
    """
    if not IDENT.match(name):
        raise SystemExit(f"refusing to interpolate {what}={name!r}: not a plain identifier")
    return name


IDX_PREFIX = "bqbench_"


def idx_prefix(args):
    """Index-name prefix, namespaced by --run-id.

    Index names are SCHEMA-scoped, not table-scoped, so two concurrent runs
    in one schema collide on `bqbench_b1` even when they use different corpus
    tables -- which is exactly what happened on 2026-09-09: one arm's
    `CREATE INDEX bqbench_b1 ON <its own table>` failed with "already
    exists" because a sibling arm owned that name, silently costing that arm
    its entire bit_width=1 leg. Pass --run-id to namespace.
    """
    return IDX_PREFIX if not args.run_id else f"{IDX_PREFIX}{args.run_id}_"


def connect(dsn):
    if psycopg is None:
        raise SystemExit("psycopg (v3) is required: pip install 'psycopg[binary]'")
    conn = psycopg.connect(dsn, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET search_path = public, turbovec")
    return conn


def drop_bench_indexes(cur, table, prefix=IDX_PREFIX):
    """Drop only indexes THIS harness created (never a PK/user index).

    `prefix` is run-scoped (see `idx_prefix`) so a run never drops a
    concurrent sibling run's indexes out from under it.
    """
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = %s", (table,))
    for (name,) in cur.fetchall():
        if name.startswith(prefix):
            cur.execute(f"DROP INDEX IF EXISTS {name}")


def env_meta(cur, args, simd):
    cur.execute("SHOW server_version")
    pg_version = cur.fetchone()[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'pg_turbovec'")
    row = cur.fetchone()
    ext_version = row[0] if row else None
    cur.execute("SHOW shared_buffers")
    shared_buffers = cur.fetchone()[0]
    gucs = {}
    for g in ("turbovec.search_k", "turbovec.probes", "turbovec.oversample",
              "turbovec.hi_dim_rerank", "turbovec.iterative_scan",
              "turbovec.out_of_core", "turbovec.scan_parallelism",
              "turbovec.bit_width_default", "turbovec.normalize_on_insert"):
        try:
            cur.execute(f"SHOW {g}")
            gucs[g] = cur.fetchone()[0]
        except Exception as exc:            # a GUC removed in a future release
            gucs[g] = f"<unavailable: {exc}>"
    cur.execute(f"SELECT count(*) FROM {args.table}")
    n_rows = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM {args.query_table}")
    n_queries = cur.fetchone()[0]
    return {
        "host": platform.node(),
        "uname": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "simd": simd,
        # Repeated at the top level of meta so no reader can miss it.
        "latency_publishable": simd["latency_publishable"],
        "latency_gate": (
            "AVX2+ required. Per AGENTS.md a pre-AVX2 host takes turbovec's "
            "scalar fallback (~1000x slower full-corpus scan); its warm p50 "
            "is not a publishable number. Recall/storage/build are "
            "CPU-independent and valid on any host."
        ),
        "pg_version": pg_version,
        "pg_turbovec_version": ext_version,
        "shared_buffers": shared_buffers,
        "guc_defaults_observed": gucs,
        "corpus": {
            "table": args.table,
            "vec_expr": args.vec_expr,
            "dim": args.dim,
            "rows": n_rows,
            "opclass": args.opclass,
            "operator": args.operator,
            "label": args.corpus_label,
        },
        "query_set": {
            "table": args.query_table,
            "n_queries": n_queries,
            "provenance": args.query_provenance,
            "caveat": (
                "queries are corpus members: rank 1 is trivially the query "
                "itself, so R@k carries a 1/k floor for ANY index"
                if args.query_provenance == "in_corpus" else
                "queries are held out of the indexed corpus"
            ),
        },
        "ground_truth": {
            "method": (
                "exact top-%d in-DB seqscan (enable_indexscan/bitmapscan off) "
                "using the SAME operator the index serves (%s), so the GT "
                "metric cannot drift from the index metric" %
                (args.gt_depth, args.operator)
            ),
            "depth": args.gt_depth,
        },
        "build_settings": {
            "maintenance_work_mem": args.maintenance_work_mem,
            "max_parallel_maintenance_workers": args.build_workers,
        },
        "protocol": {
            "k": args.k,
            "n_warm": args.n_warm,
            "latency_basis": "server-side Execution Time from EXPLAIN (ANALYZE)",
            "scan_parallelism": 0,
            "load_gate": args.load_gate,
        },
        "start_loadavg": open("/proc/loadavg").read().strip()
                          if os.path.exists("/proc/loadavg") else None,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "driver": "benches/scripts/bq/bq_frontier.py",
    }


def build_ground_truth(conn, args):
    """Exact top-`gt_depth` per query, by seqscan, with no index in the way."""
    t0 = time.time()
    with conn.cursor() as cur:
        # Drop the bench indexes so the GT scan cannot possibly use one.
        # (enable_indexscan=off also covers it; belt and braces.) Under
        # --skip-build the index IS the thing being swept, so we rely on the
        # planner GUCs alone.
        if not args.skip_build:
            drop_bench_indexes(cur, args.table, idx_prefix(args))
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute("SET max_parallel_workers_per_gather = %s" % args.gt_workers)
        cur.execute("SET parallel_setup_cost = 0")
        cur.execute("SET parallel_tuple_cost = 0")
        cur.execute("SET min_parallel_table_scan_size = 0")
        cur.execute(f"DROP TABLE IF EXISTS {args.gt_table}")
        cur.execute(f"""
            CREATE TABLE {args.gt_table} (
                qid    int  NOT NULL,
                hit_id bigint NOT NULL,
                rk     bigint NOT NULL
            )
        """)
        # ONE STATEMENT PER QUERY, with the query vector pinned by an InitPlan.
        #
        # The obvious formulation -- a CROSS JOIN LATERAL correlated on q.qvec,
        # with row_number() OVER (ORDER BY <dist>) inside -- forces a
        # `WindowAgg -> Sort -> Gather` plan: the parallel workers ship every
        # row to the leader and the LEADER sorts the whole corpus
        # single-threaded. Measured on real 1024-d data, 250k rows x 20
        # queries: `Gather (actual rows=250000, loops=20)` (5M rows to the
        # leader) with a 13.9 MB per-query quicksort, 148.2 s total.
        #
        # Looping per query with the vector as a scalar subquery makes it an
        # InitPlan constant, so each worker top-N sorts its OWN share and only
        # `gt_depth` rows per worker reach the leader:
        # `Gather Merge -> Sort (top-N heapsort, 33 kB) x 7 workers`. Same
        # inputs, 76.8 s -- 1.93x faster, and it no longer spills.
        #
        # `ORDER BY dist, id` (not just dist) makes the tie order DETERMINISTIC.
        # Ground truth is what every recall number is measured against, and
        # distances tie often at 1-bit, so an arbitrary tie order would make
        # recall figures depend on plan choice. `rk` is assigned client-side
        # from that deterministic order rather than by a window function.
        cur.execute(f"SELECT qid FROM {args.query_table} ORDER BY qid")
        qids = [r[0] for r in cur.fetchall()]
        for qid in qids:
            cur.execute(f"""
                INSERT INTO {args.gt_table} (qid, hit_id, rk)
                SELECT %s,
                       t.id,
                       row_number() OVER ()
                FROM (
                    SELECT t2.id
                    FROM {args.table} t2
                    ORDER BY {args.vec_expr.replace('t.', 't2.')}
                             OPERATOR({args.operator})
                             (SELECT qvec FROM {args.query_table} WHERE qid = %s),
                             t2.id
                    LIMIT {args.gt_depth}
                ) t
            """, (qid, qid))
        cur.execute(f"CREATE INDEX ON {args.gt_table} (qid)")
        cur.execute(f"SELECT count(*) FROM {args.gt_table}")
        n = cur.fetchone()[0]
    return {"rows": n, "seconds": round(time.time() - t0, 1)}


def load_gt(conn, args):
    with conn.cursor() as cur:
        cur.execute(f"SELECT qid, hit_id FROM {args.gt_table} ORDER BY qid, rk")
        gt = {}
        for qid, hit in cur.fetchall():
            gt.setdefault(qid, []).append(int(hit))
        cur.execute(f"SELECT qid FROM {args.query_table} ORDER BY qid")
        qids = [r[0] for r in cur.fetchall()]
    return qids, gt


def build_index(conn, args, bit_width, lists):
    """Build ONE index (all others dropped, so the planner has no choice)."""
    name = f"{idx_prefix(args)}b{bit_width}" + (f"_L{lists}" if lists else "")
    with_opts = [f"bit_width = {bit_width}"]
    if lists:
        with_opts.append(f"lists = {lists}")
    sql = (f"CREATE INDEX {name} ON {args.table} "
           f"USING turbovec ({args.vec_expr} turbovec.{args.opclass}) "
           f"WITH ({', '.join(with_opts)})")
    with conn.cursor() as cur:
        if args.skip_build:
            # Re-sweep an index that already exists (a 1M x 1536-d build is
            # minutes; also the path the plumbing smoke test takes).
            cur.execute("SELECT to_regclass(%s)", (name,))
            if cur.fetchone()[0] is None:
                return name, {"bit_width": bit_width, "lists": lists,
                              "build_sql": sql, "skip_build": True,
                              "build_error": f"--skip-build but index {name} does not exist",
                              "build_s": None, "idx_bytes": None,
                              "bytes_per_vector": None}
            cur.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s), "
                        f"(SELECT count(*) FROM {args.table})", (name, name))
            rel, tot, n = cur.fetchone()
            return name, {
                "bit_width": bit_width, "lists": lists, "build_sql": sql,
                "skip_build": True, "build_error": None, "build_s": None,
                "idx_bytes": int(rel), "idx_total_bytes": int(tot),
                "bytes_per_vector": round(rel / n, 2) if n else None,
                "n_vectors": int(n),
            }
        drop_bench_indexes(cur, args.table, idx_prefix(args))
        cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        cur.execute(f"SET max_parallel_maintenance_workers = {args.build_workers}")
        t0 = time.time()
        try:
            cur.execute(sql)
        except Exception as exc:
            # A rejected combination (bit_width=1 + lists>0, or a corpus that
            # is degenerate after mean-centering) is a RESULT, not a crash.
            return name, {"bit_width": bit_width, "lists": lists,
                          "build_sql": sql,
                          "build_error": str(exc).strip()[:400],
                          "build_s": None, "idx_bytes": None,
                          "bytes_per_vector": None}
        build_s = time.time() - t0
        cur.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s), "
                    f"(SELECT count(*) FROM {args.table})", (name, name))
        rel, tot, n = cur.fetchone()
    return name, {
        "bit_width": bit_width, "lists": lists, "build_sql": sql,
        "build_error": None,
        "build_s": round(build_s, 2),
        "idx_bytes": int(rel), "idx_total_bytes": int(tot),
        "bytes_per_vector": round(rel / n, 2) if n else None,
        "n_vectors": int(n),
    }


def query_sql(args):
    return (f"SELECT id FROM {args.table} "
            f"ORDER BY {args.vec_expr} OPERATOR({args.operator}) "
            f"(SELECT qvec FROM {args.query_table} WHERE qid = %s) "
            f"LIMIT {args.k_fetch}")


def explain_ms(cur, sql, qid):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS off, COSTS off, TIMING on) " + sql, (qid,))
    for (line,) in cur.fetchall():
        if line.startswith("Execution Time:"):
            return float(line.split()[2])
    return None


def assert_index_used(cur, sql, qid, label):
    cur.execute("EXPLAIN (COSTS OFF) " + sql, (qid,))
    plan = "\n".join(r[0] for r in cur.fetchall())
    used = "Index Scan" in plan
    return {"index_scan": used, "plan_first_line": plan.splitlines()[0] if plan else None,
            "warning": None if used else
                       f"{label}: plan is NOT an Index Scan -- this row measures "
                       f"a seq scan, not the turbovec index"}


def measure(conn, args, build, rerank_mode, search_k, probes, measure_latency,
            qids, gt):
    """One swept configuration: recall (+ latency on an AVX2 host)."""
    sql = query_sql(args)
    bw = build["bit_width"]
    window = rerank_window_predicted(rerank_mode, args.dim, bw,
                                     search_k if search_k is not None
                                     else args.default_search_k,
                                     args.oversample)
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET turbovec.scan_parallelism = 0")
        cur.execute("SET turbovec.out_of_core = off")
        cur.execute(f"SET turbovec.hi_dim_rerank = {rerank_mode}")
        cur.execute(f"SET turbovec.oversample = {args.oversample}")
        if search_k is not None:
            cur.execute(f"SET turbovec.search_k = {search_k}")
        else:
            cur.execute("RESET turbovec.search_k")
        if probes is not None:
            cur.execute(f"SET turbovec.probes = {probes}")
        plan = assert_index_used(cur, sql, qids[0], f"bw{bw}/{rerank_mode}")

        # warm the per-backend index cache + OS page cache
        for i in range(args.n_warm):
            cur.execute(sql, (qids[i % len(qids)],))
            cur.fetchall()

        before = sample_contention()
        t_wall = time.perf_counter()
        lat, r_at_k, r_at_deep, n_scored = [], [], [], 0
        for qid in qids:
            if measure_latency:
                ms = explain_ms(cur, sql, qid)
                if ms is not None:
                    lat.append(ms)
            cur.execute(sql, (qid,))
            pred = [int(r[0]) for r in cur.fetchall()]
            truth = gt.get(qid, [])
            a = recall_at(pred, truth, args.k)
            if a is not None:
                r_at_k.append(a)
                n_scored += 1
            if args.gt_depth >= args.k_deep and args.k_fetch >= args.k_deep:
                b = recall_at(pred, truth, args.k_deep)
                if b is not None:
                    r_at_deep.append(b)
        wall_s = time.perf_counter() - t_wall
        after = sample_contention()

    row = {
        "bit_width": bw,
        "lists": build["lists"],
        "rerank_mode": rerank_mode,
        "search_k": search_k,
        "search_k_effective": search_k if search_k is not None else args.default_search_k,
        "oversample": args.oversample,
        "probes": probes,
        "rerank_window_predicted": window,
        "k": args.k,
        "n_queries_scored": n_scored,
        "recall_at_k": round(statistics.mean(r_at_k), 4) if r_at_k else None,
        f"recall_at_{args.k_deep}": round(statistics.mean(r_at_deep), 4) if r_at_deep else None,
        "build_s": build["build_s"],
        "idx_bytes": build["idx_bytes"],
        "bytes_per_vector": build["bytes_per_vector"],
        "plan": plan,
    }

    if not lat:
        # No latency measured: say so in the row, do not leave a null that
        # could read as "fast".
        row["latency"] = {
            "measured": False,
            "reason": ("host kernel tier is scalar (no AVX2): latency would be "
                       "~1000x the SIMD path and is not a publishable number"
                       if not measure_latency else
                       "EXPLAIN ANALYZE returned no Execution Time"),
        }
        return row

    stats = {
        "measured": True,
        "basis": "server-side Execution Time from EXPLAIN (ANALYZE)",
        "n": len(lat),
        "min_ms": round(min(lat), 3),
        "p50_ms": round(pctl(lat, 0.50), 3),
        "p95_ms": round(pctl(lat, 0.95), 3),
        "p99_ms": round(pctl(lat, 0.99), 3),
        "max_ms": round(max(lat), 3),
        "mean_ms": round(statistics.mean(lat), 3),
        "trimmed_mean_ms": round(trimmed_mean(lat), 3),
        "qps_1conn": round(1000.0 / statistics.mean(lat), 2),
        "wall_s": round(wall_s, 2),
        "contention": {
            "loadavg_1m_before": before["loadavg_1m"],
            "loadavg_1m_after": after["loadavg_1m"],
            "mem_avail_mib_before": before.get("mem_avail_mib"),
            **delta_cpu(before, after),
            "contended_flag": bool((after["loadavg_1m"] or 0) > args.load_gate
                                   or (before["loadavg_1m"] or 0) > args.load_gate),
            "load_gate": args.load_gate,
        },
    }
    if args.force_scalar_latency and not args.simd["latency_publishable"]:
        # Structural refusal: the numbers exist but under a name that cannot
        # be quoted as a result by accident.
        row["latency_scalar_fallback_NOT_PUBLISHABLE"] = {
            **stats,
            "publishable": False,
            "reason": "measured on a pre-AVX2 host; scalar fallback, ~1000x the SIMD path",
        }
        row["latency"] = {"measured": False,
                          "reason": "see latency_scalar_fallback_NOT_PUBLISHABLE"}
    else:
        row["latency"] = stats
        # Flatten p50 for the derived iso-recall picker.
        row["p50_ms"] = stats["p50_ms"]
    return row


# --------------------------------------------------------------------------
# main sweep
# --------------------------------------------------------------------------

def run(args):
    simd = simd_profile(cpu_flags())
    if args.pretend_scalar:
        # Test hook: exercise the refusal path on an AVX2 host. Recorded in
        # the artefact so a pretend run can never be mistaken for a real one.
        simd = simd_profile(set())
        simd["pretend_scalar"] = True
    args.simd = simd
    measure_latency = simd["latency_publishable"] or args.force_scalar_latency
    if not simd["latency_publishable"]:
        print("!! host kernel tier = scalar (no AVX2). Recall/storage/build are "
              "valid; LATENCY WILL NOT BE PUBLISHED.", file=sys.stderr, flush=True)
        if args.force_scalar_latency:
            print("!! --force-scalar-latency: timings will be recorded under "
                  "`latency_scalar_fallback_NOT_PUBLISHABLE` only.",
                  file=sys.stderr, flush=True)

    conn = connect(args.dsn)
    meta = env_meta(conn.cursor(), args, simd)
    out = {
        "benchmark": "pg_turbovec 1-bit sign-BQ vs 2-bit vs 4-bit: recall / storage / latency frontier",
        "meta": meta,
        "indexes": [],
        "configs": [],
        "derived": {},
    }

    def dump():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    dump()
    print(f"[gt] exact top-{args.gt_depth} ground truth ...", flush=True)
    meta["ground_truth"].update(build_ground_truth(conn, args))
    print(f"[gt] {meta['ground_truth']['rows']} rows in "
          f"{meta['ground_truth']['seconds']}s", flush=True)
    qids, gt = load_gt(conn, args)
    dump()

    arms = [(bw, 0) for bw in args.bit_widths]
    if args.ivf_lists:
        arms += [(bw, args.ivf_lists) for bw in args.bit_widths]

    for bw, lists in arms:
        name, build = build_index(conn, args, bw, lists)
        build["index"] = name
        out["indexes"].append(build)
        dump()
        if build["build_error"]:
            print(f"[build] bw={bw} lists={lists} REJECTED: "
                  f"{build['build_error'][:120]}", flush=True)
            continue
        print(f"[build] bw={bw} lists={lists} build_s={build['build_s']} "
              f"idx_MB={build['idx_bytes']/1e6:.1f} "
              f"B/vec={build['bytes_per_vector']}", flush=True)

        probe_list = args.probes_sweep if lists else [None]
        for probes in probe_list:
            # `off` rows: the exact-rerank window is exactly what we set.
            for sk in args.search_k_sweep:
                row = measure(conn, args, build, "off", sk, probes,
                              measure_latency, qids, gt)
                out["configs"].append(row)
                report(row)
                dump()
            # `auto` row: search_k left at its default so the shipped
            # hi_dim_rerank floor is what gets measured -- the default a
            # real user gets on a 1-bit index at ANY dim.
            row = measure(conn, args, build, "auto", None, probes,
                          measure_latency, qids, gt)
            out["configs"].append(row)
            report(row)
            dump()

    flat = [r for r in out["configs"] if not r["lists"]]
    out["derived"] = {
        "iso_recall_flat": iso_recall(flat, args.recall_targets),
        "storage_per_vector": {
            str(b["bit_width"]): b["bytes_per_vector"]
            for b in out["indexes"] if not b["lists"] and not b["build_error"]
        },
        "notes": [
            "iso_recall_flat picks the CHEAPEST config per bit_width that "
            "clears each target; null means that bit_width never cleared it "
            "on this corpus -- that absence IS the result.",
            "cost_basis is warm p50 when latency was publishable, else the "
            "predicted exact-rerank window (CPU-independent work proxy).",
            "matched-STORAGE comparison is a separate run: 1-bit stores "
            "dim/8 B/vec vs 2-bit's dim/4 + 4 B, so an equal byte budget "
            "holds ~2x the rows at 1-bit. Run this driver against a 2n-row "
            "table at bit_width=1 and an n-row table at bit_width=2 and "
            "compare at equal idx_bytes; see docs/BQ_RECALL_BENCH.md.",
        ],
    }
    dump()
    conn.close()
    print(f"wrote {args.out}", flush=True)
    print("BQ_FRONTIER_DONE", flush=True)


def report(row):
    lat = row.get("latency", {})
    p50 = lat.get("p50_ms") if lat.get("measured") else "n/a(no-avx2)"
    warn = (row["plan"] or {}).get("warning")
    print(f"  bw={row['bit_width']} L{row['lists']} rr={row['rerank_mode']} "
          f"sk={row['search_k']} p={row['probes']} "
          f"window={row['rerank_window_predicted']}: "
          f"R@{row['k']}={row['recall_at_k']} p50={p50}"
          + (f"  !! {warn}" if warn else ""), flush=True)


# --------------------------------------------------------------------------
# self-check: the one runnable check for the non-trivial pure logic. No DB,
# no numpy. `bq_frontier.py --self-check`
# --------------------------------------------------------------------------

def self_check():
    # --- the AVX2 gate, the property that makes results trustworthy ---
    scalar = simd_profile({"avx", "sse4_2"})            # `meh` (Ivy Bridge)
    assert scalar["kernel_tier"] == "scalar"
    assert scalar["latency_publishable"] is False, "must refuse latency without AVX2"
    a2 = simd_profile({"avx", "avx2", "fma"})            # `arnold` (i9-12900H)
    assert a2["kernel_tier"] == "avx2" and a2["latency_publishable"] is True
    a5 = simd_profile({"avx", "avx2", "avx512f", "avx512bw"})
    assert a5["kernel_tier"] == "avx512" and a5["latency_publishable"] is True
    assert simd_profile(set())["latency_publishable"] is False

    # --- rerank window mirror vs src/guc.rs::hi_dim_rerank_tests ---
    W = rerank_window_predicted
    assert W("off", 128, 2, 32) == 32          # off_never_widens
    assert W("off", 960, 2, 32) == 32
    assert W("off", 1536, 2, 32) == 32
    assert W("off", 128, 1, 32) == 32          # off means off even at 1-bit
    assert W("auto", 255, 2, 32) == 32         # auto_does_not_touch_low_dim
    assert W("auto", 128, 2, 32) == 32
    assert W("auto", 256, 2, 32) == 256        # auto_floors_high_dim...
    assert W("auto", 960, 2, 32) == 960
    assert W("auto", 1536, 2, 32) == 1024      # capped
    assert W("auto", 128, 1, 32) == 256        # auto_widens_onebit_regardless_of_dim
    assert W("auto", 1536, 1, 32) == 1024
    assert W("auto", 960, 2, 2000) == 2000     # user_override_past_floor_wins
    assert W("auto", 960, 2, 32, 32.0) == 1024
    assert W("auto", 960, 2, 10) == 960
    assert W("on", 64, 2, 32) == 256           # on_floors_regardless_of_dim

    # --- stats ---
    assert pctl([1, 2, 3, 4, 5], 0.5) == 3
    assert pctl([], 0.5) is None
    assert abs(trimmed_mean([1, 1, 1, 100], 0.25) - 1.0) < 1e-9
    assert recall_at([1, 2, 3], [1, 9, 3], 3) == 2 / 3
    assert recall_at([1, 2, 3], [], 3) is None

    # --- iso-recall: unreachable target must report null, not the best-effort ---
    rows = [
        {"bit_width": 1, "lists": 0, "rerank_mode": "off", "search_k": 100,
         "probes": None, "rerank_window_predicted": 100, "recall_at_k": 0.80,
         "p50_ms": 5.0, "idx_bytes": 100, "bytes_per_vector": 192.0},
        {"bit_width": 1, "lists": 0, "rerank_mode": "off", "search_k": 800,
         "probes": None, "rerank_window_predicted": 800, "recall_at_k": 0.93,
         "p50_ms": 40.0, "idx_bytes": 100, "bytes_per_vector": 192.0},
        {"bit_width": 2, "lists": 0, "rerank_mode": "off", "search_k": 100,
         "probes": None, "rerank_window_predicted": 100, "recall_at_k": 0.97,
         "p50_ms": 9.0, "idx_bytes": 200, "bytes_per_vector": 388.0},
    ]
    d = iso_recall(rows, [0.90, 0.99])
    assert d["recall_at_k>=0.9"]["1"]["search_k"] == 800
    assert d["recall_at_k>=0.9"]["2"]["search_k"] == 100
    assert d["recall_at_k>=0.99"]["1"] is None, "unreachable target must be null"
    assert d["recall_at_k>=0.99"]["2"] is None
    # cheapest-clearing pick, not merely the highest recall
    d2 = iso_recall(rows, [0.90], cost_key="p50_ms")
    assert d2["recall_at_k>=0.9"]["1"]["p50_ms"] == 40.0
    # falls back to the window when latency was not measured
    nolat = [dict(r, p50_ms=None) for r in rows]
    d3 = iso_recall(nolat, [0.90])
    assert d3["recall_at_k>=0.9"]["1"]["cost_basis"] == "rerank_window_predicted"

    # --- identifier guard ---
    for bad in ("docs; DROP TABLE x", "a b", "", "1;2"):
        try:
            ident(bad, "table")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"ident() accepted {bad!r}")
    assert ident("public.docs", "table") == "public.docs"

    print("self-check OK (simd gate, rerank-window mirror, stats, iso-recall, ident)")


def dry_run_sql(args):
    """Print the statements a real run would issue. No DB connection."""
    print("-- ground truth")
    print(f"SET enable_indexscan = off; SET enable_bitmapscan = off;")
    print(f"DROP TABLE IF EXISTS {args.gt_table};")
    print(f"""CREATE TABLE {args.gt_table} AS
            SELECT q.qid, k.id AS hit_id, k.rk
            FROM {args.query_table} q
            CROSS JOIN LATERAL (
                SELECT t.id,
                       row_number() OVER (
                           ORDER BY {args.vec_expr} OPERATOR({args.operator}) q.qvec
                       ) AS rk
                FROM {args.table} t
                ORDER BY {args.vec_expr} OPERATOR({args.operator}) q.qvec
                LIMIT {args.gt_depth}
            ) k;""")
    print("\n-- index builds (one at a time; all bench indexes dropped first)")
    arms = [(bw, 0) for bw in args.bit_widths]
    if args.ivf_lists:
        arms += [(bw, args.ivf_lists) for bw in args.bit_widths]
    for bw, lists in arms:
        name = f"{idx_prefix(args)}b{bw}" + (f"_L{lists}" if lists else "")
        opts = f"bit_width = {bw}" + (f", lists = {lists}" if lists else "")
        print(f"CREATE INDEX {name} ON {args.table} "
              f"USING turbovec ({args.vec_expr} turbovec.{args.opclass}) "
              f"WITH ({opts});")
    print("\n-- per-config GUCs + query (window = predicted exact-rerank size)")
    for bw, lists in arms:
        for probes in (args.probes_sweep if lists else [None]):
            combos = [("off", sk) for sk in args.search_k_sweep] + [("auto", None)]
            for mode, sk in combos:
                w = rerank_window_predicted(
                    mode, args.dim, bw,
                    sk if sk is not None else args.default_search_k, args.oversample)
                sets = [f"SET turbovec.hi_dim_rerank = {mode}"]
                sets.append(f"SET turbovec.search_k = {sk}" if sk is not None
                            else "RESET turbovec.search_k")
                if probes is not None:
                    sets.append(f"SET turbovec.probes = {probes}")
                print(f"-- bw={bw} lists={lists} window={w}")
                print("; ".join(sets) + ";")
    print("\n-- the timed/scored query")
    print(query_sql(args).replace("%s", "<qid>") + ";")


SCHEMA_SKELETON = {
    "benchmark": "pg_turbovec 1-bit sign-BQ vs 2-bit vs 4-bit: recall / storage / latency frontier",
    "meta": {
        "host": None, "cpu_model": None,
        "simd": {"avx": None, "avx2": None, "avx512f": None,
                 "kernel_tier": None, "latency_publishable": None},
        "latency_publishable": None, "latency_gate": "<string>",
        "pg_version": None, "pg_turbovec_version": None, "shared_buffers": None,
        "guc_defaults_observed": {}, "corpus": {}, "query_set": {},
        "ground_truth": {}, "build_settings": {}, "protocol": {}, "ts": None,
    },
    "indexes": [{"bit_width": None, "lists": None, "build_s": None,
                 "idx_bytes": None, "bytes_per_vector": None,
                 "build_error": None}],
    "configs": [{"bit_width": None, "rerank_mode": None, "search_k": None,
                 "probes": None, "rerank_window_predicted": None,
                 "recall_at_k": None, "recall_at_100": None,
                 "latency": {"measured": None}, "plan": {}}],
    "derived": {"iso_recall_flat": {}, "storage_per_vector": {}},
    "_note": "SKELETON ONLY - every value is null. Not a measurement.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true",
                    help="run the pure-logic self-test and exit (no DB needed)")
    ap.add_argument("--print-schema", action="store_true",
                    help="print the all-null output skeleton and exit")
    ap.add_argument("--dry-run-sql", action="store_true",
                    help="print every SQL statement this run would execute, "
                         "then exit without touching the database")
    ap.add_argument("--dsn", help="libpq DSN, e.g. 'host=/scratch/pg port=28815 dbname=bench'")
    ap.add_argument("--table", default="docs", help="corpus table (default docs)")
    ap.add_argument("--vec-expr", default="embt",
                    help="the indexed expression, verbatim: a turbovec.vector "
                         "column name, or a parenthesised cast such as "
                         "'(emb::real[]::turbovec.vector)'")
    ap.add_argument("--dim", type=int, required=False,
                    help="corpus dimensionality (recorded, and used for the "
                         "rerank-window prediction)")
    ap.add_argument("--opclass", default="vec_cosine_ops",
                    choices=["vec_cosine_ops", "vec_l2_ops", "vec_ip_ops"])
    ap.add_argument("--operator", default="turbovec.<=>",
                    help="ORDER BY operator matching --opclass "
                         "(<=> cosine, <-> l2, <#> ip)")
    ap.add_argument("--query-table", default="bq_query_set",
                    help="table (qid int, qvec turbovec.vector)")
    ap.add_argument("--query-provenance", choices=["held_out", "in_corpus"],
                    default=None,
                    help="REQUIRED for a real run: whether the query vectors "
                         "are corpus members. Recorded in the artefact.")
    ap.add_argument("--gt-table", default="bq_gt")
    ap.add_argument("--run-id", default="",
                    help="namespace this run's index names (and, via the "
                         "runner, its query-set/GT tables) so two concurrent "
                         "arms in one schema cannot collide. Plain "
                         "identifier chars only.")
    ap.add_argument("--gt-depth", type=int, default=100)
    ap.add_argument("--gt-workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=10, help="R@k headline (default 10)")
    ap.add_argument("--k-deep", type=int, default=100, help="secondary R@k")
    ap.add_argument("--bit-widths", default="1,2,4")
    ap.add_argument("--search-k-sweep", default="32,100,256,400,800,1024,2000",
                    help="explicit exact-rerank windows swept at hi_dim_rerank=off")
    ap.add_argument("--probes-sweep", default="8,16,32,64,128",
                    help="probes swept in the optional IVF arm only")
    ap.add_argument("--ivf-lists", type=int, default=0,
                    help="also build an IVF arm with this many lists. "
                         "bit_width=1 + lists>0 is REJECTED by the extension; "
                         "the rejection is recorded as a finding.")
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--default-search-k", type=int, default=32,
                    help="the shipped turbovec.search_k default, used to "
                         "predict the window on the hi_dim_rerank=auto rows")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--load-gate", type=float, default=1.5)
    ap.add_argument("--recall-targets", default="0.90,0.95,0.99")
    ap.add_argument("--maintenance-work-mem", default="2GB")
    ap.add_argument("--build-workers", type=int, default=0)
    ap.add_argument("--skip-build", action="store_true",
                    help="do not CREATE INDEX; sweep whatever bqbench_* index "
                         "already exists (re-sweep, or plumbing smoke test)")
    ap.add_argument("--corpus-label", default=None,
                    help="human label for the corpus, e.g. 'dbpedia-openai-1M'")
    ap.add_argument("--force-scalar-latency", action="store_true",
                    help="time queries on a pre-AVX2 host anyway. Numbers are "
                         "filed under latency_scalar_fallback_NOT_PUBLISHABLE.")
    ap.add_argument("--pretend-scalar", action="store_true",
                    help="test hook: treat this host as pre-AVX2 so the "
                         "latency-refusal path can be verified anywhere. "
                         "Recorded as simd.pretend_scalar in the artefact.")
    ap.add_argument("--out", help="output JSON path")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    if args.print_schema:
        print(json.dumps(SCHEMA_SKELETON, indent=2))
        return
    missing = [f for f in ("dsn", "out", "dim", "query_provenance")
               if getattr(args, f) is None]
    if missing:
        ap.error("missing required argument(s): " +
                 ", ".join("--" + m.replace("_", "-") for m in missing))

    ident(args.table, "table")
    if args.run_id:
        ident(args.run_id, "run-id")
    ident(args.query_table, "query-table")
    ident(args.gt_table, "gt-table")
    args.bit_widths = [int(x) for x in args.bit_widths.split(",")]
    args.search_k_sweep = [int(x) for x in args.search_k_sweep.split(",")]
    args.probes_sweep = [int(x) for x in args.probes_sweep.split(",")]
    args.recall_targets = [float(x) for x in args.recall_targets.split(",")]
    # fetch enough rows to score R@k_deep too
    args.k_fetch = max(args.k, args.k_deep)
    if args.dry_run_sql:
        dry_run_sql(args)
        return
    run(args)


if __name__ == "__main__":
    main()
