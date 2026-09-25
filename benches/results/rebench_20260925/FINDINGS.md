# IVF/flat vs HNSW at matched recall — END-TO-END, publishable (2026-09-25)

Re-run of last session's Item 1 with a CORRECTED harness. Account 373102893032
(hotdog), us-east-2, c7i.8xlarge (32 vCPU, AVX2). Instance terminated, verified
clean. pg_turbovec v2.10.2 + pgvector HNSW, PG16, shared_buffers=16GB.

Corpus: Cohere/wikipedia-2023-11-embed-multilingual-v3 (en), **1M × 1024-d**,
cosine, real embeddings (resolvability 18.6%, passed). 100 held-out queries,
exact top-10 GT via CTAS. Warm p50, single connection.

## What was WRONG last session, and is fixed now

1. **Latency basis.** Last time we timed the Index-Scan NODE, which for turbovec
   excludes the reorder-queue recheck. Now we time the **top-level EXPLAIN(ANALYZE)
   Execution Time** (whole query) — IDENTICAL for both engines.
2. **The real inflation source was NOT the recheck — it was a SUBQUERY artifact.**
   The old harness put the query vector in an ORDER BY subquery
   `... ORDER BY emb <=> (SELECT emb FROM query_set WHERE qid=$N)`. Measured: that
   subquery adds ~90 ms of InitPlan/materialize overhead OUTSIDE the index scan,
   on BOTH engines (HNSW ef80: 2.3 ms with a literal vector vs 93.9 ms with the
   subquery, same plan). The corrected harness inlines the query vector as a
   LITERAL (what a real client does with a bound param).
3. **Per-query psql.** The old driver spawned a psql per query, paying the
   ~460 ms per-backend cold cache reload every time. Now all queries of an arm
   run in ONE psql -f session after a discarded warm-up.
4. **seq-fallback detector** scoped to a Seq Scan on `docs` (the harmless
   InitPlan seqscan on the 100-row query_set no longer disqualifies an arm).
5. **Planner ambiguity** (4 tv indexes, 1 opclass) removed by sweeping ONE tv
   index family per pass — no mid-sweep rebuild churn, so `contended=False` on
   all 51 arms.

## Results (all 51 arms recorded, 0 dropped)

| target | HNSW | flat bw4 | flat bw1 | IVF1024 bw4 | IVF1024 bw1 |
|---|---|---|---|---|---|
| R@10≥0.90 | 4.3 ms (0.905) | 5.2 ms (1.000) | 26.8 ms | 30.6 ms | 12.0 ms |
| R@10≥0.95 | 8.6 ms (0.953) | **5.2 ms (1.000)** | 26.8 ms | 90.6 ms | 20.8 ms |
| R@10≥0.98 | 16.6 ms (0.980) | **5.2 ms (1.000)** | 29.8 ms | unreachable | unreachable |

Index sizes: HNSW **7806 MB**, flat/IVF-bw4 **534 MB** (14.6× smaller),
flat/IVF-bw1 **134–138 MB** (58× smaller).

## The honest headline (replaces "we LOSE ~490×")

At 1M × 1024-d, warm, single-connection, this real corpus:

- **flat-bw4 is competitive-to-better than HNSW at high recall.** 5.2 ms at
  R@10 = **1.000**, flat across search_k (window 32 is already exact-quality).
  It matches HNSW at R@10 ≈ 0.90 (1.2× slower), BEATS HNSW at ≥0.95 (0.61×) and
  is 3× faster at ≥0.98 (0.32×) — where HNSW needs ef=400 to even reach 0.98 and
  flat sits at perfect recall. Reason: a quantized full scan of a 534 MB
  RAM-resident index is single-digit ms; HNSW's graph walk costs more per unit
  recall once recall is high.
- **IVF-bw1 is within 2.4–2.8× of HNSW** at 0.90–0.95 recall at **58× smaller**
  storage — a real storage/latency trade, not a 490× loss.
- **IVF-bw4 is the weakest arm** (7–10× HNSW). Confirms the project's own
  guidance: for bit_width≥2 at 1M/1024-d, FLAT beats IVF — the full quantized
  scan is so cheap that cell-pruning only adds overhead.

The old "490×" line came from a 1536-d corpus where the flat scan was ~2.5 s
AND (we now know) very likely inflated by the same subquery artifact. Dimension,
index-resident size, AND measurement method all mattered.

## Caveats to keep on the number

Warm p50, 1M rows, 1024-d, ONE corpus, single connection, all-in-RAM. NOT
QPS-under-load, NOT >10M, NOT cold. Item 2 (cold-scan) shows flat cold is
~1900 ms — the cold penalty is real and is separate work (parallel repack).

Artifacts: ivf_vs_hnsw_endtoend_result.json, iso_recall_table.txt, sweep.log,
rb_driver.py (corrected harness), rb_sweep.sh.
