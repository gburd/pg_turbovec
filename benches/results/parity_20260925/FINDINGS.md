# Parity-gap work — results (2026-09-25)

Four PARITY_GAPS items worked in parallel on EC2 (account hotdog / 373102893032,
us-east-2). Two design memos (items 3, 4), two measured benchmarks (items 1, 2).
Both instances terminated; SG + key deleted; verified clean.

---

## Item 1 — IVF/flat vs HNSW at matched recall — MEASURED, but with a
## MEASUREMENT-VALIDITY CAVEAT that blocks rewriting the scoreboard yet.

Corpus: CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (en), **1M × 1024-d**,
cosine, real embeddings (resolvability spread 18.6–69%, passed). Host c7i.8xlarge
(32 vCPU, AVX2). 100 held-out queries, exact top-10 GT via CTAS. Warm p50.

Iso-recall table AS REPORTED by the harness (Index-Scan-node time):

| target | HNSW | flat bw4 | flat bw1 | IVF bw4 | IVF bw1 |
|---|---|---|---|---|---|
| R@10≥0.90 | 3.8 ms | 4.7 ms | 24.9 ms | 29.1 ms | 10.3 ms |
| R@10≥0.95 | 12.5 ms | 4.7 ms | 24.9 ms | 86.2 ms | 18.8 ms |
| R@10≥0.98 | unreachable | 4.7 ms (R=1.000) | 27.4 ms | unreachable | unreachable |

Index sizes (the storage story, unambiguous): HNSW **7806 MB**, flat-bw4 **534 MB**
(14.6× smaller), flat-bw1 **134 MB** (58× smaller). HNSW build was minutes; the
turbovec builds seconds-to-minutes.

### DO NOT rewrite the "490× loss" scoreboard on these numbers yet.

The harness measures the **Index Scan node's** `Actual Total Time` (deliberately,
to avoid the Limit-node trap). For **turbovec** that is the quantized scan ONLY —
the `xs_recheckorderby` reorder-queue recheck (heap fetch + exact-distance
recompute for `search_k` candidates) runs in a PARENT node and is **not counted**.
HNSW's Index Scan node, by contrast, includes essentially all its work. So the
turbovec numbers **understate** end-to-end latency and are **not directly
comparable** to HNSW's as tabulated.

Evidence it matters: flat-bw4 p50 rises 4.7 ms (k=32) → 66 ms (k=2000) with
`search_k` — i.e. the recheck IS a large, growing term, exactly what the Index
Scan node omits. The old scoreboard's 2552 ms (same 1M×1024-d, `arnold`) likely
measured end-to-end `Execution Time`; that 546× gap versus this 4.7 ms is almost
certainly the uncounted recheck plus host differences, NOT a real 546× speedup.

**What is safe to conclude now:** (1) the storage/build advantage is real and
large; (2) latency scales with `search_k` (contradicting the old doc's "flat is
flat across search_k" claim — that was wrong); (3) IVF-bw1 lands in the
single-digit-to-low-double-digit ms range at 0.90–0.95 recall, plausibly within a
small constant of HNSW — but the exact ratio needs the end-to-end re-measure.

**Required before touching the scoreboard:** re-run measuring `EXPLAIN (ANALYZE)`
**Execution Time** (whole query, incl. recheck) for BOTH engines, so the
comparison is apples-to-apples. The harness's Index-Scan-node choice was right for
avoiding the Limit trap but wrong for an engine whose dominant cost is the recheck.

Artifacts: item1_ivf_vs_hnsw_result.json, item1_iso_recall_table.txt,
item1_sweep.log, item1_bench.sh, item1_driver.py.

---

## Item 2 — cold-scan latency — MEASURED, decisive.

1M × 1024-d synthetic, 4-bit, cosine, c7i.4xlarge. Three regimes, p50 ms:

| engine | R1 cold-disk | R2 cold-backend/warm-OS (pooled reality) | R3 warm-backend |
|---|---|---|---|
| HNSW | 326.8 | **1.9** | 0.6 |
| turbovec | 4571.6 | **1892.7** | 20.6 |

**The decision rule (from the design memo) resolves cleanly:**
- turbovec R2 (1892.7) ≫ R3 (20.6) → the per-backend COMPUTE term (`read_chain`
  copy + `pack::repack`) dominates cold latency by ~1870 ms. **§3a (parallelise
  `pack::repack` at cold open) IS worth coding** — S effort, additive, no wire
  change, no mmap (mmap was deleted in v1.19.0; the memo correctly killed that
  option my original Item-2 framing assumed).
- The pooled-reality gap is real and large: **turbovec 1893 ms vs HNSW 1.9 ms**
  cold-backend — ~1000×, worse than the doc's ~1.2 s line (this is 1024-d; the
  doc's was a different config).
- R1−R2 (disk term, 4572−1893 ≈ 2679 ms) is I/O the buffer manager already
  serves warm; not addressable scan-side and not the pooled case.

**Next step (code):** §3a parallel `pack::repack` at cold open, gated on this
result, with a byte-identity `#[pg_test]`. The cold penalty is compute, so
parallelising it across the 32 idle cores should cut R2 materially.

Artifacts: item2_coldscan_results.csv, item2_coldscan.log, item2_coldbench.sh.

---

## Item 3 — sparse ANN opclass — DESIGN (no build). See item3_sparse_ann_design.md.
Verdict: **no user demand found → do not build yet.** If demanded, ship Sparse
FLAT behind a new `KIND_SPARSE=4` byte (additive, REINDEX-free, reuses the
existing sparse dot kernel) — effort M; WAND is XL, gate on measured FLAT-too-slow.

## Item 4 — bulk INSERT throughput — DESIGN (no build). See item4_bulk_insert_design.md.
Verdict: commit cost is genuinely O(n_vectors) full rewrite; Z5 does NOT amortize
it (Z5 preserves cell layout, not row chains). A true fix is L/XL persist-path
work under the HARD MANDATE → **defer, gated on Item 1**. Cheap win now: document
`turbovec.ivf_max_delta_pct` + "fewer, larger transactions" for continuous ingest.
