# Roadmap: winning against Qdrant & VectorChord (incl. >RAM)

_Drafted 2026-06-16, after the v1.10.1 honesty review surfaced that
(a) IVF latency was never measured at 1M+ (only 200k in-process),
and (b) the production deployment is >5M rows, so out-of-core and
at-scale latency are not optional. This is the plan to close every
axis where a competitor currently leads._

---

## The reframing: IVF latency and >RAM are the SAME lever

The flat scan is `O(n·dim)` AND must hold the whole index in RAM.
**IVF fixes both at once**, because cells are stored contiguous on
disk (the IVF-1 layout):

- A query touches only the `probes` nearest cells' code ranges
  (~`probes/lists` of the corpus), so latency is sublinear **and**
  the working set per query is a small fraction of the index.
- If the codes are **mmap'd** (not slurped into a per-backend
  `Vec<u8>`), the OS pages in only the probed cells. The resident
  set per query is `O(probes · cell_size)`, not `O(n)`. **That is
  out-of-core search** — the index can exceed RAM; only the hot
  cells stay resident.

So the same work item ("scan only probed cells, from mmap'd
contiguous ranges") delivers the at-scale latency win **and** the
>RAM story. They are one project, not two.

---

## Honest current state (what's measured vs claimed)

| Claim | Status |
|---|---|
| Storage 7.6\u201315.2\u00d7 smaller than HNSW | \u2705 measured, 1M Cohere-wiki |
| Recall 1.000 (flat) / tunable (IVF) | \u2705 measured |
| IVF recall-vs-probes trade-off works | \u2705 measured (host-independent) |
| IVF ~5\u00d7 faster than flat scan | \u2705 measured **but only at 200k\u00d7256-d** |
| **IVF latency vs HNSW at 1M+** | \u274c **NOT measured** \u2014 the gap |
| **IVF / anything at the production 5M scale** | \u274c **NOT measured** \u2014 the priority |
| >RAM / out-of-core | \u274c not implemented (RAM-resident) |
| True metadata-filter pushdown | \u274c post-filter + probe-widening only |
| Multivector / hybrid fusion | \u274c not implemented |

The "pgvector wins latency at 1M" line in the docs was an
apples-to-oranges comparison (HNSW-1M vs IVF-200k). We do not
actually know the IVF-vs-HNSW latency at 1M+; the projection
(~40 ms at probes=16, lists\u2248\u221an) is plausible but unmeasured.

---

## Phase plan (ordered by production value)

### Phase A \u2014 IVF latency at production scale (MEASURE FIRST)

**Why first:** you run >5M in production. Before building more, we
must know where IVF actually lands vs HNSW at 1M, 5M, 10M. This is
measurement + the k-means build-speed fix that makes 5M builds
feasible.

- A-1: **Faster k-means for large builds.** A 200k\u00d7256-d/lists=448
  build took ~2.7 min (k-means dominates). At 5M that's
  prohibitive. Switch to mini-batch k-means (or sample-train +
  single-pass assign, which we already GEMM-batch) so a 5M build is
  minutes, not hours. Bench the build time at 1M/5M.
- A-2: **IVF latency frontier at 1M, then 5M, on AVX2**, isolated
  protocol (taskset, contention-gated, the v1.9.1 method).
  `lists\u2208{\u221an, 4\u221an}`, `probes\u2208{1..lists}`, vs pgvector HNSW +
  ivfflat at matched recall. Answer the real question: **at
  recall@10\u22650.95, IVF p50 vs HNSW p50 at 1M and 5M.**
- A-3: Publish to `docs/BENCHMARKS.md` + ideally a VectorDBBench
  entry. This is the credibility deliverable.

### Phase B \u2014 Out-of-core / >RAM (the architectural unlock)

**Why:** your 5M index and growth; pgvectorscale's DiskANN bet is
the one axis no PG-quantization competitor has fully ceded.

- B-1: **mmap the codes/scales chains, not just static regions.**
  Today only blocked-codes/rotation/codebook are mmap'd; the
  per-cell code ranges still come through `read_full` into a
  per-backend `Vec`. mmap the whole codes chain `MAP_PRIVATE`;
  the scan reads probed cells' ranges directly from the mapping.
  The OS page cache becomes the working-set manager.
- B-2: **Cell-granular fault-in.** Because cells are contiguous,
  a probed cell is a contiguous byte range \u2014 `madvise(WILLNEED)`
  it, scan it, let the OS evict cold cells under pressure. The
  resident set is `O(probes\u00b7cell_size)`, so an index 10\u00d7 RAM is
  fine as long as the hot probes fit.
- B-3: **Validate >RAM**: build an index larger than the box's
  RAM (or cgroup-limit RAM below index size), confirm queries
  succeed with bounded RSS and acceptable latency (cold-cell faults
  add disk latency \u2014 measure it). This is the headline "pg_turbovec
  scales past RAM" claim.
- B-4: Document the storage hierarchy: hot cells in RAM (OS cache),
  cold cells on disk, the 7\u201315\u00d7 quantization compression meaning
  far more vectors fit in RAM than HNSW's full-precision graph.
  **Our compression is our out-of-core advantage** \u2014 a 4-bit IVF
  index that fits in RAM is one where HNSW would already be
  swapping.

### Phase C \u2014 True metadata-filter pushdown (vs Qdrant/VectorChord)

**Why:** Qdrant's headline feature; VectorChord has prefilter. Our
post-filter + probe-widening is correct but scans extra candidates
under a selective predicate.

- C-1: **Filter-aware cell scan.** When the query has a pushable
  `WHERE` predicate on an indexed payload column, intersect it with
  the cell scan: skip cells (or rows within a cell) that can't match
  the filter. Requires the AM to receive the filter \u2014 PG's
  `amgettuple` gets scan keys; for a true pushdown we'd cooperate
  with a payload/bitmap. Investigate the `amcanreturn` /
  scan-key path and whether a companion bitmap (from a B-tree on the
  filter column) can prune cells.
- C-2: **Pragmatic interim:** the partial-index + iterative-probe
  combo already covers most real cases; document the pattern
  (a partial turbovec index `WHERE tenant_id = X`, or a B-tree
  prefilter feeding a bitmap). Ship docs before the deep pushdown.

### Phase D \u2014 Breadth parity (vs VectorChord)

- D-1: **Multivector / late-interaction (ColBERT)** \u2014 MaxSim over
  per-token vectors. Large; only if a user needs it. (You may \u2014
  flag it.)
- D-2: **Named vectors / hybrid dense+sparse fusion** \u2014 mostly a
  schema + query-layer concern; RRF can live in SQL. Document the
  pattern; build server-side fusion only on demand.
- D-3: **Out-of-core BUILD** (VectorChord builds 100M on a 128 GB
  box) \u2014 stream the build so the index can be built larger than
  RAM, not just queried. Pairs with Phase B's mmap.

### Phase E \u2014 Scale-correctness hardening (for the 5M production use)

- E-1: **5M+ correctness + recall** at the real dimension, on a
  real-embedding corpus, IVF + soft assignment. The production
  deployment IS the test case \u2014 mirror its shape.
- E-2: **VACUUM / aminsert at scale with IVF.** Today a vacuum
  degrades an IVF index to flat (cell contiguity breaks on
  swap-remove). For a churning 5M index that's a silent latency
  cliff. Need either incremental re-clustering or a documented
  REINDEX cadence + a `pg_stat`-visible "ivf_degraded" signal.
- E-3: **Concurrent insert/scan at scale** \u2014 the deferred-commit +
  per-backend cache under real concurrency at 5M.

---

## Priority order for the production user (>5M today)

1. **Phase A** (IVF latency at 1M/5M + fast k-means) \u2014 you need to
   know IVF works at your scale, and builds must be feasible.
2. **Phase E-2** (IVF VACUUM degradation) \u2014 a churning 5M index
   silently falling back to a 2.5 s flat scan is a production
   landmine; this is urgent for a live deployment.
3. **Phase B** (out-of-core) \u2014 as 5M grows, the RAM ceiling looms;
   the mmap-cells unlock is the answer and reuses the contiguous
   layout already shipped.
4. **Phase C** (filter pushdown) and **Phase D** (breadth) \u2014 the
   competitive-completeness items, after the deployment is solid.

---

## The winning thesis

pg_turbovec's quantization (7\u201315\u00d7 smaller than HNSW) is not just a
storage win \u2014 **it is the out-of-core advantage**: more vectors fit
in RAM, and the contiguous-cell IVF layout lets the OS page only the
hot cells. Where pgvectorscale needs DiskANN to go out-of-core,
pg_turbovec gets there by being small enough that "out-of-core" is
rarer, and by mmap'ing contiguous cells when it does happen. Where
Qdrant/VectorChord win on filtering and breadth, those are additive
features on top of an index that already beats them on storage
density. The plan above closes every gap; Phase A proves the latency
claim that makes the rest worth building.
