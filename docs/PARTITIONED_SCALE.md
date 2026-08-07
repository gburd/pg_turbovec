# Scaling `pg_turbovec` to billions of vectors with partitioning

**Status:** Phase S-0 cookbook — usable *today* with **zero AM changes**.
Partition pruning for very large partition counts is Phase S-1 (see the last
section).

`pg_turbovec` scales to **1–10 billion vectors on a big box or a small
cluster, today, with no new extension code**, by combining three things that
already work:

1. **Declarative PostgreSQL hash partitioning** — one parent table, N child
   partitions, native tuple routing on insert.
2. **One `turbovec` index per partition** — each is an independent relfile
   with its own IVF codebook; per-partition build, VACUUM, and REINDEX.
3. **PostgreSQL's native `Merge Append`** — the parent-table
   `ORDER BY emb <=> q LIMIT k` query fans out to the per-partition index
   scans and merges them into the correct global top-k, *lazily* (it pulls
   only as many rows from each partition as the global top-k needs).

The scatter→gather→global-merge that sounds like it needs the most new code
needs **none**: it is PG's native partition-wise plan over the existing
`amcanorderbyop` order-by-operator scan. This was verified against a
single-table exact top-k — overlap 10/10, and the merge pulled 13 rows total
across 4 partitions for a global top-10 (not 10 per partition). The worked
example is `benches/poc/scatter_gather_partitioned_topk.sql`.

> **The index is not the biggest cost at extreme scale — the heap is.** At
> 1024d/2-bit the *index* is ~277 B/vector (~277 TB at 1T), but the raw f32
> *heap* is ~4 KB/row (~4 PB at 1T). For billions of vectors, store `halfvec`
> (2 KB/row) or keep only quantized codes as the source of truth and put raw
> vectors in cold object storage for the rare exact re-rank. That is an
> architecture decision above the index AM; this doc covers the index side.

---

## 1. Partition sizing: 10M–50M vectors per partition

Pick the partition count so that **one partition is a bounded, testable build
unit**. A 10M-vector / 1024d / `lists=4096` IVF partition builds in a measured
~26 min and occupies ~5.3 GB. That is the recommended unit:

| Corpus | Partitions @ 10M each | Partitions @ 50M each |
|--------|----------------------:|----------------------:|
| 100M   | 10                    | 2                     |
| 1B     | 100                   | 20                    |
| 10B    | 1 000                 | 200                   |
| 1T     | 100 000               | 20 000                |

- **Smaller partitions** (10M) = more build parallelism headroom, smaller
  blast radius for a rebuild, but a larger per-query fan-out (→ §6, Phase S-1).
- **Larger partitions** (50M) = fewer index opens per query, but each build is
  a longer, less-parallel unit.
- Do **not** exceed ~50M/partition until the 100M single-partition build is
  validated (the linear 10M→100M extrapolation is currently unverified). One
  partition should always be a build unit you can rebuild in bounded time.
- Set `lists ≈ √(rows_in_partition)` per partition (the usual IVF rule):
  `lists = 4096` for 10M, `≈ 7000` for 50M.

Choose the modulus for hash partitioning = your partition count:

```sql
CREATE TABLE docs (id bigint, emb turbovec.vector) PARTITION BY HASH (id);
-- 100 partitions for a 1B corpus @ 10M each:
CREATE TABLE docs_000 PARTITION OF docs FOR VALUES WITH (MODULUS 100, REMAINDER 0);
CREATE TABLE docs_001 PARTITION OF docs FOR VALUES WITH (MODULUS 100, REMAINDER 1);
--   ... through REMAINDER 99
```

Generate the DDL rather than hand-typing it:

```sql
SELECT format(
  'CREATE TABLE docs_%1$s PARTITION OF docs FOR VALUES WITH (MODULUS 100, REMAINDER %2$s);',
  lpad(r::text, 3, '0'), r)
FROM generate_series(0, 99) r \gexec
```

---

## 2. Concurrent per-partition build (the key build-time lever)

Each partition's `CREATE INDEX` is fully independent — separate relation,
separate relfile, separate k-means, no shared state. So a partitioned build is
**embarrassingly parallel**: `wall = per_partition_build × ceil(P / concurrency)`.
This is the single biggest build-time win and it needs no extension code, only
orchestration.

On one N-core box, run `min(N_cores / build_threads, P)` builds concurrently.
`CREATE INDEX` already uses rayon threads *inside* one backend, so give each
concurrent build a slice of the cores (`turbovec` respects the rayon pool; run
a few concurrent builds rather than one per core).

```bash
# Build every partition's index, C at a time. Each psql runs one CREATE INDEX.
C=8                     # concurrent builds; tune to cores / build_threads
DB=mydb
psql -d "$DB" -Atc \
  "SELECT c.relname
     FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
     WHERE i.inhparent = 'docs'::regclass ORDER BY c.relname" \
| xargs -P "$C" -I{} psql -d "$DB" -c \
  "CREATE INDEX IF NOT EXISTS {}_emb_idx ON {} USING turbovec (emb vec_cosine_ops) WITH (lists = 4096);"
```

Across a cluster, run this per node over that node's partitions — there is **no
stitch step** (unlike the graph kind's partitioned build): `Merge Append` is
the "stitch", done per query, for free.

Projected 1T build wall (100 000 × 10M partitions, ~26 min each):

| Concurrency                              | Build wall     |
|------------------------------------------|----------------|
| 1 at a time                              | ~1 800 days ❌ |
| 32-core node, 8 concurrent              | ~224 days ❌   |
| 100-node cluster × 8 concurrent = 800×  | ~2.2 days ✅   |
| 1000-node × 8 = 8000×                    | ~5.4 h ✅      |

**1T is inherently a cluster-scale build.** Partitioning is what makes the
cluster fan-out embarrassingly parallel. `amcanbuildparallel` (intra-partition
PG parallel workers) is the *wrong* lever at scale — cross-partition
parallelism already saturates all cores; it only helps single-partition
rebuild latency.

---

## 3. Insert routing (native, nothing to do)

`INSERT INTO docs` routes each row to the correct child by `hash(id)` — this is
core PostgreSQL tuple routing, not `pg_turbovec`'s concern. The per-child
`aminsert` is the existing deferred-commit path.

**Caveat (unchanged from the single-index case):** `turbovec`'s `aminsert` is an
O(n)-per-row whole-relfile rewrite at commit (build-then-serve model). At scale
this is actually *cheaper* than a monolith — an insert rewrites only its target
partition's relfile (10M rows), not the whole corpus. Still, for heavy
ingestion: **bulk-load, then `CREATE INDEX`**, or accept the per-partition
rewrite cost, or REINDEX the churned partition periodically.

---

## 4. Per-partition VACUUM and REINDEX (native, sharded, online)

- **`VACUUM docs`** cascades to each child (`ambulkdelete` +
  `amvacuumcleanup` per partition, the generic tombstone-bitmap path — IVF and
  flat both supported). Autovacuum treats each partition as its own relation
  with its own thresholds, so vacuum work is naturally sharded and parallel.
- **`REINDEX INDEX docs_042_emb_idx`** rebuilds exactly one partition's index
  while every other partition keeps serving. This is the online-migration story:
  a wire-format bump rolls out partition-by-partition, and the
  `REINDEX`-required ERROR (from `ambeginscan`) fires per partition. A 1T
  monolith would need one catastrophic 277 TB REINDEX; the partitioned design
  reindexes 10M at a time, online, rolling.

```sql
VACUUM (VERBOSE) docs_042;              -- one partition
REINDEX INDEX CONCURRENTLY docs_042_emb_idx;   -- rebuild one, others keep serving
```

---

## 5. Query patterns

### (a) Native parent-table query — **use this** (small-to-moderate N)

```sql
SET turbovec.probes = 16;               -- per-partition IVF recall knob
SELECT id FROM docs
ORDER BY emb <=> $q               -- cosine; <-> L2, <#> inner product
LIMIT 10;
```

Plans as:

```
Limit
  -> Merge Append
       Sort Key: (docs.emb <=> $q)
       -> Index Scan using docs_000_emb_idx on docs_000
       -> Index Scan using docs_001_emb_idx on docs_001
       ...   (one per partition)
```

Each per-partition scan is the ordinary `turbovec` order-by-op scan and
streams candidates in distance order; `Merge Append` k-way-merges them and the
outer `Limit` stops early. Recall holds exactly — the merge is over exact
distances (`xs_recheckorderby` makes each stream exact), so the global top-k is
the true global top-k. **This needs no `pg_turbovec` code beyond what already
ships.**

### (b) Explicit UNION-ALL fallback — only when the planner won't do (a)

Some queries (older PG, or a filtered predicate that defeats the partition-wise
Merge Append) fall back to a per-partition `LIMIT k` then an outer merge:

```sql
WITH per_partition AS (
    (SELECT id, emb <=> $q AS d FROM docs_000 ORDER BY emb <=> $q LIMIT 10)
  UNION ALL
    (SELECT id, emb <=> $q AS d FROM docs_001 ORDER BY emb <=> $q LIMIT 10)
  -- ... one per partition
)
SELECT id FROM per_partition ORDER BY d LIMIT 10;
```

This is byte-identical in result to (a) but **strictly worse**: it forces each
partition to fully produce its own top-k (k rows each, k·N total) before the
outer merge — it cannot stop a partition early. Use it only where (a) does not
plan. `benches/poc/scatter_gather_partitioned_topk.sql` runs both and proves
they match a single-table exact top-k.

---

## 6. Partition pruning at large N (Phase S-1 — planned, next release)

The native all-partitions `Merge Append` opens one `Index Scan` node **per
partition**. That is fine up to ~a few thousand partitions; at N ≈ 10 000+ the
`O(N)` open/plan cost alone is seconds, and 1T (100 000 partitions) is
unusable. The planned fix is a **two-level IVF**: a per-partition *summary* so a
query probes only the `Kp` nearest partitions instead of all N.

> **Status:** partition pruning is designed but **not yet shipped** — it lands
> in a later release (Phase S-1), validated on real PostgreSQL. Until then, the
> free scatter-gather in §1–§5 (native `Merge Append`, no new code) scales to
> a few thousand partitions (comfortably 1–10B vectors). What follows is the
> design, not a currently-callable API.

The design: a per-partition summary = the partition's **mean vector in original
(un-rotated) space**. (Partitions train independent rotation matrices, so their
persisted per-partition centroids are not cross-comparable across partitions;
the mean is, and is scored with the same `<=>`/`<->`/`<#>` operator the user
queries with.) A `refresh_partition_summary(parent, vec_col)` computes + stores
them in a derived catalog table (additive; no wire-format change), and a
`nearest_partitions(parent, query, Kp, metric)` selector returns the `Kp`
nearest-mean partitions; the client (or, later, native runtime pruning) then
fans out only to those. This turns the `O(N)` executor fan-out into `O(Kp)`:
query cost ≈ (flat scan over the summaries, ms) + `Kp` ×
per-partition IVF scan + merge. Recall depends on `Kp` exactly as IVF recall
depends on `probes` — more probed partitions, higher recall, monotone.

### Recall / pruning tradeoff — read this before pruning

Pruning to `Kp < N` can miss a true neighbour if it lives in a partition whose
**mean** is not among the `Kp` nearest to the query. This matters for how you
partition:

- **Hash-partitioned by `id` (the default):** every partition is a uniform
  random sample of the corpus, so every partition's mean ≈ the global mean and
  the true neighbours are spread evenly across *all* partitions. Summary-based
  pruning then does **not** reliably preserve recall unless `Kp` is large —
  hash partitioning is for scatter-gather throughput and build parallelism, not
  for pruning.
- **Content-clustered partitions** (range/list-partitioned by a semantic key,
  or hash-partitioned then re-clustered so each partition holds a coherent
  region of vector space): a partition's mean is representative of its
  contents, so the `Kp` nearest-mean partitions contain the true neighbours and
  pruning is (near-)lossless.

**Guidance:** use hash partitioning for the free scatter-gather (§1–§5) up to a
few thousand partitions; when you need pruning at large N, cluster your
partitions by content and set `Kp` from a recall sweep, exactly as you'd set
IVF `probes`.

### What remains (Phase S-1 follow-up)

- **Native runtime partition pruning wiring.** Today the selector returns
  partition `regclass`es and the client fans out (via UNION-ALL of the chosen
  children, or a `WHERE`-based restriction). Driving PostgreSQL's *native*
  runtime partition pruning directly from the selector result (so the parent
  `Merge Append` opens only the `Kp` children) is a planned follow-up.
- **Sublinear summary search.** The selector currently flat-scans the summaries
  (`O(N·dim)`, single-digit ms at N=100k). A centroid-graph over the summaries
  (reusing the Phase G-1 `ivf::CentroidGraph`) makes selection sublinear if the
  flat scan itself becomes the bottleneck.
- **Persisted / auto-refreshed summaries.** The summary is refreshed manually
  today. Triggering refresh on bulk-load completion is a follow-up.

---

## Cross-node distribution (Phase S-3, beyond this doc)

True 1T needs the partitions to live on **separate hosts**: partitions become
foreign tables via `postgres_fdw` (or Citus shards), and the parent's
`Merge Append` fans out across nodes. The single-node partitioned design in
this doc is the substrate for that — same DDL, same query patterns, partitions
just move off-box.
