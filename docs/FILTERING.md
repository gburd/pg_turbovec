# Filtering & hybrid search with pg_turbovec

_Last updated: 2026-06-17 (v1.13.0). Source-of-truth read directly
from `src/knn.rs`, `src/index/scan.rs`, the turbovec kernel
(`search.rs::block_has_allowed`), and the C-2 crossover bench
(`benches/results/allowlist_crossover_floki_v1_13_0_20260617.json`)._

"Filtered ANN" / "hybrid search" means: return the `k` nearest
vectors **that also satisfy a metadata predicate** (a tenant id, a
topic, a price range, a per-query id set). pg_turbovec gives you
**three working mechanisms** for this, each with a different sweet
spot. This is the canonical guide to picking one.

If you just want the answer: **use a partial index for known,
low-cardinality filters; use `turbovec.knn(..., allowed)` for
selective, per-query id sets; use the index AM's iterative scan for
the normal `ORDER BY ... LIMIT` ergonomics with a moderately
selective `WHERE`.** The decision matrix in § 5 makes this precise.

---

## TL;DR decision matrix

| Filter shape | Selectivity | Corpus | Use |
|---|---|---|---|
| Known/enumerable value (tenant, category) | any | any | **Partial index** (§ 2) |
| Computed per query, an id set | **selective** (≤ ~5-10%) | flat-friendly (≤ ~1M) | **`knn(..., allowed)`** (§ 3) |
| Computed per query, an id set | not selective (> ~10%) | any | plain ANN, then post-filter; or partial index |
| Arbitrary `WHERE col = x` with `ORDER BY <=> q LIMIT k` | moderate | any | **Iterative scan** (§ 4) |
| Arbitrary `WHERE`, very selective, huge corpus | very low | huge | partial index if value is known; else accept the `max_scan_tuples` ceiling (§ 4) |

The honest limitation that shapes this table: the **index AM's
`ORDER BY` scan does not itself evaluate `WHERE col = x`** -- it
returns candidate TIDs and the executor rechecks the predicate. True
*in-traversal* pushdown (the index intersecting the filter with the
cell scan, Qdrant/VectorChord style) is **not** implemented on the
AM path. See § 6.

---

## 1. The three mechanisms at a glance

| | Partial index | `knn(..., allowed)` | Iterative scan |
|---|---|---|---|
| What pushes the filter | PostgreSQL planner (native) | the SIMD kernel (in-kernel) | nothing -- executor rechecks; AM just widens |
| Where the filter lives | index predicate (`WHERE` on `CREATE INDEX`) | a `bigint[]` arg per query | the query's `WHERE` clause |
| Index structure | flat **or** IVF | **flat only** (no IVF) | flat **or** IVF |
| API | `ORDER BY emb <=> q LIMIT k` | `turbovec.knn(...)` function | `ORDER BY emb <=> q LIMIT k` |
| Wins when | filter value is known & enumerable | filter is selective & per-query | you want index-scan ergonomics |
| Cost model | exact, smaller index | cheaper as filter tightens (§ 3) | extra candidates scanned under selective filter |

---

## 2. Partial index (the recommended default)

**The PostgreSQL-idiomatic answer for known, low-cardinality
categorical filters** (multi-tenant `tenant_id = X`, a fixed set of
topics, a status flag). PostgreSQL pushes the partial-index predicate
**natively** -- the index only contains rows matching its `WHERE`, so
the scan is **exact** (no under-return, no post-filter) and the index
is **smaller**.

```sql
-- One index per tenant (or per hot category):
CREATE INDEX items_emb_t5 ON items USING turbovec (embedding turbovec.vector_cosine_ops)
    WHERE tenant_id = 5;

-- The planner uses items_emb_t5 automatically when the query's WHERE
-- implies the index predicate:
SELECT id FROM items
WHERE  tenant_id = 5
ORDER  BY embedding <=> $1
LIMIT  10;
```

**When it wins:** the filter column has few distinct values you query
by, and you know them at index-build time. Each partial index is
exact and compact; there is no candidate widening and no recheck cost.

**When it loses:** high-cardinality or open-ended filters (you can't
build an index per user id), or filters computed at query time. You'd
need thousands of partial indexes -- don't. Use § 3 or § 4 instead.

**Cost model:** a partial index over a fraction `f` of the corpus is
roughly `f x` the size and `f x` the scan cost of the full index.
This is the cheapest option when it applies because the filter is
"baked in" -- zero per-query filtering work.

---

## 3. Allowlist `knn(..., allowed)` -- in-kernel pushdown

**For a filter computed per query that yields a SELECTIVE id set.**
This is the one place pg_turbovec does **true in-kernel pushdown**:
the SIMD scoring kernel skips entire 32-vector blocks whose allowed
mask is empty, *before any LUT work*. A selective allowlist therefore
gets **cheaper**, not more expensive.

Exact signature (`src/knn.rs`):

```sql
turbovec.knn(
    rel       regclass,
    id_col    text,        -- bigint key column
    vec_col   text,        -- vector column
    query     turbovec.vector,
    k         integer,
    bit_width integer        DEFAULT 4,
    allowed   bigint[]       DEFAULT NULL   -- the allowlist
) RETURNS TABLE(id bigint, score double precision)
```

Pass the id set as a `bigint[]`. `NULL` (or omitting it) does an
unfiltered search.

```sql
-- Top-10 nearest, restricted to a per-query id set:
SELECT k.id, d.body
FROM   turbovec.knn(
         'items'::regclass,
         'id', 'embedding',
         $1::turbovec.vector,
         10, 4,
         ARRAY(SELECT id FROM items WHERE tenant_id = $2 AND price < $3)::bigint[]
       ) k
JOIN   items d USING (id)
ORDER  BY k.score DESC;
```

How the pushdown works: `run_search` -> `IdMapIndex::search_with_allowlist`
-> `search_with_mask`. The mask is packed to one bit per slot;
`block_has_allowed()` tests the 32-bit window for a block and
`continue`s (skipping all LUT scoring for those 32 vectors) when it's
zero. The `turbovec::search::BLOCKS_SKIPPED_BY_MASK` counter is
incremented on each skip -- you can read it as a telemetry proxy.

### Two caveats you must know

1. **Flat only, no IVF.** `knn()` always builds/uses a **flat**
   `IdMapIndex` (confirm in `src/knn.rs`: it walks the heap via SPI or
   reuses the shared flat cache; it never consults the IVF coarse
   model). So the allowlist short-circuit operates on a flat scan.
   Practical ceiling: corpora where a flat (block-skipping) scan is
   acceptable -- roughly up to ~1M rows, less at high dim. For huge
   corpora with a known filter, prefer a partial index (§ 2).
2. **It's a function, not the `ORDER BY` operator.** `knn()` is
   `STABLE PARALLEL SAFE` and returns `(id, score)`; you join it back
   to the heap. It is *not* the index-AM `ORDER BY emb <=> q LIMIT k`
   path. If you want operator ergonomics, use § 4.

### Measured crossover (C-2)

`benches/allowlist_crossover` times `search_with_allowlist` (the exact
`knn()` kernel path) against the naive post-filter baseline (fetch
`k*oversample` unfiltered, drop ids not in the set) at varying
selectivity. Host: floki (Intel Core Ultra 7 258V, AVX2), corpus
300k x 256-d, 4-bit, k=10, oversample=4. p50 microseconds:

| allowed fraction | allowlist p50 (us) | naive post-filter p50 (us) | blocks skipped | allowlist vs baseline |
|---:|---:|---:|---:|---:|
| 100%  | 17860 | 6759 | 0      | 0.38x (slower) |
| 50%   | 14606 | 6763 | 0      | 0.46x (slower) |
| 10%   |  8894 | 7022 | 15650  | 0.79x (~break-even) |
| 1%    |  2719 | 7095 | 338200 | **2.61x faster** |
| 0.1%  |   481 | 7082 | 454050 | **14.7x faster** |

The shape is the point (absolute numbers are host-dependent): the
allowlist latency **decreases monotonically** as the filter tightens
(17.9 ms -> 0.48 ms, ~37x), while the naive post-filter is **flat**
(~7 ms regardless of selectivity, because it always scans the whole
corpus). They cross at roughly **7-10% selectivity**. Below that the
allowlist wins, and the win grows as the filter tightens.

**The corollary you must respect:** at the non-selective end
(allowlist covers most of the corpus) the allowlist path is *slower*
than plain ANN -- building and checking a full-width mask with zero
blocks skipped is pure overhead. **Only use an allowlist when it is
selective.** For non-selective filters, use a partial index or plain
ANN + a cheap SQL post-filter.

JSON: `benches/results/allowlist_crossover_floki_v1_13_0_20260617.json`.
Reproduce: `cargo bench --bench allowlist_crossover --no-default-features --features pg16 -- --json`.

---

## 4. Iterative scan + `WHERE` -- the index-AM path

**For the normal `ORDER BY emb <=> q LIMIT k` ergonomics with a
moderately selective `WHERE`.** Shipped in v1.8.0. The executor owns
the `WHERE` recheck; the AM's job is to keep feeding candidates until
the post-filter yields `k`.

```sql
SET turbovec.iterative_scan = relaxed_order;   -- default
SET turbovec.max_scan_tuples = 20000;          -- safety ceiling (default)

SELECT id FROM items
WHERE  category = 'electronics' AND in_stock
ORDER  BY embedding <=> $1
LIMIT  10;
```

How it works: when the executor post-filters a returned batch down to
empty (the `WHERE` killed every candidate), `amgettuple` re-runs the
turbovec search with a **doubled `k`** (and, for an IVF index, widens
`probes`) and feeds the new, deduplicated candidates. Ordering across
refill batches is restored by the always-on `xs_recheckorderby`
reorder queue. The widening is **capped by `turbovec.max_scan_tuples`**
(default 20000, matching pgvector) so a pathological filter can't scan
the whole corpus unbounded.

**When it wins:** you want index-scan ergonomics (operator, planner
integration, `EXPLAIN`) and the filter is moderately selective -- the
AM widens a few times and the post-filter still returns `k`.

**When it loses (the worst case):** a **very** selective filter over a
**huge** corpus. The AM may widen all the way to `max_scan_tuples`
and *still* not find `k` survivors, because the filter is independent
of vector distance -- the nearest-by-vector candidates mostly fail the
`WHERE`. You then get fewer than `k` rows (bounded, not wrong) and
paid for `max_scan_tuples` candidates. For that regime, a partial
index (§ 2, if the value is known) or an allowlist (§ 3, if it's a
selective per-query id set) is strictly better.

**Cost model:** roughly `(k / filter_selectivity)` candidates scanned
to return `k`, capped at `max_scan_tuples`. Cheap when selectivity is
moderate; degrades toward the cap as selectivity drops.

---

## 5. Choosing -- the full decision matrix

Three axes: filter **cardinality** (low/enumerable vs high/dynamic) x
**selectivity** (fraction of corpus that passes) x **corpus size**.

| Cardinality | Selectivity | Corpus | Recommended | Why |
|---|---|---|---|---|
| Low / known (tenant, category) | any | any | **Partial index** (§ 2) | exact, smaller index, zero per-query filter cost |
| High / dynamic (per-query id set) | selective (≤ ~7%) | ≤ ~1M (flat ok) | **`knn(..., allowed)`** (§ 3) | in-kernel block skip makes it *cheaper* than post-filter |
| High / dynamic (per-query id set) | not selective | any | plain ANN + SQL post-filter | allowlist overhead exceeds its benefit above the crossover |
| Arbitrary `WHERE col=x` | moderate | any | **Iterative scan** (§ 4) | operator ergonomics; AM widens to keep returning k |
| Arbitrary `WHERE col=x` | very selective | huge | partial index if value known; else § 4 with a generous `max_scan_tuples` and accept the ceiling | no in-traversal pushdown on the AM path (§ 6) |

Quick reflexes:

- **Known filter value?** Partial index. Always the first choice.
- **Per-query id set that's small?** `knn(..., allowed)`.
- **Want `ORDER BY` and the filter is moderate?** Iterative scan.
- **Per-query id set that's large, or arbitrary WHERE over millions
  with a tiny pass rate?** None of these is magic -- partial index if
  you can, otherwise accept the post-filter / widening cost.

---

## 6. The honest limitation: no in-traversal pushdown on the AM path

This is the genuine gap versus Qdrant (filterable HNSW) and
VectorChord (prefilter). On the **index-AM `ORDER BY` path**,
pg_turbovec does **not** intersect an arbitrary `WHERE` predicate with
the index traversal. The flow is:

1. The AM (`amgettuple`) returns candidate TIDs ranked by quantized
   vector distance. It receives the query's scan keys but **does not
   evaluate** the `WHERE` predicate itself.
2. The **executor** rechecks the `WHERE` against the heap tuple.
3. If the batch is post-filtered empty, the AM **widens** (§ 4) -- it
   does not *prune* the cell scan by the filter.

**Why:** the turbovec index stores only the **quantized vector codes +
the TID** (plus the rotation matrix, codebook, and IVF centroids).
**It stores no payload columns.** There is nothing inside the index to
evaluate `category = 'electronics'` against -- the category lives in
the heap. So the index physically cannot prune by an arbitrary
predicate during traversal; it can only rank by distance and let the
executor recheck.

The `knn()` allowlist (§ 3) *is* true in-kernel pushdown, but it works
because the caller pre-computes the id set and hands it in as a bit
mask -- the index intersects a *materialized id set*, not a live
predicate, and only on the flat path.

**Workarounds (covered above):** partial index (bake the filter into
the index), allowlist `knn()` (pre-materialize a selective id set), or
iterative scan (widen + recheck). Between them they cover most real
filtering needs; the gap is specifically *arbitrary `WHERE` pushed
into the IVF cell scan*.

---

## 7. Future work: true AM pushdown feasibility (C-4 sketch)

Could the `knn()` allowlist be wired into the **IVF index-AM scan
path** -- cell-scope ∧ allowlist -- so the operator path gets in-kernel
filtering too? Assessment (design sketch only; not implemented):

**Tractable in principle, with a real obstacle.** The kernel already
accepts a slot mask (`search_with_mask`) and the IVF scan already
iterates cells. Intersecting "slots in the probed cells" with "slots
in the allowlist" is a bitmap AND -- mechanically easy. The obstacle
is **getting a filter into the AM in the first place:**

- The AM receives **scan keys** but the executor owns `WHERE`
  evaluation; the AM does not (and per `AGENTS.md` / the Phase-17
  `munmap_chunk` `amrescan` crash, **must not** be casually rewired to)
  evaluate predicates. Rewriting `amgettuple`/`amrescan` to interpret
  scan keys is the XL, risky path -- explicitly out of scope.
- For true predicate pushdown the index would need either (a) a
  **companion bitmap** built from a B-tree on the filter column,
  intersected with the cell scan (a `BitmapAnd`-style cooperation --
  the planner would have to feed the AM a TID bitmap, which the
  current AM ignores), or (b) **payload columns stored in the index**
  so it can evaluate predicates itself (a wire-format change -- new
  persisted state, a `MetaPageData::version` bump, a minor release
  with a REINDEX migration per `AGENTS.md`).
- A **narrower, lower-risk** first step: expose an AM-level
  `allowed bigint[]` channel (e.g. a GUC or a function-table variant)
  that flows a pre-materialized id set into the IVF scan's
  `search_with_mask`, reusing the existing flat-path machinery without
  touching scan-key handling. This gives operator-path users the
  allowlist win on IVF without the predicate-evaluation rewrite. It is
  still a feature with a wire-compatible surface, deferred to a future
  phase.

**Verdict:** the bitmap/payload routes are each a multi-month (XL)
build with a wire-format or planner-cooperation cost; the narrow
"flow an id set into the IVF scan" route is a plausible medium-effort
follow-up that avoids the dangerous scan-key rework. None is in scope
for Phase C. This phase ships the three working patterns, the measured
crossover, and an honest map of the gap.

---

## See also

- `docs/MIGRATING_FROM_PGVECTOR.md` -- migrating filtered queries from
  pgvector's post-filter / partial-index patterns.
- an internal design note -- positioning vs Qdrant/VectorChord
  filtering.
- `docs/PRODUCTION.md` § Filtering -- operational notes.
- `src/knn.rs` -- the allowlist function source.
- `benches/allowlist_crossover.rs` -- the C-2 crossover bench.
