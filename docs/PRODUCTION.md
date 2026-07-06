# pg_turbovec in Production

A deployment guide for operating `pg_turbovec` on a real PostgreSQL
cluster. For development setup see [`BUILDING.md`](BUILDING.md). For
upgrading from pgvector see
[`MIGRATING_FROM_PGVECTOR.md`](MIGRATING_FROM_PGVECTOR.md).

---

## Audience

This document is for operators running PostgreSQL 13 or newer who
want `pg_turbovec` for low-latency, low-storage ANN search on
high-dimensional embeddings (typically 128–1536 dim).

If you're evaluating `pg_turbovec` against `pgvector`, the headline
trade-off is:

- `pg_turbovec` — **5× smaller indexes**, **2.6× faster builds**,
  **identical recall** at HNSW-`ef=40` quality, **2-3× faster warm
  scans** on RAM-constrained hosts. Loses on cold-scan latency
  (~12× slower than HNSW on first-access-per-backend).
- `pgvector` HNSW — battle-tested in many production clusters,
  faster cold scans, much larger on-disk footprint.

If your workload is connection-pooled (most production REST APIs are),
warm scans dominate and `pg_turbovec` is a clear win. If you have many
short-lived backends doing one-shot vector queries, evaluate cold-scan
latency carefully against your SLO.

---

## Installation

```bash
git clone https://codeberg.org/gregburd/pg_turbovec
cd pg_turbovec
git checkout v1.7.1
cargo pgrx install --release --pg-config $(pg_config --bindir)/pg_config
```

Prerequisites:
- Rust 1.83+ (the let-else stabilisation cliff).
- `cargo-pgrx` 0.17.0 — install with `cargo install --locked
  cargo-pgrx --version 0.17.0` then `cargo pgrx init --pg17`
  (or whichever PG version).
- LLVM/Clang for the bindgen step (`LIBCLANG_PATH` must point at
  `libclang.so`).
- OpenBLAS for the SIMD inner loop. On non-NixOS hosts you may
  need `LD_PRELOAD=/path/to/libopenblas.so.0` set on the
  postmaster (see "Operational gotchas" below).

Verify:

```sql
CREATE EXTENSION pg_turbovec;
SELECT extname, extversion FROM pg_extension WHERE extname='pg_turbovec';
-- expect: pg_turbovec | 1.7.1
```

---

## Schema setup

```sql
SET search_path = turbovec, public;

CREATE TABLE docs (
    id   bigint PRIMARY KEY,
    body text,
    emb  vector(1536)        -- dim must be a multiple of 8; 1..=16000
);

-- One index per (table, column, bit_width) combo. bit_width=4 is
-- the default and the recall/storage sweet spot for cosine. 2-bit
-- is denser (2x smaller) but recall@10 drops 5-10 points on
-- typical embedding distributions; 8-bit gives ~1% better recall
-- at 2x the storage of 4-bit.
CREATE INDEX docs_emb_idx ON docs USING turbovec
    (emb turbovec.vec_cosine_ops)
    WITH (bit_width = 4);
```

Operator class name maps to distance:

| Operator class | Distance | SQL operator |
|---|---|---|
| `vec_l2_ops` | Euclidean | `<->` |
| `vec_cosine_ops` | Cosine | `<=>` |
| `vec_ip_ops` | Inner product | `<#>` |
| `vec_l1_ops` | Manhattan | `<+>` |

Cosine is the most common for embedding search.

---

## Configuration GUCs

```sql
-- Tune at session level; or set persistently in postgresql.conf.

-- Number of candidates the SIMD kernel scores per query. Higher =
-- better recall at the cost of latency. Default 100 gives recall@10
-- ~ 0.95 on dbpedia-1M. 500 gives ~ 0.99. 1000 gives ~ 0.999.
-- Sweep this against your recall SLO; latency scales linearly.
SET turbovec.search_k = 100;

-- IVF cell probing (v1.10.0+, only for indexes built WITH (lists = N)).
-- For an IVF index, amgettuple first coarse-searches the N cell
-- centroids, picks the `probes` nearest cells, and fine-searches ONLY
-- those cells' contiguous code ranges (turbovec's blocked kernel skips
-- the unprobed ranges, so latency drops roughly proportional to
-- probes/lists). This is the IVF latency/recall dial, analogous to
-- ivfflat.probes / hnsw.ef_search:
--   * lower  = fewer cells scanned = faster, lower recall;
--   * higher = more cells scanned  = slower, higher recall;
--   * probes >= lists probes every cell and reduces EXACTLY to the
--     flat exact scan (recall ceiling, no latency win).
-- Clamped to [1, lists] at scan time. No effect on flat (lists = 0)
-- indexes, or on an IVF index degraded to flat by VACUUM swap-remove
-- (those always scan the whole corpus). Start near sqrt(lists) and
-- sweep against your recall SLO. Default 16 (since v1.22.2; was 8 --
-- the old default capped recall at ~0.40-0.80 depending on corpus,
-- well below a reasonable SLO -- see CHANGELOG.md).
SET turbovec.probes = 16;

-- Per-backend cache size for the prepared turbovec index data.
-- Each entry is the size of the index on disk (codes + scales +
-- ids + blocked + caches + rotation). Default 256 MiB; set to
-- ~ 2x the sum of hot turbovec indexes you query in one session.
-- ALSO the budget that turbovec.out_of_core = auto compares the
-- index codes against (see below).
SET turbovec.cache_size_mb = 256;

-- Out-of-core IVF serving (v1.13.0+): serve an IVF index larger
-- than RAM by caching only bounded metadata (coarse centroids,
-- cell directory, rotation, codebook, per-slot scales/ids), then
-- per query gathering ONLY the probed cells' contiguous code ranges
-- through PostgreSQL's buffer manager into a compact throwaway
-- sub-index. The per-backend resident set is then
-- O(probes * cell_size) instead of O(n) -- only the probed cells'
-- pages are read; the buffer manager + OS cache hold hot pages,
-- cold pages are read on demand. This is THE mechanism that lets a
-- >RAM IVF index be queried at all (pairs with the v1.12.0
-- out-of-core BUILD). All index data is read through the buffer
-- cache (no relfile mmap; see docs/BUFFER_CACHE_ONLY_DESIGN.md).
--
--   * auto (DEFAULT): cell-scoped only when the index codes exceed
--     0.5 * cache_size_mb -- i.e. the index that actually needs
--     out-of-core. An IVF index that fits the cache budget loads
--     whole (no per-query gather/reblock cost). This is the right
--     default: no latency tax on in-RAM indexes, automatic >RAM
--     serving when the index is large.
--   * on: always cell-scoped (pays the per-query reblock CPU even
--     on small indexes). Use only to force the memory bound.
--   * off: always load the whole index into a per-backend Arc
--     (lowest warm latency once cached, but O(n) resident -- must
--     fit in RAM). The pre-v1.13.0 behaviour.
--
-- Tradeoff: cell-scoped serving pays a per-query gather + reblock
-- of the probed cells (measured ~2.4x the warm whole-load p50 on a
-- cache-resident index), in exchange for an O(probes/lists)
-- resident set. auto only pays that when the index is too large to
-- keep whole, which is exactly when you need it. No effect on flat
-- (lists = 0) or vacuum-degraded indexes: they have no cells to
-- scope and always load whole (still O(n)-resident -- use IVF,
-- WITH (lists = N), for a >RAM corpus).
SET turbovec.out_of_core = auto;

-- turbovec.mmap_static_blocked was REMOVED in v1.22.0 (it had been
-- a deprecated no-op since v1.19.0, when pg_turbovec's relfile mmap
-- was deleted). All index data is read through PostgreSQL's
-- shared-buffer cache (ReadBufferExtended); there is no GUC to
-- toggle. Size shared_buffers to hold the hot (compressed) index for
-- best cold-fill latency -- pg_turbovec's 7-15x compression is what
-- makes "the index fits shared_buffers" achievable where fp32 HNSW
-- could not. See docs/BUFFER_CACHE_ONLY_DESIGN.md.

-- Normalise embeddings on insert. Useful if your embedding
-- producer doesn't normalise; lets you use cosine distance
-- without an explicit l2_normalize() call. Default off.
SET turbovec.normalize_on_insert = off;

-- Iterative index scan (v1.8.0+). With a selective WHERE filter +
-- ORDER BY emb <=> q LIMIT k, the executor post-filters the
-- index's candidates and can under-return if a single search_k
-- batch doesn't contain k matching rows. relaxed_order re-runs the
-- search with a doubled k and feeds the new candidates until the
-- LIMIT is satisfied or the cap below is hit; the reorder queue
-- keeps results exactly distance-ordered. off (the **default since
-- v1.20.1**) restores the pre-v1.8.0 single-batch behaviour: faster
-- by ~450x on an UNFILTERED ORDER BY (measured SIFT-1M/128d IVF:
-- ~2ms vs ~900ms), because PostgreSQL's reorder queue can only
-- return a tuple early when the AM's advertised distance is exact;
-- since we advertise NEG_INFINITY (opclass-agnostic safety) it
-- never is, so under relaxed_order the executor drove the AM's OWN
-- refill schedule to its cap (max_probes/max_scan_tuples) on EVERY
-- query regardless of LIMIT. off may under-return under a
-- SELECTIVE WHERE filter -- opt into relaxed_order for that case.
-- pgvector parity: mirrors hnsw.iterative_scan (which also defaults
-- off; strict_order is future work).
SET turbovec.iterative_scan = off;  -- off (default) | relaxed_order

-- Hard ceiling on candidates examined per iterative scan. Matches
-- pgvector's hnsw.max_scan_tuples (default 20000). Only consulted
-- when turbovec.iterative_scan != off. Raise for very selective
-- filters over large indexes; lower to bound worst-case scan work.
SET turbovec.max_scan_tuples = 20000;

-- IVF probe-widening cap (v1.10.0+, only for indexes built WITH
-- (lists = N)). Under iterative_scan = relaxed_order, when a
-- selective WHERE filter drains the cells currently probed and the
-- executor still wants tuples, the refill WIDENS the probe set
-- (probes, 2*probes, 4*probes, ...) and re-runs the cell-restricted
-- search, instead of only growing k within the initial cells. This
-- recovers true neighbours whose cell was NOT in the initial
-- `probes` nearest set -- the failure mode plain k-growth can't fix,
-- because those rows live in cells that were never scanned.
-- max_probes is the IVF analogue of ivfflat.max_probes: it caps that
-- widening at min(max_probes, lists). Clamped to lists at scan time.
-- No effect on flat (lists = 0) or vacuum-degraded indexes (no cells
-- to widen; they keep the k-growth refill). turbovec.max_scan_tuples
-- still caps total candidate work as a backstop. Default 64 (8x the
-- probes default).
--
-- The recall-knob model: `probes` is the primary IVF dial (it sets,
-- and iterative refill widens, the CELL set); `search_k`/`oversample`
-- set the candidate count WITHIN the probed cells; `max_probes` and
-- `max_scan_tuples` are the caps on those two axes respectively.
SET turbovec.max_probes = 64;

-- Oversampling (differentiator #5): tunable recall lever. The scan
-- fetches ceil(search_k * oversample) candidates ranked by the
-- lossy 2-4 bit quantized distance, then the executor's reorder
-- queue (xs_recheckorderby) re-ranks them by EXACT full-precision
-- distance and the LIMIT trims to the true top-k. Widening the
-- candidate set recovers true neighbours the quantized ranking
-- placed just outside search_k, so quantization stops being a fixed
-- accuracy point and becomes a tunable recall frontier. Matches
-- Qdrant's `oversampling` and VectorChord's rerank knob.
-- 1.0 (default) = no oversampling = pre-feature behaviour.
-- Measured (4-bit, 3000x64, search_k=10): recall@10 climbs
-- 0.81 -> 0.96 -> 0.99 -> 1.0 at oversample 1.0/1.5/2.0/4.0, with
-- p50 latency rising ~ linearly (3.8 -> 4.7 ms). Composes with
-- iterative_scan: this sets the INITIAL k, iterative refill grows
-- it from there. NOTE: there is no separate `turbovec.rescore` GUC;
-- oversampling plus the always-on reorder queue together ARE the
-- rescore mechanism (the reorder queue already re-ranks every
-- returned tuple by exact distance).
SET turbovec.oversample = 1.0;  -- 1.0 .. 100.0
```

### `turbovec.allowlist`

```sql
-- Phase C operator-path allowlist: a per-query, pre-materialized
-- row set the ORDER BY index scan restricts to, with the same
-- in-kernel 32-vector-block short-circuit pushdown turbovec.knn(
-- ..., allowed) gives -- now on the operator path AND on IVF
-- indexes (cell-scope AND allowlist). The index AM keys vectors by
-- heap TID (NOT your id column), so this is a CSV of heap TIDs
-- encoded as bigint via (block << 32) | offset; build it from ctid:
SELECT set_config('turbovec.allowlist',
  (SELECT string_agg(
     ( (split_part(btrim(ctid::text,'()'),',',1)::bigint << 32)
       | split_part(btrim(ctid::text,'()'),',',2)::bigint )::text, ',')
   FROM items WHERE tenant_id = 5),   -- your selective filter -> TIDs
  false);
SELECT id FROM items ORDER BY emb <=> $1 LIMIT 10;
RESET turbovec.allowlist;
-- Whitespace tolerated; empty tokens ignored. SET it before the
-- query and RESET it after; a leftover value silently restricts
-- later queries in the same session. Empty / unset (the default)
-- is unfiltered with ZERO added cost. A non-integer token ERRORs
-- the scan. NOT arbitrary-WHERE pushdown: the AM honours a
-- pre-materialized TID set, it never interprets scan keys. Only
-- worth it when the row set is SELECTIVE (<= ~7-10% of the corpus);
-- see docs/FILTERING.md sect 3.5. For id-column (primary-key)
-- ergonomics on a flat index, use turbovec.knn(..., allowed) instead.
```

---

## Operational tuning

### `shared_buffers`

The pgrx test cluster default is 128 MiB; production should run with
`shared_buffers = 25–40% of RAM`. For `pg_turbovec` specifically:

- **All index data is read through `shared_buffers`** (the buffer
  manager). As of v1.19.0 there is no relfile mmap; size
  `shared_buffers` to hold the hot index so cold cache-fills stay
  fast. pg_turbovec's **7–15× compression** is what makes this
  practical: the index that must fit `shared_buffers` is the
  *compressed* one, not the fp32 corpus.
- For a 10M × 1536-d × 4-bit index (~15 GiB), size `shared_buffers`
  to hold it if you query frequently (e.g. `shared_buffers = 24 GiB`
  on a 64 GiB host). For a >RAM index, use **IVF** (`WITH (lists =
  N)`) + `turbovec.out_of_core` so only the probed cells' pages are
  read — the resident set is then O(probes/lists) of the index, not
  the whole thing.
- Warm queries never touch the buffer manager (the prepared index is
  cached per-backend); `shared_buffers` sizing only affects cold
  cache-fill latency. Lower `shared_buffers` will not corrupt
  anything; it'll just make cold-fill p50 noisier as buffer-manager
  evictions force refills.

### `maintenance_work_mem`

`CREATE INDEX` and `REINDEX` use this for the heap-scan staging buffer
(v1.6.0+ Phase W streaming change). Set to **at least 1 GiB** before
large index builds; v1.6.0+ caps the staging buffer at
`min(maintenance_work_mem * 0.75, 1 GiB)`.

```sql
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 16;
CREATE INDEX docs_emb_idx ON docs USING turbovec ...;
```

Peak `CREATE INDEX` memory at 10M × 1536-d × 4-bit on v1.7.1 is ~22.5
GiB. Hosts with less than ~32 GiB free RAM may need to reduce the
corpus size or build in batches via `aminsert` (slower per-row but
bounded memory).

---

## Filtering

Three working metadata-filter patterns; pick by filter shape. Full
guide, decision matrix, and the measured allowlist crossover:
[`docs/FILTERING.md`](FILTERING.md).

- **Partial index** — the default for known, low-cardinality filters
  (`CREATE INDEX ... USING turbovec (...) WHERE tenant_id = X`). PG
  pushes the predicate natively; the index is smaller and the scan is
  exact. No GUC needed.
- **In-kernel allowlist** — `turbovec.knn(..., allowed bigint[])` for
  a selective per-query id set. The kernel skips 32-vector blocks with
  zero allowed slots, so a selective filter gets *cheaper* (crossover
  ~7–10% selectivity on AVX2). Flat-only (no IVF); use it when the
  filter is genuinely selective — a non-selective allowlist is slower
  than plain ANN.
- **Iterative scan** — `turbovec.iterative_scan = relaxed_order` +
  `turbovec.max_scan_tuples` for the `ORDER BY emb <=> q LIMIT k` AM
  path with a moderately selective `WHERE`. The AM widens the
  candidate set so the executor's post-filter still returns `k`,
  bounded by `max_scan_tuples`.

**Limitation:** there is no true in-traversal pushdown on the `ORDER
BY` AM path — the index returns distance-ranked candidates and the
executor rechecks the `WHERE` (the index stores only vector codes +
TID, no payload columns). See [`docs/FILTERING.md`](FILTERING.md) § 6.

---

## Hybrid & multivector search

Three query-layer patterns, all additive SQL surface (no wire-format
or index-AM change). Full guide with worked CTEs:
[`docs/HYBRID_SEARCH.md`](HYBRID_SEARCH.md).

- **Dense + sparse hybrid (RRF)** — fuse a dense ANN ranking with a
  keyword / full-text (or `sparsevec`) ranking using
  `turbovec.rrf_score(rank, k=60)` = `1/(k+rank)`. Each ranker emits a
  `ROW_NUMBER()` rank; sum the per-ranker `rrf_score`; order by the
  sum. A document ranked highly by both rankers wins.
- **Multivector / late interaction (MaxSim)** — `turbovec.max_sim`
  (dot) / `max_sim_cosine` (cosine) score a (query, doc) pair of
  per-token `vector[]` arrays as ColBERT MaxSim. This is a **re-rank**
  primitive: ANN-retrieve candidates on a pooled vector, then
  MaxSim-rerank the top-N. The token arrays are not indexed; recall is
  bounded by the pooled-ANN recall (index-native late interaction is a
  future phase).
- **Named vectors** — multiple `turbovec.vector` columns per row, one
  `turbovec` index each, fused at query time with RRF.

---

## Replication and standbys

`pg_turbovec` indexes are crash-safe and replicate cleanly:

- All page mutations go through `GenericXLog` → standard PG WAL.
- `ambuild` + `aminsert` + `ambulkdelete` are all WAL-logged.
- **All index data is read through PostgreSQL's buffer manager**
  (`ReadBufferExtended`) — no direct relfile `mmap`/`pread`. This is
  the correct posture for hot standbys, managed/sandboxed Postgres,
  and any environment that restricts direct file access; the buffer
  manager is the single source of truth for page reads.
- The per-backend cache (`turbovec.cache_size_mb`) is process-local,
  so primary and standby backends maintain independent caches; no
  shared-memory invalidation hazards.
- Logical replication: `pg_turbovec` indexes are not replicated by
  logical replication (PG doesn't replicate index DDL via logical
  protocol). The subscriber's table will exist without the index;
  re-create the index on the subscriber via `CREATE INDEX` once the
  initial data sync finishes.

Standby smoke test:

```sql
-- On primary:
CREATE EXTENSION pg_turbovec;
CREATE TABLE t (id bigint PRIMARY KEY, emb vector(8));
INSERT INTO t VALUES (1, '[1,0,0,0,0,0,0,0]'), (2, '[0,1,0,0,0,0,0,0]');
CREATE INDEX t_idx ON t USING turbovec (emb turbovec.vec_cosine_ops);

-- On standby (after replay catches up):
SELECT id FROM t ORDER BY emb <=> '[1,0,0,0,0,0,0,0]'::turbovec.vector LIMIT 1;
-- expect: 1
```

---

## VACUUM behavior

`pg_turbovec` implements `ambulkdelete` via swap-remove on the codes /
scales / ids chains (flat indexes), or via tombstoning (IVF indexes,
`WITH (lists = N)` — see below):

- Deletes shrink the index in place; no orphan pages.
- A subsequent autovacuum run reclaims trailing pages via
  `RelationTruncate` (Phase L hardening item 3).
- The blocked-codes chain is rebuilt lazily on the first scan after
  a delete; this triggers a one-time `pack::repack` cost (~12-15s at
  10M scale) similar to a cold-scan. Subsequent scans are warm.
- VACUUM FULL on a turbovec-indexed table works; the index is
  rebuilt from scratch (matches PG's standard behaviour for index
  AMs that don't implement `amvacuumcleanup`-only paths).

For tables with high delete churn, schedule vacuums against the
parent table normally. `pg_turbovec` does not have a separate
maintenance command.

### IVF indexes and VACUUM (`WITH (lists = N)`)

IVF indexes store their codes in **cell-contiguous** order: each
coarse cell occupies a contiguous slot range recorded in the cell
directory. The flat swap-remove above would move the last slot into a
deleted hole, crossing cell boundaries and breaking that layout.
Before v1.10.2 the vacuum path "fixed" this by blanking the IVF
metadata, which **silently degraded the index to an O(n) flat scan**
— a query that took tens of milliseconds suddenly took seconds, with
no signal to the operator. For a churning multi-million-row index
this was a production latency landmine.

**As of v1.10.2, IVF indexes survive VACUUM.** Instead of
swap-removing, the vacuum path **tombstones** deleted slots:

- The dead slot is left in place; nothing moves. `n_vectors`, the
  cell directory, and the coarse centroids stay valid, so the index
  keeps serving cell-restricted (fast) scans — `has_ivf()` stays
  true.
- The dead slot is recorded in a per-slot tombstone bitmap (a small
  on-disk chain, ~`n_vectors / 8` bytes). At scan time the
  tombstoned slots are masked out of the cell-restriction mask, so a
  deleted row is **never returned**, even though its bytes remain on
  disk until the next REINDEX.
- Dead space accumulates with churn. A `REINDEX INDEX <name>;`
  compacts the tombstones and re-clusters. Schedule a REINDEX when
  the dead fraction grows large (e.g. >20-30% of rows deleted since
  the last build) — the same cadence you would use for B-tree bloat.

Flat indexes (`lists = 0`, the default) are unaffected: they still
swap-remove, which has no contiguity invariant to protect.

### Detecting a degraded IVF index

With tombstone vacuums a healthy IVF index should never degrade. As a
safety net, any path that does invalidate the IVF metadata now leaves
the degradation **observable** instead of silent:

- A throttled scan-time `WARNING` fires once per backend per index:
  `turbovec index "<name>" was built WITH (lists > 0) but has
  degraded to a flat scan after VACUUM` with a `HINT: REINDEX INDEX
  ...`.
- A queryable signal: `turbovec.index_is_degraded(regclass)` returns
  `true` for an IVF index that has degraded to flat, `false`
  otherwise. Poll it from monitoring:

  ```sql
  SELECT c.relname,
         turbovec.index_is_degraded(c.oid) AS degraded
  FROM pg_class c
  JOIN pg_am a ON a.oid = c.relam
  WHERE a.amname = 'turbovec';
  ```

  A `degraded = true` row means: `REINDEX INDEX <name>;` to restore
  IVF (cell-restricted) query performance.

---

## Monitoring

Useful queries:

```sql
-- Index storage breakdown
SELECT
    relname AS index_name,
    pg_size_pretty(pg_relation_size(c.oid)) AS size,
    pg_relation_size(c.oid) AS size_bytes
FROM pg_class c
JOIN pg_index i ON c.oid = i.indexrelid
JOIN pg_class h ON i.indrelid = h.oid
WHERE c.relam = (SELECT oid FROM pg_am WHERE amname = 'turbovec');

-- Per-backend cache hit visibility
-- (Currently no built-in observability; the cache lives in the
-- backend's process heap. Use a debug build with the Phase U-1
-- tracepoint pattern if you need cache-miss instrumentation.
-- Production builds intentionally don't expose cache stats per
-- the "don't ship debug tracepoints" decision.)

-- Confirm wire format version of an existing index
SELECT
    pg_relation_filepath(c.oid)
FROM pg_class c
WHERE c.relname = 'docs_emb_idx';
-- Then read the meta page (block 0) byte 8..12 for the version
-- (currently 3, set in v1.4.0; see docs/UPGRADING.md).
```

---

## Operational gotchas

### `cblas_sgemm` undefined symbol

If the postmaster fails to load `pg_turbovec.so` with:

```
ERROR:  could not load library "...pg_turbovec.so": .../pg_turbovec.so:
        undefined symbol: cblas_sgemm
```

your build linked against OpenBLAS but the runtime linker can't find
`libopenblas.so` in the postmaster's `LD_LIBRARY_PATH`. Two fixes:

1. **`LD_PRELOAD` on the postmaster** (works everywhere):
   ```bash
   LD_PRELOAD=/path/to/libopenblas.so.0 pg_ctl -D ... start
   ```
2. **System-wide `ldconfig`** (Linux distros): place a `.conf` file
   in `/etc/ld.so.conf.d/openblas.conf` with the path to the
   OpenBLAS lib dir, then `ldconfig`.

This is needed on Ubuntu/Debian and other distros where OpenBLAS
isn't on the default loader path. NixOS gets it automatically via
the build-time `RUSTFLAGS="-L /nix/store/.../lib"`.

### Don't `kill -9` the postmaster

`UNLOGGED` tables are truncated on crash recovery. If you have
`pg_turbovec` indexes on `UNLOGGED` tables (common for vector caches
that can be regenerated), a `kill -9` of the postmaster will lose
the data + indexes on next startup. Always use `pg_ctl stop -m fast`
or `-m smart`.

### Codeberg HTTPS endpoint flakiness

If you `git fetch` from `https://codeberg.org/gregburd/pg_turbovec`
and get 504s, fall back to the GitHub mirror at
`https://github.com/gburd/pg_turbovec`. The release tags are
identical on both.

### Cache eviction on long-lived backends

The `enforce_cap` rule (`len() > 1`) keeps the most-recently-used
entry resident even when total cache size exceeds
`turbovec.cache_size_mb`. For a single-index workload, this means
the cache effectively never evicts the active entry. If you have
multiple turbovec indexes queried by one backend, set
`cache_size_mb` to ≥ 2× the largest one or you'll see cold-scan
latency on every cross-index switch.

---

## Upgrading

See [`UPGRADING.md`](UPGRADING.md) for the full migration matrix.
Quick reference:

```sql
-- Patch upgrade (e.g. 1.7.0 → 1.7.1): wire format frozen.
-- Just install the new .so and:
ALTER EXTENSION pg_turbovec UPDATE TO '1.7.1';

-- Minor upgrade (e.g. 1.4.x → 1.5.x → 1.6.x → 1.7.x): wire format
-- compatible across all of v1.4.0+. ALTER EXTENSION suffices.
ALTER EXTENSION pg_turbovec UPDATE TO '1.7.1';

-- Cross-major-wire-format upgrade (e.g. v1.0/1/2/3 → v1.4+):
-- the binary detects the legacy format at scan time and ERRORs
-- with a HINT: REINDEX. Run:
REINDEX INDEX CONCURRENTLY docs_emb_idx;
-- (use CONCURRENTLY if downtime is unacceptable; this avoids
-- holding an AccessExclusiveLock on the table).
```

The binary refuses to scan an out-of-date index — it ERRORs cleanly
rather than silently misbehaving. So you can safely deploy a new .so,
restart, and let the first ANN query tell you whether REINDEX is
needed.

---

## Troubleshooting

### "type modifier is not allowed for type turbovec.vector"

You wrote `vector(N)` somewhere `pg_turbovec`'s vector type doesn't
take a typmod. Use `turbovec.vector` without a parenthesised
dimension. The dim is fixed by the column's actual values + the
index's `bit_width` reloption.

### "could not access status of transaction" or "MultiXactId ... has not been created yet"

These are PG-internal errors not specific to `pg_turbovec`. Usually
indicates an inconsistent `pg_clog`/`pg_subtrans` or an interrupted
upgrade. Treat as you would for any PG cluster.

### Recall is lower than expected

- Confirm `turbovec.search_k` is set high enough. Recall@10 ≈ 0.95 at
  k=100 on typical OpenAI embeddings; raise to 500 or 1000 for higher
  recall. Latency scales linearly.
- Confirm embeddings are normalised (`l2_normalize`) if using cosine
  distance. The TurboQuant kernel assumes unit-norm input for cosine.
- Confirm `bit_width` is appropriate for your distribution. 4-bit is
  the default; 8-bit gives 1-2 percentage points better recall at 2×
  the storage cost.

### Build OOMs at scale

Set `maintenance_work_mem` to ≤ 1 GiB explicitly. v1.6.0+ caps the
staging buffer at `min(maintenance_work_mem * 0.75, 1 GiB)`. If you
inherited a session that set `maintenance_work_mem = '64GB'`, the
staging buffer is still capped at 1 GiB but the IdMapIndex's
`packed_codes` Vec grows unbounded — at 10M × 1536-d × 4-bit, that's
~7.7 GiB on its own.

The current peak RSS at 10M × 1536-d × 4-bit on `meh` is ~22.5 GiB.
Hosts with ≥ 32 GiB free RAM build cleanly. For tighter hosts,
batch via `aminsert` (slower but bounded).

### Cold scan latency unacceptable

This is the known weak point. After the first scan in a backend, the
`pack::repack` cost (~10-15 s at 1M scale, ~100-150 s at 10M scale)
is paid once and cached. Mitigations:

- Pre-warm the cache on backend startup with a dummy query.
- Use connection pooling so warm scans dominate.
- Size `shared_buffers` to hold the hot (compressed) index so cold
  cache-fills are served from the buffer cache rather than disk.

---

## Known issues

### Pre-AVX2 x86_64 wrong-results bug (FIXED in v1.7.3)

**Status:** root cause found 2026-06-15; **fixed in v1.7.3** via the
turbovec fork upgrade to v0.9.0.

**Symptom:** On x86_64 CPUs **without AVX2** (e.g. Intel
Ivy Bridge / Sandy Bridge Xeons, pre-2014), the turbovec index
AM's `ORDER BY emb_expr <=> probe LIMIT N` returned the same
`id` N times instead of the top-N nearest neighbours. First
observed on the 10 M × 1024-d Cohere wikipedia bench on `meh`
(an Intel Xeon E5-2697 v2, `avx` but no `avx2`).

**Root cause:** NOT a pg_turbovec bug. The pinned turbovec
v0.7.0 (`6e80a59`) had a kernel bug where the pre-AVX2 x86_64
scalar fallback read the perm0-interleaved (FAISS-style) SIMD
code layout as if it were sequential, producing silently-wrong
top-k. CPUs with AVX2/AVX-512 took the correct SIMD path and
were never affected — which is why it never reproduced on
AVX2 dev hosts or `arnold`, only on the pre-AVX2 `meh`.
Upstream turbovec fixed this in PR #108 (issue #106), released
in v0.8.0, with a proper `score_query_into_heap` scalar
fallback and a `FORCE_SCALAR_FALLBACK` regression test.

**Fix:** upgrade the turbovec fork to v0.9.0 (which contains the
fix). Shipping as **v1.7.3**. Wire format unchanged; no REINDEX.

**Affected users / workaround until v1.7.3:**
- Only pre-AVX2 x86_64 hosts are affected. Check with
  `grep -o avx2 /proc/cpuinfo | head -1` — if empty, you're on
  a pre-AVX2 CPU and should wait for v1.7.3 or force seq scans
  (`SET enable_indexscan = off;`).
- AVX2 (Haswell 2013+), AVX-512, and ARM NEON hosts are
  unaffected; the SIMD path was always correct there.

See `docs/RECALL.md § 2.9` for the bench artefact that
surfaced this.

## Reporting bugs

`pg_turbovec` is developed at:

- Primary: https://codeberg.org/gregburd/pg_turbovec
- Mirror: https://github.com/gburd/pg_turbovec

Issues, bug reports, and pull requests welcome on either. For bug
reports include:

- `pg_turbovec` version (`SELECT extversion FROM pg_extension WHERE
  extname='pg_turbovec';`).
- PostgreSQL version (`SELECT version();`).
- The exact `CREATE INDEX` statement that built the affected index.
- A minimal reproducer if at all possible.
- Output of `EXPLAIN (ANALYZE, BUFFERS)` for the failing query.

For performance regressions specifically, also include:
- `shared_buffers`, `maintenance_work_mem`, `turbovec.search_k`,
  `turbovec.cache_size_mb` settings.
- Approximate corpus size (`SELECT count(*), avg(array_length(emb::real[],1))
  FROM your_table;`).
- Index storage (`SELECT pg_size_pretty(pg_relation_size('your_idx'));`).
