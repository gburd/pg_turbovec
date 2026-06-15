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

-- Per-backend cache size for the prepared turbovec index data.
-- Each entry is the size of the index on disk (codes + scales +
-- ids + blocked + caches + rotation). Default 256 MiB; set to
-- ~ 2x the sum of hot turbovec indexes you query in one session.
SET turbovec.cache_size_mb = 256;

-- Use mmap-resident reads for the deterministic-after-build
-- regions of the relfile (blocked codes + rotation matrix +
-- inline codebook). Default ON in v1.5.0+; turn OFF only if you
-- hit a kernel mmap quirk. The fallback path goes through
-- shared_buffers as in v1.4.x.
SET turbovec.mmap_static_blocked = on;

-- Normalise embeddings on insert. Useful if your embedding
-- producer doesn't normalise; lets you use cosine distance
-- without an explicit l2_normalize() call. Default off.
SET turbovec.normalize_on_insert = off;
```

---

## Operational tuning

### `shared_buffers`

The pgrx test cluster default is 128 MiB; production should run with
`shared_buffers = 25–40% of RAM`. For `pg_turbovec` specifically:

- v1.5.0+ mmap-resident reads of static regions (~30–60% of an index)
  bypass `shared_buffers` entirely. They're served from the OS page
  cache via `mmap MAP_PRIVATE`.
- Mutated regions (codes/scales/ids; v1.5+ keeps these on the buffer
  manager) follow the standard rule: `shared_buffers ≥ 2× sum of all
  turbovec indexes you query in a session`.
- For a 10M × 1536-d × 4-bit index (~15 GiB), that's `shared_buffers
  ≥ 30 GiB` if you query the index frequently. On a 64 GiB host,
  `shared_buffers = 24 GiB` is a sensible production setting.

Lower `shared_buffers` will not corrupt anything; it'll just make
warm-scan p50 noisier as buffer-manager evictions force refills.

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

## Replication and standbys

`pg_turbovec` indexes are crash-safe and replicate cleanly:

- All page mutations go through `GenericXLog` → standard PG WAL.
- `ambuild` + `aminsert` + `ambulkdelete` are all WAL-logged.
- The mmap path on standbys uses `File::open(path).map(MAP_PRIVATE)`,
  which is read-only and works on hot standbys.
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
scales / ids chains:

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
- Future v1.8.0 may close the gap with a wire-format change to
  enable zero-copy mmap reads of the blocked-codes chain.

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
  `turbovec.cache_size_mb`, `turbovec.mmap_static_blocked` settings.
- Approximate corpus size (`SELECT count(*), avg(array_length(emb::real[],1))
  FROM your_table;`).
- Index storage (`SELECT pg_size_pretty(pg_relation_size('your_idx'));`).
