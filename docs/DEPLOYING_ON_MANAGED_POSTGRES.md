# Deploying pg_turbovec on managed / hosted PostgreSQL

This page consolidates the platform-relevant facts for running `pg_turbovec`
on a managed or hosted PostgreSQL service (a service that runs PostgreSQL for
customers, typically with separated storage, physical read replicas, and a
restricted-superuser role model). It is vendor-neutral: it describes the
*properties* the extension provides, not any one platform's internals.

## Durability and replication

- **All index state is WAL-logged via `GenericXLog`.** There are no
  `log_newpage`, `XLogInsert`, `smgrwrite`, `smgrextend`, `PageSetLSN`, or
  `FlushRelationBuffers` call sites — durability goes 100% through the generic
  WAL interface. This is enforced mechanically by `scripts/drift-check.sh`
  (§ 11a), so it cannot silently regress.
- Index state therefore **replicates to physical standbys** like any other
  relation, and standby queries return the same results as the primary.
- **No extension-managed files outside the relation.** All index data lives in
  the index relfile (main fork) and is read through PostgreSQL's shared-buffer
  cache; there is no side-table, no `mmap`, and no on-disk state the platform's
  backup/restore/replication does not already cover.

## Restricted-superuser compatibility

- **No function requires superuser.** The SQL surface is read-only from the
  catalog's point of view: the only index-write paths are the AM callbacks
  (`ambuild` / `aminsert` / `ambulkdelete`), which core never invokes on a read
  replica. The read functions (`turbovec.knn()`, `turbovec.colbert_search()`,
  etc.) enforce table ACLs through SPI.
- **All GUCs are per-session** (`GucContext::Userset`). There is no
  `PGC_SUSET` / `PGC_SIGHUP` / `PGC_POSTMASTER` knob, and no crash-injection or
  debug GUC in a release build. This is enforced by `scripts/drift-check.sh`
  (§ 11b). Every tuning knob is therefore settable both per session and via a
  platform parameter group with no privilege gymnastics.
- The extension installs into its own schema (`turbovec`) and is **not
  relocatable**.

## Read replicas

- Read paths work on replicas: the read functions and the index scan are
  read-only.
- **Space reclaim (relation truncation) is deferred while a reader holds the
  visibility horizon** — expected behavior on any platform with read replicas
  and `hot_standby_feedback` enabled. Truncation proceeds once the horizon is
  released; there is no wrong-results window, only deferred space return.

## Cancellation and timeouts

- `pg_turbovec` polls for interrupts (`CHECK_FOR_INTERRUPTS`) at every point it
  controls where no buffer content lock is held: on each `amgettuple` entry, at
  each `CREATE INDEX` build-stage boundary, and at the top of each
  `GenericXLog` batch in the relfile write path. So `statement_timeout` and
  `pg_cancel_backend()` (and the platform's administrative signals delivered by
  the same mechanism) take effect promptly for the iterative-scan path, builds,
  and large flushes.
- **Known limitation:** a single large *flat-kind* `search()` call is currently
  internally uninterruptible (the whole-corpus distance computation happens in
  one call into the Postgres-free `turbovec` crate). Finer-grained polling
  inside that call needs a block-scoring API from the kernel crate and is
  tracked follow-up work. In practice, prefer the IVF (`WITH (lists = N)`) or
  graph (`WITH (graph = true)`) kinds for large corpora, both because they are
  sublinear and because their scan paths are more finely cancellable.

## Memory and insert cost model

`pg_turbovec` is a build-then-query design, and operators should size for it:

- **A write transaction materializes an in-memory copy of the index.** Peak
  backend memory scales with index size for a transaction that mutates the
  index.
- **`aminsert` is O(n) per transaction, not O(1) per row** (a
  whole-relfile-rewrite at transaction commit). Many small autocommitted
  inserts into a large existing index are slow by design.
- **Recommended ingestion pattern:** bulk-load, then `REINDEX` (or build the
  index after loading), rather than trickle-inserting into a large existing
  index.
- `turbovec.cache_size_mb = 0` disables the per-backend cache (lower memory,
  higher per-query latency on a cold connection).

## Build cost and parallelism

- Parallel index build across maintenance workers is **not** enabled
  (`amcanparallel = false`); the build parallelizes internally via a rayon pool.
- `turbovec.build_parallelism` sizes that pool (default derived from
  `max_parallel_maintenance_workers`); `turbovec.graph_build_partitions` sizes
  the partitioned parallel graph build. Size maintenance windows for large
  corpora accordingly and consult `CHANGELOG.md` / `docs/BENCHMARKS.md` for
  representative build wall-clock at scale.

## Upgrades and wire-format changes

- **Patch releases never change the on-disk format.** `ALTER EXTENSION
  pg_turbovec UPDATE` is always sufficient and cannot fail on existing indexes.
- **A wire-format bump requires `REINDEX INDEX <name>`.** The binary detects a
  pre-format index and `ERROR`s at first scan with a clear `HINT: REINDEX INDEX
  <name>;` — never silent corruption.
- **A duplicate-id corrupt relfile** (which an unclean shutdown / `pg_resetwal`
  can leave) is detected at index-open and `ERROR`s with the same REINDEX hint,
  on both the read and write paths (v1.27.4 / v1.28.2+), rather than silently
  mis-serving reads while failing writes.
- The full version-to-version migration matrix is in `docs/UPGRADING.md`.

## Introspection

- `turbovec.index_is_degraded(regclass)` reports whether an IVF index has
  fallen back to a flat O(n) scan.
- `turbovec.warn_on_rebuild` surfaces the per-backend one-time costs the v7
  codes-deduplication design incurs (the `pack::repack` recompute at
  index-open).

## Supported PostgreSQL versions

PostgreSQL 13–18 are fully supported; 19 (beta) is experimental on the 1.28.x
line. See `docs/PG_VERSION_SUPPORT.md`. On a managed platform that packages
`cargo-pgrx` only up to 0.18.x, the **1.27.x line** (pinned to pgrx 0.17.0)
builds with the platform's stock `cargo-pgrx` and carries the same
correctness fixes; the 1.28.x line (pgrx 0.19.1 / Rust 1.96) adds PG19.
