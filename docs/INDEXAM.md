# Implementing the `turbovec` Index Access Method

> **Status:** Phase 4 scaffold. Builds only when the experimental
> Cargo feature is enabled; *not* exercised by the v0.3 test suite.
> Pick this up in a session that has access to a running Postgres
> dev cluster.

## TL;DR

```bash
# Default build excludes the AM scaffold:
cargo build

# Enable the experimental AM:
cargo build --features experimental_index_am

# Test it (requires cargo-pgrx):
cargo pgrx test pg17 --features experimental_index_am
```

## Why a Cargo feature gate?

The IndexAmRoutine implementation is several hundred lines of
`unsafe extern "C-unwind"` FFI. It interacts with Postgres lock
manager, snapshot machinery, and memory contexts — failure modes
include backend crashes and (worst case) heap-page corruption. The
v0.3 default build deliberately excludes this code so:

- Continuous integration stays green on every push.
- Users who don't opt in cannot accidentally `CREATE INDEX ... USING
  turbovec` and trip an unfinished code path.
- The code is still in tree, reviewable, and ready to enable as soon
  as it is validated against a real cluster.

## Module map (`src/index/`)

```
src/index/
├── mod.rs         # IndexAmRoutine builder + handler entry point
├── options.rs     # bit_width / dim reloption parser (amoptions callback)
├── opclass.rs     # extension_sql! for tvector_ip_ops + AM declaration
├── persist.rs     # SPI helpers for the turbovec.am_storage side table
├── build.rs       # ambuild + ambuildempty
├── insert.rs      # aminsert
├── scan.rs        # ambeginscan / amrescan / amgettuple / amendscan
├── vacuum.rs      # ambulkdelete + amvacuumcleanup
├── cost.rs        # amcostestimate
└── validate.rs    # amvalidate
```

## Storage strategy: side table

```sql
CREATE TABLE turbovec.am_storage (
    indexrelid  oid PRIMARY KEY,
    bit_width   int4 NOT NULL,
    dim         int4 NOT NULL,
    n_vectors   int8 NOT NULL,
    payload     bytea NOT NULL,
    version     int4 NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
```

- `payload` is the bytes produced by `IdMapIndex::write` (TVIM
  format).
- We **do not** store the index in the index relation's main fork.
  Phase 5 will move to relfile-resident pages.
- Each `aminsert` reads the current payload, deserialises, inserts
  via `IdMapIndex::add_with_ids`, serialises, writes back. Slow but
  safe. Phase 5 will introduce per-`indexrelid` mutex + lazy flush.

## Callback responsibilities (Phase 4 minimum)

| Callback              | Phase 4 implementation |
|-----------------------|------------------------|
| `ambuild`             | `IndexBuildHeapScan` on the heap relation; for each row produce a u64 from the heap TID and a Vec<f32> from the column; build `IdMapIndex`; serialise; INSERT into `am_storage`. |
| `ambuildempty`        | INSERT empty payload. |
| `aminsert`            | Load index from `am_storage`, `add_with_ids`, serialise, UPDATE. |
| `ambeginscan`         | `palloc` an `IndexScanDesc`; attach Rust-side `ScanOpaque` via `pg_sys::palloc0` cast. |
| `amrescan`            | Reset cursor; capture orderby key into `ScanOpaque.query`. |
| `amgettuple`          | On first call: load index, run `IdMapIndex::search`; cache results. On subsequent calls: pop next result, set `scan->xs_heaptid`, return `true`. Return `false` when results are drained. |
| `amendscan`           | `pfree` the `ScanOpaque`. |
| `ambulkdelete`        | For each dead heap TID, `IdMapIndex::remove(tid_to_u64)`; persist. |
| `amvacuumcleanup`     | No-op (Phase 4 has no incremental compaction). |
| `amcostestimate`      | Heuristic: `n_vectors * dim * bit_width / 64.0` for total cost; `0` startup. |
| `amoptions`           | Parse `bit_width` and `dim` reloptions; reject `bit_width ∉ {2,3,4}` and `dim % 8 != 0`. |
| `amvalidate`          | Return `true`. Phase 5 will validate operator class support. |

## Operator class plumbing

```sql
CREATE OPERATOR CLASS tvector_ip_ops
  DEFAULT FOR TYPE tvector USING turbovec AS
    OPERATOR 1 <#> (tvector, tvector) FOR ORDER BY float_ops,
    FUNCTION 1 negative_inner_product(tvector, tvector);

CREATE OPERATOR CLASS tvector_cosine_ops
  FOR TYPE tvector USING turbovec AS
    OPERATOR 1 <=> (tvector, tvector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(tvector, tvector);
```

Strategy 1 = "the order-by operator". `amcanorderbyop = true`,
`amcanorder = false` (we don't provide a total order, only nearest-
to-query ranking).

## Test plan (Phase 5)

Once the scaffold compiles and `cargo pgrx test pg17 --features
experimental_index_am` boots a cluster, the minimum acceptance
suite is:

1. `CREATE INDEX docs_emb_idx ON docs USING turbovec (embedding
   tvector_cosine_ops) WITH (bit_width = 4);` succeeds.
2. `EXPLAIN (ANALYZE) SELECT id FROM docs ORDER BY embedding <=>
   $1 LIMIT 10;` shows `Index Scan using docs_emb_idx`.
3. The same query without the index returns the same top-1 result
   (allow recall slip on top-2..10).
4. `INSERT INTO docs ...` followed by re-running the query
   reflects the new row.
5. `DELETE FROM docs WHERE id = ...` followed by `VACUUM` removes
   the row from the index.
6. `DROP INDEX docs_emb_idx` succeeds and removes the
   `am_storage` row.

## Known risks (read before enabling)

- **Memory-context lifetime.** Pgrx-allocated boxes inside callbacks
  must be transferred into Postgres's CurrentMemoryContext or
  explicitly leaked into a longer-lived context (`PortalContext` for
  scan opaque). Getting this wrong looks like SEGV during scan.
- **Lock interleaving.** `aminsert` runs under an exclusive lock on
  the heap row but only a `RowExclusiveLock` on the index. Two
  concurrent inserts can race on the side table. Phase 5 needs a
  per-`indexrelid` advisory lock, or moves to relfile pages.
- **Crash safety.** Side-table writes are WAL-logged. A crash mid-
  `aminsert` rolls back; the index then has fewer rows than the
  heap until the next `ambuild`. Phase 5 fixes this with bgworker
  reconciliation.
- **`swap_remove` index renumbering.** `IdMapIndex::remove` returns
  the slot vacated; the upstream crate's id→slot map is updated
  automatically, so external consumers (us) only ever see u64 ids.
  We rely on this — do not switch to `TurboQuantIndex::swap_remove`.

## References

- `pgvecto.rs` — production-grade pgrx index AM. Read its
  `src/index/algorithms/` and `src/index/am.rs` for working
  patterns.
- Postgres docs, [Index Access Method
  Interface](https://www.postgresql.org/docs/17/indexam.html).
- pgrx-pg-sys `pg17.rs` — search for `IndexAmRoutine` and the
  `am*_function` typedefs to see the exact ABI.

## Phase 12 known issue: forced-index-scan crashes the backend

The `experimental_index_am` build now passes 37/37 `#[pg_test]` cases
**when the planner picks the index naturally** (i.e. on small/medium
tables where `enable_seqscan = on` keeps it on a sequential scan).
The one case marked `#[ignore]` — `index_am_forced_index_scan` —
sets `enable_seqscan = off` to force the index path and
reproducibly crashes the backend with:

```
munmap_chunk(): invalid pointer
... server process (PID …) was terminated by signal 6: Aborted
```

The crash happens in glibc's `free()` somewhere between when our
`amgettuple` returns and the executor projects the row, regardless
of whether we project the distance or just the `id`. Removing every
write to `xs_orderbyvals` / `xs_orderbynulls` does not help, nor
does `Box::leak`-ing the deserialised `IdMapIndex`.

### Hypothesis

The problem is most likely an **allocator mismatch between turbovec
and the executor's memory contexts**. `turbovec` (and its transitive
deps `faer`, `openblas-src`) use the global Rust allocator
(`jemalloc` or `glibc malloc` depending on platform). The executor's
recheck-orderby path may free memory it did not allocate, or our
`amgettuple` may leak something into the wrong context.

### Workarounds for users

- **Default-plan queries work.** As long as `enable_seqscan` is on
  (the default) and the table is small enough that the planner
  prefers a sequential scan, ORDER-BY queries are fine.
- **For larger corpora**, use the function-driven `turbovec.knn()`
  instead of the index. Same SIMD kernel, no executor-recheck path:

      SELECT k.id, k.score
      FROM   turbovec.knn('docs'::regclass, 'id', 'embedding', $1, 10) k;

### Phase 13 plan

- Reach into PG's executor via gdb to identify the exact `free()`
  call site.
- Likely fix: implement a `turbovec_orderby_distance(...)` SQL
  function returning `float8` directly from u64 ids stored in the
  index, sidestepping the recheck path entirely.
- Or: switch our `amcanorderbyop` to false and expose the AM as
  `amcanorder = true` over a Btree-style strategy on the score —
  costlier but simpler.
