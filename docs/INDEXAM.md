# The `turbovec` Index Access Method (design doc)

> **Status (v1.3.0):** the `turbovec` index AM is default-on
> and the relfile-resident page format is the only storage
> strategy. The `experimental_index_am` and `relfile_storage`
> Cargo features were retired in Phase Q; the historical bits
> below are preserved for context but the build instructions
> have been brought up to date.

## TL;DR

```bash
# Default build includes the AM:
cargo build

# Test it (requires cargo-pgrx):
cargo pgrx test pg<N>

# Stripped-down build without the AM (no .so footprint for AM
# scan/insert/build code):
cargo build --no-default-features --features pg<N>
```

## Historical note: why a Cargo feature gate (and why it's gone)

The `IndexAmRoutine` implementation is several hundred lines of
`unsafe extern "C-unwind"` FFI. It interacts with the Postgres
lock manager, snapshot machinery, and memory contexts. v0.3..v0.8
shipped it behind a Cargo feature so users couldn't accidentally
`CREATE INDEX ... USING
  turbovec` and trip an unfinished code path.
- The code is still in tree, reviewable, and ready to enable as soon
  as it is validated against a real cluster.

## Module map (`src/index/`)

```
src/index/
├── mod.rs         # IndexAmRoutine builder + handler entry point
├── options.rs     # bit_width / dim reloption parser (amoptions callback)
├── opclass.rs     # extension_sql! for vec_ip_ops + AM declaration
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
CREATE OPERATOR CLASS vec_ip_ops
  DEFAULT FOR TYPE vector USING turbovec AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 negative_inner_product(vector, vector);

CREATE OPERATOR CLASS vec_cosine_ops
  FOR TYPE vector USING turbovec AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vector, vector);
```

Strategy 1 = "the order-by operator". `amcanorderbyop = true`,
`amcanorder = false` (we don't provide a total order, only nearest-
to-query ranking).

## Test plan (Phase 5)

Once the scaffold compiles and `cargo pgrx test pg17 --features
experimental_index_am` boots a cluster, the minimum acceptance
suite is:

1. `CREATE INDEX docs_emb_idx ON docs USING turbovec (embedding
   vec_cosine_ops) WITH (bit_width = 4);` succeeds.
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

## Phase 18: forced-index-scan crash, fixed

For the entire v0.4..v1.0.0-rc.1 run, the `index_am_forced_index_scan`
test case (`SET enable_seqscan = off; SELECT ... ORDER BY emb <=> q
LIMIT k`) reliably aborted the backend with:

```
munmap_chunk(): invalid pointer
... server process (PID …) was terminated by signal 6: Aborted
```

We chased a long list of red herrings — `xs_orderbyvals` allocation,
`Box::leak` lifetime tweaks, `xs_recheckorderby = true/false`,
allocator mismatches, etc. The actual bug was a one-liner in
`src/index/scan.rs::amrescan`:

```rust
// BUG (v0.4 .. v1.0.0-rc.1):
std::ptr::copy_nonoverlapping(
    orderbys,
    (*scan).orderByData,
    (norderbys as usize) * std::mem::size_of::<pg_sys::ScanKeyData>(), // wrong unit
);
```

`std::ptr::copy_nonoverlapping::<T>(src, dst, count)` takes `count`
in **elements of T**, not bytes. We were therefore copying
`norderbys * sizeof(ScanKeyData)` `ScanKeyData` *elements* into a
slot sized for `norderbys` — a buffer overrun of roughly
`sizeof(ScanKeyData)` × the requested size. That smashed the
`IndexScanDesc` and adjacent heap chunks; the actual `free()` that
tripped glibc's `munmap_chunk()` happened much later, when the scan
context was torn down and an unrelated chunk's metadata got walked.

The other 39 tests never tripped it because the planner kept
small-table ORDER BY queries on a sequential scan and `amrescan`
was never called with `norderbys > 0`. Forcing the index via
`enable_seqscan = off` was the only way to reach the buggy
codepath.

### Fix

```rust
std::ptr::copy_nonoverlapping(orderbys, (*scan).orderByData, norderbys as usize);
std::ptr::copy_nonoverlapping(keys,     (*scan).keyData,     nkeys     as usize);
```

Once the corruption was gone, the next layer of the executor
turned out to need real values in `xs_orderbyvals[0]`. The
reorder-queue path in `IndexNextWithReorder` (PG 16
`nodeIndexscan.c`) compares the AM's claimed distance against the
recomputed exact distance and `elog(ERROR, "index returned tuples
in wrong order")` if the recompute is *less than* what the AM
claimed. Setting `xs_orderbynulls[0] = true` makes the comparator
return -1 and trips that error.

We therefore write `f64::NEG_INFINITY` into `xs_orderbyvals[0]`
on every `amgettuple` — a universal lower bound that's safe for
cosine, inner-product and any future distance metric. Every tuple
goes through the reorder queue and is drained in exact order at
end-of-scan; we cap at `k = 1024` results per scan, so the queue
overhead is negligible.

### Lessons

1. **Buffer overruns in `copy_nonoverlapping` are silent until
   they're loud.** The crash was nowhere near the actual write;
   it was wherever glibc next touched the smashed arena chunk.
2. **`amcanorderbyop = true` requires monotone-or-lower-bound
   `xs_orderbyvals`.** NULLs are not safe under `cmp_orderbyvals`.
3. **Default-plan queries hide forced-plan bugs.** Always include
   one `enable_seqscan = off` test case per orderby AM.

### Workarounds (no longer needed; kept for the historical record)

- v0.4..v1.0.0-rc.1 users who hit the crash were advised to use
  the function-driven `turbovec.knn()` API instead. That still
  works identically; the index AM is now also safe.

      SELECT k.id, k.score
      FROM   turbovec.knn('docs'::regclass, 'id', 'embedding', $1, 10) k;

## Phase 14+ roadmap

### CREATE INDEX CONCURRENTLY support (Phase 14)

Postgres lets users build indexes without blocking writes via
`CREATE INDEX CONCURRENTLY`. The AM contract is that `ambuild` is
called twice:

1. First pass with a snapshot taken at the start, while writers
   continue. The result must include every row visible at that
   snapshot.
2. Second pass under a stricter snapshot, validating that no row
   inserted by writers between the two passes was missed. (PG's
   built-in machinery does the diff via `validate_index`.)

For our side-table-persisted AM, the requirements are:

- `ambuild` must be **idempotent**: running it twice over the same
  heap state must produce the same `am_storage` row. Our current
  implementation uses `INSERT ... ON CONFLICT (indexrelid) DO
  UPDATE`, so this is already true.
- `ambuild` must respect the snapshot it is given. We currently
  walk the heap via `index_build_range_scan`, which uses the
  scan's snapshot — already correct.
- We need `amcanorderbyop = true` (we have it) and we should
  *not* set `ampredlocks` (we don't).
- `aminsert` for in-flight inserts during the build window must
  arrive at the right `indexrelid`. PG drives this; nothing for us
  to add.

**Status:** untested. Phase 14 deliverable is

```rust
#[pg_test]
fn cic_concurrent_with_writes() {
    // Spawn a bgworker / background INSERT loop, kick off
    // CREATE INDEX CONCURRENTLY in the test, verify the final
    // index reflects every row including those inserted during
    // the build.
}
```

Also need to advertise CIC support in `IndexAmRoutine` — PG
automatically allows CIC when `amcanorder = false` and
`amclusterable = false`, both of which we have. So the SQL surface
should Just Work; the test is the deliverable.

### Other Phase 14+ items

- **Binary-compatible `vector` varlena layout** — replace the
  v0.x CBOR-derived storage with the pgvector-compatible
  `[i32 vl_len_, i16 dim, i16 unused, f32[dim]]` layout. Adds
  zero-copy casts to/from `pgvector.vector` when both extensions
  are installed and reduces storage overhead by ~10–15%. Pure
  data-layout change; does not need pgrx index AM expertise.
- **Parallel `ambuild`** via
  `index_build_range_scan(parallel = true)` and a shared
  `BuildState`. Material wins on multi-million-row builds.
- **Recall benchmark harness** — `benches/recall.rs` driving a
  pgrx cluster: load glove-200, openai-1536, openai-3072; build
  both pg_turbovec and pgvector hnsw indexes; compare R@k and p99
  latency at matched bit budgets. Output JSON to
  `benches/results/`.
- **HNSW-on-TurboQuant** — research: replace the flat IVF-like
  IdMapIndex with an HNSW graph whose nodes hold TurboQuant codes.
  Hierarchical structure for sub-millisecond k-NN at the cost of
  build time.
