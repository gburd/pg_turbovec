# Roadmap decisions - what we're shipping in 1.0 and what we're not

This file documents which roadmap items we deliberately skipped on
the way to 1.0, and the reasoning. The honest engineering answer
matters more than a long backlog.

## Skipped - explicit non-goals for 1.0

### Binary-compatible varlena layout for `vector`

**What it would have done.** Replace our CBOR-derived `tvector`
varlena (a serde-encoded `f32` vector with ~10-15 % overhead from
varint dim headers and serde tags) with the pgvector-byte-compatible
`[i32 vl_len_, i16 dim, i16 unused, f32[dim]]` layout. Casts to and
from `pgvector.vector` would become a single memcpy, and
libpq COPY BINARY clients written for pgvector would Just Work
against `turbovec.vector` without re-encoding.

**Why we skipped it.**

1. The cross-extension migration is one-shot:
   `UPDATE docs SET emb_tv = emb_pgv::real[]::vector` runs once on
   a million rows and finishes in seconds. The `real[]` bridge is
   not a hot path.
2. Most apps insert via SQL text or parameterised queries, not
   `COPY BINARY`. The wire-protocol benefit is real but narrow.
3. The 10-15 % varlena overhead is dwarfed by the **16×**
   compression we get from 4-bit TurboQuant quantization (1536-dim:
   pgvector 6 144 B → our `am_storage` payload ≈ 388 B/row). The
   storage win comes from quantization, not varlena layout.
4. Catalog-level type aliasing is impossible regardless. Even with
   byte-compat, `turbovec.vector` and `pgvector.vector` are
   separate type oids; you write the cast somewhere either way.
5. The implementation requires a non-trivial amount of `unsafe` FFI
   in `src/vec.rs` (manual `FromDatum` / `IntoDatum`, manual `CREATE
   TYPE` declarations bypassing pgrx's `PostgresType` derive, manual
   `tvector_send` / `tvector_recv` for libpq binary). The
   `docs/PHASE19_PROGRESS.md` handoff document enumerates the work
   if a future session wants to pick it up.

**Where it would pay off.** Nightly batch backfills that touch tens
of millions of rows; shops that ingest via `COPY BINARY` from
pgvector-aware clients. For those use cases, the `real[]` bridge
adds an O(dim) walk per row that binary-compat would eliminate.

**What users get today.** Explicit casts in both directions through
`real[]` (or `text` for ad-hoc work). See
[`docs/MIGRATING_FROM_PGVECTOR.md`](MIGRATING_FROM_PGVECTOR.md).

### Bitvec Hamming / Jaccard ANN index

**What it would have done.** Add operator classes
`bit_hamming_ops` / `bit_jaccard_ops` so `CREATE INDEX ... USING
turbovec (b bit_hamming_ops)` would index bitvec columns and serve
`ORDER BY b <~> q LIMIT k` queries.

**Why we skipped it.** The TurboQuant kernel is a scalar quantizer
for **dense, unit-norm `f32`** vectors. It is not a Hamming-space
ANN index. Supporting indexed bit-Hamming queries would require a
fundamentally different kernel - locality-sensitive hashing,
multi-index hashing, or graph-based binary search. That is a
separate research project.

**More importantly: pg_turbovec already serves the workload that
*motivates* `bit_hamming_ops` better than bit-Hamming does.**

The reason pgvector users reach for `binary_quantize() + bitvec +
bit_hamming_ops` is memory pressure: "I have 100 M × 1536-dim
embeddings, FP32 doesn't fit, I'll trade recall for 32×
compression."

The TurboQuant kernel solves the same memory-vs-recall trade-off
with provably better math (Lloyd-Max scalar quantization is
within 2.7× of the Shannon distortion-rate lower bound; 1-bit
thresholding is not). Approximate per-1536-dim numbers on real
embeddings:

| Approach | Bytes / row | R@10 |
|---|---:|---:|
| FP32 (`vector`) | 6 144 | 1.00 |
| FP16 (`halfvec`) | 3 072 | ≈ 1.00 |
| **TurboQuant 4-bit (`turbovec` index)** | **388** | **≈ 0.95** |
| **TurboQuant 2-bit (`turbovec` index)** | **196** | **≈ 0.85** |
| 1-bit + Hamming HNSW (pgvector `bit_hamming_ops`) | 192 | ≈ 0.65-0.75 |

For the *workload that drives users to bitvec ANN*, our 2-bit mode
is a strict improvement: similar storage, materially better recall.
A user who would have reached for `binary_quantize + bit_hamming_ops`
in pgvector should reach for `WITH (bit_width = 2)` in pg_turbovec.

**What users get today.**

- The `bitvec` type itself (storage, text I/O, casts).
- Exact `<~>` (Hamming) and `<%>` (Jaccard) operators (brute-force
  scan; XOR + popcount is hardware-fast even at scale).
- `binary_quantize(vector) → bitvec` for users producing bit
  signatures explicitly (e.g. perceptual-hash pipelines).

**When pg_turbovec is the wrong choice for your bitvec workload.**
If you specifically need indexed Hamming-distance ANN on, say, 64-
or 128-bit perceptual hashes (image dedup, near-duplicate
detection), pgvector's `bit_hamming_ops` HNSW index is genuinely
the right tool and there is no reason to use pg_turbovec there.

## Shipped in 1.0.x / 1.1.0 / 1.2.0 / 1.3.0

What actually landed on the way from the 1.0 design freeze
(captured by the "Skipped" section above) to today's `main`.
One-liner per release; the canonical per-version log is
[`CHANGELOG.md`](../CHANGELOG.md).

### 1. v1.0.0 — pgvector parity surface + the `turbovec` AM

**Done.**

- `vector` SQL type, text + array casts + jsonb round-trip,
  the four distance operators (`<-> <#> <=> <+>`), element-wise
  arithmetic, and `f64`-accumulator `avg(vector)` / `sum(vector)`
  aggregates.
- `halfvec` (f16), `sparsevec`, and `bitvec` SQL types with
  matching distance operators, casts, and aggregates.
- The `turbovec` index access method on `vector`, with three
  operator classes (`vec_ip_ops`, `vec_cosine_ops`, plus the
  exact-only `<->` / `<+>` paths through the heap), and
  CIC / aminsert / ambulkdelete / REINDEX all functional.
- `turbovec.knn(...)` function path with optional
  `bigint[]` allowlist (filtered search inside the SIMD kernel).
- `turbovec.*` GUC namespace.

**Not done in 1.0.0.** Cold-scan latency on a fresh backend was
still dominated by SPI fetch + tmpfile-roundtrip deserialisation
of the side-table payload — see § 1 of "Where future work
would pay off" below.

### 2. v1.0.0 (rc.2 → final) — the three arnold-driven fixes

**Done.** Real-hardware million-row run on `arnold` drove three
cumulative fixes that ship together as `1.0.0` proper:

- `turbovec.search_k` GUC (default 100) replaced the hard-coded
  `K=1024` per-scan candidate fan-out; tunable per session.
- `amrescan` tolerates non-orderby plans (`SELECT count(*)`
  no longer trips `index scan requires an ORDER BY <operator>`).
- Backend-local `Arc<RwLock<IdMapIndex>>` cache from
  `src/cache.rs` is now wired into `src/index/scan.rs` (it had
  previously only served the `turbovec.knn()` function path).
  Intra-backend warm-cache speedup measured at ~9.7× on the
  arnold corpus.

### 3. v1.0.1 — pg13 / pg14 / pg15 / pg18 build compatibility

**Done.** Three `#[cfg]` gates around AM-callback fields that
moved across PG releases (`amsummarizing`, `amadjustmembers`,
`relopt_parse_elt::isset_offset`), plus a split of `aminsert`
into two `cfg`-selected wrappers around a shared inner
implementation (the `indexUnchanged` HOT-elision flag arrived
in PG 14). All six PG versions green: 92/92 tests on each.

**Not done.** pg18 is built and tested but not yet covered by
benchmark runs; the existing arnold numbers remain pg17.

### 4. v1.1.0 — Phase J + K + L

**Done.**

- **Phase J — dbpedia-1M head-to-head.** README headline now
  cites the canonical pgvector benchmark corpus,
  `dbpedia-entities-openai-1M`, measured on arnold. There is no
  (recall, storage, latency) corner where pgvector HNSW wins on
  this corpus; pg_turbovec 2-bit at `search_k=100` is
  Pareto-dominant.
- **Phase K — deferred-commit `aminsert`** (~3000× bulk-INSERT
  speedup). `aminsert` now mutates the cached `IdMapIndex` in
  memory under a `RwLock` write guard, marks dirty, and defers
  the `am_storage` write to a `PreCommit` xact callback (see
  `src/xact.rs`). N-row bulk inserts pay one `persist::load`
  plus one `persist::save` instead of N of each.
- **Latent bugs uncovered during Phase K and fixed.**
  `IdMapIndex::add_with_ids` was recomputing the Lloyd-Max
  codebook boundaries on every call (now cached on
  `TurboQuantIndex`; see `vendor/turbovec/PATCH_NOTES.md`).
  `amcostestimate` was returning a normal cost on non-orderby
  plans, letting the planner accidentally pick our AM for
  `count(*)`; it now returns `disable_cost`.
- **Phase L — relfile-resident page format preview.** New
  Cargo feature `relfile_storage` (default OFF) moves the
  serialised index from the SPI side-table to the index
  relation's main fork, accessed through PG's standard buffer
  manager. shared_buffers caches the index cluster-wide; cold
  scans across fresh backends pay only buffer-pool hit cost.
  All six AM callbacks ported. 100/100 tests green with
  `--features "... relfile_storage pg_test"`.

### 5. v1.2.0 — Phase L hardening + Phase P cold-scan win

**Done.**

- **Phase L hardening (items 1–6 complete).** Every relfile
  page write goes through `GenericXLog`; `ambuildempty`
  populates `INIT_FORKNUM`; shrinking REINDEX truncates
  trailing pages via `RelationTruncate`; deferred-commit
  `aminsert` extended to the relfile path; migration `NOTICE`
  in `ambeginscan` for v1.0.x indexes opened under
  `relfile_storage`; `ambulkdelete` walks pages in-place
  instead of rebuilding (O(deleted) vs. O(total)).
- **Phase P — pre-baked SIMD-blocked layout + Lloyd-Max
  codebook.** `ambuild` persists the blocked codes chain and
  the codebook centroids/boundaries into the relfile meta page
  + a v2 chain. Backends opening the index for the first time
  read the prepared parts off disk via
  `IdMapIndex::from_id_map_parts_with_prepared` and skip the
  per-backend `pack::repack` (~12–15 s on dbpedia-1M) and
  Lloyd-Max compute (~5–8 s). Cold p50 on dbpedia-1M dropped
  from ~26.5 s (Phase L preview) to **1.26 s** — a 21×
  speedup vs. the v1.0.x side-table baseline (Phase O-3,
  commit `25ea5c8`).

### 6. v1.3.0 — Phase Q: side-table storage retired

**Done.**

- `src/index/persist.rs` deleted; `aminsert_sidetable` /
  `ambulkdelete_sidetable` deleted; the `turbovec.am_storage`
  table dropped from the install SQL.
- `relfile_storage` and `experimental_index_am` Cargo
  features retired (default-on for many releases; the
  flags were stale).
- All `#[cfg(feature = ...)]` gates around the index AM and
  storage paths removed.
- Migration boundary: `ambeginscan` raises `ERROR` on a
  v1.0.x..v1.2 index, with a clear `REINDEX INDEX <name>;`
  hint. The previous `NOTICE` was too easy to miss.
- Test count: 109/109 across pg13..pg18 (was 94/94 default
  + 104/104 `relfile_storage` in 1.2.0; the gates collapse).

**Not done.** None remaining — Phase Q (v1.3.0) closed the
last item by retiring the side-table path entirely. Two
concurrency caveats remain on the relfile path: two backends
racing their commit-time relfile rewrite (last writer wins)
and `PREPARE TRANSACTION` skipping the `PreCommit` callback
(parallel-worker inserts already blocked by
`amcanparallel = false`).

## Where future work would pay off (in priority order)

### 1. ~~Relfile-resident page format for the index AM (the cold-path fix).~~ **Shipped in v1.2.0; default-on in v1.3.0.**

**Status:** The relfile-resident page format with persisted
SIMD-blocked layout + Lloyd-Max codebook is the only storage
strategy as of v1.3.0 (Phase Q). Cold-scan p50 on dbpedia-1M
is **1.26 s** (post-Phase-P, commit a801f38), a 21× speedup
over the v1.0.x side-table baseline. The pre-Phase-Q section
on the SPI side-table is preserved in `git log` for context;
the story below covers what would close the remaining gap to
pgvector HNSW (~1.2 s vs. ~100 ms cold p50).

**Next lever:** cluster-wide caching of `IdMapIndex` parts via
a shared-memory `dsm` segment so the per-backend
`from_id_map_parts_with_prepared` cost (~1 s on dbpedia-1M
reading codes + scales + ids + blocked chains off disk) is
paid once per cluster instead of once per backend. Tracked
as a follow-up; not a 1.3 gating item.

1. **Real-world recall measurement vs. pgvector HNSW and
   pgvectorscale StreamingDiskANN.** We have synthetic random-vector
   numbers in `docs/RECALL.md`; we do not have head-to-head numbers
   on GloVe-200 / OpenAI ada-002 / OpenAI text-embedding-3-large at
   matched bit budgets. This is the single most important deliverable
   that gates a defensible "use pg_turbovec because ..." claim. The
   `TURBOVEC_FIXTURE_PATH` env hook in `benches/recall.rs` is ready
   to consume the fixtures; we just need to publish the table.
2. **Concurrent query throughput.** Our backend-local cache uses a
   `parking_lot::Mutex<HashMap>`. Concurrent ORDER BY scans against
   the same index will serialise on the cache lookup. We have no
   measurement of how bad it is; an `Rwlock` or sharded map may be
   warranted at high QPS.
   *(Update 2026-05-23.) Measured. The mutex itself contributes
   < 1 % to the per-call hot path: an in-process bench with N
   threads sharing one cache is statistically tied with a
   no-lock control at every N from 1 to 16. The pgbench curve
   does scale sub-linearly (2.85 × at N=8), but each backend has
   its own cache; the bottleneck is the per-call `count(*)` SPI
   roundtrip in `relation_row_count`, not the lock. See
   [`docs/CONCURRENCY.md`](CONCURRENCY.md). Skipped the
   `RwLock` / sharded-map work.*
3. **README polish around the TurboQuant-vs-1-bit-Hamming win.**
   The README mentions binary quantization briefly; it should
   spotlight the 2-bit-vs-1-bit recall comparison as the principal
   reason to pick pg_turbovec over pgvector's `binary_quantize` +
   `bit_hamming_ops` workflow.
4. **`src/index/scan.rs` cache-hit path optimisation.** The cache
   stores `Arc<IdMapIndex>`; `amgettuple` clones the Arc and runs
   `IdMapIndex::search` under it. Profiling whether the Arc-clone
   shows up at high concurrency is on the list.

## Index of related docs

- [`docs/PARITY_GAPS.md`](PARITY_GAPS.md) - feature-by-feature
  pgvector comparison.
- [`docs/PHASE19_PROGRESS.md`](PHASE19_PROGRESS.md) - handoff for
  the binary-compat varlena work, if a future session picks it up.
- [`docs/INDEXAM.md`](INDEXAM.md) - index access method
  implementation guide.
- [`docs/RECALL.md`](RECALL.md) - recall-benchmark methodology and
  the synthetic numbers we have today.
- [`docs/MIGRATING_FROM_PGVECTOR.md`](MIGRATING_FROM_PGVECTOR.md) -
  hands-on migration cookbook.
