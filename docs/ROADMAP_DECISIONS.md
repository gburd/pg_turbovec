# Roadmap decisions — what we're shipping in 1.0 and what we're not

This file documents which roadmap items we deliberately skipped on
the way to 1.0, and the reasoning. The honest engineering answer
matters more than a long backlog.

## Skipped — explicit non-goals for 1.0

### Binary-compatible varlena layout for `vector`

**What it would have done.** Replace our CBOR-derived `tvector`
varlena (a serde-encoded `f32` vector with ~10–15 % overhead from
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
3. The 10–15 % varlena overhead is dwarfed by the **16×**
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
fundamentally different kernel — locality-sensitive hashing,
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
| 1-bit + Hamming HNSW (pgvector `bit_hamming_ops`) | 192 | ≈ 0.65–0.75 |

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

## Where future work would pay off (in priority order)

1. **Real-world recall measurement vs. pgvector HNSW and
   pgvectorscale StreamingDiskANN.** We have synthetic random-vector
   numbers in `docs/RECALL.md`; we do not have head-to-head numbers
   on GloVe-200 / OpenAI ada-002 / OpenAI text-embedding-3-large at
   matched bit budgets. This is the single most important deliverable
   that gates a defensible "use pg_turbovec because …" claim. The
   `TURBOVEC_FIXTURE_PATH` env hook in `benches/recall.rs` is ready
   to consume the fixtures; we just need to publish the table.
2. **Concurrent query throughput.** Our backend-local cache uses a
   `parking_lot::Mutex<HashMap>`. Concurrent ORDER BY scans against
   the same index will serialise on the cache lookup. We have no
   measurement of how bad it is; an `Rwlock` or sharded map may be
   warranted at high QPS.
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

- [`docs/PARITY_GAPS.md`](PARITY_GAPS.md) — feature-by-feature
  pgvector comparison.
- [`docs/PHASE19_PROGRESS.md`](PHASE19_PROGRESS.md) — handoff for
  the binary-compat varlena work, if a future session picks it up.
- [`docs/INDEXAM.md`](INDEXAM.md) — index access method
  implementation guide.
- [`docs/RECALL.md`](RECALL.md) — recall-benchmark methodology and
  the synthetic numbers we have today.
- [`docs/MIGRATING_FROM_PGVECTOR.md`](MIGRATING_FROM_PGVECTOR.md) —
  hands-on migration cookbook.
