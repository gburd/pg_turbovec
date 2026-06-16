# IVF layer for pg_turbovec — design plan

_Status: PLAN (not yet implemented). Drafted 2026-06-15 after the
AVX2 latency-frontier benchmark (`docs/BENCHMARKS.md`) established
that pg_turbovec's flat `O(n·dim)` quantized scan is ~490× slower
than pgvector HNSW at 1M × 1024-d. This document lays out an
inverted-file (IVF) coarse-quantizer layer that turns the per-query
cost from `O(n·dim)` into `O((n/nlist)·dim·nprobe + nlist·dim)`,
i.e. sublinear in the corpus when `nprobe ≪ nlist`._

---

## 1. The problem this solves

pg_turbovec today scores the query against **every** vector's
quantized code on every scan. That's exact (recall 1.000) and
storage-tiny, but the latency is linear in the corpus:

| corpus | flat-scan warm p50 (AVX2, measured) |
| --- | --- |
| 1 M × 1024-d, 2-bit | ~2.5 s |

HNSW gets ~5 ms by visiting only a tiny, navigable subset of the
graph. We can't match a graph's `O(log n)` traversal with a flat
scan, but **IVF gets us most of the way**: partition the corpus into
`nlist` cells (Voronoi regions of `nlist` coarse centroids), and at
query time score only the `nprobe` nearest cells instead of the
whole corpus. With `nlist = √n` and `nprobe = 8–32`, a 1 M-row
query scans ~`nprobe/nlist` ≈ 1–3% of the codes — a 30–100× latency
cut, trading a controllable slice of recall.

This is exactly the bet VectorChord (IVF + RaBitQ) and FAISS
(`IVF…`) make. It keeps our quantization storage win **and** adds a
real sublinear-ish ANN structure on top.

---

## 2. Design overview

```
                 ┌─────────────────────────────────────────┐
   query  ──────▶│ 1. coarse search: score q against the    │
                 │    nlist coarse centroids (full f32 or    │
                 │    half-precision). Pick the nprobe        │
                 │    nearest cells.   cost: O(nlist·dim)     │
                 └───────────────────┬─────────────────────┘
                                     │ nprobe cell ids
                                     ▼
                 ┌─────────────────────────────────────────┐
   per cell ────▶│ 2. fine search: TurboQuant-score q        │
                 │    against just that cell's quantized      │
                 │    codes (the existing kernel, unchanged). │
                 │    cost: Σ O(|cell|·dim) over nprobe cells │
                 └───────────────────┬─────────────────────┘
                                     │ candidate (id, approx_dist)
                                     ▼
                 ┌─────────────────────────────────────────┐
                 │ 3. merge top-k across probed cells, hand  │
                 │    to the executor's reorder queue for     │
                 │    exact recheck (unchanged).              │
                 └─────────────────────────────────────────┘
```

The fine-search kernel (step 2) is **the existing TurboQuant
SIMD-blocked scan** — we just point it at a contiguous sub-range of
codes (one cell) instead of the whole corpus. No new distance
kernel. The novelty is all in the cell partitioning, the on-disk
layout, and the coarse search.

### Why this composes cleanly with what we already have

- **Oversampling** (`turbovec.oversample`, v1.9.0) already widens
  the candidate set; with IVF it widens *within the probed cells*.
- **Iterative scan** (v1.8.0) already refills when a selective
  `WHERE` filter under-returns; with IVF, a refill bumps `nprobe`
  (probe more cells) instead of bumping a flat `search_k`.
- **The reorder queue** (`xs_recheckorderby = true`) already does
  exact-distance recheck on returned tuples — unchanged.
- **The relfile page format** already stores codes/scales/ids as
  page chains; IVF adds cell-boundary metadata and reorders the
  codes so each cell is contiguous.

---

## 3. On-disk format (wire-format VERSION 3 → 4)

This is a **major-or-minor wire bump** (`MetaPageData::version`
3 → 4) and requires the full migration machinery per `AGENTS.md`:
an `is_legacy_v3()` predicate, an `ambeginscan` ERROR with a
`HINT: REINDEX INDEX` for v3 indexes opened by a v4 binary that was
asked to use IVF, and an `EXPECTED_WIRE_FORMAT_VERSION` bump.

**Key compatibility decision:** make IVF **opt-in per index** via a
reloption (`WITH (lists = N)`), and keep the flat layout as
`lists = 0` (the default, = today's behaviour). A v4 binary then:
- reads a v3 index as a flat index (no IVF) — **backward compatible,
  no REINDEX for existing indexes** if we keep the v3 decode path;
- reads a v4 `lists = 0` index as flat (identical bytes to v3 modulo
  the version byte);
- reads a v4 `lists > 0` index as IVF.

If we can keep the v3 flat decode path intact in the v4 binary
(very likely — the flat layout is a strict subset), then **existing
v3 indexes keep working with no REINDEX**, and only users who *want*
IVF rebuild with `WITH (lists = N)`. That's the preferred path and
keeps the bump effectively minor for non-IVF users.

### New meta-page fields (v4)

Appended after the v3 `rotation_*` fields (the meta page has room;
it's one 8 KiB block):

| field | type | meaning |
| --- | --- | --- |
| `lists` | u32 | number of coarse cells (`nlist`); 0 = flat (v3-equivalent) |
| `coarse_first` | BlockNumber | first page of the coarse-centroid chain (`nlist × dim` f32, or f16 to halve it) |
| `coarse_count` | u32 | pages in the coarse chain |
| `cell_dir_first` | BlockNumber | first page of the **cell directory**: `nlist` entries of `(code_offset: u64, n_vectors: u32)` giving each cell's contiguous range in the codes chain |
| `cell_dir_count` | u32 | pages in the cell directory |

The codes / scales / ids chains stay the same page-chain format, but
**reordered so each cell's rows are contiguous** (cell 0's rows,
then cell 1's, …). `slot_to_id` already maps slot→external id, so
reordering slots is transparent to the id layer — we just permute
the build-time slot assignment.

### Coarse centroids precision

`nlist × dim` f32 is the coarse codebook. At `nlist = 4096`,
`dim = 1024`: 4096 × 1024 × 4 = 16 MiB. Small enough to mmap (like
the existing rotation/blocked static regions) and keep resident.
Consider f16 (8 MiB) if coarse-search precision allows; the coarse
step only needs to pick candidate *cells*, so f16 is likely fine and
the fine TurboQuant step recovers precision.

---

## 4. Build path (`ambuild`)

The build gains a **coarse-clustering pre-pass** before the existing
quantize+pack:

1. **Sample** a subset of the heap-scanned vectors (e.g.
   `min(n, 256 × nlist)` rows — FAISS's rule of thumb) for k-means
   training. Reservoir-sample during the existing
   `index_build_range_scan` so we don't double-scan.
2. **Train `nlist` coarse centroids** via k-means (Lloyd's) on the
   sample, in the rotated space (apply the existing rotation first
   so cells live in the same space the fine quantizer uses). This is
   the one genuinely new compute. k-means on 256·nlist × dim is
   cheap relative to the quantize step and **parallelizes over the
   rayon pool we already added in v1.8.0**.
3. **Assign** every vector to its nearest coarse centroid (one pass;
   parallel). Produces a `slot → cell` map.
4. **Permute** the slot order so cells are contiguous, build the
   cell directory, then run the **existing** quantize + SIMD-repack
   on the permuted order. The `slot_to_id` permutes with it.
5. Persist coarse centroids + cell directory in the new meta/chain
   slots; everything else writes through the existing
   `write_full_with_prepared` path.

`nlist` defaults: `lists = 0` (flat) unless specified;
recommend `lists ≈ √n` documented in the reloption help, matching
pgvector ivfflat's guidance. Like pgvector ivfflat, the index should
be built on a populated table (k-means needs data) — `ambuildempty`
stays flat.

**Memory:** the Phase W streaming cap still applies to the quantize
phase. The k-means sample is bounded (256·nlist vectors), and the
coarse centroids are tiny. The assignment pass is one streamed
sweep. No new unbounded buffer.

---

## 5. Scan path (`amgettuple` / `amrescan`)

1. On first fetch, if `meta.lists > 0`: load coarse centroids (mmap,
   like the existing static regions) and the cell directory.
2. **Coarse search:** score the (rotated) query against the `nlist`
   centroids, take the `nprobe` nearest. `nprobe` is a new GUC
   `turbovec.probes` (default e.g. 8; pgvector ivfflat calls it
   `ivfflat.probes`). cost `O(nlist · dim)`.
3. **Fine search:** for each probed cell, call the existing
   TurboQuant search kernel over that cell's contiguous code range
   (the kernel already takes a base pointer + count — we pass the
   cell's `code_offset` + `n_vectors` instead of the whole corpus).
   Collect `(id, approx_dist)`.
4. **Merge** the per-cell top candidates into a global top-`search_k`
   heap, hand to the executor's reorder queue (unchanged).

`nprobe` is the recall/latency dial, exactly like HNSW's `ef_search`
and ivfflat's `probes`. **Iterative scan** maps onto IVF naturally:
when a selective `WHERE` under-returns, bump `nprobe` (probe more
cells) up to a `max_probes` cap, instead of bumping a flat
`search_k`. **Oversampling** widens `search_k` within the probed
cells.

### Recall knobs summary (post-IVF)

| GUC | role | analogous to |
| --- | --- | --- |
| `turbovec.probes` | cells to scan per query (the main latency/recall dial) | `ivfflat.probes`, `hnsw.ef_search` |
| `turbovec.search_k` | candidates kept from the probed cells | (existing) |
| `turbovec.oversample` | widen candidate set before exact recheck | Qdrant oversampling |
| `turbovec.max_probes` | iterative-scan cap on probe growth under selective filters | `ivfflat.max_probes` |

---

## 6. Expected performance

Rough model at 1 M × 1024-d, `nlist = 1024`, `nprobe = 16`,
even cell sizes (~1000 vectors/cell):

- **Coarse:** 1024 centroids × 1024-d ≈ 1 M f32 ops ≈ sub-ms on AVX2.
- **Fine:** 16 cells × ~1000 vectors × 1024-d ≈ the cost of a flat
  scan over ~16 000 vectors instead of 1 000 000 — **~60× fewer**.
- Projected warm p50: the measured 2.5 s flat scan / ~60 ≈ **~40 ms**
  (plus coarse overhead). Tunable down with smaller `nprobe`
  (faster, lower recall) or up (slower, higher recall) — a real
  frontier instead of a single point.

That doesn't beat HNSW's ~5 ms, but it lands pg_turbovec in the same
order of magnitude (tens of ms) while **keeping the 10–15× storage
win and the exact-on-probed-cells recall**. The recall ceiling
becomes `recall(nprobe)` — the fraction of true neighbours whose
cell was probed — exactly the IVF trade-off, tunable via `probes`.

**Recall caveat (the real IVF cost):** a true neighbour in an
un-probed cell is missed. Mitigations, all standard: larger
`nprobe`; `nlist` tuned to the data; soft assignment (assign each
vector to its top-2 cells at build time — doubles storage of the
boundary vectors but raises recall); and the iterative-scan refill
(bump `nprobe` until `LIMIT` satisfied).

---

## 7. Implementation phases

| phase | scope | risk | wire change |
| --- | --- | --- | --- |
| **IVF-1** | k-means coarse training in `ambuild` (behind `WITH (lists=N)`); persist centroids + cell directory; reorder codes by cell. Scan path STILL flat (ignores cells) — just proves the build + on-disk layout round-trips. | M | v3→v4, additive, flat-readable |
| **IVF-2** | coarse search + cell-restricted fine search in `amgettuple`; `turbovec.probes` GUC. The actual latency win. | L | none beyond IVF-1 |
| **IVF-3** | iterative-scan integration (`probes` growth + `max_probes`), oversampling-within-cells, recall/latency frontier bench on arnold (AVX2) vs flat + vs pgvector ivfflat AND hnsw. | M | none |
| **IVF-4** (optional) | soft/multi-assignment for recall; balanced cells; `lists` auto-tuning from `n`. | M | maybe (multi-assign changes cell dir) |

Ship IVF-1+2 as the minor that introduces `WITH (lists)` +
`turbovec.probes`; IVF-3 is the bench + tuning that makes it
defensible; IVF-4 is recall polish.

---

## 8. Testing requirements (lessons from the pre-AVX2 bug)

Per `docs/TESTING.md`, every new scan path needs:
- **distinct-ids assertion** on IVF scan results (the cheapest
  wrong-ranking guard).
- **recall-floor `#[pg_test]`** at the new medium scale, per
  `(bit_width, lists, probes)` — IVF recall depends on all three.
- a test that **`probes ≥ nlist` reduces to the flat scan** (exact
  recall) — the correctness anchor.
- a test that **iterative-scan refill raises `probes`** under a
  selective filter and recovers the full `LIMIT`.
- **determinism**: same table + same `lists` ⇒ byte-identical
  relfile (k-means seeded deterministically, like the rotation's
  fixed `ROTATION_SEED`).
- **AVX2 vs scalar** correctness parity is upstream turbovec's job
  for the fine kernel; the coarse search is new pg_turbovec code and
  needs its own scalar-correct + (where applicable) SIMD path, or
  just a straightforward f32 scalar coarse loop (1024 centroids is
  cheap — scalar is fine and avoids a second SIMD-correctness
  surface).

The latency-frontier validation MUST run on an **AVX2 host
(`arnold`)** with the isolation protocol from the v1.9.1 bench;
`meh` (pre-AVX2) can only validate IVF correctness/recall, not
latency.

---

## 9. Open questions / decisions to make before IVF-1

1. **Coarse centroid precision** — f32 (16 MiB @ nlist=4096,dim=1024)
   vs f16 (8 MiB). Lean f16; validate recall impact in IVF-3.
2. **k-means in-tree vs reuse a crate** — turbovec doesn't expose
   k-means. Options: hand-roll Lloyd's (small, deterministic,
   no new dep), or pull a vetted crate. Lean hand-rolled for
   determinism + zero dep surface; it's ~100 lines.
3. **Single vs soft assignment** for IVF-1 — start single
   (simplest), add soft in IVF-4 if recall needs it.
4. **`lists` default** — keep `0` (flat) so the feature is strictly
   opt-in and existing indexes/users are untouched. Document
   `lists ≈ √n` as the recommended starting point.
5. **Does keeping the v3 flat decode path in the v4 binary let us
   avoid a REINDEX for existing indexes?** Almost certainly yes
   (flat is a subset of v4). Confirm in IVF-1 — if so, the wire bump
   is painless for non-IVF users (`ALTER EXTENSION` only; only
   `WITH (lists>0)` rebuilds opt in).

---

## 10. Positioning impact

With IVF, pg_turbovec's story changes from *"compact exact
brute-force, for where O(n) fits"* to *"compact ANN with a real
latency/recall dial, 10–15× smaller than HNSW, tunable from exact
(probes=nlist) to fast-approximate (small probes)."* That's a
direct competitor to pgvector ivfflat (we'd win on storage via
quantization) and a credible answer to VectorChord's IVF+RaBitQ.
It does NOT obsolete HNSW's latency at the very low end, but it
moves us from "490× slower" into "same order of magnitude, much
smaller on disk" — the positioning that makes pg_turbovec a real
choice for latency-sensitive workloads, not just storage-bound ones.
