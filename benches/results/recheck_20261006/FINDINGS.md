# Where the per-candidate "recheck" cost goes (2026-10-06)

**Question.** The v2.11.0 findings said that after turbovec 1.1 the index scan is
a small share of a flat-index query, with "~48 µs per candidate" spent on the
heap recheck. Is that a PostgreSQL core problem worth a patch?

**Answer: no.** Core's share (heap fetch + reorder queue + executor) is
**~1–2 µs per candidate**. The other ~16 µs is ours: how the `turbovec.vector`
type is stored and decoded, and an unvectorized distance function. There is no
core patch to propose; there are three extension fixes.

## Method (and two corrections to the first pass)

- floki, Intel Core Ultra 7 258V (4 P-cores + 4 E-cores, AVX2/AVX-VNNI, no
  AVX-512). PostgreSQL **16.15 built `-O2 -fno-omit-frame-pointer`, no
  cassert**, postmaster pinned to the P-cores (`taskset -c 0-3`). pg_turbovec
  v2.11.0 (`7164153`). 200k × 1024-d real Cohere embeddings, 4-bit flat index,
  `search_k = 1024`, 200 queries per arm.
- **Attribution is from the real query**, not a proxy: `perf record -e
  task-clock -c 100000 -g --call-graph=fp` on the backend thread only, so each
  sample is 100 µs of on-CPU time regardless of which core type ran it.
  Samples are bucketed by the first matching frame anywhere in the stack
  (`attrib.py`, `reattr.py`). Two runs per arm, consistent within ~10%.
- **Correction 1.** A first pass using a TID-scan proxy reported ~50 µs per
  candidate with TOAST at ~40 µs. The proxy overstated TOAST roughly 8×,
  probably because a 1,024-element `ctid = ANY(...)` scan evaluates the array
  per page. Real-path TOAST is ~5 µs. Discard the proxy numbers.
- **Correction 2.** It is NOT true that pgvector pays the same recheck.
  pgvector's HNSW and IVFFlat set `xs_recheckorderby = false`
  (`hnswscan.c:330`, `ivfscan.c:415`): they keep full-precision vectors in the
  index, compute exact distances there, and return candidates in exact order.
  The executor fetches only the ~10 rows the LIMIT returns. pg_turbovec stores
  only 4-bit codes, so it MUST recheck from the heap, and that is a design
  trade (~8× smaller index) we chose, not a core limitation.

## Results: backend-thread µs per candidate, real query, `search_k = 1024`

| component | default storage (vector TOASTed) | `SET STORAGE PLAIN` | owner |
|---|---|---|---|
| CBOR decode of both vector args (pgrx serde) | **8.13** | **7.80** | us |
| TOAST fetch of the candidate's vector | **5.03** | 0.00 | us (type storage default) |
| exact cosine kernel | **2.93** | **2.92** | us |
| heap fetch of the candidate | 0.75 | 0.98 | core |
| reorder queue (tuple copy, pairing heap) | 0.09 | 1.00 | core |
| other executor | 0.12 | 0.04 | core |
| turbovec scan, backend thread | 0.66 | 0.65 | us |
| parse / plan / libpq | 0.30 | 0.30 | core |
| **total backend on-CPU** | **18.45 ms/query (18.0 µs/cand.)** | **14.03 ms/query (13.7)** | |

Core's share is 0.96 µs (TOASTed) to 2.02 µs (PLAIN) per candidate, 5–15% of
the backend's time. The reorder-queue cost rises under PLAIN because
`ExecCopySlotHeapTuple` now copies a 5 KB inline tuple instead of an ~80-byte
tuple holding a TOAST pointer. That is a real but small cost, and inherent to
reordering.

Wall-clock was ~31–35 ms/query; the remaining ~13–21 ms is the backend waiting
on the turbovec scan's rayon workers. On this AVX2 laptop the scan is still a
large share. The v2.11.0 claim that "the recheck is the bottleneck" holds on
Graviton4, where the 1.1 staged kernel makes the scan ~1 ms; it does not hold
on AVX2 hosts.

## Why each of our costs exists

1. **CBOR (~8 µs).** `Vector` is a pgrx `#[derive(PostgresType)]`, so every
   datum is a serde-CBOR varlena. Decoding 1,024 floats costs 3.73 µs
   (`cbor_bench.rs`: raw LE copy 0.08 µs, zero-copy view ~0). The distance
   function `cosine_distance(a: Vector, b: Vector)` takes both args by value,
   so **the constant query vector is CBOR-decoded again for every
   candidate**: two decodes per candidate, matching the measured ~8 µs. CBOR
   is also 25% larger (5,125 B vs 4,096 B).
2. **TOAST (~5 µs).** At 5,125 B a 1024-d vector exceeds the ~2 KB TOAST
   threshold, and the type's default storage is EXTENDED, so every value
   lives out of line (15 MB main heap, 2,083 MB TOAST for 200k rows). Each
   recheck does a TOAST-index probe plus chunk fetches. Float noise doesn't
   compress (`pg_column_compression` was NULL on every sampled row), so
   EXTENDED gains nothing over EXTERNAL here.
3. **Distance kernel (~2.9 µs).** `kernels::cosine_distance` accumulates in
   serial `f64`: one dependent add chain, which LLVM cannot vectorize without
   reassociation. It also recomputes the query's norm every call. Measured
   1.36 µs for ours vs **0.079 µs** for a 16-lane `f32` accumulator with the
   query norm computed once (17×, |Δ| = 2.8e-10). The in-backend 2.9 µs
   includes the pgrx `run_guarded` FFI wrapper.

## Is there anything for core?

Reading `nodeIndexscan.c::IndexNextWithReorder` (PG 16; same shape through
18) against these numbers:

- The heap fetch + `EvalOrderByExpressions` per candidate is the designed
  contract of a lossy ORDER BY (35fcb1b3, "Allow GiST distance function to
  return merely a lower-bound", 2015). We measure it at ~1 µs. Not a defect.
- `reorderqueue_push` copies every queued tuple (`ExecCopySlotHeapTuple` +
  `datumCopy`). ~0.1–1 µs. One could imagine queueing a TID + distance and
  re-fetching on pop, but that trades a copy for a second heap fetch: no clear
  win, and nothing here justifies it.
- No early exit: with our advertised `-inf` bound the executor drains all
  `search_k` candidates. That is an AM choice (TurboQuant's score isn't a
  guaranteed bound), and v1.18 already established that a tighter bound
  wouldn't skip the fetch, because `index_getnext_slot` and the recheck run
  before the bound is consulted.

Prior discussion (agora): "Comments for lossy ORDER BY are lacking" (Andres
Freund, 2019; documentation); "ORDER BY operator index scans and filtering"
(Andrew Kane, 2024; a distance *filter* not bounding the scan, a different
issue); Tom Lane (2024) on why the ORDER BY value must be computed for the
scan's targetlist. None of them is this cost. **No core patch is warranted.**

## Fixes (all extension-side), measured or bounded

| fix | removes | per-candidate effect | compatibility |
|---|---|---|---|
| A. Vectorized cosine/L2/IP kernels, query norm computed once per scan | ~2.8 µs | 2.9 → ~0.1–0.3 | code only; no format change. Changes the exact distance in the ~1e-10 range, which matters only for exact-tie ordering |
| B. Decode the query once: cache the decoded constant in `fn_extra` / `fcinfo->flinfo` | ~3.9 µs | one CBOR decode instead of two | code only |
| C. Document `ALTER TABLE … ALTER COLUMN … SET STORAGE MAIN` (or PLAIN) for ≥ ~500-d | ~5 µs TOAST | measured 18.0 → 13.7 total | user DDL, existing rows unaffected until rewritten; PLAIN caps a row at one page (~2,000-d) |
| D. Replace serde-CBOR with a raw varlena (`int16 dim, int16 unused, float4[dim]`, the pgvector layout) read zero-copy | the remaining ~4 µs decode + 25% size | → ~0 decode | **heap-type format change**: needs a version tag and a decoder that reads old CBOR transparently (HARD MANDATE #2); the index is untouched |

Together A+B+C bring backend work from ~18 µs to roughly ~6 µs per candidate,
and D to roughly ~2 µs. That leaves core's ~1–2 µs as the floor, which is the
honest cost of a heap-rechecked lossy index. **A and B are ponytail-sized, carry
no format risk, and should go first.** D is the planned "Phase 2" from the
`vec.rs` module docstring, and these numbers are the case for it.

Not measured here: end-to-end latency after A–D (re-measure on an idle host
and on Graviton4, where the scan no longer dominates).

Raw: `attrib_k1024.txt`, `reattr.txt`; scripts `attrib.py`, `reattr.py`,
`cbor_bench.rs`, `cosine_bench.rs`, `load.py`, `up.sh`. The discarded TID-scan
proxy (correction 1) is kept in `superseded_proxy/` for the record.
