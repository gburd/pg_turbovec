# Sparse ANN index kind (`KIND_SPARSE = 4`) — implementation-ready design

**Scope:** Sparse FLAT index over the existing `sparsevec` type, behind a new
`kind` byte. Exact sequential sparse-dot scan + heap rerank, mirroring the
single-vector flat path. **WAND / posting-list pruning is DEFERRED** (memo §2b,
§4 stage 2). No wire-version bump (VERSION stays 8), additive, REINDEX-free for
every existing index.

This turns the `benches/results/parity_20260925/item3_sparse_ann_design.md`
memo into a buildable plan. The memo's engineering verdict is unchanged and is
the design here: FLAT first, `KIND_SPARSE=4` additive, reuse `sparse_walk`,
defer WAND.

---

## 0. The one decision that shapes everything: variable-stride chains

Every existing chain in this AM is **fixed-stride** (codes: `dim/8*bit_width`
B/row; scales: 4 B/row; ids: 8 B/row) OR a **flat opaque byte chain**
(blocked/rotation/coarse/cell-dir/tombstone/bq_mean: `stride = 1`,
`rows_per_page = PAYLOAD_BYTES`). Sparse rows are **variable-length** (nnz
differs per row), which neither shape handles directly.

`write_chain_at` / `read_chain` require `chain_bytes.len() == n_vectors *
stride`. A sparse row breaks that invariant. The design keeps those primitives
UNTOUCHED and expresses sparse storage as **three flat opaque byte chains**
(the shape the primitives already serve with `stride = 1`), so no new chain
primitive is needed:

- **`sparse_offsets` chain** — `(n_vectors + 1)` × `u64`, a CSR row-pointer
  array into the concatenated postings. `off[i]..off[i+1]` is row `i`'s span
  (in *element* units, i.e. nnz-prefix-sum). This is the exact CSR shape the
  graph adjacency chain already uses (`(n_vectors+1)` u32 offsets), just u64
  and holding nnz-prefix-sums instead of neighbor-offsets.
- **`sparse_indices` chain** — `total_nnz` × `i32`, all rows' sorted 0-based
  coordinate indices concatenated (row `i` occupies `off[i]..off[i+1]`).
- **`sparse_values` chain** — `total_nnz` × `f32`, aligned with
  `sparse_indices`.

Plus the existing **`ids` chain** (`n_vectors` × `u64`, unchanged shape —
`slot_to_id`). Total: **3 new chains + the existing ids chain.** No codes, no
scales, no codebook, no rotation, no blocked chain (same "separate planner"
discipline as `plan_bq`).

Why offsets in *element* units (nnz counts), not bytes: `sparse_indices`
(i32) and `sparse_values` (f32) are both 4 B/element and share the same
offset array, so one prefix-sum indexes both. `dim` lives on the meta page
(single value, all rows share it — `sparsevec.dim` is a per-column constant),
so it is NOT stored per row.

**Alternative considered and rejected:** interleaved `(i32 index, f32 value)`
pairs in one chain. Rejected — separate `indices`/`values` chains let the
two-pointer `sparse_walk` read contiguous `&[i32]` / `&[f32]` slices (cache-
and SIMD-friendlier, and it is exactly the `Sparsevec` struct layout), and it
matches how the codebase already splits parallel arrays.

---

## 1. On-disk representation (new meta fields + `plan_sparse`)

Add ONE new meta group in the reserved tail (page offset 332+, all currently
zero — additive, same mechanism as v6 graph and v8 bq_mean). Encode them after
the `bq_base + 16` block in `MetaPageData::encode` and decode symmetrically in
`decode` behind a `bytes.len() >= sparse_base + N` length guard (exactly the
v6/v8 additive-decode pattern). **VERSION stays 8; only `kind = KIND_SPARSE`
discriminates.**

New `MetaPageData` fields (all zero for every non-sparse kind ⇒ existing
indexes decode byte-identically):

```
sparse_offsets_first : u32   // CSR row-pointer chain start (0 if not sparse/empty)
sparse_offsets_count : u32   // pages
sparse_offsets_bytes : u64   // (n_vectors + 1) * 8
sparse_indices_first : u32
sparse_indices_count : u32
sparse_indices_bytes : u64   // total_nnz * 4
sparse_values_first  : u32
sparse_values_count  : u32
sparse_values_bytes  : u64   // total_nnz * 4  (== sparse_indices_bytes)
sparse_total_nnz     : u64   // sum of nnz over all rows (redundant with off[n], but avoids a chain read to size allocs)
```

`dim` reuses the existing `dim: u32` meta field (the sparsevec's declared
dimension — for a 30k-dim SPLADE corpus, `dim = 30000`). `n_vectors` reuses
the existing field. `bit_width`, `stride_bytes`, `codes_*`, `scales_*`,
`rows_per_*_page`, codebook, rotation — **all zero/unused for sparse** (same
as BQ zeroes its scales/codebook/rotation). `rows_per_scales_page` must still
be non-zero (the `plan_bq` note: `read_chain` treats `rows_per_page == 0` as
corrupt; set it to `rows_per_page(4)` defensively even though sparse never
reads a scales chain).

New constructor `MetaPageData::plan_sparse(dim, n_vectors, total_nnz,
am_version)`, structured exactly like `plan_bq` (a separate planner, NOT a flag
on `plan_with_blocked`, so a bug here cannot shift a non-sparse index's
layout). Chain-start arithmetic (the FIRST family of running sums — see §2):

```
ids_first            = 1                                   // ids chain first (no codes/scales)
ids_count            = padded_pages_needed(n_vectors, rows_per_ids_page)
sparse_offsets_first = ids_first + ids_count
sparse_offsets_count = byte_pages_needed((n_vectors+1)*8)
sparse_indices_first = sparse_offsets_first + sparse_offsets_count
sparse_indices_count = byte_pages_needed(total_nnz*4)
sparse_values_first  = sparse_indices_first + sparse_indices_count
sparse_values_count  = byte_pages_needed(total_nnz*4)
```

`codes_first`/`scales_first` = 0 (absent). The three sparse chains are flat
opaque byte chains (`stride = 1`, `rows_per_page = PAYLOAD_BYTES`), so they are
NOT padded (`byte_pages_needed`, not `padded_pages_needed`) — same as every
other trailing opaque chain. The ids chain IS padded (`padded_pages_needed`),
matching the WAL-amplification fix for the one growing fixed-stride chain.

Add `is_sparse(&self) -> bool { self.kind == KIND_SPARSE }` and
`has_sparse(&self) -> bool { self.is_sparse() && self.sparse_offsets_first != 0
&& self.n_vectors > 0 }` (mirrors `is_bq`/`has_graph`).

`KIND_SPARSE: u8 = 4` in `page.rs` next to `KIND_BQ = 3`.

---

## 2. CORRUPTION-CRITICAL: every running-sum site (exhaustive)

The recurring corruption class in this repo is a chain-offset running sum that
omits a chain (bitten in v1.24.0 `graph_count`, v2.6.0 ×3 `bq_mean_count`,
v2.7.0 ×2). Adding three sparse chains means **every running sum must add all
three `sparse_*_count` fields.** Below is the COMPLETE enumeration, grep-verified
against the current tree. There are TWO families: chain-*start* arithmetic
(inside each `plan_*` / `set_*` builder) and chain-*after-all* running sums
(where a trailing chain — tombstones, graph, IVF — is placed after every prior
chain). A sparse index touches BOTH.

### Family A — `total_blocks()` (sizes the relation; must count every chain)

`src/index/page.rs :: MetaPageData::total_blocks()` (~line 810). Currently sums
`1 + codes + scales + ids + blocked + rotation + coarse + cell_dir + tombstone
+ graph + bq_mean`. **MUST add `+ sparse_offsets_count + sparse_indices_count +
sparse_values_count`.** If omitted, `extend_to(rel, total_blocks())` under-sizes
the relation and `write_chain_at` for the values chain writes past the extended
region OR `read_chain`'s `last_needed > nblk` guard trips (ERROR, not
corruption — but the write side under-extend IS corruption). This is the single
most important edit.

### Family B — chain-*start* arithmetic inside builders

1. `plan_sparse` itself (NEW, §1) — the ids/offsets/indices/values start chain.
   This is a fresh running sum; get it right at birth (see §1). It is the sparse
   analogue of `plan_bq`'s `codes_first / ids_first / mean_first` sequence.

2. `src/index/page.rs :: set_ivf_chains()` `after_rotation` sum (~line 725).
   Currently `1 + codes + scales + ids + blocked + rotation + bq_mean`.
   **Sparse is FLAT-only (see §6, `lists` rejected for sparse), so IVF chains
   are never laid out on a sparse index and this sum is never reached for
   KIND_SPARSE.** BUT — per the `set_graph_chain` precedent that added
   `bq_mean_count` "for a future graph+BQ build even though it can't happen
   today" — **add `+ sparse_*_count` here too**, guarded by the fact they're 0
   for non-sparse. Rationale: defense against a future IVF+sparse combo, and
   the project rule is "if you add a chain, add it to EVERY running sum,"
   full stop. `0` for every non-sparse kind ⇒ no behavior change today.

3. `src/index/page.rs :: set_graph_chain()` `after_every_prior_chain` sum
   (~line 690). Currently `1 + codes + scales + ids + blocked + rotation +
   coarse + cell_dir + tombstone + bq_mean`. **Add `+ sparse_offsets_count +
   sparse_indices_count + sparse_values_count`.** Graph+sparse can't co-occur
   today (both need distinct kinds; kind holds one discriminator), so this is 0
   today — but it is the EXACT site the v1.24.0 bug lived in, and the code
   comment there already documents "omitting a chain from one of these sums is
   THE recurring corruption bug." Add it.

### Family C — chain-*after-all* running sums (tombstone/trailing placement)

4. `src/index/relfile.rs :: write_full_bq_parts()` tombstone `after_all` sum
   (~line 878). BQ-only path; sparse never enters it. **Add `+ sparse_*_count`
   for the same uniformity rule** (0 today). Lower priority than C5/C6 but on
   the checklist.

5. `src/index/relfile.rs :: write_full_inner_with_tombstones()` tombstone
   `after_all` sum (~line 2152). TurboQuant-family path; sparse never enters it
   (sparse has its own writer, §3). **Add `+ sparse_*_count`** (0 today).

6. `src/index/relfile.rs :: write_tombstones_and_meta()` `after_all_other_chains`
   sum (~line 3205). **This one IS reachable for sparse** — VACUUM tombstones a
   sparse index the same way it does IVF/graph (see §3 VACUUM). **MUST add `+
   sparse_offsets_count + sparse_indices_count + sparse_values_count`,** or the
   tombstone chain lands on top of `sparse_values` on the first VACUUM of a
   sparse index — the v1.24.0 corruption reproduced exactly. **Second most
   important edit after Family A.**

### The sparse writer's OWN tombstone placement (NEW, mirrors C4/C5)

7. `write_full_sparse_parts()` (NEW, §3) re-persists an existing tombstone
   bitmap in the same rewrite (the M2 lesson: a rewrite that drops the bitmap
   resurrects deleted rows). Its `after_all` sum must be `1 + ids_count +
   sparse_offsets_count + sparse_indices_count + sparse_values_count` (plus the
   0-valued codes/scales/blocked/rotation/coarse/cell_dir/graph/bq_mean for
   uniformity with the other after_all sums). Write it identically shaped to C5
   so the two paths agree on placement.

### Summary table — what to add where

| # | Site | File:fn | Sparse reached today? | Action |
|---|------|---------|----------------------|--------|
| A | `total_blocks()` | page.rs | YES | add 3 counts — CRITICAL |
| B1 | `plan_sparse` starts | page.rs (new) | YES | new sum, get right at birth |
| B2 | `set_ivf_chains` after_rotation | page.rs | no (flat-only) | add 3 (uniformity, 0 today) |
| B3 | `set_graph_chain` after_every_prior | page.rs | no | add 3 (uniformity, 0 today) |
| C4 | `write_full_bq_parts` after_all | relfile.rs | no | add 3 (uniformity, 0 today) |
| C5 | `write_full_inner_with_tombstones` after_all | relfile.rs | no | add 3 (uniformity, 0 today) |
| C6 | `write_tombstones_and_meta` after_all_other | relfile.rs | YES | add 3 — CRITICAL |
| C7 | `write_full_sparse_parts` after_all | relfile.rs (new) | YES | new sum, mirror C5 |

Grep guard for the test/review (§5): `grep -n "bq_mean_count" src/index/page.rs
src/index/relfile.rs` currently returns these exact sites; after this change,
`grep -n "sparse_values_count"` MUST return the same set (A, B2, B3, C4, C5, C6)
plus the two new builders (B1, C7). That grep-parity is the mechanical check.

---

## 3. Build / write / read / insert / VACUUM

### Kind selection (build.rs `ambuild`)

Detect sparse from the indexed **column type** = `sparsevec`, the way ColBERT is
detected from an array type (`is_colbert_index`). Add `is_sparse_index(rel) ->
bool` reading attribute 0's `atttypid` and comparing against the `sparsevec`
type OID (look it up by name in the `turbovec` schema, or via the opclass — see
§4). Reject conflicting reloptions: `bit_width` (sparse has no quantization),
`lists`/`assign_dups` (flat-only, §6), `graph`. Dispatch to a new
`sparse_build_and_write` alongside `bq_build_and_write` / `graph_build_and_write`.

`sparse_build_and_write`: heap-scan callback decodes each `Sparsevec` (reuse the
existing `FromDatum` path + per-tuple context, exactly like the vector path),
validates `dim` consistency across rows (all rows must share the column's
declared dim — `sparsevec` already enforces sorted-unique in-range indices in
its constructor), and streams `(indices, values, nnz)` into three growing
`Vec`s plus the ids. No k-means, no rotation, no spill needed for FLAT (the
memo's whole point: sparse FLAT is cheap). At end-of-scan, build the CSR offset
prefix-sum and call `relfile::write_full_sparse`.

Note the memo's warning about the two insert timings: sparse writes
**synchronously in aminsert** if it follows the BQ model, OR defers to PreCommit
if it follows TurboQuant. **Decision D1 below** — recommend the BQ (synchronous)
model since sparse has no deferred-cache machinery and it makes `#[pg_test]`
inserts actually exercise the flush.

### Writer (relfile.rs, NEW — separate writer like `write_full_bq_parts`)

`write_full_sparse` (thin) → `write_full_sparse_parts(rel, dim, n_vectors,
offsets: &[u64], indices: &[i32], values: &[f32], slot_to_id: &[u64],
am_version, tombstones: &[u8])`:

- `plan_sparse(dim, n_vectors, total_nnz=values.len(), am_version)`.
- assert lengths: `offsets.len() == n_vectors + 1`, `indices.len() ==
  values.len() == total_nnz`, `slot_to_id.len() == n_vectors`,
  `offsets[n_vectors] == total_nnz` (the CSR invariant — assert, release-mode,
  like the `slot_to_id` HARD PERSIST-SITE GUARD).
- plan tombstone chain LAST via the C7 sum (§2).
- `extend_to(rel, meta.total_blocks().max(1))`.
- `write_chain_at` the ids chain (8 B/row, `rows_per_ids_page`), then the three
  sparse chains as flat byte chains (`stride=1, rows_per_page=PAYLOAD_BYTES`),
  reinterpreting `&[u64]`/`&[i32]`/`&[f32]` as `&[u8]` via `from_raw_parts` —
  the exact idiom `write_full_bq_parts` uses for the mean chain.
- tombstone chain if present.
- `write_meta` **LAST** (the atomic-complete crash-safety invariant — meta
  written after every chain).

### Reader (relfile.rs, NEW — `read_full_sparse`)

Under the shared rewrite lock (`lock_relfile_read` / `read_full_consistent`
pattern — a sparse read must snapshot the same consistent meta the flat path
does): `read_chain` the ids (u64), offsets (u64), indices (i32), values (f32),
reinterpreting bytes back. Return `(meta, offsets, indices, values, ids)`. Do
NOT densify. Reconstruct per-row `Sparsevec` views lazily during the scan
(slice `indices[off[i]..off[i+1]]` — zero-copy borrow, no per-row alloc).

### Scan (scan.rs)

`ambeginscan`: the `is_legacy_v7()` gate (line 297) still fires first for
pre-v8 indexes (a sparse index is v8, so it passes). Add a `KIND_SPARSE` arm.
The scan opclass carries the distance (IP/cosine — §4); a sparse index DOES
support `ORDER BY <#>` / `<=>` (unlike ColBERT), so do NOT reject it.

`amgettuple` dispatch (line 591, before the `is_bq()` branch or alongside it):
`if meta.is_sparse() { install_sparse_index(...) }`. The installed index is a
new `cache.rs` variant (or a lightweight struct held on the scan opaque — see
D2) holding the borrowed CSR arrays + ids. Its `search(query: &Sparsevec, k)`:

- For each live slot `i` (skip tombstoned via the bitmap, exactly like IVF),
  build the zero-copy row view and call the **existing `sparse_walk`** kernel
  (reuse `sparsevec_ops::sparse_walk` — expose it `pub(crate)` or lift the
  IP accumulation into a `pub(crate) fn sparse_ip(a_idx, a_val, b_idx, b_val)
  -> f64`). Accumulate top-k by IP (or cosine — precompute row norms once at
  install, or store nothing and compute norm from values during the walk).
- Emit the top-k slot ids; the executor's `xs_recheckorderby = true` path
  (already set, line 885) fetches the heap tuple and recomputes the EXACT
  `sparsevec` distance via the operator, so the index ranking need only be a
  correct candidate set. For exact FLAT it already IS exact, so recall = 1.0 —
  recheck is belt-and-braces + gives correct absolute distances.

The query `Sparsevec` arrives as the scan key datum (the `<#>` right operand),
decoded the same way the flat path decodes the query `vector`.

### Insert (insert.rs)

`aminsert` arm for `is_sparse()`: read the whole sparse relfile, append the new
row's `(indices, values)` to the three arrays + the offset + the id, rewrite
via `write_full_sparse_parts` (carrying any existing tombstone bitmap — the M2
guard). This is a whole-relfile rewrite, same as the graph insert path
(`insert.rs:332`). O(total_nnz) per insert; acceptable for FLAT (the memo
accepts FLAT's O(n) wall; WAND is the answer if that bites — deferred).
**Synchronous in aminsert (D1), like BQ.**

### VACUUM (vacuum.rs)

`ambulkdelete`: sparse joins the `is_graph() || is_bq()` tombstone branch
(line 200) — mark dead slots in the per-slot bitmap via
`write_tombstones_and_meta` (which now counts the sparse chains — C6). Do NOT
compact/rewrite the CSR arrays on vacuum (tombstone-only, like IVF/BQ/graph);
the scan masks tombstoned slots. A future `amvacuumcleanup` compaction can
rewrite via `write_full_sparse_parts` dropping dead rows, but tombstone-only is
the minimal correct behavior and matches every other kind.

---

## 4. SQL surface

Two opclasses over `sparsevec` (IP is the SPLADE-relevant one; cosine for
completeness and pgvector parity). Follow the `vec_*_ops` naming from
`options.rs` / `mod.rs`:

```sql
CREATE OPERATOR CLASS sparsevec_ip_ops
    DEFAULT FOR TYPE sparsevec USING turbovec AS
        OPERATOR 1 <#> (sparsevec, sparsevec) FOR ORDER BY float_ops,
        FUNCTION 1 sparsevec_negative_inner_product(sparsevec, sparsevec);

CREATE OPERATOR CLASS sparsevec_cosine_ops
    FOR TYPE sparsevec USING turbovec AS
        OPERATOR 1 <=> (sparsevec, sparsevec) FOR ORDER BY float_ops,
        FUNCTION 1 sparsevec_cosine_distance(sparsevec, sparsevec);
```

- Operators `<#>`, `<=>` over `sparsevec` **already exist** (sparsevec_ops.rs
  `extension_sql!`) — the opclass just references them. `<->` (L2) and `<+>`
  (L1) also exist; add `sparsevec_l2_ops` / `sparsevec_l1_ops` only if wanted
  (L2/L1 sparse ANN is niche — SKIP unless asked, YAGNI).
- `FUNCTION 1` (amsupport = 1, already the AM's `amsupport`) points at the
  existing distance functions.
- Add these `CREATE OPERATOR CLASS` blocks to the `turbovec_index_am`
  `extension_sql!` in `src/index/mod.rs`, with `requires` extended to include
  the sparsevec functions + the `sparsevec_surface` sql name so ordering is
  correct.
- **`amvalidate` (validate.rs)** is a stub returning `true` — it needs NO
  change for correctness (it validates nothing today). Leave it; changing it is
  out of scope and risks nothing.
- Column type detection in `ambuild` (§3) uses the `sparsevec` type OID. Get it
  via `pgrx`'s type registration (the `PostgresType` derive registers it) or
  `regtypein("turbovec.sparsevec")` cached once.

---

## 5. Tests (fail-before / pass-after + no-recorrupt, per the HARD MANDATE)

All `#[pg_test]` in `src/index/*.rs` or `src/lib.rs`. Every persist-path test
must drive the ACTUAL write (synchronous aminsert makes this straightforward —
D1; if deferred were chosen, use `xact::flush_to_relfile_for_test`, per the
AGENTS.md warning).

1. **`sparse_meta_round_trips`** (page.rs) — `plan_sparse` → `encode` →
   `decode` == original; assert `is_sparse()`, chain offsets non-overlapping,
   `total_blocks()` == sum of chain pages + 1. Pure, no PG.

2. **`sparse_build_scan_correctness`** — CREATE INDEX over a small `sparsevec`
   column, `ORDER BY col <#> query LIMIT k`, assert the returned ids match a
   brute-force `sparsevec_negative_inner_product` computed in SQL over the same
   rows. Exact FLAT ⇒ must match exactly (recall = 1.0). Repeat for `<=>`.

3. **`sparse_recall_vs_brute_force`** — larger synthetic Zipfian corpus (memo
   §5: ~vocab 30k, per-doc nnz ~150); assert R@10 == 1.0 against the seqscan
   `<#>` baseline (exact FLAT). This is the "recall vs brute-force sparse
   baseline" the task asks for; for FLAT it is an equality assertion, not a
   fuzzy recall bound.

4. **`sparse_insert_then_scan`** — build empty/small, INSERT rows (synchronous
   path), scan, assert new rows are found and ranked correctly. This exercises
   the aminsert whole-relfile-rewrite + the CSR append.

5. **`sparse_chain_offset_running_sum_guard`** (the corruption guard the task
   demands) — the "add-a-chain → every running sum" mechanical test. Build a
   sparse index that populates all three chains AND has ≥1 tombstoned row (so
   VACUUM's `write_tombstones_and_meta` C6 sum runs), then `turbovec_check()`
   MUST report `is_corrupt = false` and the ids MUST be unique. A pre-fix build
   (C6 sum missing the sparse counts) places the tombstone chain on top of
   `sparse_values` → the check catches duplicate/garbage ids. Assert the
   tombstone chain's first block > `sparse_values_first + sparse_values_count`
   by reading the meta. This is the fail-before/pass-after: temporarily reverting
   the C6 edit makes it fail.

6. **`sparse_no_recorrupt_under_insert_load`** (the mandated sustained-load
   validation, per the v1.28.4 lesson) — build, then N sequential INSERTs
   (each a rewrite), interleaved VACUUMs, then `turbovec_check()` clean AND a
   full scan returns exactly the live set. Run at a size that crosses several
   `PAD_PAGES`/page boundaries so chain shifts are exercised.

7. **`existing_dense_index_still_decodes`** (wire-compat) — build a plain
   `vec_ip_ops` flat index and a `bit_width=1` BQ index, then (in the same test
   binary that now knows `KIND_SPARSE`) assert they still `decode()` with
   `kind == KIND_SINGLE` / `KIND_BQ`, scan correctly, and their meta round-trips
   unchanged. Proves the additive decode: the new sparse meta fields read as 0
   on a non-sparse page and change nothing.

8. **`wire_format_version_is_stable`** (lib.rs, EXISTS) — `EXPECTED_WIRE_FORMAT_
   VERSION` stays **8**. This test must keep passing UNCHANGED, which is the
   proof that KIND_SPARSE did not bump the wire version. If it fails, the design
   was violated (someone bumped VERSION).

9. **`turbovec_check` sparse arm** — extend the kind-name map (extras.rs:434) to
   return `"sparse"` for `KIND_SPARSE`, and make `turbovec_check` validate the
   sparse chains (ids unique, `offsets` monotonic, `offsets[n] == total_nnz`,
   indices within `[0,dim)` and sorted-unique per row). Test it flags a
   deliberately corrupted sparse index.

---

## 6. Version-bump touch-list + the KIND_SPARSE additive-decode story

**This is a MINOR bump** (additive SQL surface: two new opclasses; additive wire:
new kind byte, VERSION unchanged; no REINDEX for any existing index). Per
AGENTS.md the minor requires a checked-in migration file, a generated upgrade
SQL script, an UPGRADING.md matrix row, and a CHANGELOG entry.

### Wire / decode story (the additive contract)

- `VERSION` **stays 8.** `KIND_SPARSE = 4` is the sole new discriminator, in a
  byte (offset 6) that has been present since v5. Existing flat/IVF/BQ/graph
  indexes keep their `kind` and decode byte-identically (the new
  `sparse_*` meta fields live in the reserved tail, read as 0 ⇒ "no sparse
  chains", exactly the v6-graph / v8-bq_mean additive-decode pattern).
- `is_legacy_v7()` (the live gate) is UNCHANGED — a sparse index is v8, passes
  the gate; pre-v8 indexes still get the REINDEX error. Add an
  `is_legacy_v8() -> bool { false }` **only if** you want the AGENTS.md "every
  wire bump ships an is_legacy_v{N}" slot filled — but since VERSION does NOT
  bump, this is arguably not required. **Decision D3.** Recommend: skip it
  (no version bump ⇒ no new legacy predicate needed; the existing deliberately-
  `false` predicates document that pattern).

### Full touch-list

Code:
- `src/index/page.rs` — `KIND_SPARSE` const; 10 new `MetaPageData` fields;
  `plan_sparse`; `is_sparse`/`has_sparse`; `encode`/`decode` sparse block;
  `total_blocks()` (A); `set_ivf_chains` (B2) + `set_graph_chain` (B3) sums;
  the `debug_assert!` in `encode` matches-list gets `| KIND_SPARSE`; the
  `turbovec_check`-shaped assertions.
- `src/index/relfile.rs` — `write_full_sparse` + `write_full_sparse_parts`
  (C7 sum) + `read_full_sparse` + `read_sparse_*` chain readers;
  `write_full_bq_parts` (C4), `write_full_inner_with_tombstones` (C5),
  `write_tombstones_and_meta` (C6) sums.
- `src/index/build.rs` — `is_sparse_index`; `ambuild` dispatch;
  `sparse_build_and_write`.
- `src/index/scan.rs` — `ambeginscan` allow (do NOT reject sparse);
  `amgettuple` `install_sparse_index` dispatch; the sparse search loop.
- `src/index/insert.rs` — `aminsert` sparse arm (whole-relfile rewrite).
- `src/index/vacuum.rs` — add sparse to the tombstone branch (line 200).
- `src/index/options.rs` — reject `bit_width`/`lists`/`assign_dups`/`graph`
  on a sparse column with a clear ERROR (sparse is flat-only, unquantized).
- `src/index/mod.rs` — two `CREATE OPERATOR CLASS` blocks in `turbovec_index_am`
  extension_sql + `requires`.
- `src/sparsevec_ops.rs` — expose `sparse_walk` / a `sparse_ip` as `pub(crate)`
  for the scan kernel (or lift a shared helper).
- `src/extras.rs` — `turbovec_check` kind map + sparse validation (line 434).
- `src/lib.rs` — the 9 new `#[pg_test]`s; `wire_format_version_is_stable`
  UNCHANGED.
- `src/cache.rs` — a `ReadOnlyIndex` sparse variant OR a scan-local struct (D2).

Release engineering (AGENTS.md minor checklist):
- `Cargo.toml` version bump (minor, e.g. 2.8.0).
- `migrations/NNN_pg_turbovec_v2.8.0.sql` — checked in (contains the two
  `CREATE OPERATOR CLASS` + any new function decls).
- `sql/pg_turbovec--<from>--2.8.0.sql` generated via `cargo pgrx schema` and
  committed, so `ALTER EXTENSION pg_turbovec UPDATE` creates the opclasses
  in place (the v1.28.4 lesson: the upgrade script must actually ship, or the
  opclass never gets created on in-place upgrade).
- `docs/UPGRADING.md` — new matrix row: `2.0.0–2.7.x → 2.8.0: ALTER EXTENSION
  only, no REINDEX (additive KIND_SPARSE, wire still v8)`.
- `CHANGELOG.md` — dated entry + Migration section ("ALTER EXTENSION only").
- `docs/` — a sparse-ANN usage doc (or a section in HYBRID_SEARCH.md, which
  already documents the seqscan `<#>` path — now point it at the index).
- drift-check §7 passes automatically (VERSION unchanged on a minor is fine;
  the gate only fires when VERSION moves on a PATCH). Test-count line in
  README/CHANGELOG updates by +~9.

---

## 7. Decisions to confirm (flagged for you)

- **D1 — insert timing: synchronous (BQ-style) vs deferred (TurboQuant-style).**
  RECOMMEND synchronous-in-aminsert (whole-relfile rewrite like graph/BQ).
  Reason: sparse has no deferred-cache infrastructure, and synchronous makes
  `#[pg_test]` INSERTs actually exercise the flush (the AGENTS.md trap: a plain
  INSERT in a test never hits the TurboQuant deferred path). Confirm.

- **D2 — where the installed sparse index lives: a `cache.rs` `ReadOnlyIndex`
  variant vs a scan-local struct on the scan opaque.** RECOMMEND scan-local
  (no cross-backend cache) for v1 — FLAT reload is cheap and it avoids widening
  the cache enum. Add a cache variant later only if reload cost is measured to
  bite. Confirm.

- **D3 — ship an `is_legacy_v8()` predicate?** RECOMMEND no (VERSION doesn't
  bump, so there's no new legacy tier). The AGENTS.md "every wire bump ships a
  legacy predicate" rule is about VERSION bumps, and this is a kind-byte add.
  Confirm you're OK skipping it.

- **D4 — cosine row-norm handling.** Precompute per-row norms once at install
  (O(total_nnz), stored in a scan-local `Vec<f32>`) vs recompute during each
  walk. RECOMMEND precompute at install (queries reuse it). Not a persist
  decision (norms are derived, never stored on disk), so no wire impact. Confirm
  or leave to implementer.

- **D5 — scope of opclasses: IP + cosine only, or also L2/L1?** RECOMMEND IP +
  cosine only (SPLADE ranks by IP; cosine for pgvector parity). L2/L1 sparse
  ANN is niche. Confirm before I'd add `sparsevec_l2_ops`/`l1_ops`.

- **D6 — `dim` ceiling for sparse.** `sparsevec::MAX_DIM` is 1e9 but the meta
  `dim` field is `u32` (max ~4.29e9, fine) and we never densify (§3, scan is
  sparse-native). So the 16000-dim `vector` ceiling does NOT apply — a 30k-dim
  SPLADE index is fine. Confirm you want to allow the full sparsevec dim range
  (I see no reason to cap it, since nothing densifies).

---

### Critical Files for Implementation
- /home/gburd/ws/pg_turbovec/src/index/page.rs — `KIND_SPARSE`, `plan_sparse`, the meta fields, and running-sum sites A/B2/B3 all live here; the encode/decode additive block is the wire-compat linchpin.
- /home/gburd/ws/pg_turbovec/src/index/relfile.rs — the new `write_full_sparse_parts`/`read_full_sparse` writer/reader and the corruption-critical running-sum sites C4/C5/C6/C7.
- /home/gburd/ws/pg_turbovec/src/index/build.rs — `ambuild` kind dispatch (`is_sparse_index`) and `sparse_build_and_write`, modeled on `bq_build_and_write`.
- /home/gburd/ws/pg_turbovec/src/index/scan.rs — `amgettuple` sparse dispatch + the `sparse_walk` top-k search loop reusing `sparsevec_ops`.
- /home/gburd/ws/pg_turbovec/src/index/mod.rs — the two `CREATE OPERATOR CLASS` blocks (`sparsevec_ip_ops`/`sparsevec_cosine_ops`) that make the AM indexable for `sparsevec`.
