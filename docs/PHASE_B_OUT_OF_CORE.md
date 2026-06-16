# Phase B: out-of-core / >RAM design

_Drafted 2026-06-16 alongside the Phase A-2 latency measurement.
The goal: let a pg_turbovec IVF index exceed available RAM, with the
resident set bounded by the hot (probed) cells, not the whole index.
This is the answer to pgvectorscale's DiskANN out-of-core bet, and
it's needed as the production >5M deployment grows._

---

## The current RAM ceiling (what blocks >RAM today)

`amgettuple`'s cache-miss path (src/index/scan.rs ~432):

1. `relfile::read_full()` slurps the **entire** codes/scales/ids
   chains into per-backend `Vec<u8>`s.
2. The mmap fast path (`mmap_static::load_static_regions`) maps the
   blocked-codes / rotation / codebook chains \u2014 BUT then **copies
   them into contiguous `Vec<u8>`s** (`read_chain`), because the
   relfile has a **24-byte `PageHeaderData` gap every 8168 bytes**
   (`PAYLOAD_BYTES`), so the mmap'd bytes aren't contiguous and
   turbovec's `search` (which wants a contiguous `&[u8]`) can't read
   the mapping in place.

Net: the per-backend resident set is `O(n)` \u2014 the whole index. An
index larger than RAM cannot be served. Even the "mmap" path is
really "mmap then memcpy the whole thing."

---

## The two-layer unlock

### Layer 1 (tractable now, no wire change): cell-scoped fault-in

For an **IVF** index, a query only needs the **probed cells'**
contiguous slot ranges (the cell directory gives each cell's
`[code_offset, code_offset + n_vectors)`). So:

- Don't read/copy the whole codes chain. For each probed cell, copy
  **only that cell's byte range** off the mmap (the existing
  `read_chain` walk, but bounded to the cell's pages).
- The resident set per query becomes `O(probes \u00b7 cell_size)` instead
  of `O(n)`. At `probes = 16`, `lists = \u221an`, that's ~`16/\u221an` of the
  index \u2014 e.g. ~1.6% at 1M, ~0.7% at 5M.
- The OS page cache holds the recently-touched cells; cold cells are
  faulted from disk on demand and evicted under memory pressure.
  **This is out-of-core search** \u2014 the index can exceed RAM as long
  as the working set (hot cells) fits.
- Works with the **current** header-gap layout (we still copy, but
  only the probed cells \u2014 a tiny fraction). No wire-format change,
  no REINDEX. **This is the cheap 80%.**

Caveat: we still `memcpy` the probed cells (per query, or cache the
hot ones). The copy is small (`probes \u00b7 cell_size`), so it's fine \u2014
but it's per-query work unless we cache. A per-backend LRU of
recently-scanned cell ranges (bounded by `cache_size_mb`) avoids
re-copying hot cells across queries.

### Layer 2 (deeper, v5 wire change): zero-copy gapless codes

To go *fully* zero-copy \u2014 turbovec scans the mmap'd codes in place,
no memcpy at all, resident set = OS-paged-in cells only:

- Store the codes (and the SIMD-blocked codes) in a **header-gap-free
  segment**: either a separate file/fork laid out as raw contiguous
  bytes (no per-8KB PG page header), or a relfile region we mmap and
  read past the headers via a stride-aware accessor that turbovec can
  consume. turbovec's `search` would need to accept a "blocked codes
  with stride/gap metadata" view, OR we lay the blocked codes out
  gaplessly so a plain `&[u8]` slice over the mmap works.
- This is a `MetaPageData::version` 4 \u2192 5 change (a new gapless
  codes segment), so a **minor release with the no-REINDEX-for-v4
  path** (v5 binary reads v4 by falling back to the copy path; only
  `WITH (...)` rebuilds get the gapless segment). Per the AGENTS.md
  policy, online-upgradable.
- Payoff: true zero-copy. An index 10\u00d7 RAM works with the resident
  set = exactly the faulted-in cell pages, no per-query copy. This is
  the pgvectorscale-DiskANN-equivalent capability.

**Recommendation: ship Layer 1 first** (cell-scoped fault-in, no
wire change \u2014 gets most of the >RAM benefit with low risk), measure
the cold-cell fault latency, then decide if Layer 2's zero-copy is
worth the v5 wire change. For a 7\u201315\u00d7-compressed index, Layer 1 may
be enough: the whole index is already small, so even copying probed
cells from a mostly-RAM-resident mmap is fast.

---

## Why our compression is the out-of-core advantage

pgvectorscale needs DiskANN because a full-precision HNSW graph at
100M\u00d7768-d is ~300 GB \u2014 it *must* go to disk. pg_turbovec's 4-bit
IVF index of the same corpus is ~20\u00d7 smaller (~15 GB), which **fits
in RAM on a commodity box**. So:

- For corpora where HNSW is already swapping, pg_turbovec is still
  RAM-resident. Out-of-core is rarer for us.
- When we *do* exceed RAM, Layer 1's cell-scoped fault-in means only
  the hot `probes \u00b7 cell_size` fraction needs to be resident.
- The storage hierarchy: hot cells in RAM (OS page cache), cold cells
  on disk (SSD/NVMe), the quantization meaning the on-disk footprint
  and the fault-in bytes are both 7\u201315\u00d7 smaller than a
  full-precision index would page.

**Thesis: "pg_turbovec goes out-of-core by being small enough that
it rarely has to, and by paging only hot cells when it does."**

---

## Implementation plan

### B-1: cell-scoped fault-in (Layer 1, no wire change)

- In `amgettuple`'s IVF path, replace the whole-index `read_full` /
  `load_static_regions` copy with a per-probed-cell copy: map the
  codes/scales chains, copy only the probed cells' byte ranges into
  the (much smaller) working buffer turbovec scans.
- Keep the cell ranges in a per-backend LRU (bounded by
  `turbovec.cache_size_mb`) so hot cells aren't re-copied per query.
- The blocked-codes layout: the SIMD-blocked codes are also
  cell-contiguous (cells were reordered at build), so the same
  cell-range slicing applies to the blocked codes the kernel scans.
- Resident set: `O(min(cache_size_mb, sum of hot cells))` instead of
  `O(n)`. Document the new memory model in PRODUCTION.md.

### B-2: validate >RAM

- Build an IVF index larger than a cgroup-limited RAM (or a small-RAM
  host / `systemd-run --property=MemoryMax`), confirm queries succeed
  with bounded RSS and measure the cold-cell fault latency penalty
  (first touch of a cold cell = disk read). Report warm (hot cells
  cached) vs cold (faulted) p50.
- This is the headline "pg_turbovec serves an index larger than RAM"
  claim \u2014 the pgvectorscale-parity deliverable.

### B-3 (optional, Layer 2): gapless zero-copy codes (v4\u2192v5)

- Only if B-1's per-cell copy proves a real bottleneck at scale.
- New gapless codes segment, stride-aware turbovec accessor or a
  truly contiguous mmap slice, v5 wire with no-REINDEX-for-v4
  fallback. Bigger lift; defer until B-2 shows it's needed.

### B-4 (pairs with out-of-core query): out-of-core BUILD

- VectorChord builds 100M on a 128 GB box. Our build holds the full
  flat corpus in memory for the cell permutation (the `lists > 0`
  path accumulates `ivf_flat`). Stream the permutation (sort slot \u2192
  cell on disk, or two-pass: assign in a streamed sweep, then write
  cells in a second streamed pass) so the build's peak is bounded.
  Pairs with the Phase W `maintenance_work_mem` cap. Needed to BUILD
  (not just query) indexes larger than RAM.

---

## Priority

B-1 + B-2 are the out-of-core query story \u2014 the answer to ">RAM".
B-4 (out-of-core build) is needed for the very largest corpora but is
separable. B-3 (zero-copy) is an optimization gated on B-2's
findings. Start B-1+B-2 after Phase A-2 confirms the at-scale latency
baseline (so we know the IVF query cost we're making memory-bounded).
