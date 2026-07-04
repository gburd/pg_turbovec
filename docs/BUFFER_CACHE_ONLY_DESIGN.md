# Design: all index data through the PostgreSQL buffer cache (no relfile mmap)

_Status: **IMPLEMENTED (v1.19.0, 2026-06-29).** Requirement: **every
byte of index data must be read via PostgreSQL's shared-buffer cache
(`ReadBufferExtended`)** — no direct relfile `mmap`/`pread`/`open`.
This is the correct posture for sandboxed/managed Postgres, and it
makes the buffer manager the single source of truth for page access
(consistent pinning, locking, and crash/replication semantics)._

_§1.2's GUC-removal plan ("keep the deprecated no-op for one minor,
then remove it") was also carried out: `turbovec.mmap_static_blocked`
was removed in v1.22.0. This design doc predates both changes (it was
left uncommitted after the implementation shipped); it is retained as
the design rationale, not a pending TODO — the plan below is done._

---

## 0. Why this is tractable: the buffer-manager path already exists

pg_turbovec's relfile reads were always dual-pathed. The buffer-manager
readers are present, tested, and used today as the fallback:

| Region | Buffer-manager reader (`src/index/relfile.rs`) | mmap reader to remove |
|---|---|---|
| meta page | `read_meta` (always buffer-managed) | — |
| whole codes/scales/ids | `read_full` | — |
| blocked codes (whole) | `read_blocked` | `StaticRegionsMap::read_static_chains` |
| rotation matrix | `read_rotation` | (in static-regions map) |
| scales only (OOC) | `read_scales_only` | — |
| **probed-cell codes (OOC)** | **`gather_codes_ranges`** | `StaticRegionsMap::gather_slot_ranges` |

Every reader uses `read_block` → `ReadBufferExtended(rel, MAIN_FORKNUM,
blkno, RBM_NORMAL)` → `page_data` → `UnlockReleaseBuffer`. The two
mmap call sites in `scan.rs` (`load_static_regions` for the whole-load
path, `StaticRegionsMap::open_for_ooc` for the cell-scoped path) each
**already have a buffer-manager `else` branch** (`scan.rs:709-718`,
`cache.rs:463-473`). So this is primarily a **removal**, not a rewrite.

**Correctness is unaffected:** the mmap path was always
*opt-out-able* via `turbovec.mmap_static_blocked` and the AM's
contract (heap-visibility + `xs_recheckorderby` as the source of
truth) is identical on both paths. Removing mmap removes a performance
fork, never a correctness one.

---

## 1. What changes

### 1.1 Make the buffer-manager path the ONLY path
- **Delete** `src/index/mmap_static.rs` entirely (the `File::open` +
  `memmap2` + `relfile_path` machinery — the only code that touches a
  relfile outside the buffer manager).
- **Remove** the `memmap2` dependency from `Cargo.toml`.
- **`scan.rs` whole-load fill** (`install_whole_index`, ~709): drop the
  `mmap_enabled && load_static_regions(...)` branch; always take the
  `read_blocked` / `read_rotation` / `read_full` buffer-manager branch
  that already exists below it.
- **`scan.rs` OOC install** (`try_install_ooc`, ~837): drop
  `StaticRegionsMap::open_for_ooc`; the `OocIvfIndex` carries
  `map: None` and `search_ooc` always takes the
  `gather_codes_ranges` (buffer-manager) arm at `cache.rs:473`.
- **`cache.rs`**: remove the `mmap: Option<StaticRegionsMap>` field
  from `Entry` and `OocIvfIndex`, and the `insert_with_mmap` variant
  (fold back to `insert`). The `OocIvfIndex` keeps everything else
  (cell directory, centroids, rotation, scales, slot_to_id) — those
  were already resident, not mmap'd.

### 1.2 Retire the GUC
- `turbovec.mmap_static_blocked` becomes meaningless. Options:
  - **(preferred)** remove it, and document the removal in
    `UPGRADING.md` (a removed GUC is a minor-release-with-deprecation
    concern per `AGENTS.md` — but a GUC that selected an
    now-impossible code path can be removed directly with a note; it
    never affected results). A stale `SET turbovec.mmap_static_blocked`
    in a user's session would then error — so the safer path is:
  - **(safer, recommended)** keep the GUC name as a deprecated no-op
    for one minor (accept + ignore it, log a one-time `WARNING` that
    it's deprecated and ignored), then remove it the following minor.
    This avoids breaking any startup `SET` in existing configs.

### 1.3 Nothing else moves
The codes/scales/ids/blocked/rotation/meta chains, the relfile WRITE
path (all `GenericXLog`-WAL'd through the buffer manager already), the
IVF cell layout, tombstones, determinism, and the wire format are
**unchanged**. This is a read-path-only change. **No wire change, no
REINDEX.**

---

## 2. Preserving the properties mmap gave us (the real design work)

Removing mmap reintroduces the two costs it was built to avoid. The
design must keep them bounded *through the buffer manager*:

### 2.1 The out-of-core bounded-RSS property — PRESERVED for free
The OOC win was "resident set = O(probes·cell_size), not O(n)."
`gather_codes_ranges` (the buffer-manager twin) **already reads only
the probed cells' pages** — it walks the probed `(start, count)` slot
ranges, pins/copies/releases each touched page, and never reads the
rest of the codes chain. So a >RAM IVF index stays bounded on the
buffer-manager path: the per-query working set is the probed cells'
pages (which live in shared_buffers, evicted by PG's clock-sweep under
pressure) plus the compact gathered `Vec`. **This is the key result:
out-of-core serving does NOT require mmap — it requires cell-contiguous
layout + range-scoped reads, both of which the buffer-manager path
has.**

### 2.2 The double-caching cost — accept it, and lean on compression
mmap avoided caching static pages in *both* the OS page cache and
shared_buffers. Through the buffer manager, index pages live in
shared_buffers (once). The mitigation is structural and already ours:
**pg_turbovec's 7–15× compression makes the index small enough to fit
shared_buffers** where a fp32 HNSW could not. Guidance (PRODUCTION.md):
size `shared_buffers` to hold the hot compressed index; at 4-bit a
1M×1536-d index is ~768 MB of codes, which fits a modest
`shared_buffers`. This is the honest tradeoff: no double-cache to
avoid because there's only one cache (shared_buffers), and the index
is compact enough to live in it.

### 2.3 The per-page lookup/pin/lock cost — mitigate, don't eliminate
The buffer manager's `BufTableLookup`/pin/lock per page is real
overhead mmap skipped. We cannot eliminate it (it's the price of using
the cache), but we reduce its frequency:
- **Sequential page walks** (`read_blocked`, `read_full`) touch each
  page once per cache-fill, and the result is cached in the
  per-backend `ReadOnlyIndex` `Vec` — so the per-page cost is paid
  once per (backend, am_version), not per query. Warm queries hit the
  resident `Vec`, never the buffer manager.
- **For the OOC path**, `gather_codes_ranges` pays the per-page cost
  per query (it re-gathers the probed cells), but only for the probed
  cells' pages — bounded by `probes·cell_size`, not `n`. The cell
  directory keeps a cell's pages contiguous, so the walk is a few
  sequential `read_block`s per probed cell.
- **Optional optimisation (deferred):** `ReadBufferExtended` supports
  a `BufferAccessStrategy` (ring buffer, `BAS_BULKREAD`) for large
  sequential scans so a big cache-fill doesn't evict the whole
  shared_buffers working set. The whole-load cache-fill (`read_full`/
  `read_blocked`) is exactly a bulk sequential read and is the
  candidate for `GetAccessStrategy(BAS_BULKREAD)`. This is a pure
  buffer-manager-API addition (no relfile access), keeps everything in
  the cache, and is the principled way to read a >shared_buffers index
  without thrashing the pool. Add it if profiling shows cache-fill
  evicting the hot set.

---

## 3. Performance expectation (honest)

- **Warm queries:** unchanged. Both paths cache the prepared index in
  a per-backend `Vec`; warm scans never touch the buffer manager.
  mmap only ever helped the **cache-fill** (cold) phase.
- **Cold cache-fill:** slower than mmap by the per-page
  lookup/pin/lock + the buffer-manager copy. The v1.4.0 profile that
  motivated mmap showed buffer-manager reads at ~37%+29%+28% of a cold
  warm-scan; expect cold-fill latency to rise toward those numbers on
  a >shared_buffers index. Mitigations: size shared_buffers to the
  compact index (§2.2), `BAS_BULKREAD` strategy (§2.3).
- **Out-of-core (>RAM) serving:** preserved — bounded RSS via
  `gather_codes_ranges` (§2.1). Per-query cold-cell reads now fault
  through the buffer manager (pin/copy/release) instead of an mmap
  page fault; comparable cost, fully cache-managed.
- **The win we GIVE UP:** the double-cache avoidance and the
  per-page-overhead skip on cold fills of a >shared_buffers index.
  The win we KEEP: correctness, replication/crash semantics (already
  via the buffer manager + WAL), out-of-core bounded RSS, and full
  compatibility with sandboxed/managed Postgres.

---

## 4. Tests / validation

- All 241 tests must stay green (they run with the buffer-manager path
  already reachable; removing mmap just makes it the only path).
- `relfile_wal_emits_on_build_and_insert`, the OOC tests
  (`ivf_ooc_*`), the byte-identity and recall-floor tests are the
  regression guards.
- Add/repurpose a test asserting the OOC path keeps bounded resident
  behaviour with no mmap (the existing `ivf_ooc_installs_cell_scoped_handle`
  + a check that the cache entry has no mmap handle — trivially true
  after the field is removed).
- `compile-matrix` (pg13–18) + drift-check.

---

## 5. Migration / release

- **Minor release** (read-path behaviour + a GUC deprecation). Wire
  format UNCHANGED (`MetaPageData::version` = 5; single-vector v4).
  **No REINDEX.**
- `UPGRADING.md`: note that `turbovec.mmap_static_blocked` is
  deprecated (a no-op that warns) — or removed — and that all index
  reads now go through `shared_buffers`; recommend sizing
  `shared_buffers` to the (compact) hot index.
- `PRODUCTION.md`: replace the mmap section with the shared_buffers
  sizing guidance (§2.2) and the `BAS_BULKREAD` note if added.
- `ARCHITECTURE.md`: drop the "mmap isolation contract" section;
  replace with "all reads via the buffer manager; heap-visibility +
  xs_recheckorderby remain the correctness backstops."

---

## 6. The one-line summary

mmap was a **cold-fill performance optimisation**, never a correctness
or functional requirement. Removing it and routing every read through
`ReadBufferExtended` is mostly a **deletion** (the buffer-manager
readers already exist and are tested), preserves out-of-core bounded
RSS (via `gather_codes_ranges`' cell-scoped reads), and trades cold-
fill speed on >shared_buffers indexes for full buffer-cache
discipline — a tradeoff pg_turbovec's 7–15× compression is uniquely
positioned to absorb by making the index fit shared_buffers in the
first place.
