//! GUC (Grand Unified Configuration) variables exposed by pg_turbovec.
//!
//! All variables are registered under the `turbovec` namespace. They
//! can be set per-session (`SET turbovec.bit_width_default = 2;`) or
//! in `postgresql.conf`.
//!
//! | GUC                              | Type | Default | Range          |
//! |----------------------------------|------|---------|----------------|
//! | `turbovec.bit_width_default`     | int  | 4       | 2..=4          |
//! | `turbovec.cache_size_mb`         | int  | 256     | 1..=65536      |
//! | `turbovec.warn_on_rebuild`       | bool | true    | -              |
//! | `turbovec.search_concurrency`    | int  | 1       | 1..=128        |
//! | `turbovec.normalize_on_insert`   | bool | true    | -              |
//! | `turbovec.mmap_static_blocked`   | bool | true (no-op, deprecated) | -          |
//! | `turbovec.iterative_scan`        | enum | relaxed_order | off, relaxed_order |
//! | `turbovec.max_scan_tuples`       | int  | 20000   | 1..=10_000_000 |
//! | `turbovec.build_parallelism`     | int  | 0       | 0..=128        |
//! | `turbovec.scan_parallelism`      | int  | 0       | 0..=128        |
//! | `turbovec.oversample`            | float| 1.0     | 1.0..=100.0    |
//! | `turbovec.max_probes`            | int  | 64      | 1..=65536      |
//! | `turbovec.out_of_core`           | enum | auto    | off, auto, on  |
//! | `turbovec.allowlist`             | str  | `""`    | CSV of bigint ids |

use core::ffi::CStr;
use std::ffi::CString;

use pgrx::guc::PostgresGucEnum;
use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static BIT_WIDTH_DEFAULT: GucSetting<i32> = GucSetting::<i32>::new(4);
pub static CACHE_SIZE_MB: GucSetting<i32> = GucSetting::<i32>::new(256);
pub static WARN_ON_REBUILD: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_CONCURRENCY: GucSetting<i32> = GucSetting::<i32>::new(1);
pub static NORMALIZE_ON_INSERT: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_K: GucSetting<i32> = GucSetting::<i32>::new(32);
pub static PROBES: GucSetting<i32> = GucSetting::<i32>::new(8);

/// IVF-3: iterative-scan cap on probe-set growth under a selective
/// `WHERE` filter, the IVF analogue of `ivfflat.max_probes`.
///
/// Under `turbovec.iterative_scan = relaxed_order`, when the cells
/// currently probed by an IVF scan drain and the executor still wants
/// tuples, the refill WIDENS the probe set (probes, 2·probes, 4·probes,
/// …), rebuilds the cell mask, and re-runs the cell-restricted fine
/// search — instead of only growing `k` within the initial cells. That
/// recovers true neighbours whose cell was not in the initial `probes`
/// nearest set (the failure mode IVF-2's k-growth refill couldn't fix
/// under a selective filter). `max_probes` is the ceiling on that
/// growth; the widening stops at `min(max_probes, lists)`. When the
/// probe set reaches `lists` the whole corpus has been scanned and the
/// next drain returns false (exhausted).
///
/// Default `64` mirrors a typical `ivfflat.max_probes` and is 8× the
/// `turbovec.probes` default of 8. Clamped to `lists` at scan time, so
/// a value larger than the index's cell count just means "widen to all
/// cells". No effect on flat (`lists = 0`) or vacuum-degraded indexes,
/// which have no cells to widen and keep the v1.8.0 `k`-growth refill.
/// `turbovec.max_scan_tuples` still caps total candidate work as a
/// backstop regardless of probe widening.
pub static MAX_PROBES: GucSetting<i32> = GucSetting::<i32>::new(64);

/// Phase B-1/B-2 (out-of-core query): when on (the default), an
/// IVF index scanned cold from the relfile is served **cell-scoped**
/// — the backend caches only bounded metadata (coarse centroids,
/// cell directory, rotation, codebook, and the small per-slot
/// scales/ids tables), and per query gathers ONLY the probed cells'
/// contiguous code ranges through PostgreSQL's buffer manager to
/// build a compact throwaway sub-index. The per-backend resident set
/// is then `O(probes * cell_size)` instead of `O(n)`, so an IVF
/// index larger than RAM can be
/// served: the OS page cache holds hot (recently-probed) cells and
/// cold cells fault from disk on demand.
///
/// When `off`, the scan loads the WHOLE index into a per-backend
/// `Arc` (the pre-B-1 behaviour) — lowest per-query latency once
/// warm, but resident set `O(n)`, so the index must fit in RAM.
///
/// When `auto` (**the default**), the scan goes cell-scoped ONLY
/// when the index's codes are large relative to the per-backend
/// cache budget (`turbovec.cache_size_mb`) — i.e. the index that
/// actually needs out-of-core serving. An IVF index that comfortably
/// fits the budget loads whole (no per-query gather/reblock cost),
/// so `auto` pays the cell-scoped CPU tax only when it buys the
/// memory bound. `on` forces cell-scoped regardless of size; `off`
/// forces whole-load. See [`out_of_core_cell_scoped`].
///
/// **No effect on flat (`lists = 0`) or vacuum-degraded indexes:**
/// they have no cells to scope and always load whole (and are
/// therefore still `O(n)`-resident — use IVF for a >RAM corpus). No
/// effect on the mutable (post-insert) or dirty-fallback paths,
/// which keep their in-memory mirror. Results are identical to the
/// whole-load IVF path (`probes >= lists` still reduces to the flat
/// exact scan; tombstones are still masked; soft-assign duplicates
/// are still deduped by the scan's emitted-id set).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PostgresGucEnum)]
pub enum OutOfCoreMode {
    /// Always load the whole index into the per-backend `Arc`
    /// (`O(n)` resident; lowest warm latency; must fit in RAM).
    #[name = c"off"]
    Off,
    /// Cell-scoped only when the codes are large relative to
    /// `turbovec.cache_size_mb` (the default).
    #[name = c"auto"]
    Auto,
    /// Always serve IVF cell-scoped (`O(probes*cell_size)` resident;
    /// pays the per-query gather/reblock tax regardless of size).
    #[name = c"on"]
    On,
}

pub static OUT_OF_CORE: GucSetting<OutOfCoreMode> =
    GucSetting::<OutOfCoreMode>::new(OutOfCoreMode::Auto);

/// Decide whether an IVF scan should be served cell-scoped given the
/// mode and the index's codes size. `auto` goes cell-scoped when the
/// codes exceed [`AUTO_OOC_FRACTION`] of `turbovec.cache_size_mb`
/// (the codes are the `O(n)` term the whole-load path would make
/// resident; everything else cached cell-scoped is bounded).
pub fn out_of_core_cell_scoped(codes_bytes: u64) -> bool {
    let budget_bytes = (CACHE_SIZE_MB.get() as u64).saturating_mul(1024 * 1024);
    out_of_core_decide(OUT_OF_CORE.get(), codes_bytes, budget_bytes)
}

/// Pure decision used by [`out_of_core_cell_scoped`] (factored out so
/// it can be unit-tested without touching the live GUCs). `auto`
/// goes cell-scoped when the codes exceed [`AUTO_OOC_FRACTION`] of
/// the cache budget; the codes are the `O(n)` term the whole-load
/// path would make resident.
pub fn out_of_core_decide(mode: OutOfCoreMode, codes_bytes: u64, budget_bytes: u64) -> bool {
    match mode {
        OutOfCoreMode::Off => false,
        OutOfCoreMode::On => true,
        OutOfCoreMode::Auto => {
            let threshold = (budget_bytes as f64 * AUTO_OOC_FRACTION) as u64;
            codes_bytes > threshold
        }
    }
}

/// Fraction of `turbovec.cache_size_mb` above which `auto` switches
/// an IVF index to cell-scoped serving. 0.5 means: if the codes
/// alone are more than half the cache budget, prefer the memory
/// bound over the warm-latency win.
const AUTO_OOC_FRACTION: f64 = 0.5;

/// Differentiator #5 (oversampling): candidate-set widener for tunable
/// recall, matching Qdrant's `oversampling` / VectorChord's rerank knob.
///
/// turbovec's ANN search ranks candidates by the *quantized* (2-4 bit
/// TurboQuant) distance, which is lossy. The executor's reorder queue
/// (`xs_recheckorderby = true`) already re-ranks whatever candidates we
/// return by the *exact* full-precision distance — but it can only
/// re-rank the candidates we hand it. If the true nearest neighbour
/// ranked just outside `search_k` by quantized distance, no amount of
/// reorder-queue rescoring recovers it.
///
/// `oversample` is the lever that fixes THAT: the scan fetches
/// `ceil(search_k * oversample)` quantized candidates instead of
/// `search_k`, so a true neighbour the lossy ranking placed at, say,
/// #150 enters the candidate set at `oversample >= 1.5` and the reorder
/// queue then floats it to its correct exact rank. We still only feed
/// the executor up to its `LIMIT`; oversampling only widens the pool
/// the exact re-rank draws from.
///
/// Default `1.0` is exactly the pre-feature behaviour (no oversampling).
/// Modelled as a float GUC (pgrx 0.17 `define_float_guc`), range
/// `1.0 ..= 100.0`, clamped on read.
///
/// Composition with iterative scan: `ceil(search_k * oversample)` is the
/// *initial* `k`. Iterative refill (triggered by a selective `WHERE`
/// filter draining the batch) still doubles from that starting point,
/// capped by [`MAX_SCAN_TUPLES`]. So oversample sets the floor of the
/// candidate set; iterative scan grows it from there.
pub static OVERSAMPLE: GucSetting<f64> = GucSetting::<f64>::new(1.0);

/// DEPRECATED no-op (v1.19.0+). pg_turbovec no longer mmaps the
/// relfile; every byte of index data is read through PostgreSQL's
/// shared-buffer cache (`ReadBufferExtended`). This setting is
/// ignored and retained only so an existing `SET
/// turbovec.mmap_static_blocked` does not error; it will be removed
/// in a future minor. See `docs/BUFFER_CACHE_ONLY_DESIGN.md`.
/// DEPRECATED no-op (v1.19.0+): retained only so an existing
/// `SET turbovec.mmap_static_blocked` doesn't error. All index data
/// is read through PG's buffer manager; nothing reads this. Removed
/// in a future minor. See docs/BUFFER_CACHE_ONLY_DESIGN.md.
pub static MMAP_STATIC_BLOCKED: GucSetting<bool> = GucSetting::<bool>::new(true);

/// Iterative-scan mode, modelled on pgvector's `hnsw.iterative_scan`.
///
/// * `Off` — single fixed-`search_k` batch (pre-v1.8.0 behaviour).
///   `amgettuple` returns false as soon as that batch drains, which
///   under-returns under a selective `WHERE` filter + `ORDER BY dist
///   LIMIT k`.
/// * `RelaxedOrder` (default) — when the batch drains and the
///   executor asks for more, re-run the turbovec search with a
///   doubled `k` and feed the new candidates, capped by
///   [`MAX_SCAN_TUPLES`]. Results across refill batches are only
///   approximately distance-ordered; the executor's reorder queue
///   (`xs_recheckorderby = true`) restores exact per-tuple ordering.
///
/// pgvector also exposes `strict_order` for HNSW; we defer it as
/// future work because our reorder-queue model already delivers exact
/// ordering on top of `relaxed_order`. The `#[name = ...]` attrs give
/// the lowercase pgvector-familiar spelling at the SQL surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PostgresGucEnum)]
pub enum IterativeScanMode {
    #[name = c"off"]
    Off,
    #[name = c"relaxed_order"]
    RelaxedOrder,
}

pub static ITERATIVE_SCAN: GucSetting<IterativeScanMode> =
    GucSetting::<IterativeScanMode>::new(IterativeScanMode::RelaxedOrder);

/// Hard ceiling on the total number of candidates a single scan may
/// examine when iterative refill is enabled. Matches pgvector's
/// `hnsw.max_scan_tuples` default of 20000.
pub static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(20_000);

/// Parity gap #2 (v1.8.0): caps the rayon thread pool `ambuild`
/// uses for the CPU-heavy quantize + SIMD-repack phases. turbovec's
/// `encode` and `pack::repack` are embarrassingly parallel per-row,
/// and pgvector parallelises its HNSW/IVFFlat builds across
/// `max_parallel_maintenance_workers`; this GUC is pg_turbovec's
/// equivalent knob.
///
/// `0` (the default) means "derive from `max_parallel_maintenance_workers`":
/// `ambuild` uses `max_parallel_maintenance_workers + 1` threads (the
/// leader plus its worker budget), matching how PG accounts a parallel
/// maintenance op. A positive value pins the pool size directly.
///
/// This does NOT change the on-disk bytes: rayon's parallel iterators
/// write each row's codes/scales to a fixed output index, so the
/// result is independent of thread count. The heap scan stays serial,
/// so slot ordering and the TQ+ calibration source (the first chunk)
/// are identical to a single-threaded build. The byte-for-byte
/// equality is asserted by the `build_parts_are_pool_size_invariant`
/// unit test; query-level equivalence by the
/// `parallel_build_matches_serial_query` `#[pg_test]`.
pub static BUILD_PARALLELISM: GucSetting<i32> = GucSetting::<i32>::new(0);

/// IVF fine-scan intra-query parallelism (item #2 of the IVF-scaling
/// work). The IVF out-of-core scan gathers the probed cells into one
/// compact contiguous code buffer on the backend thread, then fine-
/// scans it. At high dim and high `probes` that fine-scan is the
/// dominant per-query cost (e.g. GIST-960d, ~64k vectors at
/// probes=64) and it runs SINGLE-THREADED in one backend on one core.
/// The compact buffer is a plain `&[u8]` of contiguous rows, so the
/// scan splits it into `T` disjoint row chunks, runs the SIMD LUT
/// top-`k` per chunk in a bounded rayon pool (pure compute over owned
/// bytes — no buffer-manager / catalog / `pg_sys` access inside the
/// threads; the gather already ran on the backend thread), then
/// merges the `T` local top-`k` heaps into the global top-`k`.
///
/// Because the merge unions per-chunk top-`k` lists (a true top-`k`
/// row is always in its own chunk's top-`k`), the returned top-`k`
/// SET is identical to a serial scan of the same compact rows; the
/// executor's reorder queue (`xs_recheckorderby`) re-ranks by exact
/// distance regardless, so tie order at the k-th boundary is immaterial.
/// Asserted by `ivf_parallel_scan_matches_serial`.
///
/// **Values:** `1` = serial (no fan-out); `>1` = pin the chunk/thread
/// count; `0` (the default) = auto = `min(probed_rows_worth_it, cores,
/// AUTO_SCAN_PARALLELISM_CAP)`. The auto cap is deliberately MODEST
/// (see [`AUTO_SCAN_PARALLELISM_CAP`]): intra-query parallelism cuts
/// single-query latency (the 332ms target) but many concurrent queries
/// each grabbing every core thrashes and hurts aggregate QPS. The
/// conservative default helps the isolated-latency benchmark without
/// wrecking a high-concurrency workload; raise it explicitly for a
/// latency-bound single-query deployment, set `1` to disable.
///
/// **No effect** on the flat (non-IVF) path, the whole-load IVF path
/// (small indexes that already load whole are already fast — the win
/// is on the large / out-of-core compact path where the cell ranges
/// are explicit), or on `ambuild` (that is `turbovec.build_parallelism`).
/// Never touches the wire format — this is a pure scan-time compute knob.
pub static SCAN_PARALLELISM: GucSetting<i32> = GucSetting::<i32>::new(0);

/// Modest ceiling on `turbovec.scan_parallelism = 0` (auto) fan-out.
/// A single query going wide cuts its own latency, but N concurrent
/// queries each fanning to all cores oversubscribes badly. 4 is a
/// conservative middle ground: a meaningful latency cut on the hot
/// high-dim fine-scan without letting one query monopolise a
/// many-core box under concurrency. Pin a higher value explicitly for
/// a latency-bound, low-concurrency deployment.
const AUTO_SCAN_PARALLELISM_CAP: usize = 4;

/// Below this many compact rows the per-thread top-k + merge overhead
/// swamps the parallel IVF fine-scan; keep it serial. Used by
/// [`resolve_scan_parallelism`] to cap the chunk count so a tiny
/// compact set is never split into thread-startup-dominated slivers.
const MIN_ROWS_PER_SCAN_CHUNK: usize = 2048;

/// Resolve the IVF fine-scan chunk/thread count for a compact scan of
/// `n_compact` rows. `1` (or a compact set too small to bother
/// splitting) means run inline/serial. Auto (`0`) caps at
/// [`AUTO_SCAN_PARALLELISM_CAP`] and never exceeds the machine's core
/// count or a floor of [`MIN_ROWS_PER_SCAN_CHUNK`] rows per chunk (so
/// a tiny compact set isn't split into thread-startup-dominated
/// slivers).
pub fn resolve_scan_parallelism(n_compact: usize) -> usize {
    let configured = SCAN_PARALLELISM.get();
    let want = if configured > 0 {
        configured as usize
    } else {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        cores.min(AUTO_SCAN_PARALLELISM_CAP)
    };
    cap_scan_chunks(want, n_compact)
}

/// Cap the desired chunk count `want` by how many chunks the compact
/// set can usefully sustain (at least [`MIN_ROWS_PER_SCAN_CHUNK`] rows
/// each), never below 1. Factored out of [`resolve_scan_parallelism`]
/// so the boundary math is unit-testable without a live GUC.
fn cap_scan_chunks(want: usize, n_compact: usize) -> usize {
    let by_rows = n_compact / MIN_ROWS_PER_SCAN_CHUNK;
    want.min(by_rows).max(1)
}

/// Phase C operator-path allowlist: a pre-materialized id-set the
/// user `SET`s before an `ORDER BY emb <=> q LIMIT k` query to
/// restrict the scan to those rows — the operator-path analogue of
/// `turbovec.knn(..., allowed => ...)`. CSV of **heap TIDs** encoded
/// as bigint (`(block << 32) | offset`, the
/// `pgrx::itemptr::item_pointer_to_u64` layout). The index AM keys
/// every vector by its heap TID (it never sees a heap `id` column),
/// so the allowlist is in TID space; callers materialize it from
/// `ctid` (see docs/FILTERING.md § 3.5). Whitespace is tolerated;
/// empty tokens are ignored. Empty / unset (the default `""`) =
/// unfiltered = exact prior behaviour, zero hot-path cost.
///
/// The scan parses this ONCE per scan into a `HashSet<u64>` and ANDs
/// a by-slot bool into the slot mask before `search_masked`, so the
/// blocked kernel's 32-vector block-skip applies (the in-kernel
/// pushdown, on BOTH flat and IVF indexes; IVF-aware: the skip is
/// scoped to probed cells AND the allowlist). This is still a
/// pre-materialized id-set channel, NOT arbitrary-`WHERE` pushdown:
/// the AM never interprets scan keys. A non-integer token ERRORs the
/// scan clearly. Modelled as a string GUC (pgrx 0.17
/// `define_string_guc`), `GucContext::Userset` so it's a per-session
/// knob `SET`/`RESET` around the query.
pub static ALLOWLIST: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(None);

/// Register all `turbovec.*` GUCs with PostgreSQL.
///
/// Called from `_PG_init`. Safe to call exactly once per backend.
pub fn register_gucs() {
    GucRegistry::define_int_guc(
        c_str(b"turbovec.bit_width_default\0"),
        c_str(b"Default bit width for turbovec indexes (2, 3, or 4).\0"),
        c_str(
            b"Number of bits per coordinate used by the TurboQuant scalar quantizer when an index is created without an explicit `bit_width` reloption. Lower values save memory at the cost of recall.\0",
        ),
        &BIT_WIDTH_DEFAULT,
        2,
        4,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.cache_size_mb\0"),
        c_str(b"Backend-local cache size for materialised turbovec indexes (MiB).\0"),
        c_str(
            b"Each backend keeps recently scanned turbovec indexes in memory. When this cap is exceeded the LRU entries are evicted. Set to 0 to disable caching (forces a rebuild on every scan).\0",
        ),
        &CACHE_SIZE_MB,
        0,
        65_536,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c_str(b"turbovec.warn_on_rebuild\0"),
        c_str(b"Emit a NOTICE when a turbovec index is materialised from the heap.\0"),
        c_str(
            b"Phase 2 turbovec indexes are rebuilt from the heap on first use after a server restart. Setting this on surfaces those events so operators can decide whether to issue an explicit REINDEX.\0",
        ),
        &WARN_ON_REBUILD,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.search_concurrency\0"),
        c_str(b"Number of OS threads used inside a turbovec ANN scan.\0"),
        c_str(
            b"The TurboQuant SIMD kernel parallelises within a single query batch using rayon. This GUC caps that fan-out. 1 disables intra-query parallelism.\0",
        ),
        &SEARCH_CONCURRENCY,
        1,
        128,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c_str(b"turbovec.normalize_on_insert\0"),
        c_str(b"Unit-normalise vectors before adding them to a turbovec index.\0"),
        c_str(
            b"TurboQuant assumes unit-norm inputs; with this on (the default) we apply that normalisation transparently. Turn off only if you have a calibrated upstream that already emits unit vectors.\0",
        ),
        &NORMALIZE_ON_INSERT,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.search_k\0"),
        c_str(b"Max candidates to fetch from the index per scan (default 32).\0"),
        c_str(
            b"The kernel ranks all corpus rows; this caps how many top-scoring candidates the index returns from one amgettuple sweep. The executor then drains them under LIMIT, re-ranking by exact distance (xs_recheckorderby). Latency scales with this count: every returned candidate costs a heap-tuple fetch + an exact full-precision distance recompute in the reorder queue, which is the dominant per-query cost (the IVF scan itself is a minority of the time). The recall-vs-search_k frontier (benches/results/searchk_recall_frontier_*.json) shows recall@10 PLATEAUS by ~search_k=25 -- 25/50/100/200 give identical recall -- so the pre-v1.18 default of 100 over-provisioned the recheck ~3x for no recall gain. The default is now 32 (above the recall plateau with margin, ~3x less recheck work). RAISE it when your query's LIMIT exceeds ~20 (you need at least LIMIT candidates), or to push recall on a hard corpus; LOWER it (toward 16) for the lowest latency on small-LIMIT queries that accept slightly worse recall. Composes with turbovec.oversample (which widens the candidate set) and iterative_scan (which grows search_k on refill).\0",
        ),
        &SEARCH_K,
        1,
        100_000,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.probes\0"),
        c_str(b"IVF cells to scan per query (default 8); ignored by flat indexes.\0"),
        c_str(
            b"For an index built WITH (lists = N), amgettuple coarse-searches the N cell centroids, picks the `probes` nearest cells, and fine-searches only those cells' contiguous code ranges. This is the IVF latency/recall dial (analogous to ivfflat.probes / hnsw.ef_search): lower = faster, lower recall; higher = slower, higher recall. probes >= lists reduces to the exact flat scan. Clamped to [1, lists] at scan time. No effect on flat (lists = 0) or vacuum-degraded indexes, which always scan the whole corpus.\0",
        ),
        &PROBES,
        1,
        65_536,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.max_probes\0"),
        c_str(b"IVF iterative-scan cap on probe-set widening (default 64); ignored by flat indexes.\0"),
        c_str(
            b"For an index built WITH (lists = N) under turbovec.iterative_scan = relaxed_order, when the currently-probed cells drain and the executor still wants tuples, the refill WIDENS the probe set (probes, 2*probes, 4*probes, ...) and re-runs the cell-restricted search, recovering true neighbours whose cell was not in the initial `probes` nearest set. This is the IVF analogue of ivfflat.max_probes: it caps that widening at min(max_probes, lists). Clamped to lists at scan time. No effect on flat (lists = 0) or vacuum-degraded indexes (no cells to widen; they keep the k-growth refill). turbovec.max_scan_tuples still caps total candidate work.\0",
        ),
        &MAX_PROBES,
        1,
        65_536,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_enum_guc(
        c_str(b"turbovec.out_of_core\0"),
        c_str(b"Serve large IVF indexes cell-scoped so an index larger than RAM can be queried (off | auto | on; default auto).\0"),
        c_str(
            b"Controls out-of-core IVF serving. auto (the default) serves an IVF index (built WITH (lists > 0)) cell-scoped ONLY when its codes are large relative to turbovec.cache_size_mb (codes > 0.5 * cache_size_mb): the backend then caches only bounded metadata (coarse centroids, cell directory, rotation, codebook, per-slot scales/ids) and per query gathers only the probed cells' contiguous code ranges through PostgreSQL's buffer manager into a compact throwaway sub-index, so the per-backend resident set is O(probes * cell_size) instead of O(n) and a >RAM IVF index can be served (only the probed cells' pages are read; the buffer manager + OS cache hold hot pages, cold pages are read on demand). An IVF index that fits the cache budget loads whole under auto (no per-query gather/reblock cost). on forces cell-scoped regardless of size (pays the per-query reblock tax even on small indexes); off forces the whole-index load into a per-backend Arc (lowest warm latency, O(n) resident, must fit in RAM). No effect on flat (lists = 0) or vacuum-degraded indexes (no cells to scope; always O(n)-resident \xe2\x80\x94 use IVF for >RAM), nor on the post-insert / dirty-fallback paths. Results are identical to the whole-load IVF path.\0",
        ),
        &OUT_OF_CORE,
        GucContext::Userset,
        GucFlags::default(),
    );

    // DEPRECATED no-op (v1.19.0+): all index data is now read through
    // PostgreSQL's buffer manager; there is no relfile mmap to toggle.
    // Kept registered for one minor so an existing `SET
    // turbovec.mmap_static_blocked = ...` in a session or config
    // doesn't error; it is read by nothing and will be removed in a
    // future minor. See docs/BUFFER_CACHE_ONLY_DESIGN.md.
    GucRegistry::define_bool_guc(
        c_str(b"turbovec.mmap_static_blocked\0"),
        c_str(b"DEPRECATED no-op: all index reads now go through PostgreSQL's buffer manager; this setting is ignored.\0"),
        c_str(
            b"Deprecated and ignored as of v1.19.0. pg_turbovec no longer mmaps the relfile; every byte of index data is read through PostgreSQL's shared-buffer cache (ReadBufferExtended). This GUC is retained as a no-op for one minor release so existing `SET` statements do not error, and will be removed. See docs/BUFFER_CACHE_ONLY_DESIGN.md.\0",
        ),
        &MMAP_STATIC_BLOCKED,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_enum_guc(
        c_str(b"turbovec.iterative_scan\0"),
        c_str(b"Iterative index scan mode (off | relaxed_order).\0"),
        c_str(
            b"When relaxed_order (the default), amgettuple re-runs the search with a doubled k and feeds new candidates if the executor exhausts the current batch under a selective WHERE filter + ORDER BY dist LIMIT k, capped by turbovec.max_scan_tuples. off restores the pre-v1.8.0 single-batch behaviour. strict_order (pgvector parity) is future work; our reorder queue already restores exact ordering on top of relaxed_order.\0",
        ),
        &ITERATIVE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.max_scan_tuples\0"),
        c_str(b"Max candidates examined per iterative scan (default 20000).\0"),
        c_str(
            b"Hard ceiling on the total number of candidates a single iterative scan may examine before returning false. Matches pgvector's hnsw.max_scan_tuples. Only consulted when turbovec.iterative_scan != off. Raise for very selective filters over large indexes; lower to bound worst-case scan work.\0",
        ),
        &MAX_SCAN_TUPLES,
        1,
        10_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_float_guc(
        c_str(b"turbovec.oversample\0"),
        c_str(b"Quantized-candidate oversampling multiplier for tunable recall (default 1.0).\0"),
        c_str(
            b"The scan fetches ceil(search_k * oversample) candidates ranked by the lossy quantized distance, then the executor's reorder queue (xs_recheckorderby) re-ranks them by exact full-precision distance and the LIMIT trims to the true top-k. Widening the candidate set recovers true neighbours the quantized ranking placed just outside search_k, turning quantization from a fixed accuracy point into a tunable recall frontier (matches Qdrant oversampling / VectorChord rerank). 1.0 (the default) is the pre-feature behaviour. Composes with iterative_scan: this sets the initial k, iterative refill grows it from there. Latency scales roughly linearly with the candidate count.\0",
        ),
        &OVERSAMPLE,
        1.0,
        100.0,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.build_parallelism\0"),
        c_str(b"OS threads used to quantize + repack vectors during CREATE INDEX / REINDEX (0 = auto).\0"),
        c_str(
            b"ambuild's encode and SIMD-repack phases are embarrassingly parallel per vector. This caps the rayon thread pool sizing those phases. 0 (the default) derives the pool size from max_parallel_maintenance_workers + 1 (leader + worker budget). A positive value pins the thread count. The on-disk index bytes are identical regardless of this value \xe2\x80\x94 only build wall-clock changes.\0",
        ),
        &BUILD_PARALLELISM,
        0,
        128,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c_str(b"turbovec.scan_parallelism\0"),
        c_str(b"OS threads used to fine-scan probed IVF cells per query (0 = auto, capped modest; 1 = serial).\0"),
        c_str(
            b"The out-of-core IVF scan gathers the probed cells into one compact contiguous code buffer on the backend thread, then fine-scans it for the top-k. At high dim and high probes that fine-scan dominates query latency and runs single-threaded on one core. This GUC splits the compact rows into N disjoint chunks, runs the SIMD LUT top-k per chunk in a bounded rayon pool (pure compute over owned bytes; no buffer-manager access inside threads), and merges the per-chunk top-k heaps into the global top-k. 1 = serial (no fan-out); a positive value pins the chunk/thread count; 0 (the default) = auto = min(cores, 4), a deliberately MODEST cap: intra-query parallelism cuts single-query latency but many concurrent queries each grabbing every core thrashes and hurts aggregate QPS, so the default favours the isolated-latency case without wrecking high concurrency (raise it for a latency-bound low-concurrency deployment). The returned top-k SET is identical to a serial scan (the executor re-ranks by exact distance regardless of tie order). No effect on the flat (non-IVF) path, the whole-load IVF path (small indexes already load whole and are fast), or ambuild (that is turbovec.build_parallelism). Never changes on-disk bytes.\0",
        ),
        &SCAN_PARALLELISM,
        0,
        128,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_string_guc(
        c_str(b"turbovec.allowlist\0"),
        c_str(b"Per-query allowlist of heap TIDs for ORDER BY scans (CSV of bigints; empty = unfiltered).\0"),
        c_str(
            b"A pre-materialized id-set the operator-path scan restricts to, the ORDER BY analogue of turbovec.knn(..., allowed). The index AM keys vectors by heap TID (it never sees a heap id column), so the allowlist is a CSV of heap TIDs encoded as bigint via (block << 32) | offset (the pgrx item_pointer_to_u64 layout); build it from ctid. SET it before an ORDER BY emb <=> q LIMIT k query and the scan returns only those rows, with the same in-kernel 32-vector-block short-circuit pushdown the knn() function gets, on both flat and IVF indexes (IVF-aware: scoped to probed cells AND the allowlist). Whitespace is tolerated; empty tokens are ignored; a non-integer token ERRORs the scan. Empty / unset (the default) is unfiltered with zero added cost. This is NOT arbitrary-WHERE pushdown: the AM never interprets scan keys, it only honours this pre-materialized TID set. RESET turbovec.allowlist after the query.\0",
        ),
        &ALLOWLIST,
        GucContext::Userset,
        GucFlags::default(),
    );
}

/// Const-fold a `&'static [u8]` containing a trailing NUL into a
/// `&'static CStr`. Pgrx 0.17 wants `&CStr` for GUC names.
#[inline]
const fn c_str(bytes: &'static [u8]) -> &'static CStr {
    match CStr::from_bytes_with_nul(bytes) {
        Ok(s) => s,
        Err(_) => panic!("missing trailing NUL in GUC string"),
    }
}

#[cfg(test)]
mod scan_parallelism_tests {
    use super::{cap_scan_chunks, MIN_ROWS_PER_SCAN_CHUNK};

    /// The chunk-count cap: never split below MIN_ROWS_PER_SCAN_CHUNK
    /// rows per chunk, never drop below 1, and honour the ceiling.
    #[test]
    fn cap_respects_row_floor_and_ceiling() {
        let m = MIN_ROWS_PER_SCAN_CHUNK;
        // Tiny compact set -> serial regardless of want.
        assert_eq!(cap_scan_chunks(8, 0), 1);
        assert_eq!(cap_scan_chunks(8, m - 1), 1);
        // Exactly one chunk's worth -> still 1 (can't sustain 2).
        assert_eq!(cap_scan_chunks(8, m), 1);
        // Two chunks' worth -> at most 2.
        assert_eq!(cap_scan_chunks(8, 2 * m), 2);
        // want below the row-supported count wins (the ceiling).
        assert_eq!(cap_scan_chunks(4, 100 * m), 4);
        // want=1 always serial.
        assert_eq!(cap_scan_chunks(1, 100 * m), 1);
    }
}
