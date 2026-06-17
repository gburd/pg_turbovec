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
//! | `turbovec.mmap_static_blocked`   | bool | true    | -              |
//! | `turbovec.iterative_scan`        | enum | relaxed_order | off, relaxed_order |
//! | `turbovec.max_scan_tuples`       | int  | 20000   | 1..=10_000_000 |
//! | `turbovec.build_parallelism`     | int  | 0       | 0..=128        |
//! | `turbovec.oversample`            | float| 1.0     | 1.0..=100.0    |
//! | `turbovec.max_probes`            | int  | 64      | 1..=65536      |
//! | `turbovec.out_of_core`           | enum | auto    | off, auto, on  |

use core::ffi::CStr;

use pgrx::guc::PostgresGucEnum;
use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static BIT_WIDTH_DEFAULT: GucSetting<i32> = GucSetting::<i32>::new(4);
pub static CACHE_SIZE_MB: GucSetting<i32> = GucSetting::<i32>::new(256);
pub static WARN_ON_REBUILD: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_CONCURRENCY: GucSetting<i32> = GucSetting::<i32>::new(1);
pub static NORMALIZE_ON_INSERT: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_K: GucSetting<i32> = GucSetting::<i32>::new(100);
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
/// scales/ids tables) plus a `MAP_PRIVATE` mmap of the relfile, and
/// per query copies ONLY the probed cells' contiguous code ranges
/// off the mmap to build a compact throwaway sub-index. The
/// per-backend resident set is then `O(probes * cell_size + faulted
/// pages)` instead of `O(n)`, so an IVF index larger than RAM can be
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

/// Phase R-3: when on (the default), `ambeginscan` mmap-loads the
/// deterministic-after-`ambuild` regions of the relfile (blocked
/// codes + persisted rotation matrix) instead of pulling the
/// chains through `ReadBufferExtended` / shared_buffers. The
/// codebook is read straight from the meta page either way.
///
/// Mmap is `MAP_PRIVATE`, read-only, lives for the
/// backend-local cache entry's lifetime, and is invalidated
/// when the cache entry's `(relfilenode, am_version)` mismatch
/// rolls forward (REINDEX or any committed mutation). Heap
/// visibility + `xs_recheckorderby = true` remain the MVCC
/// backstops; see `docs/ARCHITECTURE.md` § "Index AM · mmap
/// isolation contract" for the full argument.
///
/// Set off only if you observe weirdness on a custom storage
/// substrate (e.g. tablespace on a filesystem that doesn't
/// support shared mappings) and need to fall back to the
/// buffer-manager-only read path.
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
        c_str(b"Max candidates to fetch from the index per scan (default 100).\0"),
        c_str(
            b"The kernel ranks all corpus rows; this caps how many top-scoring candidates the index returns from one amgettuple sweep. The executor then drains them under LIMIT/recheck-orderby. Set higher for queries with LIMIT > 100 or when xs_recheckorderby semantics require oversampling. Set lower for lower-latency queries that accept slightly worse recall.\0",
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
            b"Controls out-of-core IVF serving. auto (the default) serves an IVF index (built WITH (lists > 0)) cell-scoped ONLY when its codes are large relative to turbovec.cache_size_mb (codes > 0.5 * cache_size_mb): the backend then caches only bounded metadata (coarse centroids, cell directory, rotation, codebook, per-slot scales/ids) plus a MAP_PRIVATE mmap of the relfile, and per query copies only the probed cells' contiguous code ranges off the mmap into a compact throwaway sub-index, so the per-backend resident set is O(probes * cell_size + faulted pages) instead of O(n) and a >RAM IVF index can be served (hot cells stay in the OS page cache; cold cells fault on demand). An IVF index that fits the cache budget loads whole under auto (no per-query gather/reblock cost). on forces cell-scoped regardless of size (pays the per-query reblock tax even on small indexes); off forces the whole-index load into a per-backend Arc (lowest warm latency, O(n) resident, must fit in RAM). No effect on flat (lists = 0) or vacuum-degraded indexes (no cells to scope; always O(n)-resident \xe2\x80\x94 use IVF for >RAM), nor on the post-insert / dirty-fallback paths. Results are identical to the whole-load IVF path.\0",
        ),
        &OUT_OF_CORE,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c_str(b"turbovec.mmap_static_blocked\0"),
        c_str(b"Mmap the deterministic static regions of a turbovec relfile (default on).\0"),
        c_str(
            b"When on, ambeginscan mmaps the persisted SIMD-blocked codes and rotation matrix RO into the backend address space, bypassing PG's buffer manager for those bytes. Halves warm-scan latency on indexes that don't fit in shared_buffers. The cache entry holds the Mmap so it lives until the cache invalidates (REINDEX / am_version bump / backend exit). Codes / scales / ids chains keep going through the buffer manager because VACUUM swap-remove mutates them in place. Heap visibility + xs_recheckorderby remain the MVCC backstops; see docs/ARCHITECTURE.md.\0",
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
