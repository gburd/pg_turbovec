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

use core::ffi::CStr;

use pgrx::guc::PostgresGucEnum;
use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static BIT_WIDTH_DEFAULT: GucSetting<i32> = GucSetting::<i32>::new(4);
pub static CACHE_SIZE_MB: GucSetting<i32> = GucSetting::<i32>::new(256);
pub static WARN_ON_REBUILD: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_CONCURRENCY: GucSetting<i32> = GucSetting::<i32>::new(1);
pub static NORMALIZE_ON_INSERT: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_K: GucSetting<i32> = GucSetting::<i32>::new(100);

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
