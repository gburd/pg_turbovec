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

use core::ffi::CStr;

use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static BIT_WIDTH_DEFAULT: GucSetting<i32> = GucSetting::<i32>::new(4);
pub static CACHE_SIZE_MB: GucSetting<i32> = GucSetting::<i32>::new(256);
pub static WARN_ON_REBUILD: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_CONCURRENCY: GucSetting<i32> = GucSetting::<i32>::new(1);
pub static NORMALIZE_ON_INSERT: GucSetting<bool> = GucSetting::<bool>::new(true);
pub static SEARCH_K: GucSetting<i32> = GucSetting::<i32>::new(100);

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
