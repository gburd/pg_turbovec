//! Phase 4 — `turbovec` PostgreSQL index access method.
//!
//! **EXPERIMENTAL** — only built when the `experimental_index_am`
//! Cargo feature is enabled. See `docs/INDEXAM.md` for the design
//! and the test plan that gates promotion to v0.5 default-on.
//!
//! Module map:
//! - `mod.rs` (this file) — `IndexAmRoutine` builder, the
//!   `turbovec_index_handler` SQL function, and the `extension_sql!`
//!   block that creates the access method and operator classes.
//! - `options.rs` — `bit_width` / `dim` reloption parser
//!   (`amoptions` callback).
//! - `persist.rs` — SPI helpers backing the `turbovec.am_storage`
//!   side table.
//! - `build.rs` — `ambuild` / `ambuildempty`.
//! - `insert.rs` — `aminsert`.
//! - `scan.rs` — `ambeginscan` / `amrescan` / `amgettuple` /
//!   `amendscan` (and the per-scan `ScanOpaque`).
//! - `vacuum.rs` — `ambulkdelete` / `amvacuumcleanup`.
//! - `cost.rs` — `amcostestimate`.
//! - `validate.rs` — `amvalidate`.
//!
//! ## SQL surface
//!
//! ```sql
//! CREATE FUNCTION turbovec_index_handler(internal)
//!     RETURNS index_am_handler
//!     LANGUAGE c AS '$libdir/pg_turbovec', 'turbovec_index_handler';
//!
//! CREATE ACCESS METHOD turbovec
//!     TYPE INDEX HANDLER turbovec_index_handler;
//!
//! CREATE OPERATOR CLASS tvector_ip_ops ...
//! CREATE OPERATOR CLASS tvector_cosine_ops ...
//! ```

use pgrx::prelude::*;

mod build;
mod cost;
mod insert;
mod options;
mod persist;
mod scan;
mod vacuum;
mod validate;

/// Strategy number for the order-by operator inside both
/// `tvector_ip_ops` (`<#>`) and `tvector_cosine_ops` (`<=>`).
#[allow(dead_code)]
pub(crate) const STRAT_ORDER_BY: u16 = 1;

/// Number of support functions per operator class. We expose:
/// 1. The distance function used to evaluate the order-by clause.
pub(crate) const SUPPORT_FN_COUNT: u16 = 1;

/// Build the `IndexAmRoutine` PostgreSQL hands back from
/// `turbovec_index_handler`. The returned pointer must be allocated
/// in `CurrentMemoryContext` per the
/// [Index Access Method Interface](https://www.postgresql.org/docs/17/indexam.html#INDEX-AM-FUNCTIONS).
///
/// # Safety
///
/// Caller must be running inside a Postgres backend (this is invoked
/// only via the `turbovec_index_handler(internal)` SQL function
/// during `CREATE ACCESS METHOD` resolution and thereafter on every
/// connection that touches a `turbovec` index).
unsafe fn make_routine() -> *mut pg_sys::IndexAmRoutine {
    let routine = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexAmRoutine>())
        as *mut pg_sys::IndexAmRoutine;
    if routine.is_null() {
        error!("turbovec: failed to allocate IndexAmRoutine");
    }

    (*routine).type_ = pg_sys::NodeTag::T_IndexAmRoutine;
    (*routine).amstrategies = 0; // we only have one order-by strategy
    (*routine).amsupport = SUPPORT_FN_COUNT;
    (*routine).amoptsprocnum = 0;

    // Capabilities — see comments above each flag for rationale.
    (*routine).amcanorder = false;
    (*routine).amcanorderbyop = true;
    (*routine).amcanbackward = false;
    (*routine).amcanunique = false;
    (*routine).amcanmulticol = false;
    (*routine).amoptionalkey = true;
    (*routine).amsearcharray = false;
    (*routine).amsearchnulls = false;
    (*routine).amstorage = true;
    (*routine).amclusterable = false;
    (*routine).ampredlocks = false;
    (*routine).amcanparallel = false;
    (*routine).amcaninclude = false;
    (*routine).amusemaintenanceworkmem = true;
    (*routine).amsummarizing = false;
    (*routine).amparallelvacuumoptions = 0;
    (*routine).amkeytype = pg_sys::InvalidOid;

    // Fields that only exist on pg17+. Feature-gate so the same
    // module compiles cleanly across pgrx's pg13…pg18 features.
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    {
        (*routine).amcanbuildparallel = false;
    }

    (*routine).ambuild = Some(build::ambuild);
    (*routine).ambuildempty = Some(build::ambuildempty);
    (*routine).aminsert = Some(insert::aminsert);
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    {
        (*routine).aminsertcleanup = None;
    }
    (*routine).ambulkdelete = Some(vacuum::ambulkdelete);
    (*routine).amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    (*routine).amcanreturn = None;
    (*routine).amcostestimate = Some(cost::amcostestimate);
    (*routine).amoptions = Some(options::amoptions);
    (*routine).amproperty = None;
    (*routine).ambuildphasename = None;
    (*routine).amvalidate = Some(validate::amvalidate);
    (*routine).amadjustmembers = None;
    (*routine).ambeginscan = Some(scan::ambeginscan);
    (*routine).amrescan = Some(scan::amrescan);
    (*routine).amgettuple = Some(scan::amgettuple);
    (*routine).amgetbitmap = None;
    (*routine).amendscan = Some(scan::amendscan);
    (*routine).ammarkpos = None;
    (*routine).amrestrpos = None;
    (*routine).amestimateparallelscan = None;
    (*routine).aminitparallelscan = None;
    (*routine).amparallelrescan = None;

    routine
}

/// `turbovec_index_handler(internal) RETURNS index_am_handler` — the
/// SQL-callable hook PostgreSQL invokes to fetch the routine.
///
/// We expose it via a raw `extern "C-unwind"` wrapper rather than
/// pgrx's `#[pg_extern]` because the return type is a Postgres
/// `Datum` carrying an `IndexAmRoutine *`, which pgrx's higher-level
/// SQL machinery doesn't model directly. The companion
/// `pg_finfo_turbovec_index_handler_wrapper` is what PG looks up
/// during `CREATE FUNCTION ... LANGUAGE c` resolution.
#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn turbovec_index_handler_wrapper(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    unsafe { pg_sys::Datum::from(make_routine()) }
}

/// PG_FUNCTION_INFO_V1 companion (auto-generated by pgrx for normal
/// `#[pg_extern]` functions; emitted manually here because we side-
/// stepped pg_extern).
#[unsafe(no_mangle)]
#[doc(hidden)]
pub extern "C" fn pg_finfo_turbovec_index_handler_wrapper() -> &'static pg_sys::Pg_finfo_record {
    const V1: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
    &V1
}

extension_sql!(
    r"
    CREATE FUNCTION turbovec_index_handler(internal) RETURNS index_am_handler
        AS 'MODULE_PATHNAME', 'turbovec_index_handler_wrapper'
        LANGUAGE c;
    ",
    name = "turbovec_index_handler_decl",
);

extension_sql!(
    r"
    -- Side table backing the `turbovec` access method.
    CREATE TABLE IF NOT EXISTS turbovec.am_storage (
        indexrelid  oid PRIMARY KEY,
        bit_width   int4 NOT NULL,
        dim         int4 NOT NULL,
        n_vectors   int8 NOT NULL,
        payload     bytea NOT NULL,
        version     int4 NOT NULL,
        live_ids    bytea NOT NULL DEFAULT ''::bytea,
        updated_at  timestamptz NOT NULL DEFAULT now()
    );
    -- Live u64 ids in little-endian; one row per index.
    -- Backwards-compat: existing v0.x rows that lack this column
    -- get an empty bytea, which means ambulkdelete falls back to
    -- a no-op (same behaviour as v0.4..v0.14).
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'turbovec'
              AND table_name = 'am_storage'
              AND column_name = 'live_ids'
        ) THEN
            ALTER TABLE turbovec.am_storage
                ADD COLUMN live_ids bytea NOT NULL DEFAULT ''::bytea;
        END IF;
    END$$;
    -- payload can be very large; force out-of-line uncompressed
    -- storage so we never accidentally PGLZ-compress turbovec's
    -- already-quantised bytes.
    ALTER TABLE turbovec.am_storage ALTER COLUMN payload SET STORAGE EXTERNAL;
    ALTER TABLE turbovec.am_storage ALTER COLUMN live_ids SET STORAGE EXTERNAL;

    -- Register the access method.
    CREATE ACCESS METHOD turbovec
        TYPE INDEX HANDLER turbovec_index_handler;

    -- Operator classes. Strategy 1 is the order-by operator.
    CREATE OPERATOR CLASS tvector_ip_ops
        DEFAULT FOR TYPE tvector USING turbovec AS
            OPERATOR 1 <#> (tvector, tvector) FOR ORDER BY float_ops,
            FUNCTION 1 negative_inner_product(tvector, tvector);

    CREATE OPERATOR CLASS tvector_cosine_ops
        FOR TYPE tvector USING turbovec AS
            OPERATOR 1 <=> (tvector, tvector) FOR ORDER BY float_ops,
            FUNCTION 1 cosine_distance(tvector, tvector);
    ",
    name = "turbovec_index_am",
    requires = ["turbovec_index_handler_decl", "tvector_operators"]
);
