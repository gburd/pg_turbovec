//! Phase 4 — `turbovec` PostgreSQL index access method.
//!
//! Module map:
//! - `mod.rs` (this file) — `IndexAmRoutine` builder, the
//!   `turbovec_index_handler` SQL function, and the `extension_sql!`
//!   block that creates the access method and operator classes.
//! - `options.rs` — `bit_width` / `dim` / `lists` reloption parser
//!   (`amoptions` callback).
//! - `ivf.rs` — IVF coarse-quantizer k-means + cell-directory types
//!   (Postgres-free, IVF-1).
//! - `page.rs` — meta-page byte layout for the relfile main fork.
//! - `relfile.rs` — buffer-manager I/O for the relfile pages.
//! - `mmap_static.rs` — mmap-based reads of the deterministic
//!   static regions (Phase R-3, v1.5.0).
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
//! CREATE OPERATOR CLASS vec_ip_ops ...
//! CREATE OPERATOR CLASS vec_cosine_ops ...
//! ```

use pgrx::prelude::*;

mod build;
mod build_pool;
mod cost;
mod insert;
pub(crate) mod ivf;
pub(crate) mod mmap_static;
mod options;
pub(crate) mod page;
pub(crate) mod relfile;
pub(crate) mod scan;
pub(crate) mod vacuum;
mod validate;

/// Strategy number for the order-by operator inside both
/// `vec_ip_ops` (`<#>`) and `vec_cosine_ops` (`<=>`).
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
    // `amsummarizing` was added in PG 16 (BRIN summarising indexes).
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    {
        (*routine).amsummarizing = false;
    }
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
    // `amadjustmembers` was added in PG 14 (op family adjust callback).
    #[cfg(not(feature = "pg13"))]
    {
        (*routine).amadjustmembers = None;
    }
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
    -- Phase Q (v1.3.0): the side-table `turbovec.am_storage` is
    -- gone. All index state lives in the index relation's main
    -- fork via the relfile path (the only storage strategy).
    -- Drop any leftover row from a v1.0.x..v1.2.0 install. Users
    -- with existing turbovec indexes must `REINDEX INDEX <name>;`
    -- after upgrade — `ambeginscan` errors loudly otherwise.
    DROP TABLE IF EXISTS turbovec.am_storage CASCADE;

    -- Register the access method.
    CREATE ACCESS METHOD turbovec
        TYPE INDEX HANDLER turbovec_index_handler;

    -- Operator classes. Strategy 1 is the order-by operator.
    CREATE OPERATOR CLASS vec_ip_ops
        DEFAULT FOR TYPE vector USING turbovec AS
            OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 negative_inner_product(vector, vector);

    CREATE OPERATOR CLASS vec_cosine_ops
        FOR TYPE vector USING turbovec AS
            OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 cosine_distance(vector, vector);

    -- L2 / L1 operator classes. The TurboQuant kernel ranks by
    -- inner-product internally, but our amgettuple sets
    -- xs_recheckorderby = true so the executor recomputes the
    -- exact distance against the heap tuple. For unit-norm
    -- vectors (the recommended insert mode), inner-product and L2
    -- ranking are equivalent (L2 = sqrt(2 - 2*IP)), so the
    -- candidate set quality matches the cosine path. For L1 the
    -- candidate set is noisier but recheck guarantees exact final
    -- ordering.
    CREATE OPERATOR CLASS vec_l2_ops
        FOR TYPE vector USING turbovec AS
            OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 l2_distance(vector, vector);

    CREATE OPERATOR CLASS vec_l1_ops
        FOR TYPE vector USING turbovec AS
            OPERATOR 1 <+> (vector, vector) FOR ORDER BY float_ops,
            FUNCTION 1 l1_distance(vector, vector);

    -- Phase F-2: the ColBERT / multivector token index kind. The
    -- opclass is over the ARRAY type `turbovec.vector[]` (a column of
    -- per-doc token arrays). It registers NO order-by operator and NO
    -- `<=>`-style binary operator: ColBERT has no single-vector
    -- orderby semantics, so the planner has nothing to match and will
    -- NEVER pick this index for an `ORDER BY ... <=> q` scan (the
    -- forbidden amrescan path). The opclass exists purely so
    --   CREATE INDEX ... USING turbovec (tokens vec_colbert_ops)
    -- is accepted and routes to the v5 persistent-token-index build.
    -- Reads go through turbovec.colbert_search(), not the planner;
    -- ambeginscan/amrescan REJECT an ORDER BY scan on a kind=colbert
    -- index with a HINT to use colbert_search. Support function 1 is
    -- max_sim (the Phase D MaxSim kernel) purely to satisfy the AM's
    -- amsupport = 1; it is never invoked by the (rejected) scan path.
    CREATE OPERATOR CLASS vec_colbert_ops
        FOR TYPE vector[] USING turbovec AS
            FUNCTION 1 max_sim(vector[], vector[]);
    ",
    name = "turbovec_index_am",
    requires = ["turbovec_index_handler_decl", "vec_operators", max_sim]
);
