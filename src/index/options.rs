//! `amoptions` — parse and validate per-index reloptions.
//!
//! Supported reloptions:
//!
//! | Name        | Type | Default | Range  |
//! |-------------|------|---------|--------|
//! | `bit_width` | int  | (GUC `turbovec.bit_width_default`) | 2..=4 |
//! | `dim`       | int  | 0 (auto-detect from first row)     | {0} ∪ ((>0) ∧ multiple of 8) |
//! | `lists`     | int  | 0 (flat; v3-equivalent)            | 0..=1_000_000 |
//! | `assign_dups` | int | 1 (single assignment)             | 1..=4 |
//! | `graph`     | bool | false (flat; not a graph index)   | – |
//!
//! `lists` is the IVF coarse-cell count (`nlist`). `0` (the default)
//! keeps the flat layout — byte-identical to the v3 wire format
//! modulo the version byte — so existing indexes and non-IVF users
//! are untouched. `lists > 0` opts into the IVF build path
//! (an internal design note). Recommended starting point: `lists ≈ √n`.
//!
//! `assign_dups` (IVF-4a) is the soft-assignment multiplicity `M`.
//! `1` (the default) is single assignment (each vector in exactly one
//! cell). `M > 1` enables soft assignment: a boundary vector (within
//! `ivf::BOUNDARY_FACTOR` of its nearest cell) is also stored in its
//! 2nd..Mth nearest cells, raising recall@10 at a fixed
//! `turbovec.probes` at a bounded storage cost. Only meaningful when
//! `lists > 0`; ignored for flat indexes.
//!
//! `graph` (Phase G-2a, an internal design note) opts a
//! build into the Vamana-style navigable-graph index kind
//! (`KIND_GRAPH`, wire v6) instead of the flat/IVF layout — the
//! analogue of how `lists = N` opts into IVF. `false` (the default)
//! is the ordinary flat/IVF build, fully backward compatible.
//! Mutually exclusive with `lists > 0` (validated below): a graph
//! index is never IVF-backed.
//!
//! The reloption byte payload is owned by Postgres and read back via
//! `IndexRelationGetReloptions`. We use `add_local_int_reloption` /
//! `build_local_reloptions` from `pg_sys`.

use pgrx::pg_sys;

use crate::guc;

/// Reloption layout — must match the field offsets passed to
/// `add_local_int_reloption`.
#[repr(C)]
pub(crate) struct TurbovecRelopts {
    /// Standard `bytea` header that wraps every reloption blob.
    pub vl_len_: i32,
    pub bit_width: i32,
    pub dim: i32,
    /// IVF coarse-cell count (`nlist`); 0 = flat (v3-equivalent).
    pub lists: i32,
    /// IVF-4a soft-assignment multiplicity (M); 1 = single assignment.
    pub assign_dups: i32,
    /// Phase G-2a: build a Vamana graph index (`KIND_GRAPH`) instead
    /// of flat/IVF. `false` = ordinary flat/IVF build (default).
    pub graph: bool,
}

/// Upper bound on `assign_dups` (IVF-4a soft assignment). `M > 4`
/// rarely helps recall enough to justify the storage blow-up; the
/// ceiling keeps the per-vector dup list small and bounds worst-case
/// index growth at ~4×.
pub(crate) const MAX_ASSIGN_DUPS: i32 = 4;

/// Upper bound on `lists`. A sane ceiling so a fat-fingered
/// `WITH (lists = 2000000000)` doesn't try to train two billion
/// centroids; well above any realistic `nlist ≈ √n` (√n = 1e6 needs
/// n = 1e12 rows).
pub(crate) const MAX_LISTS: i32 = 1_000_000;

/// `amoptions` callback. Receives the raw `reloptions` Datum
/// (a `text[]` of `key=value` strings) plus a `validate` flag, and
/// returns a palloc'd `bytea` whose payload is a `TurbovecRelopts`.
///
/// # Safety
///
/// Caller is PostgreSQL's reloption machinery; pointer ownership
/// follows standard Postgres rules.
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // Static reloption table is registered lazily on first call.
    static mut RELOPT_KIND: pg_sys::relopt_kind::Type = 0;
    static mut INITIALISED: bool = false;

    if !INITIALISED {
        RELOPT_KIND = pg_sys::add_reloption_kind();

        let bit_width_default = guc::BIT_WIDTH_DEFAULT.get();
        pg_sys::add_int_reloption(
            RELOPT_KIND,
            c"bit_width".as_ptr(),
            c"TurboQuant bit width per coordinate (2, 3, or 4)".as_ptr(),
            bit_width_default,
            2,
            4,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            RELOPT_KIND,
            c"dim".as_ptr(),
            c"Vector dimension (0 = auto-detect on first build)".as_ptr(),
            0,
            0,
            crate::vec::MAX_DIM as i32,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            RELOPT_KIND,
            c"lists".as_ptr(),
            c"IVF coarse-cell count (nlist); 0 = flat. Recommend ~sqrt(n).".as_ptr(),
            0,
            0,
            MAX_LISTS,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_int_reloption(
            RELOPT_KIND,
            c"assign_dups".as_ptr(),
            c"IVF soft-assignment multiplicity M (1 = single; M>1 stores boundary vectors in their top-M nearest cells to raise recall). Needs lists>0.".as_ptr(),
            1,
            1,
            MAX_ASSIGN_DUPS,
            pg_sys::AccessExclusiveLock as i32,
        );
        pg_sys::add_bool_reloption(
            RELOPT_KIND,
            c"graph".as_ptr(),
            c"Build a Vamana-style navigable-graph index (KIND_GRAPH) instead of flat/IVF. Mutually exclusive with lists>0.".as_ptr(),
            false,
            pg_sys::AccessExclusiveLock as i32,
        );

        INITIALISED = true;
    }

    // PG 18 added an `isset_offset` field to `relopt_parse_elt` so
    // callers can distinguish "explicitly set to default" from "never
    // set". We don't track that distinction; pass -1 (unused).
    let tab = [
        pg_sys::relopt_parse_elt {
            optname: c"bit_width".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, bit_width) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: -1,
        },
        pg_sys::relopt_parse_elt {
            optname: c"dim".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, dim) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: -1,
        },
        pg_sys::relopt_parse_elt {
            optname: c"lists".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, lists) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: -1,
        },
        pg_sys::relopt_parse_elt {
            optname: c"assign_dups".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, assign_dups) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: -1,
        },
        pg_sys::relopt_parse_elt {
            optname: c"graph".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_BOOL,
            offset: std::mem::offset_of!(TurbovecRelopts, graph) as i32,
            #[cfg(feature = "pg18")]
            isset_offset: -1,
        },
    ];

    let opts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND,
        std::mem::size_of::<TurbovecRelopts>(),
        tab.as_ptr(),
        tab.len() as i32,
    ) as *mut TurbovecRelopts;

    if !opts.is_null() && validate {
        let o = &*opts;
        if !(2..=4).contains(&o.bit_width) {
            pgrx::error!("turbovec: bit_width must be in 2..=4 (got {})", o.bit_width);
        }
        if o.dim != 0 && (o.dim <= 0 || o.dim % 8 != 0) {
            pgrx::error!(
                "turbovec: dim must be 0 (auto) or a positive multiple of 8 (got {})",
                o.dim
            );
        }
        if !(0..=MAX_LISTS).contains(&o.lists) {
            pgrx::error!(
                "turbovec: lists must be in 0..={} (got {})",
                MAX_LISTS,
                o.lists
            );
        }
        if !(1..=MAX_ASSIGN_DUPS).contains(&o.assign_dups) {
            pgrx::error!(
                "turbovec: assign_dups must be in 1..={} (got {})",
                MAX_ASSIGN_DUPS,
                o.assign_dups
            );
        }
        if o.graph && o.lists > 0 {
            pgrx::error!(
                "turbovec: graph = true is mutually exclusive with lists > 0 (a graph index is never IVF-backed)"
            );
        }
    }

    opts as *mut pg_sys::bytea
}

/// Resolve effective options for a given index relation. Falls back
/// to the GUC defaults when the relation has no reloptions set.
/// Returns `(bit_width, dim, lists, assign_dups, graph)`.
pub(crate) unsafe fn read(rel: pg_sys::Relation) -> (i32, i32, i32, i32, bool) {
    let raw = (*rel).rd_options as *const TurbovecRelopts;
    if raw.is_null() {
        (guc::BIT_WIDTH_DEFAULT.get(), 0, 0, 1, false)
    } else {
        let o = &*raw;
        (o.bit_width, o.dim, o.lists, o.assign_dups, o.graph)
    }
}
