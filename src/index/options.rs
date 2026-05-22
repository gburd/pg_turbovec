//! `amoptions` — parse and validate per-index reloptions.
//!
//! Supported reloptions:
//!
//! | Name        | Type | Default | Range  |
//! |-------------|------|---------|--------|
//! | `bit_width` | int  | (GUC `turbovec.bit_width_default`) | 2..=4 |
//! | `dim`       | int  | 0 (auto-detect from first row)     | {0} ∪ ((>0) ∧ multiple of 8) |
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
}

/// `amoptions` callback. Receives the raw `reloptions` Datum
/// (a `text[]` of `key=value` strings) plus a `validate` flag, and
/// returns a palloc'd `bytea` whose payload is a `TurbovecRelopts`.
///
/// # Safety
///
/// Caller is PostgreSQL's reloption machinery; pointer ownership
/// follows standard Postgres rules.
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
            crate::tvector::MAX_DIM as i32,
            pg_sys::AccessExclusiveLock as i32,
        );

        INITIALISED = true;
    }

    let tab = [
        pg_sys::relopt_parse_elt {
            optname: c"bit_width".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, bit_width) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: c"dim".as_ptr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(TurbovecRelopts, dim) as i32,
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
            pgrx::error!(
                "turbovec: bit_width must be in 2..=4 (got {})",
                o.bit_width
            );
        }
        if o.dim != 0 && (o.dim <= 0 || o.dim % 8 != 0) {
            pgrx::error!(
                "turbovec: dim must be 0 (auto) or a positive multiple of 8 (got {})",
                o.dim
            );
        }
    }

    opts as *mut pg_sys::bytea
}

/// Resolve effective options for a given index relation. Falls back
/// to the GUC defaults when the relation has no reloptions set.
pub(crate) unsafe fn read(rel: pg_sys::Relation) -> (i32, i32) {
    let raw = (*rel).rd_options as *const TurbovecRelopts;
    if raw.is_null() {
        (guc::BIT_WIDTH_DEFAULT.get(), 0)
    } else {
        let o = &*raw;
        (o.bit_width, o.dim)
    }
}
