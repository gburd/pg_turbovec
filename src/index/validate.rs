//! `amvalidate` — opclass sanity check. Phase 4 returns true
//! unconditionally; Phase 5 will verify operator class strategy
//! numbers and support function signatures.

use pgrx::pg_sys;

#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}
