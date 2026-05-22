//! `amcostestimate` — cheap heuristic so the planner picks our index
//! for `ORDER BY emb <=> $1 LIMIT k` patterns.
//!
//! Postgres reads four numbers back: startup cost, total cost,
//! selectivity, and correlation. v0.4 returns:
//!
//! * `indexStartupCost` = small constant (we have to load the
//!   payload from `am_storage` once per scan).
//! * `indexTotalCost` ≈ `n_vectors * dim * bit_width / 8 / SIMD_WORD`.
//! * `indexSelectivity` = 0.0 (we always return at most LIMIT k).
//! * `indexCorrelation` = 0.0.
//!
//! v0.5 should pull the actual `n_vectors` and `dim` from
//! `am_storage` rather than guessing.

use pgrx::pg_sys;

pub(crate) unsafe extern "C-unwind" fn amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    _path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // Tiny constants — the goal is to be cheaper than any
    // alternative `Sort` plan over a full sequential scan.
    *index_startup_cost = 1.0;
    *index_total_cost = 10.0;
    *index_selectivity = 0.0;
    *index_correlation = 0.0;
    *index_pages = 1.0;
}
