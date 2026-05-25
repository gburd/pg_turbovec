//! `amcostestimate` — inform the planner about the cost of an
//! ANN scan so it can compare against alternative plans (Sort over
//! Seq Scan, Bitmap Heap Scan, etc.).
//!
//! v1.3.0 reads the actual `n_vectors`, `dim`, and `bit_width`
//! straight off the relfile meta page (block 0 of the index's
//! main fork) and computes a cost proportional to the SIMD work
//! the kernel will do for one batched search. The previous
//! versions read these out of the SPI side-table; that's gone in
//! v1.3.0. The relfile read is cheap — one buffer-pool hit on a
//! pinned shared-buffer page \u2014 and avoids the SPI round-trip
//! that would otherwise re-enter the executor mid-plan.

use pgrx::pg_sys;
#[allow(unused_imports)]
use pgrx::prelude::*;

use crate::index::relfile;

#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // Pull the index oid from the IndexPath.
    let indexrelid: Option<pg_sys::Oid> = if !path.is_null() {
        let info = (*path).indexinfo;
        if !info.is_null() {
            Some((*info).indexoid)
        } else {
            None
        }
    } else {
        None
    };

    // Read n_vectors / dim / bit_width straight off the relfile
    // meta page. AccessShareLock is the lightest lock we can take
    // (compatible with everything except AccessExclusive); a
    // failed open or an empty meta page falls through to the
    // pessimistic default below so the planner doesn't crash on
    // partially-built indexes.
    let (n_vectors, dim, bit_width): (i64, i32, i32) = if let Some(oid) = indexrelid {
        let rel = pg_sys::index_open(oid, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            (1_000, 384, 4)
        } else {
            let v = match relfile::read_meta(rel) {
                Some(m) => (m.n_vectors as i64, m.dim as i32, m.bit_width as i32),
                None => (1_000, 384, 4),
            };
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            v
        }
    } else {
        (1_000, 384, 4)
    };

    // SIMD throughput model. The kernel processes 32 vectors per
    // SIMD block; each block does `dim * bit_width / 8` byte loads
    // plus a small constant for the LUT. Real-world numbers from
    // the upstream paper (and our `cargo bench --bench distance`)
    // are 5–10 ns per scored vector at 4-bit / dim=1536 on AVX2.
    // We use 8 ns/vector as a portable default and scale linearly
    // with `dim * bit_width`.
    let bits_per_vec = (dim as f64) * (bit_width as f64);
    let nanos_per_vec = 8.0 * (bits_per_vec / (1536.0 * 4.0));
    let total_nanos = (n_vectors as f64) * nanos_per_vec;

    // Postgres expresses cost in "page reads". 1 page ≈ 4 µs on a
    // SATA SSD per the default cost_constants. Convert ns → page
    // equivalents via the executor's `cpu_operator_cost` (default
    // 0.0025 per row-op).
    let cpu_cost = (total_nanos / 1_000_000_000.0) / 0.0025;
    let startup_cost = 1.0 + (n_vectors as f64).log2().max(1.0);

    // If the planner is considering this index without any
    // ORDER BY operator (e.g. a `count(*)` or a non-distance
    // restriction qual), we can't actually serve the scan — our
    // `amrescan` short-circuits to an empty result set in that
    // case. Advertise a cost large enough to lose every realistic
    // alternative so the planner picks a seq scan or a btree
    // primary-key scan instead. Without this, a 1 k-row INSERT
    // can pick our AM for self-checks like `SELECT count(*)` and
    // see zero rows, which surfaces as the bulk-insert
    // "committed-but-invisible" symptom.
    let has_orderby = !path.is_null() && !(*path).indexorderbys.is_null();
    if !has_orderby {
        *index_startup_cost = pg_sys::disable_cost;
        *index_total_cost = pg_sys::disable_cost;
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 1.0;
        return;
    }

    *index_startup_cost = startup_cost;
    *index_total_cost = startup_cost + cpu_cost;

    // The index always returns up to `k` rows (LIMIT is applied at
    // a higher level), so selectivity from the planner's point of
    // view is ~ 0 (very few rows match). Correlation 0 because the
    // index ordering has no relationship with heap order.
    *index_selectivity = 0.0;
    *index_correlation = 0.0;

    // Approximate page count: bytes-per-vector / 8 KiB.
    let bytes_per_vec = (bits_per_vec / 8.0) + 4.0; // + 4-byte scale
    *index_pages = ((n_vectors as f64) * bytes_per_vec / 8192.0).max(1.0);
}
