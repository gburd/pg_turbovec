//! `amcostestimate` — inform the planner about the cost of an
//! ANN scan so it can compare against alternative plans (Sort over
//! Seq Scan, Bitmap Heap Scan, etc.).
//!
//! v0.16 reads the actual `n_vectors` and `dim` from
//! `turbovec.am_storage` and computes a cost proportional to the
//! SIMD work the kernel will do for one batched search. Phase 17
//! will fold in `loop_count` for nested-loop join estimation.

use pgrx::pg_sys;
#[allow(unused_imports)]
use pgrx::prelude::*;

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

    // Pull n_vectors and dim from am_storage. We catch any error
    // (table not yet created, row missing) and fall back to a
    // pessimistic estimate.
    let (n_vectors, dim, bit_width): (i64, i32, i32) = if let Some(oid) = indexrelid {
        let row = pgrx::Spi::connect(|client| {
            let sql =
                "SELECT n_vectors, dim, bit_width FROM turbovec.am_storage \
                 WHERE indexrelid = $1";
            let mut iter = match client.select(sql, Some(1), &[oid.into()]) {
                Ok(t) => t,
                Err(_) => return None,
            };
            let row = iter.next()?;
            let nv: Option<i64> = row.get(1).ok().flatten();
            let dim: Option<i32> = row.get(2).ok().flatten();
            let bw: Option<i32> = row.get(3).ok().flatten();
            match (nv, dim, bw) {
                (Some(nv), Some(dim), Some(bw)) => Some((nv, dim, bw)),
                _ => None,
            }
        });
        row.unwrap_or((1_000, 384, 4))
    } else {
        (1_000, 384, 4)
    };

    // SIMD throughput model. The kernel processes 32 vectors per
    // SIMD block; each block does `dim * bit_width / 8` byte loads
    // plus a small constant for the LUT. Real-world numbers from
    // the upstream paper (and our `cargo bench --bench distance`)
    // are 5–10 ns per scored vector at 4-bit / dim=1536 on AVX2.
    // We use 8 ns/vec as a portable default and scale linearly
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
