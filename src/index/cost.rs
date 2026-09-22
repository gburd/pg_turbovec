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
//! pinned shared-buffer page — and avoids the SPI round-trip
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
    // Phase Z4: also read `lists` and whether the IVF cells are still
    // live, so an IVF index can be costed for the cells it PROBES rather
    // than the whole corpus. `lists == 0` (flat) and a degraded IVF index
    // both scan everything, so both get `effective_lists = 0`.
    let (n_vectors, dim, bit_width, lists): (i64, i32, i32, u32) = if let Some(oid) = indexrelid {
        let rel = pg_sys::index_open(oid, pg_sys::AccessShareLock as i32);
        if rel.is_null() {
            (1_000, 384, 4, 0)
        } else {
            let v = match relfile::read_meta(rel) {
                Some(m) => {
                    // A degraded IVF index takes the FLAT fallback (see
                    // `MetaPageData::is_degraded` and Phase Z1), so cost it
                    // as flat -- otherwise the planner would keep believing
                    // in cell pruning that no longer happens.
                    let effective_lists = if m.is_degraded() { 0 } else { m.lists };
                    (
                        m.n_vectors as i64,
                        m.dim as i32,
                        m.bit_width as i32,
                        effective_lists,
                    )
                }
                None => (1_000, 384, 4, 0),
            };
            pg_sys::index_close(rel, pg_sys::AccessShareLock as i32);
            v
        }
    } else {
        (1_000, 384, 4, 0)
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

    // Phase Z4: how many vectors does this scan actually SCORE?
    //
    // Flat (`lists == 0`): all of them -- that is the O(n) wall IVF
    // exists to break, and the planner should see it.
    //
    // IVF (`lists > 0`): the scan clamps to `turbovec.probes` cells
    // (`scan.rs`: `PROBES.get().clamp(1, lists)`), so it scores roughly
    // `n_vectors * probes / lists`. Before Z4 this model charged the
    // full corpus for every kind, so an IVF index probing 1 of 1024
    // cells was costed identically to a flat scan of everything and the
    // planner could not see IVF's whole point.
    //
    // Cells are assumed evenly populated. k-means does not guarantee
    // that, but the alternative is reading the cell directory on every
    // planner call -- a real I/O cost to sharpen an estimate the
    // planner then compares against its own approximations. If skew
    // ever demonstrably misleads the planner, read the directory here.
    let scored_vectors = if lists > 0 {
        let probes = (crate::guc::PROBES.get() as f64).clamp(1.0, lists as f64);
        let fraction = probes / (lists as f64);
        // Never estimate below one cell's worth: a probe always reads a
        // whole cell, and a zero-cost estimate would let the planner pick
        // this path for free.
        ((n_vectors as f64) * fraction).max((n_vectors as f64) / (lists as f64))
    } else {
        n_vectors as f64
    };

    // Modelled wall-clock of the scan, kept for reference/debugging; the
    // planner cost below is derived from `nanos_per_vec` directly.
    let _total_nanos = scored_vectors * nanos_per_vec;

    // Convert the modelled CPU time into PostgreSQL's cost unit.
    //
    // PG's unit is "one sequential page read" (`seq_page_cost = 1.0`), with
    // `cpu_operator_cost = 0.0025` per simple operator evaluation. The
    // natural anchor for a scan that evaluates one distance per candidate
    // is therefore `cpu_operator_cost` per scored vector, scaled by how
    // much work that distance actually is relative to a simple operator.
    //
    // The pre-Z4 line divided SECONDS by `cpu_operator_cost`, which is a
    // unit error: it made a full 1M x 1024-d flat scan cost ~23 while
    // PostgreSQL costs the equivalent sequential scan at ~73,000. Being
    // ~3000x too cheap is why an ANN path could win against plans that are
    // genuinely faster, and why tests need `enable_seqscan = off` to force
    // the comparison at all.
    //
    // `nanos_per_vec` is calibrated at 8 ns for dim=1536 / 4-bit (matches
    // our published 1M x 1024-d 4-bit scan: the model says 5.3 ms, measured
    // is 6.08 ms), so normalising by that reference keeps the dim and
    // bit_width scaling while landing on a defensible absolute cost.
    const REF_NANOS_PER_VEC: f64 = 8.0;
    let work_per_vec = nanos_per_vec / REF_NANOS_PER_VEC;
    let cpu_cost = scored_vectors * work_per_vec * pg_sys::cpu_operator_cost;
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

    // Phase Z4: report the fraction of the table this scan will return,
    // instead of a hardcoded 0.0.
    //
    // The old `0.0` told the planner "essentially no rows match" for EVERY
    // query, which is both wrong and unhelpful: it cannot then weigh an ANN
    // scan against filter-first alternatives, and `0.0` propagates into
    // downstream join/row estimates.
    //
    // Use the planner's OWN work rather than recomputing it: for a base
    // relation it has already reduced `rel->rows` from `rel->tuples` by the
    // restriction quals' combined selectivity. Dividing recovers exactly
    // the filter selectivity it believes, with no risk of disagreeing with
    // it and no `clauselist_selectivity` call of our own.
    //
    // An ANN scan is an ordered scan: the executor stops pulling once the
    // LIMIT is satisfied, so the useful figure is "what fraction survives
    // the quals", not "how many rows does the kernel score".
    let filter_selectivity = if !path.is_null() {
        let rel = (*path).path.parent;
        if !rel.is_null() && (*rel).tuples > 0.0 {
            ((*rel).rows / (*rel).tuples).clamp(0.0, 1.0)
        } else {
            1.0
        }
    } else {
        1.0
    };
    // Floor at one row's worth so a highly selective qual never reports a
    // literal 0.0 (which would resurrect the old "no rows" problem and can
    // make downstream estimates collapse to 1).
    let min_sel = if n_vectors > 0 {
        1.0 / (n_vectors as f64)
    } else {
        0.0
    };
    *index_selectivity = filter_selectivity.max(min_sel);

    // Correlation 0: ANN ordering (by distance) has no relationship with
    // heap order.
    *index_correlation = 0.0;

    // Approximate page count: bytes-per-vector / 8 KiB. Phase Z4 scopes
    // this to the pages actually touched, so an IVF scan that probes a
    // few cells is not charged for reading the entire codes chain.
    let bytes_per_vec = (bits_per_vec / 8.0) + 4.0; // + 4-byte scale
    *index_pages = (scored_vectors * bytes_per_vec / 8192.0).max(1.0);
}
