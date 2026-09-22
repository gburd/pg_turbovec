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

/// How many vectors will one scan actually SCORE?
///
/// Flat (`lists == 0`): the whole corpus -- the O(n) wall IVF exists to
/// break, and the planner should see it. IVF (`lists > 0`): the scan clamps
/// to `turbovec.probes` cells (`scan.rs`:
/// `PROBES.get().clamp(1, lists)`), so roughly `n_vectors * probes / lists`,
/// never less than one cell's worth (a probe always reads a whole cell, and
/// a zero estimate would let the planner take this path for free).
///
/// Cells are assumed evenly populated. k-means does not guarantee that, but
/// the alternative is reading the cell directory on every planner call --
/// real I/O to sharpen an estimate the planner then compares against its own
/// approximations. If skew ever demonstrably misleads the planner, read the
/// directory here.
///
/// Pure so it can be unit-tested: the arithmetic is NOT observable through
/// `EXPLAIN`, because the index-scan node's cost also carries PostgreSQL's
/// heap-fetch and qual costs, which are far larger and swamp it.
pub(crate) fn scored_vectors(n_vectors: i64, lists: u32, probes: i32) -> f64 {
    let n = n_vectors as f64;
    if lists == 0 {
        return n;
    }
    let lists_f = lists as f64;
    let p = (probes as f64).clamp(1.0, lists_f);
    (n * p / lists_f).max(n / lists_f)
}

/// CPU cost, in PostgreSQL cost units, of scoring `scored` vectors at the
/// given `dim` / `bit_width`.
///
/// PG's unit is "one sequential page read" (`seq_page_cost = 1.0`), with
/// `cpu_operator_cost` per simple operator evaluation. A distance
/// evaluation per candidate anchors naturally at `cpu_operator_cost`,
/// scaled by how much work that distance is relative to the reference.
///
/// The reference is 8 ns/vector at dim=1536 / 4-bit, which validates
/// against our own published measurement (this model says 5.3 ms for
/// 1M x 1024-d 4-bit; measured is 6.08 ms).
pub(crate) fn scan_cpu_cost(scored: f64, dim: i32, bit_width: i32, cpu_operator_cost: f64) -> f64 {
    const REF_BITS: f64 = 1536.0 * 4.0;
    let bits_per_vec = (dim as f64) * (bit_width as f64);
    // work relative to the reference vector
    let work_per_vec = bits_per_vec / REF_BITS;
    scored * work_per_vec * cpu_operator_cost
}

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
    let scored_vectors = scored_vectors(n_vectors, lists, crate::guc::PROBES.get());

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
    let cpu_cost = scan_cpu_cost(scored_vectors, dim, bit_width, pg_sys::cpu_operator_cost);
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

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::{scan_cpu_cost, scored_vectors};

    /// Phase Z4: IVF must be costed for the cells it PROBES.
    ///
    /// Before Z4 the model charged the full corpus for every kind, so an
    /// index probing 1 of 1024 cells cost exactly as much as a flat scan of
    /// everything and the planner could not see IVF's whole point.
    ///
    /// Tested here rather than through `EXPLAIN`: the index-scan node's cost
    /// also carries PostgreSQL's heap-fetch and qual costs, which are ~1482
    /// on a 20k-row fixture and swamp the ~0.5 the AM contributes. Asserting
    /// on the EXPLAIN total measures PG's heap model, not ours.
    #[test]
    fn ivf_scored_vectors_scales_with_probes() {
        let n = 1_000_000;
        let lists = 1024;
        // Flat scores everything.
        assert_eq!(scored_vectors(n, 0, 16), 1_000_000.0);
        // Probing every cell equals a full scan.
        assert_eq!(scored_vectors(n, lists, lists as i32), 1_000_000.0);
        // 16 of 1024 cells -> ~1/64th of the corpus.
        assert_eq!(scored_vectors(n, lists, 16), 1_000_000.0 * 16.0 / 1024.0);
        // Monotone in probes.
        let mut prev = 0.0;
        for p in [1, 2, 8, 64, 512, 1024] {
            let v = scored_vectors(n, lists, p);
            assert!(
                v > prev,
                "scored must grow with probes ({p}: {v} <= {prev})"
            );
            prev = v;
        }
        // A 1-probe scan is dramatically cheaper than a full scan.
        assert!(scored_vectors(n, lists, 1) < scored_vectors(n, 0, 1) / 100.0);
    }

    /// Never estimate below one cell's worth, and never above the corpus:
    /// a probe reads a whole cell, and a zero estimate would let the planner
    /// take the ANN path for free.
    #[test]
    fn scored_vectors_is_clamped() {
        let n = 10_000;
        let lists = 100;
        // probes <= 0 is clamped up to one cell.
        assert_eq!(scored_vectors(n, lists, 0), 100.0);
        assert_eq!(scored_vectors(n, lists, -5), 100.0);
        // probes above `lists` is clamped down to a full scan.
        assert_eq!(scored_vectors(n, lists, 10_000), 10_000.0);
        // An empty index costs nothing to score but must not go negative.
        assert_eq!(scored_vectors(0, lists, 8), 0.0);
    }

    /// The unit conversion: a full 1M x 1024-d 4-bit flat scan must land in
    /// the same ballpark as PostgreSQL's own cost for touching that much
    /// data, not ~3000x cheaper.
    ///
    /// Pre-Z4 the model divided SECONDS by `cpu_operator_cost`, producing
    /// ~23 for this scan while PG costs the equivalent sequential scan at
    /// ~73,000 -- which is why an ANN path could beat plans that are
    /// genuinely faster.
    #[test]
    fn scan_cost_is_in_postgres_units() {
        const CPU_OP: f64 = 0.0025;
        let flat = scan_cpu_cost(1_000_000.0, 1024, 4, CPU_OP);
        // Pre-fix value was ~2.1 (plus a ~21 startup term). Anything in that
        // range means the unit error is back.
        assert!(
            flat > 100.0,
            "a 1M x 1024-d flat scan costing {flat} is implausibly cheap -- \
             the ns->cost unit error has regressed"
        );
        // And it must not overshoot into absurdity either.
        assert!(
            flat < 100_000.0,
            "a 1M x 1024-d flat scan costing {flat} is implausibly expensive"
        );
        // Wider vectors and more bits cost strictly more.
        assert!(scan_cpu_cost(1_000.0, 1536, 4, CPU_OP) > scan_cpu_cost(1_000.0, 768, 4, CPU_OP));
        assert!(scan_cpu_cost(1_000.0, 1024, 4, CPU_OP) > scan_cpu_cost(1_000.0, 1024, 1, CPU_OP));
        // Scoring fewer vectors costs proportionally less.
        let full = scan_cpu_cost(1_000_000.0, 1024, 4, CPU_OP);
        let probed = scan_cpu_cost(1_000_000.0 * 16.0 / 1024.0, 1024, 4, CPU_OP);
        assert!(
            (full / probed - 64.0).abs() < 1e-6,
            "cost must be linear in scored"
        );
    }
}
