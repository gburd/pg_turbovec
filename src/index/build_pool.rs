//! Bounded rayon thread pool for the `ambuild` quantize + repack phases.
//!
//! Parity gap #2 (v1.8.0). pgvector parallelises its HNSW / IVFFlat
//! builds across `max_parallel_maintenance_workers`; pg_turbovec's
//! `ambuild` was single-threaded *from PG's point of view* — it never
//! launched maintenance workers and never sized its CPU fan-out off
//! PG's parallel-maintenance budget.
//!
//! turbovec's `encode` and `pack::repack` are already
//! embarrassingly-parallel per vector (rayon `par_iter` / `par_chunks`
//! inside the crate), but they fan out across rayon's *global* default
//! pool — every core on the box — which both ignores
//! `max_parallel_maintenance_workers` and oversubscribes on a busy
//! host. This module gives `ambuild` an explicit, GUC-sized pool to
//! `install()` that work into.
//!
//! ## Why this preserves byte-identical relfiles
//!
//! The parallelism here is purely *intra-phase*: rayon splits the
//! per-vector encode and the per-block repack across threads, but every
//! row writes its codes/scales to a fixed output index and every block
//! writes to a fixed offset. The result is therefore independent of the
//! thread count (data-parallel map, no reduction order dependence). The
//! heap scan itself stays serial, so:
//!
//! - slot ordering is identical to a serial build (heap-scan order), and
//! - the TQ+ calibration is fit on the same first chunk.
//!
//! Two builds of the same table — serial or parallel, any pool size —
//! produce byte-identical `packed_codes` / `scales` / `slot_to_id` and
//! thus byte-identical relfiles. Asserted directly by the
//! `build_parts_are_pool_size_invariant` unit test below.

use pgrx::pg_sys;

use crate::guc;

/// Resolve the build pool size from `turbovec.build_parallelism`,
/// falling back to PG's `max_parallel_maintenance_workers + 1` (leader
/// plus worker budget) when the GUC is 0 (auto). Clamped to at least 1
/// so a zero-thread `ThreadPoolBuilder` (which rayon reads as "use the
/// global default pool") can never sneak in.
pub(crate) fn resolve_pool_size() -> usize {
    let configured = guc::BUILD_PARALLELISM.get();
    if configured > 0 {
        return configured as usize;
    }
    // SAFETY: a C global int set at postmaster start; reading it from a
    // backend is safe. The `+ 1` accounts for the leader, mirroring how
    // PG sizes a parallel maintenance operation (leader participates).
    let workers = unsafe { pg_sys::max_parallel_maintenance_workers }.max(0) as usize;
    (workers + 1).max(1)
}

/// Build a rayon thread pool of the resolved size. Returns `None` on
/// the degenerate single-thread case so the caller can run the work
/// inline (avoids spawning a pool just to use one thread, and avoids
/// the rayon-in-rayon nesting cost if turbovec's global pool is already
/// active).
pub(crate) fn make_pool() -> Option<rayon::ThreadPool> {
    let n = resolve_pool_size();
    // Always build a real pool of the resolved size (>=1). Returning
    // a pool even for n==1 (instead of None -> inline) is important
    // now that the IVF k-means path uses rayon `par_iter` internally:
    // with `None`, `install` runs the closure inline and any nested
    // `par_iter` escapes to rayon's GLOBAL pool (all machine cores),
    // violating the `build_parallelism = 1` resource-control contract.
    // A size-1 pool confines that nested parallelism to a single
    // thread. (Determinism is independent of thread count -- the
    // k-means reduction is fixed-order -- so this only affects
    // resource use, not results.)
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .thread_name(|i| format!("turbovec-build-{i}"))
        .build()
        .ok()
}

/// Build a rayon thread pool of exactly `n` threads for the IVF
/// per-query fine-scan (item #2 of the IVF-scaling work), or `None`
/// for the degenerate `n <= 1` case so the caller runs inline. Unlike
/// [`make_pool`], the size is passed in already-resolved (by
/// `guc::resolve_scan_parallelism`, which caps it modestly for
/// concurrency safety) rather than derived from the build GUC — a scan
/// and a build have different fan-out budgets. The threads do pure
/// compute over owned code bytes; no PG state is touched inside them.
pub(crate) fn scan_pool(n: usize) -> Option<rayon::ThreadPool> {
    if n <= 1 {
        return None;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("turbovec-scan-{i}"))
        .build()
        .ok()
}

/// Run `f` on `pool` if present, else inline. The closure is where the
/// turbovec calls that fan out via rayon (`add_with_ids` → `encode`,
/// `prepare_eager` → `repack`) execute, so they pick up `pool` as the
/// ambient pool via `install`.
#[inline]
pub(crate) fn install<R: Send>(
    pool: Option<&rayon::ThreadPool>,
    f: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use turbovec::IdMapIndex;

    /// Build an `IdMapIndex` from the same vectors via a rayon pool of
    /// `n_threads` and return its persisted parts (the bytes that land
    /// in the relfile: packed_codes, scales, blocked_codes, slot_to_id).
    fn build_parts(
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        bit_width: usize,
        n_threads: usize,
    ) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<u64>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut idx = IdMapIndex::new(dim, bit_width).unwrap();
            idx.add_with_ids(vectors, ids).unwrap();
            idx.prepare_eager();
            (
                idx.packed_codes().to_vec(),
                idx.scales().to_vec(),
                idx.blocked_codes().to_vec(),
                idx.slot_to_id().to_vec(),
            )
        })
    }

    /// The load-bearing determinism guarantee: the relfile-bound parts
    /// (`packed_codes` / `scales` / `blocked_codes` / `slot_to_id`) are
    /// byte-for-byte independent of the rayon pool size. This is what
    /// lets a parallel `ambuild` produce a relfile logically identical
    /// to a serial build — the quantize/repack fan-out is a pure
    /// data-parallel map writing each row/block to a fixed index.
    ///
    /// We need >= 1000 vectors so the TQ+ calibration path (which only
    /// engages at `TQPLUS_MIN_SAMPLES`) is exercised under both pool
    /// sizes; below that threshold encode falls back to identity
    /// calibration and the test would miss the calibration code path.
    #[test]
    fn build_parts_are_pool_size_invariant() {
        let dim = 64usize;
        let n = 1500usize;
        let bit_width = 4usize;
        // Deterministic pseudo-random vectors (same LCG the repack test
        // uses), so the fixture itself doesn't introduce nondeterminism.
        let mut s = 0x9e37_79b9u32;
        let mut vectors = vec![0.0f32; n * dim];
        for v in vectors.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // map to [-1, 1)
            *v = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let ids: Vec<u64> = (0..n as u64).collect();

        let serial = build_parts(&vectors, &ids, dim, bit_width, 1);
        let parallel = build_parts(&vectors, &ids, dim, bit_width, 4);

        assert_eq!(serial.0, parallel.0, "packed_codes differ by pool size");
        assert_eq!(serial.1, parallel.1, "scales differ by pool size");
        assert_eq!(
            serial.2, parallel.2,
            "blocked_codes differ by pool size"
        );
        assert_eq!(serial.3, parallel.3, "slot_to_id differs by pool size");
    }
}
