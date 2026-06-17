//! Allowlist selectivity crossover bench (Phase C / C-2).
//!
//! Measures the user-facing claim "a selective allowlist makes
//! `turbovec.knn(..., allowed)` CHEAPER, not more expensive" by
//! timing `IdMapIndex::search_with_allowlist` — the exact kernel
//! path `src/knn.rs::run_search` takes — at varying allowlist
//! selectivity (fraction of corpus allowed), against the naive
//! post-filter baseline (unfiltered search at k*oversample, filter
//! the candidate ids in a hash set).
//!
//! `knn()` always builds a FLAT `IdMapIndex` (no IVF cells), so this
//! bench is faithful to the shipped path: same struct, same
//! `search_with_allowlist`, same 32-vector block short-circuit
//! (`turbovec::search::block_has_allowed`).
//!
//! Run (no pgrx / no pg_config needed):
//!
//! ```bash
//! cargo bench --bench allowlist_crossover --no-default-features \
//!     -- --json > benches/results/<file>.json
//! ```
//!
//! Without `--json` it prints a human table to stderr. Env knobs:
//! `TV_N` (corpus rows), `TV_DIM`, `TV_K`, `TV_BITS`, `TV_OVERSAMPLE`,
//! `TV_ITERS`, `TV_SEED`.

use std::time::Instant;

use rand::prelude::*;
use rand::rngs::StdRng;
use turbovec::IdMapIndex;
use turbovec::search::{blocks_skipped_by_mask, reset_blocks_skipped_by_mask};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn random_unit(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// p50 of a Vec of microsecond latencies.
fn p50(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let n = env_usize("TV_N", 300_000);
    let dim = env_usize("TV_DIM", 256);
    let k = env_usize("TV_K", 10);
    let bits = env_usize("TV_BITS", 4);
    let oversample = env_f64("TV_OVERSAMPLE", 4.0);
    let iters = env_usize("TV_ITERS", 50);
    let seed = env_usize("TV_SEED", 0xC0FFEE) as u64;
    let json = std::env::args().any(|a| a == "--json");

    eprintln!(
        "allowlist_crossover: n={n} dim={dim} k={k} bits={bits} \
         oversample={oversample} iters={iters} seed={seed:#x}"
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // Build the flat IdMapIndex exactly as knn() does: ids 0..n,
    // add_with_ids, then prepare the search caches once.
    eprintln!("building corpus ({n} x {dim}) ...");
    let build_t0 = Instant::now();
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        flat.extend_from_slice(&random_unit(dim, &mut rng));
        ids.push(i as u64);
    }
    let mut idx = IdMapIndex::new(dim, bits).expect("IdMapIndex::new");
    idx.add_with_ids(&flat, &ids).expect("add_with_ids");
    // Warm the rotation/centroid/blocked caches (knn() warms via the
    // first search; we time steady state, not first-touch).
    let warm_q = random_unit(dim, &mut rng);
    let _ = idx.search(&warm_q, k);
    eprintln!("built + warmed in {:.1}s", build_t0.elapsed().as_secs_f64());

    // Held-out query vectors (not in the corpus, but same distribution).
    let queries: Vec<Vec<f32>> = (0..iters).map(|_| random_unit(dim, &mut rng)).collect();

    // Selectivity points: fraction of corpus in the allowlist.
    let fractions = [1.0_f64, 0.5, 0.1, 0.01, 0.001];

    // Build a stable shuffled id permutation; the allowlist for
    // fraction f is the first ceil(f*n) ids of that permutation.
    let mut perm: Vec<u64> = (0..n as u64).collect();
    perm.shuffle(&mut rng);

    #[derive(Clone)]
    struct Row {
        fraction: f64,
        allowed: usize,
        allowlist_p50_us: f64,
        baseline_p50_us: f64,
        blocks_skipped_total: u64,
        returned: usize,
    }
    let mut rows: Vec<Row> = Vec::new();

    for &f in &fractions {
        let m = ((f * n as f64).ceil() as usize).max(k).min(n);
        let mut allow: Vec<u64> = perm[..m].to_vec();
        allow.sort_unstable();
        let allow_set: std::collections::HashSet<u64> = allow.iter().copied().collect();

        // --- in-kernel allowlist path (what knn(..., allowed) runs) ---
        reset_blocks_skipped_by_mask();
        let mut allow_lat = Vec::with_capacity(iters);
        let mut returned = 0usize;
        for q in &queries {
            let take = k.min(m);
            let t0 = Instant::now();
            let (_s, hits) = idx.search_with_allowlist(q, take, Some(&allow));
            allow_lat.push(t0.elapsed().as_secs_f64() * 1e6);
            returned = hits.len();
        }
        let blocks_skipped = blocks_skipped_by_mask();

        // --- naive post-filter baseline: fetch k*oversample
        //     unfiltered, drop ids not in the set, keep top-k ---
        let fetch = ((k as f64 * oversample).ceil() as usize).min(n);
        let mut base_lat = Vec::with_capacity(iters);
        for q in &queries {
            let t0 = Instant::now();
            let (_s, hits) = idx.search(q, fetch);
            let kept: Vec<u64> = hits.into_iter().filter(|id| allow_set.contains(id)).take(k).collect();
            base_lat.push(t0.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(kept);
        }

        rows.push(Row {
            fraction: f,
            allowed: m,
            allowlist_p50_us: p50(allow_lat),
            baseline_p50_us: p50(base_lat),
            blocks_skipped_total: blocks_skipped,
            returned,
        });
    }

    // Human table.
    eprintln!();
    eprintln!(
        "{:>9}  {:>10}  {:>16}  {:>16}  {:>14}  {:>8}",
        "fraction", "allowed", "allowlist p50 us", "baseline p50 us", "blks skipped", "returned"
    );
    for r in &rows {
        eprintln!(
            "{:>9.3}  {:>10}  {:>16.1}  {:>16.1}  {:>14}  {:>8}",
            r.fraction,
            r.allowed,
            r.allowlist_p50_us,
            r.baseline_p50_us,
            r.blocks_skipped_total,
            r.returned,
        );
    }

    if json {
        let simd = if cfg!(target_feature = "avx2") {
            "avx2 (compile-time)"
        } else {
            "scalar/sse (no avx2 at compile time)"
        };
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str("  \"benchmark\": \"allowlist selectivity crossover (in-kernel pushdown vs naive post-filter)\",\n");
        s.push_str("  \"schema_version\": 1,\n");
        s.push_str(&format!("  \"date\": \"{}\",\n", env_date()));
        s.push_str("  \"path_measured\": \"turbovec::IdMapIndex::search_with_allowlist (identical to src/knn.rs run_search; FLAT index, no IVF)\",\n");
        s.push_str(&format!(
            "  \"corpus\": {{ \"rows\": {n}, \"dim\": {dim}, \"distribution\": \"random unit vectors\", \"bit_width\": {bits} }},\n"
        ));
        s.push_str(&format!(
            "  \"params\": {{ \"k\": {k}, \"baseline_oversample\": {oversample}, \"iters\": {iters}, \"seed\": {seed} }},\n"
        ));
        s.push_str(&format!("  \"simd\": \"{simd}\",\n"));
        s.push_str("  \"points\": [\n");
        for (i, r) in rows.iter().enumerate() {
            s.push_str(&format!(
                "    {{ \"fraction\": {:.4}, \"allowed\": {}, \"allowlist_p50_us\": {:.1}, \"baseline_p50_us\": {:.1}, \"blocks_skipped_total\": {}, \"returned\": {} }}{}\n",
                r.fraction,
                r.allowed,
                r.allowlist_p50_us,
                r.baseline_p50_us,
                r.blocks_skipped_total,
                r.returned,
                if i + 1 == rows.len() { "" } else { "," },
            ));
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        println!("{s}");
    }
}

fn env_date() -> String {
    // Avoid a chrono dep: shell out to `date`. Lazy but adequate for a
    // bench artifact timestamp.
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
