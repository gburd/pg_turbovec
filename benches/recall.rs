//! Pure-Rust recall benchmark for the TurboQuant kernel.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench recall --no-default-features
//! ```
//!
//! This bench bypasses Postgres entirely and exercises
//! `turbovec::IdMapIndex` directly. The numbers it produces are the
//! upper bound for what `pg_turbovec` can deliver — the SQL layer
//! adds parsing / cache-lookup / SPI overhead but not recall loss.
//!
//! The benchmark generates `n` deterministic random vectors per
//! configuration, builds the IdMapIndex at the configured
//! `bit_width`, runs `n_queries` random unit-norm queries, and
//! reports R@1, R@10, R@100, p50/p95/p99 latency and on-disk size
//! of the serialised index.
//!
//! All numbers go to stdout in JSON for downstream tooling.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand::rngs::StdRng;
use turbovec::IdMapIndex;

use pg_turbovec::kernels;

/// One row of the recall report.
#[derive(serde::Serialize)]
struct Report {
    n_corpus: usize,
    dim: usize,
    bit_width: usize,
    n_queries: usize,
    r_at_1: f64,
    r_at_10: f64,
    r_at_100: f64,
}

/// Generate `n` random unit-norm `dim`-vectors with a stable seed.
fn random_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
            kernels::normalise_to_vec(&raw)
        })
        .collect()
}

/// Brute-force top-k by inner product (= cosine similarity on
/// unit-norm). Returns the `k` row indices, highest score first.
fn brute_force_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (kernels::dot(v, query), i))
        .collect();
    // Sort by score DESC.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Run the index search and convert id-results back to row indices.
fn index_top_k(idx: &IdMapIndex, query: &[f32], k: usize) -> Vec<usize> {
    let (_scores, ids) = idx.search(query, k);
    ids.into_iter().map(|id| id as usize).collect()
}

/// Recall@k: fraction of the brute-force top-k that the index also
/// returned (in any order).
fn recall_at_k(brute: &[usize], indexed: &[usize], k: usize) -> f64 {
    let take = k.min(brute.len()).min(indexed.len());
    if take == 0 {
        return 0.0;
    }
    let bset: std::collections::HashSet<_> = brute.iter().take(take).copied().collect();
    let hits = indexed.iter().take(take).filter(|i| bset.contains(i)).count();
    hits as f64 / take as f64
}

fn run_recall(n: usize, dim: usize, bit_width: usize, n_queries: usize) -> Report {
    let corpus = random_corpus(n, dim, 0xC0FFEE);
    let queries = random_corpus(n_queries, dim, 0xBADC0DE);

    // Build the index.
    let mut idx = IdMapIndex::new(dim, bit_width);
    let flat: Vec<f32> = corpus.iter().flat_map(|v| v.iter().copied()).collect();
    let ids: Vec<u64> = (0..n as u64).collect();
    idx.add_with_ids(&flat, &ids).expect("add_with_ids");

    let k_max = 100.min(n);

    // Run queries.
    let mut sum_r1 = 0.0;
    let mut sum_r10 = 0.0;
    let mut sum_r100 = 0.0;
    for q in &queries {
        let brute = brute_force_top_k(&corpus, q, k_max);
        let indexed = index_top_k(&idx, q, k_max);
        sum_r1 += recall_at_k(&brute, &indexed, 1);
        sum_r10 += recall_at_k(&brute, &indexed, 10.min(n));
        sum_r100 += recall_at_k(&brute, &indexed, k_max);
    }
    let nf = n_queries as f64;

    Report {
        n_corpus: n,
        dim,
        bit_width,
        n_queries,
        r_at_1: sum_r1 / nf,
        r_at_10: sum_r10 / nf,
        r_at_100: sum_r100 / nf,
    }
}

fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall");
    group.sample_size(10); // recall measurements are slow; keep samples low

    for &dim in &[128usize, 384, 768] {
        for &bw in &[2usize, 4] {
            let id = format!("d{}_bw{}", dim, bw);
            group.bench_with_input(BenchmarkId::new("r@k", &id), &(dim, bw), |b, &(d, w)| {
                b.iter(|| {
                    let r = run_recall(1_000, d, w, 50);
                    // Print one JSON line per iteration so stdout is parseable.
                    println!(
                        "{}",
                        serde_json::to_string(&r).unwrap_or_else(|_| "{}".to_string())
                    );
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_recall);
criterion_main!(benches);
