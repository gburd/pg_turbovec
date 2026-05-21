//! Pure-Rust micro-benchmarks for the distance kernels.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench distance --no-default-features
//! ```
//!
//! `--no-default-features` is needed because the default `pg17`
//! feature pulls in pgrx's pg_sys bindgen, which requires
//! `pg_config`. The bench only touches `pg_turbovec::kernels` and
//! does not need any Postgres infrastructure.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pg_turbovec::kernels;
use rand::prelude::*;
use rand::rngs::StdRng;

fn random_unit(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    kernels::normalise_to_vec(&raw)
}

fn bench_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    for &dim in &[128usize, 384, 768, 1536, 3072] {
        let a = random_unit(dim, &mut rng);
        let b = random_unit(dim, &mut rng);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bencher, _| {
            bencher.iter(|| kernels::dot(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("l2_sq", dim), &dim, |bencher, _| {
            bencher.iter(|| kernels::l2_sq(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("l1_abs", dim), &dim, |bencher, _| {
            bencher.iter(|| kernels::l1_abs(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bencher, _| {
            bencher.iter(|| kernels::cosine_distance(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
