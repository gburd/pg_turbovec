// pg_turbovec's exact cosine (src/kernels.rs, verbatim) vs a vectorizable f32 one.
fn dot(a: &[f32], b: &[f32]) -> f64 { let mut acc = 0.0f64; for (x, y) in a.iter().zip(b) { acc += f64::from(*x) * f64::from(*y); } acc }
fn norm2(a: &[f32]) -> f64 { let mut acc = 0.0f64; for x in a { acc += f64::from(*x) * f64::from(*x); } acc }
fn cosine_ours(a: &[f32], b: &[f32]) -> f64 { let (na, nb) = (norm2(a), norm2(b)); 1.0 - (dot(a, b) / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0) }
// 8 independent f32 lanes -> LLVM emits packed FMAs; query norm passed in (cacheable per scan).
fn cosine_lanes(a: &[f32], b: &[f32], qn: f64) -> f64 {
    let (mut d, mut n) = ([0f32; 16], [0f32; 16]);
    for (ca, cb) in a.chunks_exact(16).zip(b.chunks_exact(16)) { for i in 0..16 { d[i] += ca[i] * cb[i]; n[i] += ca[i] * ca[i]; } }
    let (d, n): (f64, f64) = (d.iter().map(|&x| x as f64).sum(), n.iter().map(|&x| x as f64).sum());
    1.0 - (d / (n.sqrt() * qn)).clamp(-1.0, 1.0)
}
fn main() {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.11).cos()).collect();
    let n = 2_000_000; let mut s = 0.0;
    let t = std::time::Instant::now(); for _ in 0..n { s += cosine_ours(std::hint::black_box(&a), std::hint::black_box(&b)); }
    let o = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    let qn = norm2(&b).sqrt();
    let t = std::time::Instant::now(); for _ in 0..n { s += cosine_lanes(std::hint::black_box(&a), std::hint::black_box(&b), qn); }
    let l = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("1024-d cosine: ours (serial f64, query norm per call) {o:.3} us | 16-lane f32, cached query norm {l:.3} us | diff {:.2e} ({s:.1})",
        (cosine_ours(&a, &b) - cosine_lanes(&a, &b, qn)).abs());
}
