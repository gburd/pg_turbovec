// Kernel-only single-query latency: the exact turbovec search pg_turbovec's
// flat scan calls (TurboQuantIndex::from_parts + prepare + search) on the real
// corpus codes, no PostgreSQL in the loop. Also times cold-open
// (from_parts + prepare = the per-backend cache build) and dumps ids so the
// staged id set can be compared to the whole-index scan.
use std::io::Read;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (base, qf, n, bits) = (&a[1], &a[2], a[3].parse::<usize>().unwrap(), a[4].parse::<usize>().unwrap());
    let dim = 1024usize;
    let threads: usize = a[5].parse().unwrap();
    let ids_out = a.get(6).cloned();
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let rd = |p: &str| { let mut b = Vec::new(); std::fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>() };
    let v = rd(base); let q = rd(qf); assert_eq!(v.len(), n * dim);
    let nq = (q.len() / dim).min(200);
    let mut enc = turbovec::TurboQuantIndex::new(dim, bits).unwrap();
    enc.add(&v);
    let (codes, scales) = (enc.packed_codes().to_vec(), enc.scales().to_vec());
    drop(enc); drop(v);
    let mut cold = Vec::new();
    let mut idx = None;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let ix = turbovec::TurboQuantIndex::from_parts(Some(dim), bits, n, codes.clone(), scales.clone(), vec![], vec![]).unwrap();
        ix.prepare();
        cold.push(t.elapsed().as_secs_f64() * 1e3);
        idx = Some(ix);
    }
    let idx = idx.unwrap();
    cold.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print!("{{\"threads\":{threads},\"bits\":{bits},\"cold_open_ms\":{:.1}", cold[1]);
    let mut dump = String::new();
    for k in [10usize, 32, 100, 256, 1024] {
        for i in 0..10 { let _ = idx.search(&q[i * dim..(i + 1) * dim], k); }
        let mut t = Vec::new();
        for i in 0..nq {
            let s = std::time::Instant::now();
            let r = idx.search(&q[i * dim..(i + 1) * dim], k);
            t.push(s.elapsed().as_secs_f64() * 1e3);
            if k == 10 || k == 100 { dump += &format!("{k} {i} {:?}\n", r.indices); }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        print!(",\"k{k}_p50_ms\":{:.3}", t[t.len() / 2]);
    }
    println!("}}");
    if let Some(p) = ids_out { std::fs::write(p, dump).unwrap(); }
}
