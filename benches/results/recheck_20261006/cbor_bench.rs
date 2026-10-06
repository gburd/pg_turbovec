// Isolate CBOR decode of a 1024-d Vector vs. a raw f32 memcpy, same bytes.
fn main() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Vector { data: Vec<f32> }
    let v = Vector { data: (0..1024).map(|i| (i as f32 * 0.37).sin()).collect() };
    let enc = serde_cbor::to_vec(&v).unwrap();
    let raw: Vec<u8> = v.data.iter().flat_map(|x| x.to_le_bytes()).collect();
    println!("cbor bytes={} raw bytes={}", enc.len(), raw.len());
    let n = 200_000;
    let t = std::time::Instant::now(); let mut s = 0.0f32;
    for _ in 0..n { let d: Vector = serde_cbor::from_slice(&enc).unwrap(); s += d.data[7]; }
    let cb = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    let t = std::time::Instant::now();
    for _ in 0..n { let d: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(); s += d[7]; }
    let rw = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    let t = std::time::Instant::now();
    for _ in 0..n { let d: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, 1024) }; s += std::hint::black_box(d)[7]; }
    let zc = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("per 1024-d decode: cbor={cb:.2} us  raw-copy={rw:.3} us  zero-copy={zc:.4} us  ({s})");
}
