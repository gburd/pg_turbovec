// Quick smoke check of Phase P prepared-layout savings.
// Runs: cargo run --release --example phase_p_smoke
// (or --debug for the slower, more telling debug-mode timing).

use std::time::Instant;

fn main() {
    let n = 1000;
    let dim = 384;
    let bit_width = 4;

    // Cheap pseudo-random vectors.
    let mut vectors = vec![0.0_f32; n * dim];
    for i in 0..n {
        for k in 0..dim {
            let h = (i.wrapping_mul(2654435761) ^ k.wrapping_mul(40503)) & 0x7FFF;
            vectors[i * dim + k] = (h as f32 / 16384.0) - 1.0;
        }
    }
    let ids: Vec<u64> = (0..n as u64).collect();

    // Build the index.
    let mut idx = turbovec::IdMapIndex::new(dim, bit_width);
    idx.add_with_ids(&vectors, &ids).unwrap();

    // Capture prepared parts.
    let t0 = Instant::now();
    idx.prepare_eager();
    let prep_us = t0.elapsed().as_micros();
    let blocked = idx.blocked_codes().to_vec();
    let n_blocks = idx.n_blocks();
    let centroids = idx.centroids().to_vec();
    let boundaries = idx.boundaries().to_vec();
    let bit_width_inner = idx.bit_width();
    let dim_inner = idx.dim();
    let n_inner = idx.len();

    // Snapshot raw parts via packed_codes / scales / slot_to_id.
    let codes = idx.packed_codes().to_vec();
    let scales = idx.scales().to_vec();
    let slot_to_id = idx.slot_to_id().to_vec();

    drop(idx);

    let q: Vec<f32> = vectors[..dim].to_vec();

    // Plain reload (no prepared parts) — first search pays repack + codebook.
    let t1 = Instant::now();
    let plain = turbovec::IdMapIndex::from_id_map_parts(
        bit_width_inner, dim_inner, n_inner,
        codes.clone(), scales.clone(), slot_to_id.clone(),
    ).unwrap();
    let plain_ctor_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let (_, ids_plain) = plain.search(&q, 1);
    let plain_search_us = t2.elapsed().as_micros();

    // Prepared reload — first search reads OnceLocks and skips repack/codebook.
    let t3 = Instant::now();
    let prep = turbovec::IdMapIndex::from_id_map_parts_with_prepared(
        bit_width_inner, dim_inner, n_inner,
        codes, scales, slot_to_id,
        blocked, n_blocks, centroids, boundaries,
    ).unwrap();
    let prep_ctor_us = t3.elapsed().as_micros();

    let t4 = Instant::now();
    let (_, ids_prep) = prep.search(&q, 1);
    let prep_search_us = t4.elapsed().as_micros();

    println!("phase-p smoke (n={n}, dim={dim}, bit_width={bit_width}):");
    println!("  prepare_eager (writer side): {prep_us} us");
    println!("  PLAIN  ctor: {plain_ctor_us:>6} us  first-search: {plain_search_us:>6} us");
    println!("  PREP   ctor: {prep_ctor_us:>6} us  first-search: {prep_search_us:>6} us");
    println!("  speedup on first-search: {:.1}x", plain_search_us as f64 / prep_search_us.max(1) as f64);
    assert_eq!(ids_plain, ids_prep, "results must agree");
}
