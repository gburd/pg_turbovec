//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Repack bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
pub fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let blocked_size = n_blocks * n_byte_groups * BLOCK;
    let bytes_per_row = bits * bytes_per_plane;

    let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

    // Step 1: Extract packed nibble bytes per vector per group
    let mut codes_flat = vec![vec![0u8; n_byte_groups]; n_vectors];
    for vec_idx in 0..n_vectors {
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                byte_val |= code << shift;
            }
            codes_flat[vec_idx][g] = byte_val;
        }
    }

    // Step 2: Pack into platform-specific layout
    let blocked = pack_blocked(n_vectors, n_blocks, n_byte_groups, blocked_size, &codes_flat, &perm0);
    (blocked, n_blocks)
}

#[cfg(target_arch = "x86_64")]
fn pack_blocked(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[Vec<u8>],
    perm0: &[usize; 16],
) -> Vec<u8> {
    // FAISS layout: split each byte into hi/lo nibbles, interleave with perm0.
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va = base_vec + perm0[j];
                let vb = base_vec + perm0[j] + 16;
                let ba = if va < n { codes_flat[va][g] } else { 0 };
                let bb = if vb < n { codes_flat[vb][g] } else { 0 };
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    blocked
}

#[cfg(not(target_arch = "x86_64"))]
fn pack_blocked(
    n: usize,
    n_blocks: usize,
    n_byte_groups: usize,
    blocked_size: usize,
    codes_flat: &[Vec<u8>],
    _perm0: &[usize; 16],
) -> Vec<u8> {
    // Sequential layout: each byte stored as-is, vectors in order.
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    blocked[out_offset + lane] = codes_flat[vi][g];
                }
            }
        }
    }
    blocked
}

/// Repack 3-bit codes into two blocked arrays:
/// - sub_codes: 2-bit nibble format from planes 0,1
/// - plane2: packed bits blocked by 32 vectors
pub fn repack_3bit(
    packed_codes: &[u8],
    n_vectors: usize,
    dim: usize,
) -> (Vec<u8>, Vec<u8>, usize) {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = 3 * bytes_per_plane;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;

    let sub_byte_groups = dim / 4;
    let mut sub_codes = vec![0u8; n_blocks * sub_byte_groups * BLOCK];

    let plane2_byte_groups = bytes_per_plane;
    let mut plane2_blocked = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;

        for g in 0..sub_byte_groups {
            let out_offset = (block_idx * sub_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }

                let mut byte_val = 0u8;
                let dim_start = g * 4;
                for c in 0..4usize {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..2usize {
                        let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                        if plane_byte & mask != 0 { code |= 1 << p; }
                    }
                    byte_val |= code << ((3 - c) * 2);
                }
                sub_codes[out_offset + lane] = byte_val;
            }
        }

        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }
                plane2_blocked[out_offset + lane] = packed_codes[vec_idx * bytes_per_row + 2 * bytes_per_plane + g];
            }
        }
    }

    (sub_codes, plane2_blocked, n_blocks)
}
