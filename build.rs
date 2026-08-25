fn main() {
    // turbovec 1.0.0's v5 block-Hadamard rotation is pure Rust and
    // drops the OpenBLAS dependency (upstream #, "drops the 42 MB
    // OpenBLAS dependency"). pg_turbovec's own IVF k-means uses the
    // pure-Rust `gemm` crate (no external BLAS), so nothing in the
    // shared object references `cblas_sgemm` anymore. The old
    // `cargo:rustc-link-lib=openblas` / Accelerate re-emit is dead and
    // would add a spurious DT_NEEDED (a hard `LOAD` failure on a host
    // without libopenblas). Removed for the 2.0.0 port.
}
