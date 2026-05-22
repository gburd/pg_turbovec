fn main() {
    // turbovec depends on `openblas-src` for SIMD-accelerated routines.
    // Because our crate is a `cdylib`, rustc does not automatically
    // propagate link directives from indirect dependencies into the
    // shared object's `DT_NEEDED` list. Re-emit them here so the
    // resulting `pg_turbovec.so` carries an explicit dependency on
    // OpenBLAS — without this, `LOAD 'pg_turbovec'` fails with
    // "undefined symbol: cblas_sgemm".
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=openblas");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // On macOS turbovec uses Apple Accelerate via the
        // `accelerate` blas-src feature.
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
