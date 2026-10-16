# pg_turbovec build env on Debian 12 / LLVM 14 (host: k-s0spike)
. ~/.cargo/env
export LIBCLANG_PATH=/usr/lib/llvm-14/lib
# clang-14 ships its own resource headers; glibc headers are in the standard prefix.
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/include -isystem /usr/lib/llvm-14/lib/clang/14.0.6/include"
# Debian default linker is GNU ld.bfd; no openblas needed (pure-Rust gemm since v2.0.0).
