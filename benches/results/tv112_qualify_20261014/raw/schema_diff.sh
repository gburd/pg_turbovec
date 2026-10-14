set -uo pipefail
TC=/home/gburd/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu
export PATH=$TC/bin:$PATH RUSTC=$TC/bin/rustc
export LIBCLANG_PATH=/nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /nix/store/x8lqlydsxbrwvf6p7v18gws8kn1xl37f-glibc-2.38-23-dev/include -isystem /nix/store/10y7v0cqr8xqsqlqnfzw6i9s42f6f8rd-clang-17.0.6-lib/lib/clang/17/include"
export RUSTFLAGS="-L /nix/store/qbq20d6v6qf87cnlv5k55i0hnpzy00hq-openblas-0.3.30/lib -C link-arg=-fuse-ld=bfd"
W=/tmp/schemadiff
cd $W/main && CARGO_TARGET_DIR=$W/t-main $W/pgrx191/bin/cargo-pgrx pgrx schema pg16 -o $W/main.sql > $W/main.log 2>&1; echo "main rc=$?"
norm() { grep -v -E '^\s*--|^\s*$|^/\*|^\*/|This file is auto generated|The ordering of items' "$1" | sed -E 's#/\* [^*]*\*/##g; s/[[:space:]]+$//' | sort; }
norm $W/main.sql > $W/main.norm; norm $W/new.sql > $W/new.norm
wc -l $W/main.norm $W/new.norm
diff $W/main.norm $W/new.norm > $W/schema.diff; echo "diff lines: $(wc -l < $W/schema.diff)"; head -60 $W/schema.diff
