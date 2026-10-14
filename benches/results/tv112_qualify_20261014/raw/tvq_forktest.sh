set -uo pipefail
. ~/.cargo/env
D=/work/turbovec-$(date +%s); git clone -q -b pgtv-2.12.1-port ~/tv-port.bundle $D && cd $D
git log --oneline -1
run() { echo "=== $1"; shift; env "$@" cargo test --release -p turbovec --all-targets 2>&1 | tee /work/tvtest-$TAG.log | grep -E '^test result|FAILED|panicked|^error' | awk '/^test result/{p+=$4;f+=$6;i+=$8;n++} !/^test result/{print} END{print "passed",p,"failed",f,"ignored",i,n" binaries"}'; }
if grep -q -w avx512_vnni /proc/cpuinfo; then
  TAG=asm run "forced asm (TURBOVEC_DPBUSD_ASM=1)" TURBOVEC_DPBUSD_ASM=1 CARGO_TARGET_DIR=/work/t-asm
  grep -c 'dpbusd_matches_the_scalar_definition ... ok' /work/tvtest-asm.log
  TAG=intr run "intrinsic (LLVM 22)" CARGO_TARGET_DIR=/work/t-intr
  grep -c 'dpbusd_matches_the_scalar_definition ... ok' /work/tvtest-intr.log
else
  TAG=arm run "aarch64" CARGO_TARGET_DIR=/work/t-arm
fi
