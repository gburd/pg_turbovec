# Source this before running: activates venv + NixOS shared-lib paths for faiss/numpy.
cd /tmp/gapb-investigation
. .venv/bin/activate
GCC=/nix/store/0p8b2lqk47fvxm9hc6c8mnln5l8x51q1-gcc-14.3.0-lib/lib
GCC2=/nix/store/7c0v0kbrrdc2cqgisi78jdqxn73n3401-gcc-14.2.1.20250322-lib/lib
ZLIB=/nix/store/ig0kkzw4n2pws12dj7szjm71f1a43if6-zlib-1.3/lib
OB=/nix/store/qbq20d6v6qf87cnlv5k55i0hnpzy00hq-openblas-0.3.30/lib
export LD_LIBRARY_PATH="$GCC:$GCC2:$ZLIB:$OB:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=8
