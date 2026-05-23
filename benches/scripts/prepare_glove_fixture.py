#!/usr/bin/env python3
"""Convert an ann-benchmarks GloVe HDF5 file into the pg_turbovec
fixture format documented in docs/RECALL.md § 6.1.

Inputs (positional):
    1. path to glove-{25,50,100,200}-angular.hdf5
    2. output directory (created if absent)
    3. corpus_n: number of training rows to extract (default 100000)
    4. queries_n: number of test queries to extract (default 1000)

Outputs (in <outdir>):
    corpus.bin          <u32 dim><u32 n><f32 dim*n>   — corpus vectors
    queries.bin         <u32 dim><u32 n><f32 dim*n>   — query vectors
    ground_truth.bin    <u32 k><u32 n><u32 k*n>       — exact top-k per query
                        recomputed against the *subset* corpus (not the full
                        ann-benchmarks corpus), since the published neighbors
                        index into the full 1.18M-row training set.
    fixture.json        metadata (dim, sizes, source)

The vectors are unit-normalized in place — `glove-*-angular` already
ships pretty close to unit norm but we re-normalize defensively so
both pgvector cosine and pg_turbovec (which assumes unit-norm input)
see the same data.

Run via:
    nix-shell -p python3Packages.numpy python3Packages.h5py \\
        --run "python3 prepare_glove_fixture.py glove-100-angular.hdf5 ./out 100000 1000"
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np


def write_fixture(path: Path, mat: np.ndarray) -> None:
    """Write a (n, dim) float32 matrix in `<u32 dim><u32 n><f32...>` format."""
    n, dim = mat.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", dim, n))
        mat.astype("<f4", copy=False).tofile(f)


def write_ground_truth(path: Path, gt: np.ndarray) -> None:
    """Write a (n_queries, k) uint32 matrix in `<u32 k><u32 n><u32...>` format."""
    n, k = gt.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", k, n))
        gt.astype("<u4", copy=False).tofile(f)


def unit_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__ or "")
        return 2

    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    corpus_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
    queries_n = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src}")
    with h5py.File(src, "r") as f:
        train = np.asarray(f["train"][:corpus_n], dtype=np.float32)
        test = np.asarray(f["test"][:queries_n], dtype=np.float32)
        # NOTE: f['neighbors'] is over the FULL training set; we
        # recompute below against our subset.

    n_corpus, dim = train.shape
    n_queries, _ = test.shape
    print(f"Corpus: {n_corpus} x {dim}, queries: {n_queries} x {dim}")

    print("Unit-normalizing corpus and queries (cosine == inner product)")
    train = unit_normalize(train)
    test = unit_normalize(test)

    print("Computing exact top-100 ground truth on subset...")
    # Inner product on unit-norm == cosine similarity, descending order.
    # For numerical stability and speed, do it in chunks.
    k_gt = 100
    gt = np.empty((n_queries, k_gt), dtype=np.uint32)
    chunk = 256
    for i in range(0, n_queries, chunk):
        q = test[i : i + chunk]                  # (b, d)
        scores = q @ train.T                     # (b, n_corpus)
        # argsort descending; use argpartition for speed.
        part = np.argpartition(-scores, k_gt - 1, axis=1)[:, :k_gt]
        # sort the top-k portion exactly.
        rows = np.arange(part.shape[0])[:, None]
        part_sorted = part[rows, np.argsort(-scores[rows, part], axis=1)]
        gt[i : i + chunk] = part_sorted.astype(np.uint32)

    print(f"Writing {outdir}/corpus.bin")
    write_fixture(outdir / "corpus.bin", train)
    print(f"Writing {outdir}/queries.bin")
    write_fixture(outdir / "queries.bin", test)
    print(f"Writing {outdir}/ground_truth.bin")
    write_ground_truth(outdir / "ground_truth.bin", gt)

    meta = {
        "source": str(src),
        "dim": int(dim),
        "corpus_n": int(n_corpus),
        "queries_n": int(n_queries),
        "ground_truth_k": int(k_gt),
        "metric": "cosine (inner product on unit-norm)",
        "note": (
            "Ground truth is recomputed against the subset, NOT the published "
            "ann-benchmarks neighbors (which index the full 1.18M-row train set)."
        ),
    }
    with open(outdir / "fixture.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {outdir}/fixture.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
