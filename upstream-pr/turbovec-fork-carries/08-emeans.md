# Carry #8: deterministic Euclidean k-means (Lloyd + e-means)

Fork branch `carry/emeans`, commit 662084c (on top of fork main 1202a9e =
upstream 1.1.2 + carries #1-#7). New module `turbovec/src/cluster.rs`.

## What it is

A general vector clustering primitive turbovec lacks. The only clustering in
the crate today is `codebook`'s one-dimensional Lloyd-Max scalar quantizer (a
density quantizer for the Beta marginal), which is a different problem from
Euclidean k-means over high-dimensional vectors. An inverted-file (IVF) coarse
quantizer needs the latter; pg_turbovec currently carries its own Lloyd in
`src/index/ivf.rs` and will switch to this module.

Public API:
  pub fn cluster_lloyd(sample, n, k, dim, max_iter, seed) -> ClusterResult
  pub fn cluster_emeans(sample, n, k, dim, epochs, seed) -> ClusterResult
  pub fn cluster_emeans_tuned(sample, n, k, dim, epochs, b, alpha0, seed)
  pub fn emeans_batch_size(n, k, t) -> usize
  pub fn emeans_alpha0(n, k, t, b) -> f32
  pub fn mse(sample, n, centroids, k, dim) -> f64

e-means is Bessa et al., "e-means: outperforming k-means in one-tenth the
epochs" (Meta FAIR, arXiv 2611.09501, 2026), reference code in
faiss/contrib/e_means (faiss is MIT; this is an independent Rust
implementation of the published algorithm, not copied code). Implemented
faithfully: EMA centroid statistics S<-(1-a)S+a*s, N<-(1-a)N+a*n, c=S/N;
cosine-annealed step a(b)=a0*(0.1+0.9*0.5*(1+cos(pi*b/U))); automatic batch
B=0.035*k^0.13*n^0.88*T^0.81 (/8 for T>2, nearest power of two) and peak step
a0=min(0.824*k^-0.46*B^0.90*n^-0.45*T^-0.11, 1) from the paper's fitted scaling
laws; dead-cluster revival every 5 batches; last-epoch-mean finalisation.

One deliberate deviation, documented in the module: the paper revives a dead
cluster by splitting a donor with a tiny [1+-eps] (BF16 eps) perturbation and
relying on later batches to pull the children apart. At IVF-coarse scale (large
k relative to the batch, few epochs) that drift does not complete within the
budget and left ~30% of a k=1024 codebook dead in our runs. We instead reseed a
dead cluster to the farthest well-separated point in the current batch (the
paper's own ablated "resample from batch" alternative, Table 14, sharpened to
pick the worst-covered point). That keeps essentially all clusters live (8 dead
of 1024 vs 300+). Open to matching upstream's exact revival if preferred.

## Determinism

Both solvers are bit-for-bit deterministic in the supplied seed: all randomness
from a seeded ChaCha8Rng, all f64 reductions over a fixed partition in a fixed
order independent of thread count. This matches the crate's existing
determinism discipline (rotation, pack) and is what lets a consumer persist
centroids and gate on byte equality (pg_turbovec's IVF relfile does).

## Measured (EC2 c7i, synthetic Gaussian mixtures at IVF-coarse scale)

| n      | k    | dim | e-means 4ep | Lloyd-25 | speedup | loss (em/lloyd) | dead |
|--------|------|-----|-------------|----------|---------|-----------------|------|
| 102400 | 256  | 32  | 1.2 s       | 6.5 s    | 5.3x    | 0.825           | 9    |
| 204800 | 1024 | 64  | 20.0 s      | 96.2 s   | 4.8x    | 1.048           | 8    |
| 102400 | 256  | 128 | 5.9 s       | 26.5 s   | 4.5x    | 0.691           | 0    |

loss < 1 means e-means reaches a lower MSE than Lloyd-25. Faster everywhere, at
equal or lower loss except a ~5% gap at k=1024 on this synthetic, which is an
unusually Lloyd-friendly case (k true blobs = k requested clusters). The paper
reports e-means winning outright on real embedding datasets (DEEP, SIFT, VIBE);
a real-data A/B is the natural next measurement before quoting a headline.

8 tests: determinism (same seed; across rayon pool sizes), scaling-law sanity,
IVF-scale quality vs Lloyd, no-dead-clusters on imbalanced data, edge cases.
clippy + fmt clean on the crate's bar.

## Upstream path (issue-first, per CONTRIBUTING)

turbovec takes an issue before a PR. Propose as: "a deterministic Euclidean
k-means module (Lloyd + e-means) for IVF/codebook training". It is additive
(new module, no change to existing APIs), MIT-compatible, and a capability the
crate does not have. If the maintainer prefers the paper's exact donor-split
revival, we adapt. Not yet filed.
