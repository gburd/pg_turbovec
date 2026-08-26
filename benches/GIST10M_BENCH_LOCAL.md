# GIST-1M + 10M competitive leg — pg_turbovec 2.0.0 (LOCAL working log)

Box: i-0393e90bc0c2c7b54 @ 18.217.136.19 (i4i.8xlarge, 32 vCPU AVX-512, 247 GiB, /mnt/nvme 3.4T).
Canonical checkpoint on box: /mnt/nvme/GIST10M_BENCH.md + /mnt/nvme/results/*.json

## Setup state (2026-08-26)
- OS Ubuntu 24.04. rustup 1.98.0, cargo-pgrx 0.19.1, PG18.6 (pgrx download).
  pg_config: /home/ubuntu/.pgrx/18.6/pgrx-install/bin/pg_config
- pg_turbovec branch feat/turbovec-1.0.0-port @ 25ab885; turbovec pinned f29a2f2 (fork). crate ver 1.29.6, wire target v8.
- GIST-960 HDF5 downloaded (1M x 960, 1000 test, 100-nn GT). SIFT already done (prior leg).
- Clones: pgvector v0.8.0, VectorChord, pgvectorscale on box.
- Harness copied to /mnt/nvme/src: bench_lib.py g0_driver.py tv_leg.py leg2_10m.py synth10m.py load_corpus.py leg3_driver.py (user patched ec2-user->ubuntu).

## Baseline to beat (v1.29.3, 2026-08-15, benches/results/competitive_20260815)
GIST-1M @ R~0.95: Qdrant 2.31ms/6852MB, HNSW 7.91ms/8051MB, vchord 16.34ms/4403MB,
  turbovec IVF capped R=0.9497@32.6ms (just under bar), flat R=0.991@78ms/502MB, IVFFlat 115ms, DiskANN never 0.95.
GIST-1M @ R~0.99: Qdrant 4.5ms, HNSW 20.8ms, turbovec flat 78ms/498MB, vchord 28.29ms, IVFFlat 223ms, DiskANN never.
GIST-10M build-in-budget(2h): turbovec IVF 846s(R up to .991), vchord 740s(.999), IVFFlat 4616s(unusable latency),
  HNSW DNF 7203s, DiskANN DNF 7201s. Storage: tv 4.7GB vs vchord 42.9GB (9x).
Verdict: turbovec = storage+build-at-scale winner; NOT latency leader at 960d.

## Plan
1. Start PG18, create vecbench DB, extensions.
2. Load GIST-1M dual-column corpus (gist_corpus: emb pgvector, embt turbovec).
3. GIST-1M sweeps: turbovec (flat L0 + IVF L1000/L4000, rr off/auto), HNSW m32/efc256,
   IVFFlat, vchord, diskann(if builds). Qdrant: 1 try, 5-min timeout, else DNF.
4. Synth 10M corpus + BLAS GT (synth10m.py) -> gist10m_corpus.
5. 10M leg: turbovec IVF + HNSW + vchord (2h budget each). IVFFlat if time.
6. Compare to baseline; write frontier tables.

## Progress
- [DONE] turbovec build+install, wire_version=8 CONFIRMED (smoke index).
- [DONE] PG18.6 running on port 28818, socket /mnt/nvme/pg, db vecbench. pgvector 0.8.1 (0.8.0 fails PG18 compile - API change; 0.8.1 = closest PG18-compatible, same HNSW/IVFFlat algo).
- [running] GIST-1M load (load_gist1m.log) ~17min ETA
- [running] synth10m corpus+GT (synth10m.log)
- [running] VectorChord build w/ its pinned cargo-pgrx 0.17.0 (build_vchord.log)
- [pending] pgvectorscale build (pgrx 0.16.1 pin)
- NOTE: pgvector bumped 0.8.0->0.8.1 for PG18 compat. Baseline used PG17+pgvector0.8.0. Report the delta.
