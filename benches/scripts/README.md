# Benchmark scripts

Scripts that produce the data committed under `benches/results/`.
They are not run by `cargo bench`; they're driven by hand or by CI.

## `prepare_glove_fixture.py`

Converts an [ann-benchmarks](http://ann-benchmarks.com/) GloVe HDF5
file into the binary fixture format documented in
`docs/RECALL.md` § 6.1.

```bash
mkdir -p ../../fixtures && cd ../../fixtures
curl -L -O http://ann-benchmarks.com/glove-100-angular.hdf5
nix-shell -p python3Packages.numpy python3Packages.h5py --run \
    "python3 ../benches/scripts/prepare_glove_fixture.py \
        glove-100-angular.hdf5 ./glove-100 100000 1000"
```

Produces under `fixtures/glove-100/`:

- `corpus.bin` — 100 000 × 100-dim FP32 unit-norm GloVe vectors.
- `queries.bin` — 1 000 × 100-dim FP32 query vectors.
- `ground_truth.bin` — top-100 exact cosine neighbours for each
  query, **recomputed** against the subset (the published
  `neighbors` index into the full 1.18 M-row training set and are
  not directly applicable).
- `fixture.json` — metadata.

Pick a different size by passing `corpus_n` and `queries_n` as
positional args 3 and 4.

## `run_recall_vs_pgvector.py`

Drives a Postgres cluster that has both `pg_turbovec` and pgvector
loaded. Builds three indexes on the same corpus table — pgvector
HNSW, pg_turbovec at `bit_width = 4`, pg_turbovec at
`bit_width = 2` — runs identical query workloads against each, and
records R@1 / R@10 / R@100 plus p50/p95/p99 latency at LIMIT=10
versus the exact ground truth.

```bash
nix-shell -p python3Packages.numpy python3Packages.psycopg2 --run "
    python3 benches/scripts/run_recall_vs_pgvector.py \
        fixtures/glove-100 \
        benches/results/recall_vs_pgvector_$(date -u +%Y_%m_%d).json \
        --pg-bin /home/gburd/.pgrx/install-pg16/bin \
        --pg-data /home/gburd/.pgrx/data-16 \
        --port 28816 --socket-dir /home/gburd/.pgrx \
        --ef-search 40,80,200
"
```

Notes:

- The driver always restarts the cluster with `TMPDIR=/tmp` so
  `pg_turbovec`'s persist layer (which writes serialised
  `IdMapIndex` files via `std::env::temp_dir()` before encoding
  them into the side-table BYTEA) doesn't inherit a now-defunct
  `nix-shell` TMPDIR.
- pgvector and pg_turbovec both register a type called `vector`
  in different schemas. The driver disambiguates with explicit
  `OPERATOR(public.<=>)` / `OPERATOR(turbovec.<=>)` casts.
- pg_turbovec requires `dim % 8 == 0`; the driver zero-pads the
  `turbovec.vector` column to the next multiple of 8 (104 for
  GloVe-100). Zero-padding is identity-preserving for cosine on
  unit-norm input, so recall is unaffected. pgvector sees the
  un-padded data.

## Installing pgvector into the pgrx cluster

The pgrx-managed cluster doesn't ship pgvector by default. On
Nix, the easiest path is:

```bash
PGV=$(nix-build '<nixpkgs>' -A postgresql16Packages.pgvector --no-out-link)
PGRX=/home/gburd/.pgrx/install-pg16
cp -f $PGV/lib/vector.so       $PGRX/lib/
cp -f $PGV/share/postgresql/extension/vector*  $PGRX/share/postgresql/extension/
```

PG 16's ABI is stable across point releases, so a pgvector built
against 16.14 loads cleanly into a 16.9 cluster.
