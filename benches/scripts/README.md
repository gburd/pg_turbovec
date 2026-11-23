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

## `bq/` — 1-bit sign-BQ frontier

`bq/bq_frontier.py` sweeps `bit_width` in {1, 2, 4} over one corpus and
reports R@k vs exact in-DB ground truth, `pg_relation_size`, bytes/vector,
build wall-clock and — **only on an AVX2+ host** — warm p50/p95/p99 + QPS.
`bq/run_bq_frontier.sh` is the phase runner (`preflight` / `queryset` /
`sweep`); `bq/smoke_stub.sql` is a plumbing-only fixture that measures
nothing.

The latency gate is structural, not a convention: on a pre-AVX2 host
(`meh`) turbovec takes its scalar fallback (~1000× slower full-corpus scan)
and the driver **will not emit a latency number at all**. Read
[`docs/BQ_RECALL_BENCH.md`](../../docs/BQ_RECALL_BENCH.md) before running it
— it has the runbook, the host rules, the re-rank-window controls, and the
predictions the run is meant to confirm or falsify.

No-database self-checks:

```bash
python3 benches/scripts/bq/bq_frontier.py --self-check     # pure logic
python3 benches/scripts/bq/bq_frontier.py --print-schema   # output shape
python3 benches/scripts/bq/bq_frontier.py --dry-run-sql \
    --dsn x --out /tmp/x.json --dim 1536 --query-provenance held_out
```

## `prep_hn.py` — HackerNews (HN768) vector corpus

Adds HackerNews to the test corpora.  Unlike the standing corpora (CO1024,
OA1536, CW768), which are *pre-embedded* text datasets that ship an `emb`
column, HackerNews is published as raw text, so `prep_hn.py` embeds it: it
streams HN story/comment text from a pinned HF dataset revision
(`julien040/hacker-news-posts`), embeds each doc with a pinned
sentence-transformers model (default `BAAI/bge-base-en-v1.5`, 768-d, chosen to
match the 768-d CW768 corpus so HN is directly comparable), L2-normalises to
fp32, and emits the **same on-disk shape** the bench loaders consume:
`corpus.npy`, `queries.npy`, exact fp64 `gt_ids.npy`/`gt_dist.npy`, 16-way
`%.9g` text COPY files, and a pinned `datasets.lock.json` (text revision +
model + md5s, so the corpus is byte-reproducible).

```bash
# 1M regression corpus (RAM-resident; a laptop/CI-sized smoke of the HN shape)
python3 benches/scripts/prep_hn.py 1000000 5000 /scratch/pg_turbovec-bench/hn768_1m

# 150M field-scale corpus (the scale the pg_fts field report used; needs a
# GPU/large host and hours -- run on EC2, never the workstation)
python3 benches/scripts/prep_hn.py 150000000 5000 /scratch/pg_turbovec-bench/hn768_150m
```

The output directory is a drop-in for the vs-bench loader
(`load_ds.sh <dir> 768 hn768`) and the single-client harness
(`bench.py --data <dir>`).  Deps: `sentence-transformers`, `torch`,
`numpy`, `pyarrow`, `huggingface_hub`.

| id | dim | source text | embedding model | scales | why |
|----|-----|-------------|-----------------|--------|-----|
| **HN768** | 768 | `julien040/hacker-news-posts` (HF, pinned) | `BAAI/bge-base-en-v1.5` (pinned) | 1M, **150M** | the field corpus; short real-world docs; 768-d matches CW768 for cross-corpus comparison |

`prep_hn.py` embeds locally (no API key, no per-call cost); the vectors are
ours, so unlike the vendor-embedded corpora HN may be regenerated and, if
wanted, redistributed.  Changing the text revision or the model is a **new
dataset**, not a silent drift -- both are pinned in `datasets.lock.json`.

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
