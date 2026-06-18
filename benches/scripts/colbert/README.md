# ColBERT Phase F-1 gate harness

Measures whether index-native ColBERT stage-1 (`turbovec.colbert_search`)
delivers a real recall gain over the Phase-D pooled-vector + `max_sim`
rerank baseline, on a real ColBERT corpus with ground-truth qrels. This
is the GATE for funding Phase F-2 (the persistent token index).
See an internal design note -> "F-1 benchmark (the gate for F-2)".

## Files

- `colbert_encoder.py`  Lightweight transformers-only ColBERTv2 encoder.
  Reproduces `colbert-ir/colbertv2.0` token embeddings (BERT 768-d ->
  `linear.weight` (128,768) -> L2-normalise) without the heavy colbert-ir
  framework. Query augmentation ([CLS][unused0]..pad-with-[MASK]..[SEP]),
  doc punctuation-skiplist + pad filtering ([CLS][unused1]..[SEP]).
- `embed_scifact.py`    Downloads BEIR/SciFact, embeds corpus + 300 test
  queries in batches, writes incremental `.npy` shards (concat tokens +
  offsets + pooled + ids) to `$OUTDIR`.
- `run_f1_sweep.py`     Loads shards into pg16 (schema `cb`), builds the
  pooled baseline index, runs both arms over the config sweep + the exact
  brute-force ceiling, writes the results JSON.

## Corpus

SciFact: 5183 docs (avg 157 tokens/doc, capped 180), 300 test queries
(32 tokens each via query augmentation), 339 qrel judgments.

## Memory discipline (floki: 30 GiB RAM, ZERO swap)

Two distinct OOM hazards, both handled:

1. **Embedding (torch).** Run under a hard cgroup cap so an overrun is
   contained, not cascaded into the postmaster:
   ```
   systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
       python embed_scifact.py
   ```
   (Fallback if no `systemd-run --user`: `ulimit -v 12000000` in a
   subshell.) CPU torch only, `torch.set_num_threads(4)`, batches of 32,
   `gc.collect()` between batches, incremental writes to disk.

2. **The sweep / `colbert_search` per-call leak.** `colbert_search`
   leaks ~28 MB of backend RSS per call (the backend-cached token-index
   workspace is not freed between calls in a session); left unchecked it
   climbed to 18 GiB and nearly OOM'd the postmaster. `run_f1_sweep.py`
   reconnects every `RECONNECT_EVERY=40` queries so each backend exits
   and reclaims, capping peak backend RSS at ~3.1 GiB. A memory watchdog
   aborts the sweep (writing partial results) if `MemAvailable` drops
   below 6 GiB. **NB this is a real bug in `colbert_search` worth fixing
   in the F-2 work; F-1 just works around it in the harness.**

## Run

```bash
source /tmp/colbert-venv/bin/activate   # torch-cpu, transformers, psycopg, datasets
export OUTDIR=/tmp/colbert-scifact
export DSN="host=/home/gburd/.pgrx port=28816 dbname=postgres"

# 1. embed (under the cap, heartbeat-wrapped, ~15 min on floki CPU)
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
    python embed_scifact.py

# 2. load into pg16 (~1 min)
python run_f1_sweep.py --load

# 3. ceiling + sweep -> JSON (~50 min; reconnects to stay memory-safe)
python run_f1_sweep.py --ceiling --sweep \
    --out ../../results/colbert_f1_gate_floki_scifact_<date>.json
```

The turbovec extension on this cluster is installed as a bare schema +
AM (no `pg_extension` row / control file), so the harness does NOT run
`CREATE EXTENSION` — the `turbovec.*` objects already exist.

## Reading the result

`verdict` in the JSON answers the gate's two questions explicitly:
(a) best recall operating point + quantization erosion, (b) the
recall/latency delta vs the Phase-D baseline, and a GO/NO-GO with the
honest single-corpus caveat.
