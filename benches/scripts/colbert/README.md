# ColBERT Phase F-1 gate harness

Measures whether index-native ColBERT stage-1 (`turbovec.colbert_search`)
delivers a real recall gain over the Phase-D pooled-vector + `max_sim`
rerank baseline, on a real ColBERT corpus with ground-truth qrels. This
is the GATE for funding Phase F-2 (the persistent token index).
 -> "F-1 benchmark (the gate for F-2)".

## Files

- `colbert_encoder.py`  Lightweight transformers-only ColBERTv2 encoder.
  (shared by both corpora; unchanged.)
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

### Phase F-2 confirmation (second corpus, NFCorpus)

- `embed_nfcorpus.py`   Like `embed_scifact.py` but for BeIR/nfcorpus
  (medical/nutrition, out-of-domain vs SciFact, entity-heavier). One
  substantive difference: NFCorpus `_id`s are STRINGS (`MED-2427`,
  `PLAIN-2`), so docs get synthetic int64 ids (written to `doc_id_map.json`)
  and queries keep their real string id (`queries_qids.json`); qrels.json
  is rewritten with synthetic int doc-ids.
- `run_f2_sweep.py`     Like `run_f1_sweep.py` but (a) schema `cb2`, (b)
  BUILDS THE PERSISTENT `vec_colbert_ops` index (`--colbert-lists N`,
  N~=sqrt(n_tokens)) so Arm A exercises the F-2 persistent read path, not
  the F-1 backend-cache rebuild; (c) adds a `persistent_read_probe` that
  issues 60 `colbert_search` calls on ONE backend with NO reconnect and
  tracks RSS, proving the persistent path does NOT rebuild per call (RSS
  plateaus flat vs F-1's ~28 MB/call climb); (d) reports the persistent
  index build time + on-disk size via `pg_relation_size`.

  Run (after `source /tmp/colbert-venv/bin/activate`,
  `export OUTDIR=/tmp/colbert-nfcorpus`):
  ```bash
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
      python embed_nfcorpus.py            # ~17 min on floki CPU
  python run_f2_sweep.py --load           # ~1 min
  python run_f2_sweep.py --ceiling --sweep --colbert-lists 749 \
      --out ../../results/colbert_f2_confirm_floki_nfcorpus_<date>.json
  ```
  Verdict (2026-06-18): the SciFact recall gain REPLICATES on NFCorpus
  (same sign, same low-candidate-budget shape, signal intact under 2-bit);
  the persistent F-2 index built cleanly, reached the exact ceiling, and
  eliminated the F-1 per-call leak. See the result JSON's `verdict` block.

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
