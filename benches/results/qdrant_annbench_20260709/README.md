# Qdrant + ANN-Benchmarks head-to-head — raw results (2026-07-09)

a benchmark host, PG17.5-from-source, pg_turbovec v1.25.0 (commit
`2ef2388`), pgvector 0.8.0, Qdrant 1.18.0. See
`an internal benchmark note` for the full write-up.

## Layout

- `raw/` — per-config JSON (engine, recall@10, p50, qps_1conn,
  qps_8conn, build_s, idx_bytes) for each engine × dataset.
  - `*_sift1m.json`, `*_gist1m.json` — Leg 1 (1M), in-RAM.
  - `*_gist10m.json` — Leg 2 (10M semi-synthetic); `turbovec`/`hnsw`
    entries record `build_FAILED` (90-min budget cap); `qdrant`
    completed.
  - `turbovec_*_OOC*.json` — the discarded out-of-core Leg-1 pass
    (preserved for the record; NOT used in the write-up).
- `scripts/` — harness (see write-up § Reproducibility).
- `logs/` — heartbeat-wrapped run logs for every stage.

## Headline

- `turbovec.hi_dim_rerank=auto` (v1.25.0) lifts GIST-960-1M R@10 from
  0.876 (`off` ceiling) to 0.953 — crosses R@10≥0.90 and ≥0.95.
- Qdrant wins raw latency at every band; pg_turbovec wins storage
  (SIFT 7–8×, GIST 5.4× smaller @ 1M).
- @ 10M×960 only Qdrant built in budget (32 min); turbovec IVF and
  pgvector HNSW both exceeded the 90-min cap.
