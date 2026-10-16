# Pooled benchmark harness — `benches/scripts/pooled/`

> ## ⚠️ READ FIRST: collapsed single-host runs are DIAGNOSTICS, never headlines
>
> `benches/METHODOLOGY.md` is canonical. The **headline** pg_turbovec
> benchmark is a **three-tier distributed rig**: PostgreSQL + pg_turbovec on
> one **bare-metal** on-demand instance, a connection pooler on its **own**
> on-demand instance, and **N spot** load-generator instances, over a
> **working set larger than RAM**, on **tuned storage** (ZFS + ZIL/SLOG +
> L2ARC on NVMe). Lead with: **sustained QPS + QPS/core**, **tail latency
> (p95/p99/p99.9) under the pooler**, **memory-per-backend × backend-count**.
> Single-client warm p50 is a **diagnostic row only**.
>
> **Any run with the DB, pooler, and driver on one host, or without a pooler,
> or with a RAM-resident corpus, is a DIAGNOSTIC — label it so.** The
> `run_smoke.sh` orchestrator here is exactly that: it proves the harness
> works and gives a provisional before-number for Phase S1. It is **not** a
> publishable result.

## What's here

| file | role | runs on |
|------|------|---------|
| `common.sh` | shared config; all tier endpoints + corpus/index knobs are env vars (nothing host-specific) | sourced by all |
| `provision.md` | how the real three-tier EC2 rig is launched (SG scoped to /32 + intra-rig, `run=<id>` tag, teardown-by-tag). Documentation; S0 does not launch it. | — |
| `setup_db.sh` | tune PG (shared_buffers, effective_cache_size, max_connections, wal_level, huge_pages note), create extension, load corpus, build flat/IVF index, EXPLAIN-verify index use | DB tier |
| `setup_pooler.sh` | install/configure PgBouncer, **transaction** pooling (`pool_mode`, `default_pool_size`, `max_client_conn`, `server_lifetime`), point at DB | pooler tier |
| `run_load.sh` | the driver: pgbench through the pooler, **closed-loop** (max QPS) **and open-loop** (`-R` fixed arrival, tail latency); captures QPS, QPS/core, p50/p95/p99/p99.9, backend CPU from `/proc`; emits per-driver JSON | each load node |
| `fresh_backend_cost.sh` | first-query cost on a freshly-recycled backend (our per-backend cache builds on first query); estimates how often a pooled workload pays it | DB tier (direct) |
| `aggregate.py` | folds per-driver JSONs into one QPS/latency series; emits the headline markdown table | anywhere |
| `run_smoke.sh` | **collapsed single-host DIAGNOSTIC** orchestrator; wires the above on one box for both `scan_parallelism=auto` and `=1` | one dev host |

## The build recipe that worked (Debian 12 / LLVM 14)

The S0 dev host is Debian 12, 8 vCPU (AVX-512 VNNI), rust stable 1.99,
cargo-pgrx 0.19.3, **LLVM/clang 14** (older than the project's usual). It
built v2.13.0 cleanly:

```bash
# No OpenBLAS needed since v2.0.0 (pure-Rust `gemm`); the only env is for
# bindgen to find Debian's libclang + clang resource headers.
. ~/.cargo/env
export LIBCLANG_PATH=/usr/lib/llvm-14/lib
export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/include -isystem /usr/lib/llvm-14/lib/clang/14.0.6/include"
# Debian default linker is GNU ld.bfd; no -fuse-ld override needed.

cd ~/pg_turbovec && git checkout v2.13.0
cargo build --release                      # ~minutes; turbovec fork handles LLVM<22
cargo pgrx init --pg18 download            # downloads + compiles PG 18.6 from source
cargo pgrx install --release --pg-config ~/.pgrx/18.6/pgrx-install/bin/pg_config
```

turbovec's `build.rs` detects LLVM < 22 and emits `vpdpbusd` via inline asm,
so the AVX-512 VNNI path builds under clang-14 with no extra flags.

> **SQL gotcha:** the opclass and distance operator are in the `turbovec`
> schema. Either `SET search_path = turbovec, public` (the harness sets
> `PGOPTIONS=-c search_path=turbovec,public`), or schema-qualify:
> `turbovec.vec_cosine_ops` and `OPERATOR(turbovec.<=>)`. `run_load.sh`
> schema-qualifies the operator so it survives a transaction-mode pooler
> that drops `SET`.

## Quick start — collapsed smoke (diagnostic)

```bash
# cluster must be up on DB_PORT with pg_turbovec installed; pgbouncer present.
DB_PORT=28818 \
CORPUS_ROWS=500000 DIM=768 INDEX_KIND=flat BIT_WIDTH=4 \
SHARED_BUFFERS=256MB CLIENTS="1 2 4 8 16 32 64" DURATION=20 \
bash run_smoke.sh
```

Results land under `benches/results/`: per-`scan_parallelism` markdown tables
(`smoke_<kind>_sp<auto|1>_<ts>.md`), the raw per-run JSONs, and a
fresh-backend JSON. `run_smoke.sh` sweeps **both** `scan_parallelism=auto`
(default, single-client latency) and `=1` (the pooled QPS/core setting) — the
equal-CPU pair METHODOLOGY.md requires.

## Running the real three-tier rig later

The scripts are tier-agnostic: point each at the right host via env and run
them where `provision.md` says. There is no code change between the smoke and
the headline run — only the topology (separate metal DB, isolated pooler, spot
fleet, >RAM corpus, tuned storage) and the `TOPOLOGY` label change. See
`provision.md` for the EC2 launch + AWS burner rules + teardown-by-tag.

## What the harness still needs for the real rig

- **Multi-driver clock sync + staggered start** across the spot fleet (chrony;
  `run_load.sh` already tags each JSON with `driver_id`, `aggregate.py` folds
  them — the open-loop percentile fold across drivers is currently a
  conservative max, exact needs the raw `-l` logs shipped and re-percentiled).
- **A real >RAM corpus loader** — `setup_db.sh`'s batched random generator is
  for the smoke; TB-scale needs a streaming loader (reuse
  `benches/scripts/load_dbpedia_1M.py`) and a recall ground-truth table.
- **Recall at the operating point** — the smoke only sanity-checks that the
  index returns k rows; the headline needs matched-recall Pareto operating
  points (discarded calibration pass) per the fairness rules.
- **ZFS/ZIL/L2ARC storage setup + huge-pages reservation** on the metal DB
  host (documented in `provision.md`/`setup_db.sh`, not scripted).
- **Memory-per-backend × backend-count** capture — add RSS sampling of DB
  backends at full pool size (the methodology's headline metric #4); the
  fresh-backend probe measures the per-backend build but not aggregate RSS.
