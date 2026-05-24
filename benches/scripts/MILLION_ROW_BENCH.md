# Million-row recall + latency bench scripts

The three files below produce
[`benches/results/recall_lat_million_2026_05_24.json`](../results/recall_lat_million_2026_05_24.json)
on a host that already has:

* a running PG 17 cluster on port 28815, socket dir `/scratch/pg_turbovec-bench`,
* `pgvector 0.8.0` and `pg_turbovec 1.0.0-rc.2` installed in that cluster,
* a database called `bench` with the empty 3-row schema:

  ```sql
  CREATE TABLE docs      (id bigint PRIMARY KEY, emb vector(384));
  CREATE TABLE query_set (qid int PRIMARY KEY, doc_id bigint, emb vector);
  CREATE TABLE gt_top10  (qid int, hit_id bigint, rk int);
  ```

Set `LD_LIBRARY_PATH=/lib64` and `$HOME/.pgrx/17.9/pgrx-install/bin`
on PATH first.

## 1. `rebuild_corpus_million.sh`

Drops any existing turbovec / hnsw indexes, TRUNCATEs `docs`, and
inserts 1 M random unit-norm 384-d vectors using a `VOLATILE`
`rand_vec(d)` SQL function (the obvious correlated-subquery form
that `random()` takes will get constant-folded into a single
vector by the planner; the `VOLATILE` function form is the
workaround). Then refreshes `query_set.emb` from `docs`,
recomputes `gt_top10` by exact brute-force seq scan, and rebuilds
all three indexes (HNSW, turbovec 4-bit, turbovec 2-bit).

Wall time on `arnold` (i9-12900H): ~14 minutes total.

## 2. `bench_million_setup.sql`

Defines the `bench_runs` results table and the helper functions:

* `bench_one_query_pgv(qid)` — one timed pgvector query
* `bench_one_query_tv(qid)`  — one timed pg_turbovec query
* `bench_run_config(label, engine)` — sweep all 50 qids
* `bench_recall_at_10(label)` — R@10 vs `gt_top10`
* `bench_summary(label)` — min / p50 / p95 / max / mean / R@10

Run once after `rebuild_corpus_million.sh`.

## 3. `run_bench_sweep_million.sh`

Runs the full latency + recall sweep in 6 phases:

| Phase | Config                          |
|------:|---------------------------------|
| A     | HNSW ef_search=40               |
| B     | HNSW ef_search=200              |
| C     | turbovec 2-bit, search_k=100    |
| D     | turbovec 2-bit, search_k=200    |
| E     | turbovec 4-bit, search_k=100    |
| F     | turbovec 4-bit, search_k=200    |

Phases C/D run with both turbovec indexes present (the planner
picks the smaller 2-bit one). E/F drop the 2-bit index so the
planner falls back to 4-bit, then the 2-bit is rebuilt at the end.

Wall time on `arnold`: ~22 minutes.

The summary at the end of the run is the basis for the table in
[`docs/RECALL.md` § 2.1.0](../../docs/RECALL.md).
