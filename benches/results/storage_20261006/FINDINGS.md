# Column storage for `turbovec.vector`: what TOAST costs, what inline storage costs (2026-10-06)

**Question.** The recheck attribution (`../recheck_20261006/FINDINGS.md`)
found ~5 µs of each candidate's ~18 µs going to the TOAST fetch of the
candidate's vector. Users can remove that with
`ALTER TABLE … ALTER COLUMN … SET STORAGE MAIN` (or `PLAIN`). How much does it
save, and what does it cost elsewhere? Where are the limits? This supports
the user guidance in `docs/PRODUCTION.md` § "Column storage for vector
columns". Nothing here changes code or the type's default storage.

**Answer.**

- At 1024-d, `MAIN` and `PLAIN` save **~4.9 µs per rechecked candidate**
  (16.5 → 11.4 ms at `search_k = 1024`, −31%). `EXTERNAL` saves nothing over
  the default `EXTENDED`. It also saves ~3.8 µs/candidate at 768-d and
  ~5.3 µs at 1536-d, and **nothing at 384-d**, because a 384-d vector is
  already stored inline.
- The cost is the heap. It goes from 6 MB to 781 MB for 100k × 1024-d, so a
  cold sequential scan that never reads the vector goes from **38 ms to
  4,312 ms** (113×). Updating any other column of a row with an inline vector
  writes a whole new copy of the vector to WAL: **34× the WAL** in the test
  below.
- Inline storage stops at **1,623-d** for this table shape. Above that `MAIN`
  silently falls back to TOAST, and `PLAIN` raises `row is too big`.
- `SET STORAGE` on a populated table moves nothing. Existing rows move only
  when they are rewritten (`VACUUM FULL`, `pg_repack`, or an `UPDATE` that
  produces a new datum).
- Under `MAIN`, every other toastable value in the row longer than ~24 bytes
  (a 40-byte title, a 300-byte body) goes to TOAST instead (§ 4.1).
- Leaving the column alone and setting the table's `toast_tuple_target =
  8160` gave the same latency as `MAIN` (11.07 vs 11.02 ms, same run) and
  kept those text columns inline (§ 2.2, § 4.1).

## Setup

- EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, Sapphire Rapids, 16 vCPU, AVX-512
  VBMI+VNNI, so turbovec 1.1's staged search engages; 30 GiB), Debian 13, gp3 EBS.
- PostgreSQL 16.15 built `-O2 -g -fno-omit-frame-pointer`, **no cassert**
  (`/work/pg16rel` copied to `/work/pg16rel_c`, so no other agent's install
  could replace the library mid-run). pg_turbovec at `3ff7768`
  (`perf/recheck-abc` = v2.11.0 + docs), `cargo pgrx install --release`,
  `pg_turbovec.so` md5 `a3019fd05887629c7bb83fdc63352ad2` (`raw_env.txt`).
- The review follow-up runs (2026-10-07: `inline_limit.sql` text cases,
  `dump_trap.sh`, `ivf_degrade.sql`, `ttt_props.sh`, `run_ttt.sh`) used the
  same build (`pg_turbovec.so` md5 unchanged) and cluster. The box's other
  benchmark (`/work/ab`) had been idle for ~5 h, no `cargo pgrx test` was
  running, and loadavg was 0.20 when `run_ttt.sh` started (1.18 at the end,
  from the run itself; `raw_run_ttt.txt`), so those timings were not
  contended.
- Cluster (`up.sh`): `shared_buffers = 8GB`, `jit = off`,
  `max_parallel_workers_per_gather = 0`, `turbovec.cache_size_mb = 2048`,
  `default_toast_compression = pglz` (the build has no lz4). Own port and
  data dir.
- Data (`gen.py`): unit-norm Gaussian vectors, seed `20261006 + dim`, 100k rows
  per table, plus 200 held-out queries. Synthetic data is fine here because
  storage behaviour depends only on value size and compressibility, and
  float32 noise does not compress, the same as real embeddings
  (`pg_column_compression` was NULL on every row of the four 1024-d tables;
  the floki run found the same on real Cohere vectors). Table shape:
  `(id bigint PRIMARY KEY, tv turbovec.vector)`. `SET STORAGE` was applied
  **before** `COPY` (`load.py`), then `VACUUM ANALYZE`. Index: flat 4-bit
  `vec_cosine_ops`.
- The vector datum is serde-CBOR: **13 + 5·d bytes** for full-precision
  float32 values (5,133 B at 1024-d, against 4,096 B of raw f32). CBOR
  stores an element in 3 bytes when it is exactly representable as f16, so
  the size depends on the values: a 1024-d vector rounded through
  `turbovec.halfvec` is **3,085 B** (13 + 3·d; last query in
  `raw_inline_limit.txt`), and every dimension limit in § 4 moves up by
  about 5/3 for such data (computed, not measured). How many elements of a
  real embedding are f16-exact was not measured on any corpus; check yours
  with `SELECT avg(pg_column_size(emb)) FROM docs;`.

## 1. Sizes (100k rows; `raw_sizes.txt`, `raw_load_*.txt`)

| dim | storage | main heap | TOAST | heap + TOAST | turbovec index |
|---:|---|---:|---:|---:|---:|
| 384 | EXTENDED (default) | 195.5 MB | 0 | 195.6 MB | 21.2 MB |
| 384 | MAIN | 195.5 MB | 0 | 195.6 MB | 21.2 MB |
| 768 | EXTENDED | 6.0 MB | 390.6 MB | 401.1 MB | 39.2 MB |
| 768 | MAIN | 391.0 MB | 0 | 391.1 MB | 39.2 MB |
| 1024 | EXTENDED | 5.8 MB | 558.0 MB | 570.5 MB | 55.2 MB |
| 1024 | EXTERNAL | 6.0 MB | 558.0 MB | 570.7 MB | 55.2 MB |
| 1024 | MAIN | 781.3 MB | 0 | 781.5 MB | 55.2 MB |
| 1024 | PLAIN | 781.3 MB | 0 | 781.5 MB | 55.2 MB |
| 1024 | EXTENDED + `toast_tuple_target = 8160` | 781.6 MB | 0 | 781.9 MB | 55.2 MB |
| 1536 | EXTENDED | 6.0 MB | 781.3 MB | 796.2 MB | 81.2 MB |
| 1536 | MAIN | 781.5 MB | 0 | 781.8 MB | 81.2 MB |

- Sizes are from `raw_sizes.txt`, taken at the end of the session (the
  `toast_tuple_target` row from `raw_sizes_ttt.txt`, loaded from the same
  100k vectors on 2026-10-07; `raw_load_ttt.txt`). A later
  `VACUUM` truncated a few empty tail pages, so some heap figures are ~0.3 MB
  below the right-after-load values quoted elsewhere (6.0 / 781.6 MB).
- With the default storage every vector ≥ 398-d lives in TOAST and the main
  heap is ~60 bytes/row. Inline, it is one row per 8 KB page at 1024-d and
  1536-d (5.1 KB and 7.7 KB tuples), and two per page at 768-d.
- **Inline is 37% bigger in total at 1024-d** (782 vs 571 MB). That is not
  TOAST overhead: a 5.1 KB tuple wastes the remaining ~3 KB of its page.
  TOAST packs chunks densely (3 chunks per value, 4 chunks per page).
  At 768-d and 1536-d the two layouts come out about the same size.
- The turbovec index is identical in every case. It stores 4-bit codes and
  never reads the column's storage setting.

## 2. Warm kNN latency (`bench.py`; `raw_warm_*.json`, `raw_summary.txt`)

Query: `SELECT id FROM t ORDER BY tv <=> $q LIMIT 10` with
`turbovec.search_k ∈ {32, 256, 1024}`, `oversample = 1.0`,
`hi_dim_rerank = off`, flat 4-bit index. Metric: `EXPLAIN (ANALYZE, TIMING
OFF)` Execution Time, warm (one backend per variant, every query run once
first). 200 queries × 5 rounds at 1024-d (3 rounds at the other dims), with
the variant order rotated every round. The 1024-d set was run three times
(`run1`, `default`, `pinned`; see § 2.1). The table pools all 3,000 queries
per cell.

**1024-d, median ms/query (n = 3,000 each):**

| `search_k` | EXTENDED (default) | EXTERNAL | MAIN | PLAIN |
|---:|---:|---:|---:|---:|
| 32 | 1.23 | 1.20 | 1.00 | 1.01 |
| 256 | 5.01 | 4.99 | 3.63 | 3.62 |
| 1024 | **16.48** | 16.56 | **11.44** | 11.45 |

Per-candidate effect, derived two ways for each of the three runs:

| vs EXTENDED | (k=1024 delta) / 1024 | slope k=256→1024, minus EXTENDED's slope |
|---|---|---|
| EXTERNAL | −0.13, +0.01, +0.45 µs | −0.15, +0.10, +0.58 µs (noise) |
| MAIN | **−4.79, −4.90, −5.08 µs** | −4.69, −4.64, −4.95 µs |
| PLAIN | −4.72, −5.11, −5.05 µs | −4.54, −4.87, −4.96 µs |

The per-candidate total (slope from k=256 to k=1024) is ~15.0 µs with TOAST
and ~10.2 µs inline. The 2026-10-06 floki attribution (different host, real
embeddings, `perf`) put the TOAST fetch at 5.0 µs and measured 18.0 → 13.7
µs/candidate. This end-to-end A/B agrees on the ~5 µs. All four variants
returned the same top-10 for 50/50 queries, as expected, since storage
does not change values.

**By dimension, EXTENDED → MAIN at `search_k = 1024`** (malloc-pinned run, § 2.1):

| dim | EXTENDED | MAIN | delta | per candidate |
|---:|---:|---:|---:|---:|
| 384 | 6.08 ms | 6.03 ms | −0.05 ms | ~0 (already inline) |
| 768 | 13.11 ms | 9.24 ms | −3.88 ms | −3.8 µs |
| 1024 | 16.55 ms | 11.35 ms | −5.20 ms | −5.1 µs |
| 1536 | 20.78 ms | 15.33 ms | −5.45 ms | −5.3 µs |

With the defaults left alone (`search_k = 32`, `hi_dim_rerank = auto`), a
1024-d index still rechecks 1,024 candidates: `auto` raises the window to
`min(dim, 1024)` for dim ≥ 256. Measured: EXTENDED 16.47 ms, MAIN 11.35 ms
(`raw_bench_1024_defaults.json`). So the saving applies to an untuned
install, not only to a hand-tuned large `search_k`.

### 2.1 A measurement trap: glibc heap trimming (`raw_faults*.txt`, `raw_wall_glibc.txt`, `raw_strace_alloc.txt`)

Under default glibc malloc, a warm backend sometimes ends up in a state where
each `search_k = 1024` query takes **~2,000 minor page faults** and **2–3 ms
longer** (1024-d MAIN: 14.6 ms in that state, 11.4 ms without). strace shows
~7 `brk` calls per query: glibc trims the top of its heap at the end of the
query and faults it back in on the next. Whether a backend enters the state
depends on its allocation history, so it is bimodal per backend: in one run
the k=1024-only MAIN backend hit it every time, while the backend that also
ran k=32/256 never did. It showed up with inline storage at 1024/1536-d
(~2,000 faults/query), and with both storage modes at 384-d (~515
faults/query). Starting the postmaster with
`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=67108864:glibc.malloc.trim_threshold=268435456`
removed it (0 faults, deterministic). `bench.py` therefore records faults per
query, and the "pinned" run uses those tunables. In the default-malloc run
three cells were affected: 1536-d MAIN at k=1024 (18.23 ms, vs 15.33 pinned)
and both 384-d k=1024 cells (+0.8 ms each, so the 384-d delta is unaffected).
Every other cell agrees with the pinned run within 0.4 ms. **This is not a storage effect. It is a hazard for
any latency A/B on this extension** (flagged for the step-5 A/B in
`../perf_abc_20261006/`). Setting only
`trim_threshold` made it worse (`raw_wall_glibc.txt`), so that is not the fix.

### 2.2 `toast_tuple_target = 8160` instead of `MAIN` (`run_ttt.sh`; `raw_warm_ttt_1024.json`, `raw_run_ttt.txt`)

Same `bench.py` (200 queries × 5 rotated rounds, n = 1,000 per cell; the
postmaster's malloc settings were not recorded, but every cell had 0 minor
faults per query, so the § 2.1 trim state did not occur), with a third table `t1024_ttt`: the column left at the default
`EXTENDED`, the table set to `toast_tuple_target = 8160` before the `COPY`
(`load.py ... ttt`). Same 100k vectors as `t1024_ext` (0 differing,
`raw_sizes_ttt.txt`).

| `search_k` | EXTENDED | MAIN | EXTENDED + `toast_tuple_target = 8160` |
|---:|---:|---:|---:|
| 32 | 1.23 | 0.94 | 0.96 |
| 256 | 4.97 | 3.47 | 3.49 |
| 1024 | 16.19 | 11.02 | 11.07 |

Per candidate against EXTENDED: MAIN −5.04 µs ((k=1024 delta) / 1024) and
−4.78 µs (slope), `toast_tuple_target` −4.99 and −4.74 µs. The vector sits
inline either way, so the two are the same within noise. 0 minor faults per
query in every cell. The absolute numbers are a little below the earlier
runs (16.48 / 11.44 ms); compare within a run.

## 3. Cold cost (`cold.py`; `raw_cold_1024.json`)

Before **every** measurement: `pg_ctl stop -m fast`, `sync; echo 3 >
/proc/sys/vm/drop_caches`, `pg_ctl start`, a fresh backend. 3 reps,
variants rotated, 1024-d, 100k rows.

| query | EXTENDED | EXTERNAL | MAIN | PLAIN |
|---|---:|---:|---:|---:|
| `SELECT count(*), sum(id)` (Seq Scan), cold | **38 ms** (736 blocks) | 35 ms (768) | **4,312 ms** (100,000) | 4,377 ms (100,000) |
| same query, warm (second run) | 5.0 ms | 4.9 ms | 46 ms | 45 ms |
| first kNN of a fresh backend, `search_k = 1024`, cold | 2,628 ms (9,568 blocks) | 2,467 ms | 1,445 ms (7,838) | 1,395 ms |
| same query, warm | 16.3 ms | 17.1 ms | 13.7 ms | 14.0 ms |

- A scan that never touches the vector reads **136× the blocks** with inline
  storage and is **113× slower cold**, **9× slower warm**. This is the
  downside: every heap page holds one row instead of ~130. The same goes for
  anything else that walks the heap, e.g. a filtered `WHERE` without a
  supporting index, `VACUUM`, or a `count(*)` that is not index-only.
- The cold first kNN is **faster** inline (1.4 s vs 2.6 s). Each of the 1,024
  candidates costs one heap page read, instead of a heap page plus a TOAST
  index probe plus TOAST chunk pages: 1,730 fewer blocks and ~1.2 s less I/O
  wait on this gp3 volume. Both numbers include loading the 55 MB index into
  the backend cache, which is the same for every variant.
- `superseded_raw_cold_1024_no_vm.json`: the first attempt let the planner
  pick an Index Only Scan with an unset visibility map (94k heap fetches),
  which measured a different thing. It was re-run with the scan forced to
  Seq Scan after a `VACUUM` (`raw_cold_1024.json`, log
  `raw_cold_1024.txt`). The superseded file is kept for the record.

## 4. Where inline storage stops fitting (`inline_limit.sql`; `raw_inline_limit.txt`)

Single-row inserts of `(random()*2-1)::real` vectors (full mantissa, so 5
bytes/element), 8 KB pages, table `(id bigint, tv turbovec.vector)`:

| storage | stays inline up to | first dim that doesn't | what happens then |
|---|---:|---:|---|
| EXTENDED (default), EXTERNAL | **397-d** (1,998-byte datum) | 398-d | moved to TOAST (2 chunks) |
| MAIN | **1,623-d** (8,128-byte datum) | 1,624-d | **silently** moved to TOAST (5 chunks) |
| PLAIN | **1,623-d** | 1,624-d | `ERROR: row is too big: size 8168, maximum size 8160` |

- The boundary can move by ±1 dim from row to row. An element that happens
  to be exactly representable in f16 is encoded in 3 bytes instead of 5. In
  the committed run (re-run 2026-10-07 with `setseed`), MAIN at 1604-d with
  12 extra bigints came out 8,031 bytes instead of 8,033 and fit inline (an
  8,159-byte tuple), one dim past the 1,603-d limit; the first run of the
  same script had put it in TOAST. An earlier probe run with a different
  value generator fit 1,624-d under PLAIN. (`vec_bytes` of a TOASTed value
  is 4 bytes below 13 + 5·d because `pg_column_size` reports the external
  size without the varlena header.)
- The EXTENDED/EXTERNAL limit is the ~2 KB `TOAST_TUPLE_TARGET` (2,032-byte
  tuple). The MAIN/PLAIN limit is the 8,160-byte maximum heap tuple. Both
  depend on the row's other columns. With 12 extra `bigint NOT NULL` columns
  (96 bytes) the limits measured **378-d** and **1,603-d**: 96 bytes of
  other data cost 19–20 dims at 5 bytes per dim, as predicted.
- The limits above are for fixed-width neighbours. Text neighbours behave
  differently (§ 4.1).
- The turbovec index needs dim to be a multiple of 8, so the largest
  indexable dim that stays inline is **1,616**. 1536-d (OpenAI
  `text-embedding-3-small` / `ada-002`) fits. 3072-d
  (`text-embedding-3-large`) fits under neither mode: it stays in TOAST with
  `MAIN` and errors with `PLAIN`.
- So `MAIN` is the safe choice: past the limit it degrades to today's
  behaviour. `PLAIN` turns the same situation into failed INSERTs.
- These limits are for the current serde-CBOR encoding (5 bytes/element). A
  smaller encoding would move them.

### 4.1 Text columns next to the vector (`inline_limit.sql`, second half; `raw_inline_limit.txt`)

Table `(id bigint, title text, body text, tv turbovec.vector)`, one row,
random hex text (incompressible). Placement read from the raw heap tuple
with `heap_page_item_attrs`: an 18-byte value starting with byte `0x01` is a
TOAST pointer.

| vector storage | `toast_tuple_target` | dim | 40-byte title | 300-byte body | vector |
|---|---:|---:|---|---|---|
| EXTENDED | default | 384, 1024, 1536, 3072 | inline | inline | TOAST |
| MAIN | default | 384 | inline | **TOAST** | inline |
| MAIN | default | 1024, 1536 | **TOAST** | **TOAST** | inline |
| MAIN | default | 3072 | TOAST | TOAST | TOAST |
| EXTENDED | 8160 | 384, 1024, 1536 | inline | inline | inline |
| EXTENDED | 8160 | 3072 | inline | inline | TOAST |
| MAIN | 8160 | 1024 | inline | inline | inline |

- Under `MAIN`, PostgreSQL's first two TOAST rounds
  (`heap_toast_insert_or_update`) move every non-MAIN toastable value larger
  than ~24 bytes out of line (after a compression attempt), largest first,
  while the tuple is over `TOAST_TUPLE_TARGET` (2,032 bytes), and only then
  consider MAIN values. An inline 400+-dim vector keeps the tuple over
  2,032 bytes, so its text neighbours are always pushed to TOAST. At 384-d
  the tuple (2,313 bytes with both texts) dropped under the target once the
  body was gone, so the title stayed.
- Under the default storage the neighbours stay inline because the vector is
  the value that gets moved.
- The default-storage lower bound depends on the row too: a 384-d vector next
  to a 200-byte text column went to TOAST.
- With `toast_tuple_target = 8160` the target is the whole page, so nothing
  moves while the tuple fits. When it doesn't, the largest EXTENDED value
  goes first, which is usually the vector. 1024-d + a 3,500-byte body:
  EXTENDED + 8160 put the **vector** in TOAST and kept the body; MAIN +
  default put both texts in TOAST; MAIN + 8160 moved only the body.

## 5. Changing storage on an existing table (`alter_after.sql`, `rewrite_paths.sql`, `inplace*.sql`, `repack.sh`)

100k × 1024-d loaded with the default storage (`raw_alter_after.txt`):

| step | attstorage | heap | TOAST | rows in TOAST |
|---|---|---:|---:|---:|
| loaded | x | 6.0 MB | 558.0 MB | 100,000 |
| `ALTER … SET STORAGE MAIN` (0.5 ms, catalog only) | m | 6.0 MB | 558.0 MB | 100,000 |
| `UPDATE … SET id = id` and `SET tv = tv` (1,000 rows each), `VACUUM` | m | 6.0 MB | 558.0 MB | 100,000 |
| `UPDATE … SET tv = l2_normalize(tv)` (1,000 rows), `VACUUM` | m | 13.6 MB | 558.0 MB | 99,000 |
| `VACUUM FULL` (4.8 s) | m | 781.3 MB | 0 | 0 |

- `SET STORAGE` only changes `pg_attribute.attstorage`. **Existing rows stay
  where they are.** An `UPDATE` that leaves the vector unchanged keeps the
  old TOAST pointer, and that includes `SET tv = tv`. Only a new datum is
  stored inline. `tv = tv::real[]::turbovec.vector` is a value-preserving way
  to make one (`raw_rewrite_paths.txt` shows the values are identical).
- `VACUUM FULL` rewrites the table under the new setting and rebuilds the
  turbovec index from the heap. `turbovec_check` was clean afterwards and kNN
  returned results. **`pg_repack` 1.5.3** (`raw_repack.txt`) did the same
  rewrite with only brief exclusive locks. Both a flat and an IVF index came
  back with all 20k entries and the IVF index not degraded. That was an idle
  table: writes made during a repack are replayed into the new table as
  ordinary inserts, which is the append path that degrades an IVF index past
  `ivf_max_delta_pct`, so on a busy table check `index_is_degraded()`
  afterwards (not measured). The new inline copy is ~1.4× the old heap +
  TOAST at 1024-d (781.5 vs 570.5 MB, § 1), plus its indexes.
- `CREATE TABLE … AS SELECT` gives the new column the **type default**
  (EXTENDED), not the source column's MAIN. `CREATE TABLE … (LIKE src
  INCLUDING ALL)` (or `INCLUDING STORAGE`) copies MAIN, and `INSERT … SELECT`
  into it stores the values inline (`raw_rewrite_paths.txt`).
- But `LIKE … INCLUDING ALL` also copies an IVF index definition onto the
  empty table, and **an IVF index built on an empty table never gets cells
  and is not reported as degraded** (`raw_ivf_degrade.txt`): after
  `CREATE INDEX … WITH (lists = 16)` on an empty table and a 3,000-row COPY,
  `index_degradation` reported `lists = 0`, `degraded = f`,
  `scan_fraction = 1.0`, and a kNN scan raised no WARNING. The `LIKE` copy
  of a `lists = 64` index loaded with 20,000 rows read the same. A `TRUNCATE`
  does the same to a healthy IVF index (`lists` 16 → 0 at the `TRUNCATE`,
  still 0 after reloading 3,000 rows).
  `REINDEX` after the load restored the cells. This is a product gap
  (silent loss of trained structure), not yet filed as an issue; it is not caused
  by column storage. Copy with `EXCLUDING INDEXES` and create the indexes
  after loading.
- **In-place batched UPDATE is a poor migration path** (`raw_inplace.txt`,
  `raw_inplace_vacuum.txt`; 20k rows, `SET STORAGE MAIN` then `UPDATE … SET
  tv = tv::real[]::turbovec.vector` in batches with `VACUUM` between):
  - every updated row adds a turbovec index entry. On an IVF index, **the
    25% first batch took it past `turbovec.ivf_max_delta_pct` (10%) and
    degraded it** to a flat scan (`index_is_degraded = t`) until `REINDEX`.
    Smaller batches don't avoid this, they only postpone it
    (`ivf_degrade.sql`, `raw_ivf_degrade.txt`, 20k rows, `lists = 64`):
    5% and 10% cumulative left it healthy, 15% degraded it. A full
    migration rewrites every row, so it always crosses the bound;
  - after the update the heap is 30× larger, so the dead rows sit on < 2% of
    its pages and PostgreSQL's VACUUM *index bypass* skips index vacuuming
    (`ambulkdelete`; `amvacuumcleanup` still runs). The
    flat index still held **40,000 entries for 20,000 rows** after the final
    `VACUUM`. `VACUUM (INDEX_CLEANUP ON)` removed them. Dead entries are
    filtered at the heap, so results are correct, but each one scores the
    same as its live successor, so they compete for `search_k` slots until
    cleaned (inference, not measured);
  - the end state matched a rewrite (157 MB heap, 0 TOAST), but it took the
    WAL of rewriting every vector plus the steps above. Use `pg_repack` or
    `VACUUM FULL` instead.
- `pg_dump` writes `ALTER TABLE ONLY … SET STORAGE MAIN` only for a column
  whose storage differs from its type's **current** storage
  (`dump_trap.sh`, `raw_dump_trap.txt`, two dumps of one database):
  - with the type at its default (`x`), a column set to MAIN by `ALTER TABLE`
    is dumped with its `SET STORAGE MAIN` line, and restoring that dump into
    a fresh database gives `m`;
  - after `ALTER TYPE turbovec.vector SET (STORAGE = main)`, neither that
    column nor a column created afterwards gets a `SET STORAGE` line, and
    restoring into a fresh database (fresh `CREATE EXTENSION`, so the type
    is `x` again) gives `x` for both.

  So changing the type also drops the per-column settings from every later
  dump. Set storage per column and leave the type alone.
- Lock: `ALTER TABLE … SET STORAGE` takes `ACCESS EXCLUSIVE`
  (`tablecmds.c`, "may add toast tables"; checked in the PG 16 source), but
  only for a catalog update. PG 16 also accepts `STORAGE MAIN` inline in
  `CREATE TABLE`; on 13–15 use `ALTER TABLE` right after creating it.
  `VACUUM FULL` holds `ACCESS EXCLUSIVE` for the whole rewrite. `pg_repack`
  holds it only briefly at the start and end.

## 6. Write-side costs (`copytime.py`, `update_cost.sql`, `hot.sql`)

- **COPY** of 100k × 1024-d, median of 3 rounds (`raw_copytime_1024.json`):
  EXTENDED 10.54 s, MAIN 10.35 s, EXTERNAL 9.48 s, PLAIN 9.43 s.
  The ~1.1 s difference between the compressing modes (EXTENDED, MAIN) and
  the non-compressing ones (EXTERNAL, PLAIN) is consistent with pglz trying
  and failing to compress every value (~10 µs/row). This is an inference from
  the grouping, not a profile.
- **Updating another column** (`raw_update_cost.txt`): `UPDATE … SET meta =
  meta + 1` on 10,000 of 100k rows, right after a `CHECKPOINT`:

  | | EXTENDED | MAIN |
  |---|---:|---:|
  | WAL written | 2.8 MB | **95.9 MB** |
  | statement time | 379 ms | 576 ms |
  | heap growth | +0.3 MB | +77.8 MB |
  | turbovec index | +4.1 MB (10k new entries) | +4.1 MB |

  The new row version carries a full copy of the inline vector. With TOAST it
  carries a ~20-byte pointer to the same, unchanged TOAST value. With the
  column at EXTENDED and `toast_tuple_target = 8160` the vector is inline too,
  and the result matched MAIN: 95.9 MB of WAL, 480 ms, heap +77.8 MB, 0 HOT
  (`raw_update_cost_ttt.txt`, run 2026-10-07).
- **HOT updates** (`raw_hot.txt`, 20k rows, `fillfactor = 50`): 2,000 of
  2,000 updates were HOT with EXTENDED and **0 of 2,000** with MAIN. A 5 KB
  tuple can't fit a second version on its page even at fillfactor 50. With
  MAIN, every update of any column adds an entry to every index on the
  table, the turbovec index included (a new vector the flat index stores and
  scans until VACUUM removes the old one). With TOASTed vectors and some
  free space per page, those updates can be HOT and touch no index.
  `toast_tuple_target = 8160` behaved like MAIN: 0 of 2,000 HOT
  (`raw_ttt_props.txt`).

## What this means for the guidance

Inline storage (`MAIN`) is a **latency-for-everything-else** trade. It pays
when the workload is dominated by large-candidate kNN on a table that stays
cached and is rarely scanned or updated for other reasons. It hurts
sequential scans, non-vector updates (WAL, HOT), and any working set that
doesn't fit in RAM. It does nothing below ~400-d (already inline) or above
~1,620-d (doesn't fit). `EXTERNAL` (pgvector's default) is equivalent to
today's `EXTENDED` for latency and size. It skips the futile compression
attempt, worth ~10 µs per inserted row.

`toast_tuple_target = 8160` with the column left at its default gets the
same latency and the same costs as `MAIN`, keeps other columns inline while
the row fits a page, takes a weaker lock (`SHARE UPDATE EXCLUSIVE`,
`raw_ttt_props.txt`) and is written by `pg_dump` as
`WITH (toast_tuple_target='8160')`. It applies to the whole table, isn't
copied by `LIKE … INCLUDING ALL` or `CREATE TABLE … AS`, needs the same
rewrite for existing rows (`VACUUM FULL` and `pg_repack` both honoured it;
an `UPDATE` that leaves the vector unchanged did not move it), and, unlike
`MAIN`, gives up the vector first when a row is over ~8 KB anyway.

Not measured: real embeddings (storage behaviour is content-independent
except for compressibility, which was nil here as it was for the real Cohere
corpus); tables much larger than RAM; PG versions other than 16 (the TOAST
thresholds are the same from 13 through 18); Graviton (no reason for a
different result: the saving is a TOAST fetch, not SIMD code); other
`bit_width` / IVF settings for the latency A/B (the recheck path is the same);
latency of `MAIN` and `toast_tuple_target = 8160` set together (placement
only, § 4.1); `pg_repack` on a table taking writes; pg_turbovec 2.12.0 (the
millisecond totals here are v2.11.0 and will be lower once the 2.12.0
distance-kernel and query-decoding changes land; those don't touch the TOAST
fetch, so the per-candidate saving should carry over, but that is untested).

## Files

Scripts: `up.sh` (cluster), `gen.py` (data), `load.py` (tables), `sizes.sql`,
`bench.py` (warm A/B), `run_bench.sh` (warm A/B under both malloc configs),
`run_ttt.sh` (§ 2.2 arm), `cold.py`, `inline_limit.sql`, `alter_after.sql`,
`rewrite_paths.sql`, `inplace.sql`, `inplace_vacuum.sql`, `repack.sh`,
`dump_trap.sh`, `ivf_degrade.sql`, `ttt_props.sh`, `copytime.py`,
`update_cost.sql` (`-v ttt=8160` for the § 6 row), `hot.sql`, `wall.py` /
`faults.py` / `faults2.py` / `strace_alloc.py` / `prof.sh` (§ 2.1
investigation), `summarize.py` (prints `raw_summary.txt` from the raw JSON).
`raw_wall_glibc.txt` and `raw_faults_glibc.txt` are `wall.py` / `faults.py`
run from a shell loop that restarted the postmaster with and without
`GLIBC_TUNABLES` (the settings are printed in each file).
Raw: every `raw_*` file, named after the script that produced it.
