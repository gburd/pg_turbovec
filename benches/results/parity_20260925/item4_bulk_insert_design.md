# Item 4 — Bulk `INSERT … SELECT` throughput: design + feasibility memo

**Scope:** read-only analysis. No code, no EC2. Repo: `/home/gburd/ws/pg_turbovec` @ v2.10.0.
**Verdict up front:** the commit-time cost is genuinely `O(n_vectors)` total rewrite (not `O(touched)`). Z5 does **not** amortize it — Z5 preserves the IVF *cell layout*, not the row chains. A true amortization is an **L/XL** effort that trips squarely into the HARD MANDATE on persist-path corruption. **Recommendation: defer**, gate on Item 1, ship the cheap doc + knob win now.

---

## 1. Trace the actual cost — it is `O(n_vectors)`, confirmed

The PreCommit callback drains dirty entries and calls `flush_to_relfile` per index (`src/xact.rs:245-268`), which calls `reconcile_and_write_flush` (`src/xact.rs:187`).

Inside `reconcile_and_write_flush` (`src/index/relfile.rs:1214`), the entire on-disk row state is read back **in full** under the exclusive rewrite lock:

- `read_chain(... m.n_vectors)` for codes, scales, ids — all three chains, all `n_vectors` rows (`src/index/relfile.rs:1300-1327`).
- `reconcile_flush_image` (`src/index/relfile.rs:1123`) starts from `disk_codes.to_vec()`, `disk_scales.to_vec()`, `disk_ids.to_vec()` (`:1134-1136`) — a full copy of every existing row — then UPDATEs in place / APPENDs at the tail for each `touched_id` (`:1168-1185`).
- The merged image goes to `write_full_inner` → `write_full_inner_with_tombstones` (`src/index/relfile.rs:1415`, `:1912`, `:1957`), which re-plans the meta and **rewrites all three chains from offset 0** every commit.

So the commit cost is `O(n_vectors)` in both **I/O** (all pages of codes+scales+ids re-emitted through `GenericXLog`) and **CPU** (the release-active `HashSet` bijection guard at `src/index/relfile.rs:2023-2042` is `O(n_vectors)` per flush; the reconcile builds two `O(n_vectors)` `HashMap`s at `:1141-1152`).

**What gets rewritten:** codes chain, scales chain, ids chain, and the meta page (written **last**, per the invariant). Not `O(touched)` — `touched_ids` only selects which rows to *splice*, but the write emits the whole merged array. For a 1.7M-row index, one `INSERT … SELECT` of any size pays a full ~1.7M-row (codes+scales+ids) rewrite at commit.

This exactly matches `docs/PARITY_GAPS.md:139-146` ("we still pay one full relfile rewrite at commit time, which is O(n_vectors)").

---

## 2. Interaction with Z5 (v2.10.0 bounded delta) — orthogonal; does NOT amortize the rewrite

Z5's mechanism (`docs/PARITY_GAPS.md:583-593`, code at `src/index/relfile.rs:1246-1290`, `:1379-1401`, `delta_within_bound` at `:3014`):

- On flush of an IVF index, read the existing coarse centroids + cell directory and **write them back unchanged** (`preserved_ivf`, `:1267`).
- Because `reconcile_flush_image` keeps every existing row at its slot and appends new rows at the tail, slots `[0, D)` stay cell-partitioned and `[D, n_live)` is a delta the scan sweeps exhaustively. `D = CellDirectory::total_vectors()`, so the delta length is *derivable* — **no format change** (`:1254-1260`).
- `ivf_delta_within_bound` (`:3034`) drops the layout past `turbovec.ivf_max_delta_pct`, degrading to flat (reportably, via Z1).

**What Z5 saved:** the IVF *trained structure* (centroids + directory), which pre-Z5 was dropped on the first insert, causing an immediate flat-scan cliff. Measured 11%/22% median win at 1M×256-d (`docs/PARITY_GAPS.md:585-590`).

**What Z5 did NOT touch:** the row chains. `reconcile_flush_image` still `.to_vec()`s all of disk (`:1134-1136`) and `write_full_inner` still rewrites all three chains from zero (§1). Z5 preserves *~lists×dim floats + a small directory* across the flush; the `O(n_vectors)` codes/scales/ids rewrite is unchanged. The two are orthogonal: Z5 removed a *recall/latency* cliff, not a *write-amplification* cost. **The whole relfile is still rewritten each commit.**

The one thing Z5 *does* give the amortization design for free: it proved the **append-at-tail** slot discipline is sound and needs no meta field (delta length derivable). That is the reusable primitive.

---

## 3. Sketch of a true amortization (append-only delta region + consolidation), and what breaks

### The zvec lesson (`docs/PARITY_GAPS.md:410-427`)
zvec keeps trained indexes immutable, lands writes in a Flat mutable segment, seals+rolls it over, and runs `optimize()` (build+merge) outside the exclusive lock, publishing atomically. Mapped onto us: **don't rewrite the base chains on every commit — append the txn's new rows to a physically separate tail region, and consolidate periodically.**

### Concrete shape
- **Base region** `[0, B)`: the consolidated chains, rewritten only at consolidation.
- **Delta region** `[B, n_live)`: physically appended pages. A commit writes only *its own* new rows here (`O(touched)` pages) + the meta page. No re-read of the base.
- **Consolidation**: an explicit step (piggyback on VACUUM, or a `turbovec_consolidate()` SQL fn) folds the delta into the base and, for IVF, re-cell-assigns the delta rows. `O(n_vectors)` but amortized over many commits.
- **Scan**: base (cell-probed for IVF) + delta (exhaustive) — Z5's fan-out shape already, just made durable/incremental instead of recomputed each flush.

### What breaks — and who already solved it

| Concern | Status | Notes |
|---|---|---|
| **Crash-safety: chains-then-meta invariant** (AGENTS.md HARD MANDATE #1) | **NEW, hardest** | Today the meta is written last after a *complete* rewrite, so the on-disk image is always internally consistent. An incremental append means the meta must atomically flip `n_vectors`/delta-length only after the delta pages are durable — the same ordering, but now the base must remain valid if the append tears. A torn delta append that the meta doesn't yet reference is safe (rows just not indexed); a meta bumped before delta pages are durable is corruption. Needs its own repro test that fails-before/passes-after **and** a sustained-insert-load re-corruption check (the v1.28.4 lesson, AGENTS.md). |
| **VACUUM / tombstones** | **Mostly solved by Z5/E-2** | IVF already tombstones rather than swap-removes (`src/index/relfile.rs:1263-1265`), so slot indices are stable and the delta's slots don't shift under VACUUM. Consolidation must compact tombstoned slots — that's just what a full rewrite already does. |
| **MVCC / rollback** | **Solved** | The Abort + AbortSub callbacks already invalidate the whole dirty cache (`src/xact.rs:295-317`); a rolled-back txn's appended delta would simply never get its meta bump. The deferred-commit model already handles "in-memory mutated, not yet durable." |
| **Lost-update under concurrent VACUUM** | **Solved by v1.29.1 reconcile** | `reconcile_and_write_flush` re-reads current disk under the exclusive lock (`src/index/relfile.rs:1214-1327`). An append design keeps that lock; it re-reads only the *tail cursor*, not all rows. |
| **On-disk bijection guard cost** | **Regression risk** | The `O(n_vectors)` dup/id-0 `HashSet` guard (`src/index/relfile.rs:2023`) runs per flush; an append path must guard only the *delta* to actually get `O(touched)`, without weakening the invariant. |
| **Wire format** | **NEW** | A durable delta cursor distinct from `total_vectors()` likely needs a meta field → additive bump or a new `kind` byte (AGENTS.md prefers a new `kind`). Must read old format transparently (no REINDEX for a minor). |

Net: MVCC and VACUUM interaction are largely **already solved**; **crash-safety of an incremental (non-full-rewrite) meta commit is the new, load-bearing risk** and is exactly the class of change AGENTS.md HARD MANDATE gates hardest.

---

## 4. Effort + recommendation

**Effort: L (realistically L→XL).** Not the append itself — the append primitive exists (Z5 proved tail-append + derivable delta length). The cost is the **crash-safety re-verification**: a new torn-write model for incremental meta commits, a repro test, and a *sustained-insert re-corruption* validation on an AVX2 host, per the HARD MANDATE. That validation, not the code, is the long pole (v1.28.4 shipped on reasoning alone and re-corrupted in production — AGENTS.md forbids repeating that).

**Recommendation: DEFER. Keep the `CREATE INDEX`-after-bulk guidance. Gate on Item 1.**

Justification the evidence supports:
- The gap only bites **continuous high-ingest into an already-large index** (>1.7M rows, sustained single writer). A one-shot bulk load already has a correct, cheaper answer (`CREATE INDEX` after the `INSERT`), documented at `docs/PARITY_GAPS.md:145-146`.
- Whether that continuous-ingest profile even *chooses* us is unknown until **Item 1 (IVF-vs-HNSW at matched recall/latency)** resolves. If IVF isn't latency-competitive at that profile, the workload lands on HNSW regardless and this rewrite is dead-weight risk against the persist path — the most corruption-sensitive code in the tree.
- Building an L/XL persist-path change *on spec*, before Item 1 tells us the profile is real for us, directly contradicts both YAGNI and the HARD MANDATE's "when in doubt, do not ship."

So: **defer, revisit only if Item 1 shows we're competitive on latency at the continuous-ingest profile.** If it does, the append-delta design above is the plan of record, sized L→XL, gated on the sustained-load re-corruption test.

---

## 5. The cheap partial win (ship now, no new feature)

**5a. Document the amortization knob that already exists.** `turbovec.ivf_max_delta_pct` (`src/index/relfile.rs:3014-3035`, GUC `IVF_MAX_DELTA_PCT`) is the existing lever for the continuous-ingest profile: it bounds how large the appended delta grows before the index degrades to flat. Raising it lets an IVF index absorb more sustained inserts before a REINDEX is needed, at a bounded scan-latency cost. This is under-documented for the *ingest* use case — `docs/PARITY_GAPS.md` frames it only as a Z5 recall knob. One paragraph in the INSERT-throughput section (and `docs/UPGRADING.md`/tuning docs) turning "load via CREATE INDEX" into "for continuous ingest, tune `ivf_max_delta_pct` and REINDEX on a cadence" is a real, honest, zero-risk win.

**5b. Sharpen the guidance itself.** The current one-liner ("load via CREATE INDEX after the bulk INSERT") doesn't address the *continuous* profile at all — that workload can't stop to `CREATE INDEX`. The honest guidance is: batch inserts into larger transactions (the `O(n_vectors)` rewrite is *per-commit*, so fewer/bigger commits amortize it directly — this is already true today, no code needed), keep `ivf_max_delta_pct` tuned, and REINDEX on a scheduled cadence rather than never. Naming "fewer, larger transactions" explicitly is the single cheapest latency lever available and costs one doc paragraph.

**Do NOT** build a batching GUC or an auto-consolidation daemon on spec — transaction batching is already user-controllable, and auto-consolidation is the deferred L/XL feature wearing a smaller hat.

---

### Citations index
- Commit cost `O(n_vectors)`: `src/xact.rs:187,245-268`; `src/index/relfile.rs:1214,1300-1327,1134-1136,1168-1185,1415,2023-2042`.
- Z5 delta (orthogonal): `docs/PARITY_GAPS.md:583-593`; `src/index/relfile.rs:1246-1290,1379-1401,3014-3035`.
- Crash-safety invariant + mandate: `AGENTS.md` HARD MANDATE #1; `src/index/relfile.rs:2044-2052` (meta-last lock), `:1957` (tombstones folded into one meta write).
- MVCC/abort already handled: `src/xact.rs:295-317`.
- Existing knob to document: `src/index/relfile.rs:3014-3035` (`ivf_max_delta_pct`).
