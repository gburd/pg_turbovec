# Corruption fix — concurrent VACUUM / deferred-flush lost-update repro

This harness reproduces (on the pre-fix binary) and proves the absence of
(on the fixed binary) the `.tvim` id-table corruption reported 2026-08-11:
continuous concurrent inserts + VACUUM into a ~1.76M-row 768d flat
`turbovec.vector` index produced "duplicate ids (id 0 appears in more than
one slot)" (XX001) and SIGABRT-crashing backends.

## Scripts

- `load-corpus.sh`  — build a ~1.76M-row 768d corpus of DISTINCT vectors
  (`NROWS` env, default 1,760,000). The per-element expression references
  the outer row id so the planner cannot hoist the array subquery into one
  shared value (the `n_distinct=1` test-data trap).
- `build-index.sh`  — `CREATE INDEX ... USING turbovec (emb turbovec.vec_l2_ops)`.
- `run2.sh`         — the comprehensive sustained-load harness (below).
- `mini-repro.sh`   — a minimal 2-writer crash repro with per-backend stderr
  capture (diagnostics).
- `bench.sh`        — build/warm-kNN/insert-throughput regression bench.

## `run2.sh` — the comprehensive workload

Concurrently, against the flat index:

- VACUUM `corpus` every 3s (swap-remove shrink of the flat relfile).
- `NWRITERS` writers (default 4), each a bounded persistent psql running a
  loop of one-transaction-per-batch (16 then 128 rows) via
  `INSERT ... ON CONFLICT (id) DO UPDATE`; every `RECONNECT_EVERY` (8) cycles
  the writer exits and a fresh backend starts (exercises the first-insert
  cold-load path) — bounded connection count, no runaway spawn.
- 3 scanner loops (`ORDER BY emb <-> q LIMIT 10`, probes=16).
- A monitor that (a) continuously scans the PG log for the DEFINITIVE
  corruption signatures (dup-id "more than one slot", "duplicate ids",
  `signal 6`/SIGABRT, "database system is in recovery") at zero DB cost, and
  (b) every `CHECK_EVERY` (90s) briefly quiesces writers+VACUUM so the shared
  rewrite lock drains, runs an UNCONTENDED `turbovec.turbovec_check`, and
  records `is_corrupt`.

FAIL if any dup-id / SIGABRT / recovery / `is_corrupt=true` is observed.

## Result (2026-08-13, i4i.8xlarge, PG18)

- **Pre-fix (meta-lock base, commit 21a18c7): FAIL at 60s** — `turbovec_check`
  reports `duplicate_id=0, is_corrupt=true`; 35 SIGABRT / recovery events.
- **Fixed (branch `fix/corruption-vacuum-race`): PASS at 70.5 min** — 46/46
  quiesced `turbovec_check` reads clean (`count_matches=true`, no duplicate,
  `is_corrupt=false`); 0 dup-id, 0 SIGABRT, 0 recovery; ~19,470 write
  transactions concurrent with VACUUM-every-3s.

## Reproduce

```bash
# one-time
NROWS=1760000 ./load-corpus.sh
./build-index.sh
# comprehensive run (65+ min)
DUR=4200 NWRITERS=4 CHECK_EVERY=90 OUT=/tmp/run-$(date +%s) ./run2.sh
cat /tmp/run-*/verdict.txt      # VERDICT=PASS
```
