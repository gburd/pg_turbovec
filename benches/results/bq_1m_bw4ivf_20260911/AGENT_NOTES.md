# Notes for the lead on the bw4+IVF 1M arm (2026-09-11)

Only items that correct or sharpen what was stated back to me. Everything
here is measured from
`benches/results/bq_1m_bw4ivf_20260911/bq_1m_bw4_flat_ivf1024.json`.

## 1. CORRECTION: "3.6x the load" is cpu_busy, not loadavg

Both numbers are real but they are different multiples, and the stronger
one should not be attached to the wrong metric:

| metric | flat (8 rows) | ivf (40 rows) | ratio |
|---|---:|---:|---:|
| mean `loadavg_1m_before` | 6.00 | 3.20 | **1.88x** |
| mean `cpu_busy_pct`      | 22.0 % | 6.1 % | **3.6x** |

So: "flat ran at 3.6x the CPU-busy and 1.9x the loadavg of IVF." Quoting
3.6x against loadavg would be wrong.

## 2. The conclusion does not actually depend on the latency numbers at all

This is a stronger claim than the bias argument and it should be the one
that carries the section, because it needs no contention reasoning:

**IVF cannot reach R@10 >= 0.98 at any setting in the sweep.** Max R@10 by
probes = 0.703 / 0.785 / 0.875 / 0.936 / **0.959** (p=8/16/32/64/128), and
widening the rerank window 32 -> 2000 at p=128 leaves recall at 0.959 for
every one of the 7 windows -- it does not move by a single query. Flat is
at **1.000**.

Recall is CPU-independent (BQ_RECALL_BENCH.md § 2). So even if every p50 in
the artefact were thrown away as contended, bw4 IVF still loses at the two
highest targets on recall alone. The latency table is corroboration, not
the load-bearing evidence. I would lead with the ceiling and treat the
62 % latency gap as secondary.

## 3. Report this row loudly: the ONE place IVF's p50 is below flat's

Iso-knob, w=2000: flat 122.19 ms vs IVF p=128 **116.77 ms** (0.96x). It is
the only such row in 48. It is **not** a win -- it is 0.959 recall against
flat's 1.000, so it fails matched-recall -- but it exists, and it is better
that you surface it than that a reader later finds it and wonders what else
was filtered. Every other row at every other window has IVF slower
(2.65x / 1.85x / 1.56x / 1.38x / 1.16x / 1.11x at w=32/100/256/400/800/1024).

Tightest honest framing of "never wins": **IVF's fastest configuration
anywhere in the sweep is 15.76 ms (p=32, w=32, R@10 = 0.875) -- 2.6x slower
than flat's cheapest configuration (6.08 ms), which is simultaneously its
most accurate (R@10 = 1.000).** Flat's cheapest row is its best row, so
there is no target at which IVF is given an opening.

## 4. Two secondary measured facts, and one cross-run comparison NOT to make

- **Build time: flat 14.89 s vs IVF 101.22 s (6.8x).** Worth stating next to
  the RSS figure -- the user is weighing enabling `lists` on a production
  index, and 6.8x build for a latency loss is part of the cost.
- **Storage overhead of IVF at bw4: +0.75 %** (559.95 -> 564.17 B/vec).
  Do NOT compare that to § 0.6e's "+3.1 % at bw1" as a bit-width effect:
  that is cross-run, and per the resolvability caveat the two runs may not
  share corpus geometry. Within this run it is +0.75 %, full stop.
- **GT build was 438.4 s**, not the ~275 s the brief quoted. Firmly in the
  parallel-CTAS regime (not the 3755 s pre-fix regime), so the code was
  current at v2.8.1 -- but it is 1.6x the reference figure and I did not
  investigate why. Non-blocking; flagging so it is not read as a harness
  regression if someone diffs it.

## 5. What this arm does NOT license

- One corpus, one dim (1024), one list count (1024), one bit width (4),
  **100 queries** (§ 5 asks for >= 200).
- Says nothing about bw4 + IVF at n > 1M, where flat's O(n) wall keeps
  growing and IVF's ceiling does not move. The measured ceiling (0.959 at
  p=128) is a property of cell-restricted search, so a user with a >= 0.98
  target is answered at any scale; a user with a 0.90 target at 10M is
  **not** answered by this arm.
- `lists = 1024` only. The brief correctly excluded 4096 (measured worse at
  1M for bw1), but no other list count was tried at bw4.
