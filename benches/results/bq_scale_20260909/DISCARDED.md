# DISCARDED — the 1M arm's recall numbers are a CORPUS artefact, not a result

**Do not cite the recall figures in these artefacts.** They are kept only as
a record of the failure mode.

## What the run reported

1 000 000 × 768-d synthetic corpus on `meh`, `bit_width = 1`:

| rerank window | R@10 | R@100 |
|---:|---:|---:|
| 32 | 0.031 | 0.015 |
| 1024 (`auto`) | 0.212 | 0.132 |
| 8000 | 0.509 | 0.383 |

Against the published 250k × 1024-d real-corpus run (R@10 = 0.744 at
window 32), that looks like a catastrophic scale collapse.

## It isn't. The corpus was unrankable.

Resolvability probe — exact cosine distance of the true 1st vs 100th
neighbour, 5 held-out queries, measured on both corpora:

| corpus | nn1 | nn100 | spread |
|---|---:|---:|---:|
| synthetic 1M × 768-d (this run) | 0.0866–0.0989 | 0.0928–0.1055 | **6.6–10.4 %** |
| real Cohere-wiki 250k × 1024-d | 0.135–0.353 | 0.455–0.527 | **37–268 %** |

When the 1st and 100th neighbours differ by <10 % in distance, the top-100
is essentially a tie and **no quantizer can rank it** — a lossy code has
nothing to preserve. The measured "recall" is then mostly the tie-breaking
order, which is exactly what a 1-bit code destroys first.

## Why the generator did this — and one wrong diagnosis, ruled out

**The clustering worked.** Verified directly on the loaded corpus:

- `centres`: 200 rows, **200 distinct** `cvec` (not 1).
- `docs`: 200 distinct `cid`, **5000 distinct `embt` per cid**.
- same-cluster mean distance **0.108** vs cross-cluster **0.990** — a ~9×
  separation. A query's nearest centre is 0.056; its farthest, 1.079.

**The defect is inside each cluster, not between them.** Within a single
cluster, the 100 nearest neighbours of a member (excluding itself) span
**0.089767 → 0.096247, a 7.22 % spread**. That is the tie. 5000 iid Gaussian
points at d = 768 are mutually near-equidistant: the pairwise separation norm
concentrates at ≈ `0.35 * sqrt(2 * 768)` ≈ 13.7 with a relative spread of only
`1/sqrt(2d)` ≈ 2.6 %. Ranking inside that is ranking noise, and a 1-bit code
is the first thing to lose it.

So the dispatch prompt's warning was heeded at the level it was written —
clustering *was* implemented — but σ = 0.35 per coordinate at d = 768
re-created distance concentration **inside** each cluster, which is the same
failure one level down.

### Ruled out: the uncorrelated-subquery hoist

A plausible alternative diagnosis was raised and is **disproven for this
corpus**, but it is worth recording because the pattern really is present in
the SQL:

```sql
-- gen_corpus.sql, the centres table: the inner subquery never references `c`
CREATE TABLE centres AS
SELECT c AS cid,
       ARRAY(SELECT randn()::real FROM generate_series(1, 768)) AS cvec
FROM generate_series(1, 200) c;
```

That is textbook uncorrelated-subquery shape — the same class as the v1.24.0
test-harness bug in `AGENTS.md`, where an uncorrelated `random()` was hoisted
and made every row identical. Confirmed minimally on the same cluster:

| form | distinct vectors / rows |
|---|---|
| `ARRAY(SELECT random() FROM generate_series(1,4))` | **1 / 5** |
| same, plus a correlating `WHERE c = c` | **5 / 5** |

So the hazard is genuine and this SQL invites it. But the *loaded* `centres`
table has 200 distinct rows, so PostgreSQL did **not** hoist it here — the
volatility of `randn()` (a `VOLATILE` SQL function wrapping `random()`) forced
per-row evaluation. **Do not "fix" the recall numbers by correlating the
subquery; that is not what is wrong with them.** Correlate it anyway for
safety, since the behaviour is plan-dependent and one PostgreSQL version's
choice is not a guarantee.

## What IS salvageable from this arm

Storage and build, which don't depend on rankability:

| | bytes/vec | index | build |
|---|---:|---:|---:|
| bw1 | 104.9 | 104.9 MB | 171.1 s |
| bw2 | 209.7 | 209.7 MB | 154.7 s |

`104.9 = 768/8 + ~9` B/vector confirms the `dim/8` sign-code stride at 1 M
rows, and the **2.00× exact ratio** between 1-bit and 2-bit holds at this
scale. Build is ~171 s for 1 M × 768-d on a 24-core pre-AVX2 host.

Also: ground truth took **23 084 s (6.4 h)** — an exact 100-query × 1 M × 768-d
seqscan on a scalar host. That cost is the real reason this arm is expensive,
and it is a `meh` property (no AVX2), not a turbovec one.

## The actual open question is still open

**Does the rerank window needed for a given recall grow with n?** This run
cannot answer it. Answering it needs a *real* 1 M-row corpus — e.g. the
1 M × 1024-d Cohere-wiki table already on `arnold` — with the harness's
per-row-cast trap avoided (materialise a native `turbovec.vector` column
first; see `docs/BQ_RECALL_BENCH.md` § 0.6).

## Lesson for future synthetic corpora

Before trusting any recall number from generated data, run the resolvability
probe above and require the nn1→nn100 spread to be **comparable to a real
corpus** (tens of percent, not single digits). A synthetic corpus that passes
`is_degenerate()` can still be statistically unrankable — those are different
checks, and only the second one predicts whether recall means anything.
