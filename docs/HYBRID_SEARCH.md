# Hybrid & multivector search

The breadth-parity guide (vs VectorChord / Qdrant): how to do
**multivector / late-interaction (ColBERT) re-ranking**, **hybrid
dense+sparse fusion (RRF)**, and **named vectors (multiple vector
columns)** with pg_turbovec.

All three are *additive SQL surface* — they do not change the index
wire format or the index AM. pg_turbovec indexes **one vector per
row**; the multivector and hybrid pieces are query-layer constructs
on top of that single-vector index.

> **Phase D (v1.13.x).** `turbovec.max_sim` / `turbovec.max_sim_cosine`
> (MaxSim re-rank) and `turbovec.rrf_score` (reciprocal rank fusion)
> ship as SQL functions. Named vectors are a documented schema
> pattern. Index-native late interaction is a documented future
> phase — see [the limitation](#the-honest-limitation).

---

## 1. Multivector / late interaction (MaxSim)

### What it is

Late-interaction models (ColBERT and friends) represent a query and a
document each as a **set of per-token vectors**, not a single pooled
vector. The relevance score is **MaxSim**:

```
MaxSim(Q, D) = sum over query-token vectors q in Q of
                 ( max over doc-token vectors d in D of sim(q, d) )
```

Each query token finds its single best-matching document token; the
score is the sum of those per-token maxima. This captures fine-grained
term-level matches that a single pooled embedding averages away.

### The functions

```sql
turbovec.max_sim(query turbovec.vector[], doc turbovec.vector[])
  -> double precision    -- dot-product MaxSim

turbovec.max_sim_cosine(query turbovec.vector[], doc turbovec.vector[])
  -> double precision    -- cosine MaxSim
```

Both take **arrays of `turbovec.vector`** — one entry per token.

**Conventions (read these):**

- **Similarity, not distance.** `max_sim` uses raw **dot product** as
  the per-pair similarity (higher = more similar). `max_sim_cosine`
  uses **cosine similarity = `1 - cosine_distance`** (range `[-1, 1]`,
  higher = more similar). ColBERT token vectors are normally
  L2-normalised, in which case `max_sim` (dot) already equals cosine —
  prefer `max_sim` for normalised tokens (it skips the per-pair norm).
- **Dimension.** Every token vector in **both** arrays must share one
  dimension. A mismatch raises `ERROR: different vector dimensions N
  and M in 'max_sim'`.
- **Empty arrays score `0.0`.** An empty query (nothing to match) and
  an empty doc (nothing to match against) both yield `0.0`. This is
  the ColBERT convention and lets you `LEFT JOIN` candidate docs
  without `NULL` surprises.
- **Determinism.** The outer sum walks the query tokens in array
  order; no reassociation, so the `f64` result is reproducible.
- **Cost.** `O(|Q| · |D| · dim)` per pair. This is a **re-rank**
  primitive for a small candidate set (top-N), not an index scan.

```sql
SELECT turbovec.max_sim(
  ARRAY['[1,0]','[0,1]']::turbovec.vector[],          -- query tokens
  ARRAY['[1,0]','[0,1]','[1,1]']::turbovec.vector[]   -- doc tokens
);
-- q1=[1,0]: max dot over doc = 1 ; q2=[0,1]: max dot over doc = 1 ; sum = 2
```

### The re-rank usage pattern

MaxSim over every (query, doc) pair in a corpus is `O(corpus)` — too
slow to scan. The standard ColBERT serving recipe is **two-stage
retrieve-then-rerank**:

1. **Stage 1 — ANN retrieve candidates** on a *single* pooled /
   centroid vector per document. Store one mean-pooled `turbovec.vector`
   per doc, index it with a `turbovec` index, and pull the top-N
   (say N = 100–1000) candidate documents with a normal `<=>` / `<#>`
   ANN query.
2. **Stage 2 — MaxSim re-rank** the N candidates using their full
   per-token arrays, then take the final top-k.

```sql
-- Schema: one pooled vector (indexed) + the per-token array (re-rank only)
CREATE TABLE docs (
  id           bigint PRIMARY KEY,
  pooled       turbovec.vector,          -- mean of token vectors; indexed
  tokens       turbovec.vector[]         -- per-token ColBERT vectors
);
CREATE INDEX ON docs USING turbovec (pooled vec_cosine_ops);

-- Query: :q_pooled is the pooled query vector; :q_tokens the token array.
WITH candidates AS (
  SELECT id, tokens
  FROM docs
  ORDER BY pooled <=> :q_pooled        -- stage 1: ANN on pooled vector
  LIMIT 200                            -- candidate set N
)
SELECT id,
       turbovec.max_sim(:q_tokens, tokens) AS score   -- stage 2: rerank
FROM candidates
ORDER BY score DESC                                    -- MaxSim is a SIMILARITY
LIMIT 10;
```

The pooled-vector ANN is sublinear (index-accelerated); the MaxSim
re-rank is `O(N · |Q| · |D| · dim)` over just the N candidates, which
is cheap for N in the hundreds.

### The honest limitation — and what `colbert_search` (F-1) adds

**pg_turbovec's persistent index stores one vector per row.**
`max_sim` is a re-rank primitive over candidate documents' token
arrays; the pooled-vector + `max_sim` pattern above is **not** an
index-accelerated late-interaction scan, and its recall is capped by
the pooled-vector ANN: if the right document's *pooled* vector isn't
in the top-N candidates, MaxSim never sees it.

**`turbovec.colbert_search` (v1.16.0, Phase F-1) closes exactly that
recall gap** — it retrieves candidates by **best single token**, not
pooled mean:

```sql
turbovec.colbert_search(
    rel         regclass,
    id_col      text,                 -- bigint doc key
    token_col   text,                 -- a turbovec.vector[] column (per-doc tokens)
    query       turbovec.vector[],    -- the query's token vectors
    k           integer,              -- final top-k docs
    per_token_k integer DEFAULT 64,   -- stage-1 hits per query token
    candidate_n integer DEFAULT 256,  -- max candidate docs into stage 2
    bit_width   integer DEFAULT 4
) RETURNS TABLE(id bigint, score double precision)
```

```sql
-- docs(id bigint, tokens turbovec.vector[])  -- per-doc token arrays
SELECT id, score
FROM turbovec.colbert_search('docs'::regclass, 'id', 'tokens',
                             :q_tokens::turbovec.vector[], 10);
```

How it works (two stages, same shape as the pattern above but with an
indexed stage 1 over **all** tokens):

1. **Stage 1 (index-accelerated).** A backend-cached flat **token**
   index — one slot per token across all docs, the slot's id being its
   doc id — is batch-searched with all `|Q|` query tokens; the hit
   doc-ids are unioned into a candidate set (capped at `candidate_n`,
   keeping the docs with the best stage-1 token score). Because every
   token is indexed, a doc whose *pooled* vector is far but which has
   **one** token near a query token is still retrieved — the recall
   the pooled-vector pattern cannot get.
2. **Stage 2 (exact).** Each candidate's full token array is read from
   the heap and scored with the exact `max_sim` kernel; the top `k`
   are returned.

**What it is and isn't.** `colbert_search` is the *index-native
stage-1* over `max_sim`'s *exact stage-2*. It is **not** the full
persistent multivector index AM (per-token relfile, MaxSim-aware
scan, PLAID centroid pruning) — the token index lives only in the
backend cache (like `turbovec.knn`), so there is **no wire-format
change** and no `CREATE INDEX` for it. It rebuilds per backend on a
cold cache (fine for moderate corpora; the build is the same
quantize-pack as `knn`). For very large token corpora a persistent
token-index AM (Phase F-2) is the next step — gated on a measured
recall/latency win over this F-1 path (see
`docs/PHASE_F_COLBERT_PLAN.md`).

**Tuning.** `per_token_k` (stage-1 hits per query token) and
`candidate_n` (max docs into stage 2) trade recall for work; under
heavy token quantization (2–3 bit) raise `per_token_k`. `bit_width`
4 is the safe default for 128-d ColBERT tokens. The crossover where
`colbert_search` beats the pooled-vector pattern is **workload-
dependent** (it wins on entity / rare-term / long-doc queries where
the pooled mean washes out the matching token; it ties on
pooled-friendly semantic queries).

---

## 2. Hybrid dense + sparse (Reciprocal Rank Fusion)

### What it is

Hybrid search fuses a **dense** semantic ranking (vector ANN) with a
**sparse / lexical** ranking (BM25 / full-text `ts_rank`, or a
`turbovec.sparsevec` inner-product score). The two rankers disagree on
scale, so you fuse by **rank**, not by raw score, using **Reciprocal
Rank Fusion (RRF)**:

```
score(d) = sum over rankers r of  1 / (k + rank_r(d))
```

`k` (default 60, from Cormack et al. 2009) damps the contribution of
low ranks. A document ranked highly by *both* rankers wins.

### The helper

```sql
turbovec.rrf_score(rank integer, k integer DEFAULT 60)
  -> double precision    -- = 1.0 / (k + rank)
```

The arithmetic is trivial; the helper exists so the formula and its
`k` default live in one tested place instead of being retyped in every
query. `k + rank` must be positive (a non-positive denominator raises
`ERROR`). Use a consistent rank base across rankers (this guide uses
**0-based** ranks via `ROW_NUMBER() - 1`, but 1-based works as long as
both rankers agree).

```sql
SELECT turbovec.rrf_score(0);      -- 1/60  ≈ 0.016667
SELECT turbovec.rrf_score(1, 10);  -- 1/11  ≈ 0.090909
```

### The recipe (dense ANN + full-text)

```sql
-- Schema: one dense vector column + a tsvector for lexical search.
CREATE TABLE docs (
  id      bigint PRIMARY KEY,
  body    text,
  emb     turbovec.vector,
  body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
CREATE INDEX ON docs USING turbovec (emb vec_cosine_ops);
CREATE INDEX ON docs USING gin (body_tsv);

-- Query: :q_emb is the dense query vector; :q_text the keyword query.
WITH dense AS (
  SELECT id,
         ROW_NUMBER() OVER (ORDER BY emb <=> :q_emb) - 1 AS rk
  FROM docs
  ORDER BY emb <=> :q_emb
  LIMIT 100                                      -- dense candidate pool
),
sparse AS (
  SELECT id,
         ROW_NUMBER() OVER (
           ORDER BY ts_rank(body_tsv, plainto_tsquery('english', :q_text)) DESC
         ) - 1 AS rk
  FROM docs
  WHERE body_tsv @@ plainto_tsquery('english', :q_text)
  LIMIT 100                                      -- lexical candidate pool
),
fused AS (
  SELECT id, SUM(s) AS score FROM (
    SELECT id, turbovec.rrf_score(rk) AS s FROM dense
    UNION ALL
    SELECT id, turbovec.rrf_score(rk) AS s FROM sparse
  ) u
  GROUP BY id
)
SELECT d.id, d.body, fused.score
FROM fused JOIN docs d USING (id)
ORDER BY fused.score DESC
LIMIT 10;
```

Each ranker produces a 0-based rank via `ROW_NUMBER()`; `rrf_score`
turns rank into a fusion term; the `UNION ALL` + `GROUP BY ... SUM`
sums the terms per document; the final `ORDER BY score DESC` returns
the fused ranking. A document present in only one ranker still
contributes that ranker's term (it just doesn't get the boost of
appearing in both).

### Sparsevec variant

If you keep a learned sparse / SPLADE vector in a `turbovec.sparsevec`
column, swap the `sparse` CTE's ordering for the sparse inner product
(`<#>` is **negative** inner product, so smaller is more similar —
order ascending):

```sql
sparse AS (
  SELECT id,
         ROW_NUMBER() OVER (ORDER BY svec <#> :q_svec) - 1 AS rk
  FROM docs
  ORDER BY svec <#> :q_svec
  LIMIT 100
)
```

### Tuning `k`

`k = 60` is the canonical default and a fine starting point. Lower `k`
(e.g. 10–20) sharpens the contribution of the very top ranks (the #1
result dominates); higher `k` (100+) flattens the curve so deeper
ranks matter more. Tune on your own relevance judgments; the default
is robust across corpora.

### Why a scalar helper, not a server-side fusion operator

The roadmap calls for "server-side fusion only on demand." The scalar
`rrf_score` + the CTE recipe above is the pragmatic core: it composes
with arbitrary rankers, `WHERE` filters, and `JOIN`s, and it is fully
inspectable in `EXPLAIN`. A bespoke two-array fusion aggregate would
be less flexible and is not provided until a concrete need appears.

---

## 3. Named vectors (multiple vector columns)

"Named vectors" — Qdrant's term for multiple distinct embeddings per
record (e.g. a title embedding *and* a body embedding, or a CLIP image
embedding *and* a text embedding) — is a **schema pattern** in
PostgreSQL, not a feature. Each named vector is just another
`turbovec.vector` column with its own index:

```sql
CREATE TABLE products (
  id        bigint PRIMARY KEY,
  title_emb turbovec.vector,
  body_emb  turbovec.vector,
  image_emb turbovec.vector
);
CREATE INDEX ON products USING turbovec (title_emb vec_cosine_ops);
CREATE INDEX ON products USING turbovec (body_emb  vec_cosine_ops);
CREATE INDEX ON products USING turbovec (image_emb vec_cosine_ops);
```

Query a single named vector with a normal ANN query (`ORDER BY
title_emb <=> :q`). To combine multiple named vectors into one
ranking, **fuse them at query time with RRF** — exactly the recipe in
§2, with each named-vector ANN as one ranker:

```sql
WITH by_title AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY title_emb <=> :q_title) - 1 AS rk
  FROM products ORDER BY title_emb <=> :q_title LIMIT 100
),
by_body AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY body_emb <=> :q_body) - 1 AS rk
  FROM products ORDER BY body_emb <=> :q_body LIMIT 100
)
SELECT id, SUM(s) AS score FROM (
  SELECT id, turbovec.rrf_score(rk) AS s FROM by_title
  UNION ALL
  SELECT id, turbovec.rrf_score(rk) AS s FROM by_body
) u
GROUP BY id ORDER BY score DESC LIMIT 10;
```

Each column has its own index, so each sub-query is index-accelerated;
RRF fuses the per-column rankings without any cross-column distance
needing to be defined.

---

## See also

- [`FILTERING.md`](FILTERING.md) — metadata filtering & filtered ANN.
- [`PARITY_GAPS.md`](PARITY_GAPS.md) — full pgvector parity tracker.
- [`COMPETITIVE_ANALYSIS.md`](COMPETITIVE_ANALYSIS.md) — vs Qdrant /
  VectorChord, incl. the multivector / hybrid scoreboard rows.
- [`MIGRATING_FROM_PGVECTOR.md`](MIGRATING_FROM_PGVECTOR.md) —
  coexistence & migration.
