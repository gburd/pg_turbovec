# Upstream proposal #1 — re-expose `pack::repack` as `pub`

**Fork carry:** `537a289` (branch `pgtv-2.0.0-port`, on top of turbovec 1.0.0 `ccab9f3`)
**Target file:** `turbovec/src/pack.rs`
**Process note:** turbovec is issue-first, invitation-only for PRs. This is a
GitHub **issue** proposal, not a cold PR. It slots naturally as a follow-up on
the closed `#70` thread (the maintainer explicitly invited "comment and I'll
reopen if the cache-fill path isn't covered").

---

## (a) Motivation — from turbovec's own perspective

`pack::repack` turns row-major packed codes into turbovec's SIMD-blocked search
layout. Today it's `pub(crate)`, so the only way an out-of-tree consumer can get
a blocked layout is to let `from_parts` build it internally and keep it resident
alongside the packed codes. That forces any embedder that stores turbovec codes
in its *own* container (a database page cache, an mmap'd column, an RPC payload)
to either (1) persist both the packed **and** the blocked bytes — the blocked
layout is the larger of the two — or (2) hold both in memory after load.

`repack` is the natural public seam for "I already have `packed_codes()`; give me
the blocked layout on demand." It's a pure function of `(packed_codes, n_vectors,
bits, dim)` with no hidden state, its inputs are already reachable via the public
`packed_codes()` accessor, and `pack` is already `pub mod`. Exposing it lets
storage-backed embedders persist **only** the compact packed codes and reconstruct
the blocked layout lazily — a strictly smaller on-disk/on-wire footprint, which is
squarely in line with turbovec's headline "fits it in 4 GB" compression story.
It also composes cleanly with the round-trip API `#204`/`#210` already shipped:
`from_parts` gives you the codes back, `repack` gives you the search layout back.

## (b) Exact diff summary

One visibility change plus a doc comment. No signature change, no behaviour change.

```diff
-pub(crate) fn repack(
+pub fn repack(
     packed_codes: &[u8],
     n_vectors: usize,
     bits: usize,
     dim: usize,
 ) -> (Vec<u8>, usize) {
```

Doc comment added noting the intended use (recompute the blocked layout from
`packed_codes()` at load, to avoid persisting the layout twice). The existing
safety note — "callers must pass validated inputs; construct through `from_parts`
which validates first" — is retained.

## (c) API-stability / maintenance concerns the maintainer would raise

- **New public surface = new stability commitment.** `repack`'s return shape
  `(Vec<u8>, usize)` and the block geometry it encodes become semver-load-bearing.
  Mitigant: the signature is already stable internally (used by `from_parts` and
  the tests), and the geometry is already implicitly public via the on-disk format.
- **Precondition footgun.** `repack` assumes validated inputs; passing mismatched
  `n_vectors`/`bits`/`dim` produces a wrong (not panicking) layout. `from_parts`
  validates before it repacks. The doc comment must steer callers to validate via
  `from_parts` first, or the maintainer may prefer a thin validating wrapper.
- **Alternative the maintainer might counter-propose:** instead of exposing the
  free function, add a `TurboQuantIndex::blocked_codes()` accessor (symmetric with
  `packed_codes()`), so embedders read the layout off a constructed index rather
  than recomputing it. That is *more* memory at load, not less — it defeats the
  "persist only packed" win — so the free `repack` is the one that serves the
  compact-storage use case. Worth stating explicitly to preempt the counter.
- **Relationship to proposal #3.** #3 parallelizes `repack`'s body. If the
  maintainer prefers to keep `repack` private and instead expose the parallel
  reconstruction some other way, #1 is moot. See the strategy note — #1 is the
  prerequisite; #3 is an internal-speed improvement to whatever #1 exposes.

## (d) Proposed issue/PR text

**Title:** `Expose pack::repack as pub for storage-backed embedders`

**Body:**

> Following up on #70 (the cache-fill / out-of-tree-storage path).
>
> `pack::repack` is `pub(crate)` (pack.rs). `pack` itself is already `pub mod`,
> and `TurboQuantIndex::packed_codes()` is public as of #204. The one gap for an
> embedder that stores turbovec codes in its own container is turning those
> packed codes back into the SIMD-blocked search layout without persisting the
> layout too.
>
> Concrete consumer: pg_turbovec stores only the row-major packed codes in
> PostgreSQL relation pages (halving the on-disk footprint vs. persisting the
> blocked layout as well) and recomputes the blocked layout once per backend at
> index-open via `repack`. With `repack` private, that isn't expressible against
> released turbovec — it's the one line we carry as a fork delta.
>
> Would you take a one-line visibility change `pub(crate)` → `pub` on `repack`,
> now that #142 added its input validation and #204 made `packed_codes()` public?
> It's a pure function of `(packed_codes, n_vectors, bits, dim)`; the doc would
> steer callers to construct through `from_parts` (which validates) when they
> don't already hold validated parts.
>
> Happy to write it up as a focused PR if you're open to it — flagging as an
> issue first per CONTRIBUTING. `[skip changelog]` would not apply; this is
> user-visible public surface, so it'd get an `## [Unreleased]` line under the
> Rust crate.

**Test plan (if invited to PR):** `cargo test -p turbovec --release` (no new
tests strictly needed — a doctest showing `packed_codes()` → `repack` round-trip
would satisfy the mutation gate by asserting the layout matches a `from_parts`
build).
