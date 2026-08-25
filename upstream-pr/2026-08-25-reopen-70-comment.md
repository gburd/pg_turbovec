# Upstream turbovec — proposed comment on #70 (reopen ask) + follow-up issues

**Status: DRAFT for Greg to review + post as `gburd`.** Upstream is
invitation-only for PRs and explicitly wary of cold AI-assisted PRs, so
the right move is a well-framed comment/issue with the design context —
NOT a cold PR. Ryan closed #70 inviting exactly this: *"if anything
doesn't cover the pg_turbovec cache-fill path in practice, comment and
I'll reopen."*

`gh` is authenticated as `gburd` with `repo` scope, so once approved:
`gh issue comment 70 --repo RyanCodrai/turbovec --body-file <this text>`
(or open focused issues). Left for Greg to send — posting in his name to
a third party shouldn't be automated.

---

## Comment to post on #70

Thanks for landing #204 + #210 — `from_parts` + the accessors +
`to_bytes`/`from_bytes` cover the round-trip half cleanly, and the
validated-constructor invariant (accepted index survives its own
write→load) is exactly what an embedder wants.

Now that I'm looking at porting pg_turbovec onto 1.0.0, three small
gaps remain on the **cache-fill / out-of-tree-storage** path. pg_turbovec
doesn't use turbovec's file I/O at all — it stores the quantized codes in
PostgreSQL's own relation pages (through PG's buffer manager) and
reconstructs a search-ready index per backend at index-open. So its needs
are specifically about *reconstructing from caller-owned bytes without a
turbovec file*:

1. **`pack::repack` is `pub(crate)` (pack.rs:59).** pg_turbovec persists
   only the row-major packed codes on disk (halving the footprint) and
   recomputes the SIMD-blocked layout once per backend at open via
   `repack`. With it private, an out-of-tree consumer can't rebuild the
   blocked layout from `packed_codes()`. Would you take a one-line
   visibility change (`pub(crate)` → `pub`) now that #142 added its input
   validation? `pack` is already `pub mod`.

2. **A borrowed / zero-copy reconstruction path.** `from_parts` takes
   `Vec<u8>` (owned). pg_turbovec reads codes straight out of pinned
   buffer-manager pages and would like to build a *read-only* search
   index over `&[u8]` without cloning the codes (they're the bulk of the
   index — ~40 GB at 100M×768d×4-bit). Is a `from_parts_borrowed` (or a
   `TurboQuantIndex<Cow<[u8]>>`-shaped) API something you'd consider, or
   should that stay a fork concern like `from_id_map_parts`? Framing it
   as a question, not a PR — happy to write up the exact shape in a
   focused issue if you're open to it.

3. **`make_rotation_matrix` removal (#344) + the v5 block-Hadamard
   rotation.** pg_turbovec (on the 0.9.0 line) calls the old free
   `rotation::make_rotation_matrix`; 1.0.0 replaced it with the
   `Rotation` struct and the v5 block-Hadamard k=2 rotation that
   "altered every encoded byte." That's a hard wire-format break for us
   (existing pg_turbovec indexes would need re-encoding), which is fine
   as a documented major migration on our side — but it means the port is
   a 2.0-scale effort, not a version bump. No ask here; just confirming my
   understanding so the two projects' formats don't silently diverge.
   (Your `convert` tool + the v5/v6→v7 path is a good precedent for how
   we'll handle the pg_turbovec-side migration.)

Everything else in #70 is genuinely covered — thanks again.

---

## Notes for the pg_turbovec side (not for upstream)
- If Ryan takes (1), pg_turbovec's Track-B port drops one fork patch.
- (2) is the load-bearing one for the cache-fill perf story; if declined,
  it stays a fork carry (the borrowed ReadOnlyIndex path).
- (3) confirms Track B is a pg_turbovec 2.0.0 (wire break, REINDEX or an
  offline converter per the HARD MANDATE).
