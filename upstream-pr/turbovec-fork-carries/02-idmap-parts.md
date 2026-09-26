# Upstream proposal #2 — `IdMapIndex` parts API (accessors + `from_id_map_parts`)

**Fork carry:** `f29a2f2` (branch `pgtv-2.0.0-port`, on top of turbovec 1.0.0 `ccab9f3`)
**Target file:** `turbovec/src/id_map.rs`
**Process note:** issue-first. This is the **riskiest** of the three to get
accepted upstream (see strategy note) — the maintainer already signalled on #70
that the `from_id_map_parts*` family is "arguably pg_turbovec-specific." Frame it
as a design question, not a PR, and be ready for a "no, keep it a fork concern."

---

## (a) Motivation — from turbovec's own perspective

`TurboQuantIndex` gained a public parts round-trip in #204/#210 (`from_parts` +
`packed_codes()`/`scales()`/`tqplus_shift()`/`tqplus_scale()`). `IdMapIndex`
wraps a `TurboQuantIndex` plus a `slot_to_id` table, but after 1.0.0 sealed its
`inner` field it has **no** equivalent parts surface — you can round-trip the
inner quantizer but not the id-mapped index that most consumers actually hold.

The general utility: any embedder that persists an `IdMapIndex` into its own
storage (not a turbovec `.tvim` file) needs to read the id table back out and
reconstruct a mutable, search-ready `IdMapIndex` from decoded parts. That's the
same shape #204 already blessed one level down — this proposal is "finish the
parts API at the `IdMapIndex` layer so the two constructors are symmetric."
The codebook and rotation in 1.0.0 are deterministic functions of
`(bit_width, dim)`, so an embedder only has to persist codes + scales + the id
table + the (possibly empty) TQ+ calibration arrays; `prepare` rebuilds the rest.

## (b) Exact diff summary

Purely additive; `+66` lines on `impl IdMapIndex`, no existing signature touched.
Five thin borrowing accessors that delegate to the inner `TurboQuantIndex` (plus
the already-owned `slot_to_id`), and one constructor:

```
pub fn packed_codes(&self) -> &[u8]        // -> self.inner.packed_codes()
pub fn scales(&self) -> &[f32]             // -> self.inner.scales()
pub fn slot_to_id(&self) -> &[u64]         // -> &self.slot_to_id
pub fn tqplus_shift(&self) -> &[f32]       // -> self.inner.tqplus_shift()
pub fn tqplus_scale(&self) -> &[f32]       // -> self.inner.tqplus_scale()

pub fn from_id_map_parts(
    bit_width, dim, n_vectors,
    packed_codes: Vec<u8>, scales: Vec<f32>, slot_to_id: Vec<u64>,
    tqplus_shift: Vec<f32>, tqplus_scale: Vec<f32>,
) -> std::io::Result<Self>
```

`from_id_map_parts` builds the inner index via the already-public
`TurboQuantIndex::from_parts` (so it inherits #204's validation), maps its error
into `io::Error(InvalidData)`, then calls the existing private
`from_index_and_ids(inner, slot_to_id)`. `dim == 0` maps to `None` (the
unset-dim convention). TQ+ arrays may be empty (identity/uncalibrated).

## (c) API-stability / maintenance concerns the maintainer would raise

- **This is the widest new surface of the three** — five accessors plus a
  seven-argument constructor. Every one is a semver commitment, and the arg list
  is exactly the kind of positional-boolean-adjacent signature that ages badly
  (eight scalars, two of which are "usually empty"). The maintainer may want a
  `Parts` struct or a builder instead of a wide positional constructor.
- **Duplication of the `TurboQuantIndex` surface.** The five accessors are pure
  delegations. The maintainer may reasonably ask "why not expose `inner()`
  once?" — a single `pub fn inner(&self) -> &TurboQuantIndex` would give all five
  accessors for free and be a smaller commitment. Counter-point: exposing `inner`
  leaks the composition and lets callers reach past the id layer; five named
  read-only accessors are the more conservative surface. Worth offering both and
  letting the maintainer pick.
- **`std::io::Error` in a non-IO constructor.** `from_id_map_parts` returns
  `io::Result` purely to match the existing load path's error type; a maintainer
  building a fresh API might prefer a domain error enum. It's cosmetic but it's
  the kind of thing that gets bikeshedded.
- **The #70 signal.** On #70 the maintainer's stated read was that the
  `from_id_map_parts*` family "threads pre-baked SIMD-blocked layout + rotation +
  codebook through, which is our relfile-resident optimisation … may be better
  left as a fork concern." Note that the 1.0.0-era carry is *narrower* than that
  older description — it no longer threads a pre-baked blocked layout (that's
  proposal #1's `repack`), only the deterministic parts — which weakens the
  "too pg_turbovec-specific" objection. Lead with that narrowing.
- **Mutation gate.** A round-trip test (`add_with_ids` → read parts →
  `from_id_map_parts` → identical `search` results) is needed, and it must
  assert the reconstructed index *searches identically*, not merely constructs,
  or the mutation gate will flag the constructor body.

## (d) Proposed issue/PR text

**Title:** `Symmetric parts API on IdMapIndex (accessors + from_id_map_parts)`

**Body:**

> Follow-up to #70 / #204. #204 gave `TurboQuantIndex` a public parts round-trip
> (`from_parts` + the four accessors). `IdMapIndex` wraps a `TurboQuantIndex` +
> a `slot_to_id` table but has no matching surface since 1.0.0 sealed `inner`,
> so an embedder that stores an id-mapped index in its own container can round-
> trip the quantizer but not the id table.
>
> Proposing the symmetric surface at the `IdMapIndex` layer:
> - borrowing accessors `packed_codes`/`scales`/`slot_to_id`/`tqplus_shift`/
>   `tqplus_scale` (delegating to the inner index; `slot_to_id` is owned here),
> - `from_id_map_parts(bit_width, dim, n_vectors, packed_codes, scales,
>   slot_to_id, tqplus_shift, tqplus_scale)`, building the inner index via the
>   existing public `from_parts` (inherits its validation) and reusing the
>   private `from_index_and_ids`.
>
> Note this is narrower than what I described on #70 — it no longer threads a
> pre-baked blocked layout (that's the separate `repack` ask), only the parts
> that are deterministic from `(bit_width, dim)`. So it's "the same parts round-
> trip #204 shipped, one layer up," not a bespoke optimisation hook.
>
> Two open design choices I'd defer to you on: (1) wide positional constructor
> vs. a `Parts` struct / builder; (2) five named accessors vs. a single
> `inner(&self) -> &TurboQuantIndex`. Happy either way. Flagging as an issue
> per CONTRIBUTING; if this is better left a fork concern, that's a fine answer
> and closes it for me.

**Test plan (if invited to PR):** `cargo test -p turbovec --release` plus a new
`add_with_ids → parts → from_id_map_parts → search-identical` round-trip test;
`## [Unreleased]` changelog line under the Rust crate (public surface).
