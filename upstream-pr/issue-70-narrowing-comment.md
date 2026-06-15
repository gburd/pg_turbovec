# Draft comment for turbovec issue #70 (narrowing scope after v0.9.0)

> **POSTED 2026-06-15:** https://github.com/RyanCodrai/turbovec/issues/70#issuecomment-4711296306

Post this manually — the GitHub MCP toolset available to the agent is
read-only (no issue-comment write). Paste into
https://github.com/RyanCodrai/turbovec/issues/70

---

Update from the `pg_turbovec` side, now that the fork tracks v0.9.0.

Thanks @ZeyadMohamad for re-implementing this against current `main`
with the TQ+ v3 fields and the #108 hardening — that's exactly the
right basis, and it's much more useful than my original v0.6.0 draft.
And thanks @faysou; the #68 caching work is complementary.

One scope note that should make a PR smaller and easier to accept: of
the three asks in the original issue, **`pg_turbovec` now only needs
the first.**

1. **`pub TurboQuantIndex::from_parts` + `packed_codes()` / `scales()`
   accessors** — still needed, and the single highest-value item. Any
   database-storage embedder (we use PostgreSQL relfile pages) wants to
   construct an index from already-decoded bytes without a tmpfile
   round-trip. `pg_turbovec`'s entire cache-fill path is built on this.

2. **`from_id_map_parts*` constructors on `IdMapIndex`** — we carry these
   in our fork, but they're arguably `pg_turbovec`-specific (they thread
   pre-baked SIMD-blocked layout + rotation + codebook through, which is
   our relfile-resident optimisation). I'm not sure they belong upstream
   unless another embedder wants the same "skip prepare on load" path.
   Happy to discuss; they may be better left as a fork concern.

3. **The `Read`/`Write` IO API (`write_to`/`load_from` etc.)** — we
   **no longer need this.** `pg_turbovec` reads straight from PG buffer
   pages via `from_parts`, never from `.tv`/`.tvim` files, so the
   streaming-IO half of the original issue is moot for us. It's only
   worth doing if another consumer wants in-memory `.tvim` (de)serialisation.

So if you're inclined to take anything, **#1 alone** (the `pub from_parts`
+ accessors that @ZeyadMohamad already implemented) closes the issue for
`pg_turbovec`'s purposes. The rest is optional.

Separately — thank you for #108. The pre-AVX2 scalar-fallback fix (the
perm0 de-interleave) was a silently-wrong-top-k bug that `pg_turbovec`
hit in production on a pre-AVX2 Xeon bench host; upgrading our fork to
v0.9.0 fixed it. The audit was real-world load-bearing for us.
