# Upstream submission strategy — turbovec fork carries → upstream

Drafts: `/tmp/upstream_pr_1_pub_repack.md`, `/tmp/upstream_pr_2_idmap_parts.md`,
`/tmp/upstream_pr_3_parallel_repack.md`.

Fork: `gburd/turbovec@pgtv-2.0.0-port` = upstream 1.0.0 (`ccab9f3`) + exactly
three additive deltas. Goal: land them upstream so the fork can be retired.

## The process constraint shapes everything

turbovec's `CONTRIBUTING.md` is **issue-first, invitation-only for PRs**, and
explicitly wary of cold AI-assisted PRs ("when the cognitive load of review
exceeds the cognitive load of writing the change, the PR is a net loss").
`CODEOWNERS` = `* @RyanCodrai`; only he merges to `main`. There is a changelog
gate (any shipped-code change must add an `## [Unreleased]` line) and a mutation
gate (a test that passes on the un-fixed code fails CI). Two automated
escape-hatch markers exist (`[skip changelog]` / `[skip mutants]`, whole-line,
outside code fences) but neither applies to us — #1/#2 are new public surface,
#3 is a behaviour-preserving perf change that the gate deliberately still covers.

**Therefore: these go up as GitHub _issues_ first, framed as design questions
with the pg_turbovec context, NOT as cold PRs.** The natural home is the closed
`#70` thread (cache-fill / out-of-tree-storage), which the maintainer invited
reopening on. There is prior, healthy interaction history here: upstream already
landed `from_parts` + accessors (#204/#210) partly in response to this exact
consumer, and the pg_turbovec side has posted well-received context on #70
before. That relationship is the asset; don't spend it on a cold PR.

## (b) Recommended strategy: bundle vs. separate, and order

**Separate issues, not one bundle** — CONTRIBUTING says "one logical change per
PR; refactors get their own PR." The three are genuinely different logical
changes (a visibility change, a new API surface, a perf change) and have
different acceptance odds; bundling drags the easy wins down to the pace of the
risky one and violates the stated convention.

But **group them as two conversations**, because #1 and #3 are coupled (both are
`repack` in `pack.rs`):

1. **First: #3 (parallel repack) + #1 (pub repack) as a linked pair, #3
   leading.** #3 is the strongest card — pure perf, byte-identical, mirrors
   existing in-file machinery, benefits turbovec's own load path independent of
   any embedder. Lead with it to establish the `repack`-is-hot framing on its own
   merits. Then #1 rides in as "…and while repack is in focus, expose it so
   storage-backed embedders get this speedup at load too." If the maintainer
   takes #3, #1 becomes an easy yes; if he takes #1 first, #3 is the obvious
   follow-up. Either order within the pair works; the pairing is the point.
2. **Second, separately: #2 (IdMapIndex parts API).** Different file, different
   (wider) surface, and the maintainer already flagged this family as maybe
   "a fork concern" on #70. Send it after #1/#3 have (re)built goodwill, framed as
   "finish the #204 parts round-trip one layer up," and lead with the fact that
   the 1.0.0-era carry is *narrower* than what he saw on #70 (no pre-baked
   blocked layout threaded through anymore). Be ready to accept "keep it a fork
   concern" as a clean answer.

## Does #3 landing obviate #1?

**No — they're orthogonal, and #1 is still needed even if #3 lands.**
- #1 is *visibility*: can an out-of-tree consumer **call** `repack` at all?
- #3 is *speed*: how fast does `repack` run once called?
- If #3 lands but `repack` stays `pub(crate)`, pg_turbovec still can't call it —
  it would get a faster *internal* repack (inside turbovec's own `from_parts`),
  but pg_turbovec deliberately persists **only** packed codes and calls `repack`
  itself at open; that call site requires #1's `pub`.
- Conversely, #1 without #3 works but is slow at the cold-open (the 1766 ms
  floor). So for pg_turbovec's specific "persist packed only, repack at open"
  design, **#1 is the load-bearing one and #3 is the speedup on top.**
- The only way #3 fully obviates #1 is if the maintainer counter-proposes
  "keep `repack` private, but have `from_parts` build the blocked layout in
  parallel and expose it via `blocked_codes()`." That *would* let pg_turbovec
  drop both carries — but at the cost of holding the blocked layout resident at
  load (more memory, defeating the persist-packed-only win). Flag this as the
  likely counter and state why the free `repack` is preferred for compact
  storage. (This is the one real "bundling insight": #1 exists **because**
  pg_turbovec wants the blocked layout *out* of memory, not persisted.)

## (c) Riskiest carry to get accepted, and why

**#2 (IdMapIndex parts API) is the riskiest**, for three compounding reasons:
1. **Widest surface** — five accessors + a 7-arg positional constructor, every
   item a semver commitment. #1 is one visibility bit; #3 is zero public surface.
   #2 is a whole new API the maintainer has to own forever.
2. **Prior explicit hesitation** — on #70 the maintainer's stated read was that
   `from_id_map_parts*` "may be better left as a fork concern." That's a soft no
   already on record. (Mitigant: the 1.0.0 carry is narrower than what he saw.)
3. **Design bikeshed surface** — wide positional ctor vs. `Parts` struct,
   five accessors vs. one `inner()`, `io::Error` in a non-IO path. Each is a
   legitimate objection that stalls acceptance even if the intent is agreed.

**Least risky: #3** (byte-identical perf, existing machinery, turbovec's own
benefit). **Middle: #1** (trivial diff, but adds a permanent public commitment
and depends on the maintainer agreeing embedders should touch `repack` directly).

## (d) What could not be determined

- **Whether the maintainer will accept ANY new public API given the
  invitation-only stance.** #204 landing suggests yes when the issue-side context
  is strong, but that's inference, not confirmation.
- **The live state of issue #70** (open/closed, latest comments) — read from
  local notes/drafts (drafts dated 2026-06-15 and 2026-08-25 reference it as
  closed-and-reopenable), not verified against github.com in this session. Check
  before posting; the framing assumes #70 is the thread to reopen.
- **Whether `#142` / `#204` / `#210` numbers are exactly right** — taken from the
  pg_turbovec-side draft comments, not cross-checked against the upstream issue
  tracker. Verify issue numbers before citing them upstream.
- **The 250k×1024d "~6 s → ~250 ms" micro-number** — cited from the task brief /
  commit message; the *end-to-end* cold-scan 1766→566 ms is independently
  confirmed from `benches/results/rebench_20260925/coldscan_*.log`. Re-run the
  isolated repack micro-bench on an AVX2 host (arnold) before quoting the 24×
  micro-figure in a public PR; per AGENTS.md, meh/rv are correctness-only, not
  latency.
- **Which upstream release these would target** — 1.0.0 is current `main`; no
  visibility into an in-flight next release that might move `pack.rs`.
