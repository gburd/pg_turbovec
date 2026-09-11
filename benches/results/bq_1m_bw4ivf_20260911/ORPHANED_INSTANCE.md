# Orphaned instance i-064245e785d2d3a57 — billing assessment

**Account 292759875395 (`bene`) EXPIRED Fri Sep 11 12:03:00 UTC 2026.**
The instance launched 11:15:35 UTC, so it ran ~48 minutes inside the account's
remaining life.

## Is it still costing money?

**Almost certainly not, and not something I can act on either way.** When a
burner/sandbox account's credentials are revoked or the account is closed:

- I cannot reach it — every key for 292759875395 is invalid (verified `bene`
  plus `chiuso`/`lala` from the older backup; all `InvalidClientTokenId`).
- `lava` is account **769093516156**, a different account, and returns
  `InvalidInstanceID.NotFound` for that ID — as expected, not a bug.
- So no session on this box can terminate it. If the account is closed, AWS
  terminates its resources as part of closure; if it is merely
  credential-revoked, only the account owner can act.

**Worst-case exposure is bounded and small**: ~48 min of `r7i.8xlarge`
on-demand ≈ **$1.70** if billing stopped at expiry, and the run had already
finished by then.

## Nothing of value is lost

Every artefact was pulled to the local box before access was lost and is
archived in-repo at `benches/results/bq_1m_bw4ivf_20260911/`:
JSON (48 configs), sweep.log, gates.log, pass1_buildmem.log, rss_pass1.tsv.gz,
AGENT_NOTES.md. The measurement is complete and published.

## For Greg, if the account is NOT closed

These would need cleanup from an owner session in 292759875395 / us-east-2:
- instance `i-064245e785d2d3a57`
- security group `sg-0e1805e73f55d6cc7`
- key pair `pgtv-bw4ivf-20260911-111509`

## Going forward

New burner is **`lava`** → account **769093516156**, region `us-east-2`,
already configured in `~/.aws/config`. AGENTS.md updated so the next dispatch
uses it and does not repeat this. Note there is an untagged `i4i.metal` running
in `lava` that is NOT mine — per the touch-only-your-own-tag rule, leave it be.
