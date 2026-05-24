---
name: drift-check
description: Audit pg_turbovec docs/ against actual implementation. Flags stale test counts, version numbers, file structure, feature flags, and benchmark numbers that no longer match the code.
---

# pg_turbovec docs-drift check

This project ships a docs-heavy repo (`docs/ARCHITECTURE.md`, `docs/INDEXAM.md`,
`docs/PARITY_GAPS.md`, `docs/RECALL.md`, `docs/PG_VERSION_SUPPORT.md`,
`docs/ROADMAP_DECISIONS.md`, plus README + CHANGELOG). They drift fast.
This skill is the routine to keep them honest.

## When to run

- After tagging a release.
- Before tagging a release.
- After landing any change that touches: pgrx version, the test count, a
  PG version's support status, a benchmark number, a public type, or
  the file layout under `src/`.
- Periodically (every ~5 commits).

## Workflow

For each item in the checklist below, compare the documented value to
the actual value. If they disagree, edit the doc and surface the diff
in your reply (don't silently fix without telling the user — they
might want to fix the *code* instead).

### 1. Version numbers

```bash
grep -E "^version" Cargo.toml                              # source of truth
grep -E "default_version" pg_turbovec.control              # SQL surface
grep -E "^## \[" CHANGELOG.md | head                       # newest entry first
git tag --list 'v*' --sort=-creatordate | head -3          # last three tags
```

Expected: `Cargo.toml` == `pg_turbovec.control` == top of CHANGELOG.md
(once "Unreleased" stanza is removed) == latest tag.

> Note: `--sort=-v:refname` puts pre-release tags (`v1.0.0-rc.1`)
> *ahead* of their final `v1.0.0` because of how Git's version-sort
> handles SemVer pre-release suffixes. Use `--sort=-creatordate` for
> chronological order.

### 2. Test count

```bash
# pgrx mounts BOTH '#[test]' (regular Rust) and '#[pg_test]' (Postgres
# backend) under one binary at `cargo pgrx test`. Count both.
ANN=$(grep -rE '#\[(test|pg_test|pgrx::pg_test)\]' src/ | wc -l)
echo "actual annotations: $ANN"
# A few may be cfg-gated; the run reports the runnable set:
# Run quickly against pg16 (default) to get the truth-of-the-day:
#   cargo pgrx test pg16 2>&1 | grep -E "^test result: ok\.\s+[0-9]+ passed"
grep -ohE "\b[0-9]+/[0-9]+\s*(passing|tests|test result|cases)" \
    docs/*.md README.md CHANGELOG.md 2>/dev/null \
    | sort | uniq -c | sort -rn | head
```

The annotation count is an upper bound; the test runner prints the
actual passed/failed count after `--features pg<N> experimental_index_am
pg_test`. Documented numbers should match the runner output, not the
annotation count.

### 3. PG version support

```bash
grep -E '^pg[0-9]+ = ' Cargo.toml                          # what we claim
cat docs/PG_VERSION_SUPPORT.md | grep -E '^\| [0-9]+'      # what we document
ls ~/.pgrx                                                  # what we test
```

Cargo.toml's feature list must match `docs/PG_VERSION_SUPPORT.md`'s
table. The `.forgejo/workflows/test.yml` matrix must also match.

### 4. File layout vs ARCHITECTURE.md

```bash
ls src/
ls src/index/
grep -E '^- `src/' docs/ARCHITECTURE.md | head -40
```

Every `*.rs` file under `src/` should have a description in
`docs/ARCHITECTURE.md`. Any file mentioned in ARCHITECTURE.md that
doesn't exist anymore should be removed.

### 5. Feature flags

```bash
grep -A 20 "^\[features\]" Cargo.toml
grep -rE 'feature = "[^"]+"' README.md docs/USAGE.md docs/BUILDING.md
```

Documented build incantations like `cargo build --features pg17` or
`--no-default-features --features "pg17 experimental_index_am"` must
work as written. Run a few to verify.

### 6. Benchmark numbers (the most fragile section)

`docs/RECALL.md` has the canonical numbers. Cross-check:

- README "headline" table (typically near the top) against
  `docs/RECALL.md` section headers.
- `benches/results/*.json` files referenced in RECALL.md must exist
  and contain the cited numbers.
- "post-fix" / "after commit X" claims should reference a real
  short-SHA reachable from `main`.

```bash
ls benches/results/
grep -E "(commit|p50|R@10|warm|cold).*[0-9]" docs/RECALL.md README.md \
    | head -20
```

### 7. Roadmap claims

```bash
cat docs/ROADMAP_DECISIONS.md | grep -E '^### [12]\. |^- \*\*' | head
git log --oneline -20
```

If a "future work" item from `ROADMAP_DECISIONS.md` has actually
landed (look for matching commit messages), move it from the future
section to the "What we shipped" section.

### 8. CHANGELOG vs git log

```bash
git log --oneline --since="$(git log -1 --format=%ai $(git tag --sort=-v:refname | head -2 | tail -1))" | head -20
grep -A 200 '^## \[' CHANGELOG.md | head -100
```

Every commit since the previous tag should be reflected in the
CHANGELOG entry for the current/next version. Bench, docs, and
chore commits can be summarised in one line; user-visible behaviour
or fixes need their own bullet.

### 9. Vendored dependencies

```bash
ls vendor/ 2>/dev/null
cat vendor/*/PATCH_NOTES.md 2>/dev/null | head -10
```

Anything in `vendor/` should have a `PATCH_NOTES.md` describing
exactly what was changed vs upstream and the upstreaming status.

### 10. CI matrix

```bash
cat .forgejo/workflows/test.yml | grep -A 1 'matrix:'
cat .github/workflows/test.yml | grep -A 1 'matrix:' 2>/dev/null
grep -E '^pg[0-9]+ = ' Cargo.toml
```

Every PG version in `Cargo.toml`'s features must appear in **both**
CI matrices.

## Output format

After auditing, produce a single markdown report:

```
# pg_turbovec docs drift report — <YYYY-MM-DD>

## Drift found

- **<file>:<line> — <one-line description>**
  Documented: `<old value>`
  Actual:     `<new value>`
  Suggested fix: <action>

## Drift fixed

(any drift the agent fixed in-place, with the diff)

## All clear

(sections that audited clean)

## Suggestions

(non-drift improvements the audit surfaced)
```

Commit any in-place fixes with a `docs(drift):` prefix and reference
the commits / values that introduced the drift.

## Anti-patterns

- **Don't silently update benchmark numbers in docs without re-running
  the bench.** A "drift fix" that overwrites a bench number with
  whatever the code currently does, without re-measuring, is fraud.
- **Don't remove a doc section that describes deprecated behaviour
  without checking whether real users still rely on it.** The
  `docs/MIGRATING_FROM_PGVECTOR.md` cookbook describes user-facing
  behaviour and changing it is a breaking change.
- **Don't drop a PG version from the docs without dropping it from
  CI and `Cargo.toml` too.** That's the failure mode that gave us
  v1.0.0 → v1.0.1.
