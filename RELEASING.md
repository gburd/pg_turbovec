# Releasing `pg_turbovec`

This document describes how to cut a release of `pg_turbovec`. The
process is intentionally manual so each release gets a real human
reading the diff, the test output, and the changelog.

## Versioning

`pg_turbovec` follows [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`:

* **Patch** (`1.0.0` → `1.0.1`) — bug fixes that don't change the
  public SQL surface or the on-disk format.
* **Minor** (`1.0.0` → `1.1.0`) — additive SQL surface changes
  (new functions, new GUCs, new reloptions). Old applications must
  continue to work without modification.
* **Major** (`1.x` → `2.0.0`) — breaking SQL changes, on-disk
  format changes that require dump/restore, or removed APIs.

Pre-1.0 release candidates are cut as `1.0.0-rc.N`; we do **not**
consider them compatible with each other or with 1.0.0 final until
the on-disk format is frozen.

## Pre-flight checklist

Before tagging:

* [ ] `cargo pgrx test pg16` — full feature build, every test green.
* [ ] `cargo pgrx test pg16 --no-default-features --features pg16`
  — kernel-only build, every test green.
* [ ] `cargo clippy --features pg16 --tests -- -D warnings` — clean.
* [ ] `cargo clippy --no-default-features --features pg16 --tests
  -- -D warnings` — clean.
* [ ] `cargo fmt --all -- --check` — formatted.
* [ ] `cargo bench --bench distance --no-default-features
  --features pg16 --no-run` — benches still compile.
* [ ] `cargo bench --bench recall   --no-default-features
  --features pg16 --no-run` — recall bench still compiles.
* [ ] `bash scripts/drift-check.sh` — zero drift; the script
  enforces version alignment, PG-version-matrix consistency,
  bench-result references, broken-link detection, vendor patch
  notes presence, wire-format-version compatibility for patch
  releases, and PARITY_GAPS scoreboard freshness.
* [ ] **Read `docs/PARITY_GAPS.md` § "Performance gaps" line by
  line.** drift-check §8 catches `TBD` and "we lose Nx" without a
  phase qualifier, but it cannot catch a row whose number is
  numerically wrong (e.g. "~70 ms" when the latest measurement is
  90 ms). Eyeball every row's number against the most recent
  bench JSON in `benches/results/`. If a row hasn't been
  re-measured since the relevant phase landed, either re-measure
  or annotate the row with the bench commit it last reflected.
* [ ] **Read `README.md` table at the top.** Same eyeball test for
  the headline numbers; the README is what users see first.
* [ ] `CHANGELOG.md` entry written with phase label, summary,
  highlights, and a `[<version>]` link at the bottom.
* [ ] If this is a minor or major bump that changed the on-disk
  format, the CHANGELOG entry has a Migration section and
  `docs/UPGRADING.md` has a new row in its migration matrix.
* [ ] `README.md` status banner updated to reflect the new state
  (test count, known limitations, RC vs final).
* [ ] Any GUC range or default that changed in this release is
  documented in `docs/USAGE.md`.

## Release steps

1. **Bump version**:

   * `Cargo.toml`: `version = "X.Y.Z"`.
   * `pg_turbovec.control`: `default_version = 'X.Y.Z'`.
   * Run `cargo build --features pg16` to refresh `Cargo.lock`.

2. **Regenerate the SQL schema** so consumers can install from the
   tarball without running pgrx:

   ```bash
   cargo pgrx schema --features pg16 \
       --out sql/pg_turbovec--X.Y.Z.sql
   ```

   Commit the regenerated `.sql` file.

3. **Append the migration script** if this release changes the SQL
   surface. Naming is `migrations/pg_turbovec--<from>--<to>.sql`.

4. **Update `CHANGELOG.md`**:

   * Move the new section's heading from `[X.Y.Z] — Unreleased` to
     `[X.Y.Z] — YYYY-MM-DD`.
   * Append the new version to the link list at the bottom.

5. **Commit the release**:

   ```bash
   git add Cargo.toml Cargo.lock pg_turbovec.control \
           CHANGELOG.md README.md \
           sql/pg_turbovec--X.Y.Z.sql \
           migrations/pg_turbovec--*--X.Y.Z.sql
   git commit -m "Release vX.Y.Z"
   ```

6. **Tag and push** — this TRIGGERS the automated publish:

   ```bash
   git tag -s -m "Release vX.Y.Z" vX.Y.Z
   git push origin main
   git push origin vX.Y.Z
   git push github main
   git push github vX.Y.Z
   ```

   Pushing the `vX.Y.Z` tag to Codeberg (`origin`) fires
   `.forgejo/workflows/release.yml` on the **self-hosted Forgejo
   runner**, which:
   1. compiles + runs `scripts/drift-check.sh` (the release gate — the
      full `cargo pgrx test` matrix runs separately on the GitHub
      mirror, so a tag should only be cut once that matrix is green),
   2. builds the PGXN **source** dist zip via `scripts/make-dist.sh`
      (renders `META.json` from `META.json.in`, runs `cargo pgrx
      schema` to generate the install SQL, zips a PGXN-layout source
      archive),
   3. attaches the zip to a Codeberg release,
   4. uploads the dist to **PGXN** (`manager.pgxn.org/upload`),
   5. drafts + submits a **postgresql.org news** announcement (feeds
      pgsql-announce) via `ci/announce.sh`.

   The GitHub mirror does NOT publish to PGXN or announce — those run
   ONCE, on the Codeberg side.

### One-time setup for the automated publish

The release workflow needs a **self-hosted Forgejo runner** (Codeberg's
hosted runners have a 10-minute job cap that a pgrx build blows past)
and these repo secrets (Codeberg → Settings → Actions → Secrets):

| Secret | Purpose | Required? |
|---|---|---|
| `RELEASE_TOKEN` | Codeberg token, repo/release write (attach the dist) | yes |
| `PGXN_USER`, `PGXN_PASSWORD` | PGXN Manager credentials (username/password — PGXN has no API tokens) | for PGXN upload |
| `PGORG_USER`, `PGORG_PASSWORD` | postgresql.org password-login account | for the announce |
| `PGORG_ORG_ID`, `PGORG_EMAIL_ID`, `PGORG_TAGS` | postgresql.org approved-org id, confirmed-email id, space-separated NewsTag ids | for the announce |

Register the runner with `bash scripts/install-forgejo-runner.sh` (see
`docs/CI.md`). If a secret group is unset, that step no-ops cleanly
(PGXN skips; announce prints the drafted text to paste by hand at
<https://www.postgresql.org/account/news/new/>) — a release never fails
for lack of publish config. **`turbovec` is a pgrx extension, so the
PGXN dist is a SOURCE archive built with `cargo pgrx install`, NOT a
`pgxn install`-able package** (PGXN's model assumes a PGXS `Makefile`;
pgrx has none). It is published for discoverability + version-pinning;
the caveat is stated in the dist's README and the Codeberg release
notes.

If the self-hosted runner is NOT yet registered, do the publish steps
by hand: run `bash scripts/make-dist.sh pg16` locally, attach the zip
to a Codeberg release, `curl` it to PGXN (see the workflow's
"Publish to PGXN" step for the exact command), and run
`bash ci/announce.sh X.Y.Z` to get the announcement draft.

## crates.io publish (optional, post-1.0.0)

`pg_turbovec` is **not** currently published on crates.io. The
extension is consumed via `cargo pgrx` against a checkout, and a
crates.io publish would publish the cdylib stub and the kernels
crate but not the SQL schema, which is the actually-useful artifact.
If we ever do publish, the checklist is:

* [ ] `cargo publish --dry-run` — manifest validates, all required
  files are in the tarball.
* [ ] `Cargo.toml` `description`, `license`, `repository`,
  `homepage`, `documentation`, `keywords`, `categories`,
  `rust-version` all populated. (They are, as of 1.0.0-rc.2.)
* [ ] `README.md` rendered with absolute links so it makes sense
  when displayed on crates.io.
* [ ] `LICENSE` is present at the crate root.
* [ ] No path/git dependencies — only registry dependencies.
* [ ] `cargo doc --no-deps` builds without errors so docs.rs gets
  a clean build.
* [ ] All bench harnesses gated behind dev-dependencies.
* [ ] `cargo publish` (after `cargo login`).

If we *do* publish, the version published is the kernels-only build
(no `experimental_index_am` feature in the default published set),
because `IndexAmRoutine` makes no sense in a non-pgrx context.

## Hot-fix release

For a hot-fix off a released tag:

```bash
git checkout vX.Y.Z
git checkout -b release/X.Y
# apply the fix, run the pre-flight checklist
git commit
git tag -s -m "Release vX.Y.(Z+1)" vX.Y.(Z+1)
git push origin release/X.Y vX.Y.(Z+1)
```

Then forward-port the fix to `main` if it isn't already there.

## Yanking a release

If a published release turns out to be broken:

1. Mark it pre-release on Codeberg / GitHub with a banner pointing
   at the fixed successor.
2. Add a `WARNING` line to the affected `[X.Y.Z]` section in
   `CHANGELOG.md` explaining what was wrong and which release
   supersedes it.
3. Do *not* delete the tag — downstream consumers may already have
   it pinned.
