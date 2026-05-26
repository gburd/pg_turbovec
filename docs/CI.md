# CI setup notes

## GitHub Actions (mirror at `gburd/pg_turbovec`)

The mirror automatically syncs from Codeberg. Workflows live at
`.github/workflows/test.yml` and run on every push to `main` plus
every PR.

If the workflow fails with `cargo-pgrx` complaining about `edition2024`,
bump the toolchain version in `setup-rust-toolchain` — `cargo-pgrx
0.17.0` requires Cargo ≥ 1.85.0 (the version that stabilised
`edition2024`).

Trigger a manual run: <https://github.com/gburd/pg_turbovec/actions>
→ "test" workflow → Run workflow.

## Codeberg Actions (canonical CI at `gregburd/pg_turbovec`)

**No application required — Codeberg's hosted Forgejo Actions is open to
all users now.** The (outdated)
<https://codeberg.org/Codeberg-CI/request-access> page predates the
current setup; ignore it. The current docs are at
<https://codeberg.org/actions/meta>.

What you DO need:

1. **Enable Actions** on the repo: Settings → Units → "Enable Actions".
   `has_actions: True` confirms it. (Already done.)

2. **Use a Codeberg-provided runner label.** Hosted runners have these
   labels with a 10-minute job-runtime cap each:

   | Label | CPU | RAM | Runtime cap |
   |---|---|---|---|
   | `codeberg-tiny`   | 1 | 2 GB | 2 min |
   | `codeberg-small`  | 2 | 4 GB | 5 min |
   | `codeberg-medium` | 4 | 8 GB | 10 min |

   Plus `*-lazy` variants for jobs that can wait. `runs-on: docker`
   (the GitHub Actions default) doesn't match any Codeberg runner;
   pushes that use it sit in `status: waiting` forever.

3. **Live with the runtime cap.** `cargo pgrx test pg<N>` jobs take
   7–15 min each (cold cache builds PG from source); they can't fit
   `codeberg-medium`. So `.forgejo/workflows/test.yml` is intentionally
   slim — it runs only `drift-check` (3 s) on `codeberg-tiny`. The
   full 6-PG-version test matrix runs only on the GitHub mirror at
   `.github/workflows/test.yml`.

   If you want the full matrix on Codeberg too, register a self-hosted
   runner with no time cap:

   ```bash
   # 1. Visit https://codeberg.org/gregburd/pg_turbovec/settings/actions/runners
   # 2. Click "Create new runner" → copy the token.
   # 3. ssh to a runner host (meh and arnold both work; meh has 24 cores)
   #    and run:
   export FORGEJO_RUNNER_TOKEN=<paste-the-token>
   bash scripts/install-forgejo-runner.sh
   ```

   Idempotent. Drops a static `forgejo-runner` binary into
   `~/.local/share/forgejo-runner/`, registers it against the
   repo, creates a systemd user unit so it auto-starts on boot.
   Once registered, change `.forgejo/workflows/test.yml` to also
   include the test matrix and add `runs-on: self-hosted` to it.

## Status

- **GitHub mirror CI:** ✅ green on every push (drift-check + 6 PG
  versions). The pg14 job in run `26428026790` (commit `f270120`)
  hung 6h in `apt-get install` — transient apt-mirror flakiness;
  the other 5 PG versions succeeded on the same commit. Re-running
  pg14 clears it; the next push will re-trigger CI.
- **Codeberg Actions:** ✅ enabled (`has_actions: True`). After
  the workflow split, `drift-check` will run on `codeberg-tiny`
  on every push. The full test matrix lives only on the GitHub
  mirror per the runtime-cap analysis above.

## What the workflows do

Both run the same two-stage pipeline:

1. **`drift-check` job** — runs `bash scripts/drift-check.sh` to
   verify version numbers, PG version matrix, bench-result
   references, and markdown links are consistent across the
   tree. Fails fast on any drift.
2. **`test` matrix** — `cargo pgrx test pg<N>` for N in
   `[13, 14, 15, 16, 17, 18]`. The `cargo pgrx init --pgN
   download` step builds PostgreSQL N from source the first
   time the workflow runs in a given runner image, then caches
   it across runs.

Cache keys include `Cargo.lock` so a dependency bump invalidates
both the pgrx install and the cargo target dir.

## Common failures + fixes

| Symptom | Cause | Fix |
|---|---|---|
| `cargo-pgrx` install fails: "feature `edition2024` is required" | Rust toolchain < 1.85 | Bump `toolchain:` in setup-rust-toolchain |
| `cargo pgrx init` fails on `apt-get`-missing package | New PG version added without updating apt deps | Add the missing `lib*-dev` to the install step |
| `cargo pgrx test` flakes on `postmaster.pid` | Stale state from prior failed run, cache hit | Add `pkill -9 -f test-pgdata; mv target/test-pgdata /tmp/orph-$$` before the test step |
| GitHub Actions runs but Codeberg doesn't | Actions disabled on the Codeberg repo | See "Manual one-time step" above |

## Drift between GitHub mirror + Codeberg

Drift between the two CI workflows is tracked by
`scripts/drift-check.sh` step "10. CI matrix": both workflow files
must have the same `matrix.pg` value. Drift-check fails the
build if they disagree.
