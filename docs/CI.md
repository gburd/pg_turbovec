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

**Manual one-time step:** Codeberg ships Forgejo Actions but
disables it by default for new repos. To turn it on:

1. Visit <https://codeberg.org/gregburd/pg_turbovec/settings>.
2. Scroll to **Advanced Settings → Actions**.
3. Tick "Enable Repository Actions".
4. Save.

Verify via API (after enabling):

```bash
curl -s "https://codeberg.org/api/v1/repos/gregburd/pg_turbovec" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['has_actions'])"
# Expected: True
```

The workflow file at `.forgejo/workflows/test.yml` already exists and
uses fully-qualified Docker image names (`docker.io/library/debian:bookworm-slim`,
`docker.io/library/rust:1-bookworm`) so it works on the default
Forgejo runner image without needing a custom registry. Until
Actions is flipped on by hand, the GitHub mirror at
`gburd/pg_turbovec` carries the canonical CI green badge.

## Status

- **GitHub mirror CI:** ✅ green (verified on commits `5dbe3aa`, `a291219`,
  `9046f2c`; drift-check + 6 PG versions pass on every push).
- **Codeberg Actions:** ⚠️ enabled at the repo level (`has_actions: True`)
  but the workflow run sits in `status: waiting` because no Forgejo
  runner has picked it up. Codeberg's [shared runner pool](https://docs.codeberg.org/ci/actions/) is opt-in;
  you can either:
  - **Apply for shared-runner access** at <https://codeberg.org/Codeberg-CI/request-access>
    (free, requires a brief review for abuse prevention).
  - **Run your own runner.** One-token install via the new
    `scripts/install-forgejo-runner.sh`:

    ```bash
    # 1. Visit https://codeberg.org/gregburd/pg_turbovec/settings/actions/runners
    # 2. Click "Create new runner" → copy the token.
    # 3. ssh to the runner host (arnold is a good pick) and run:
    export FORGEJO_RUNNER_TOKEN=<paste-the-token>
    bash scripts/install-forgejo-runner.sh
    ```

    Idempotent. Drops a static `forgejo-runner` binary into
    `~/.local/share/forgejo-runner/`, registers it against the
    repo, and creates a systemd user unit so it auto-starts on
    boot. Next push to `origin/main` triggers CI.

  Until one of those lands, the GitHub mirror at `gburd/pg_turbovec`
  carries the canonical green badge. Both workflows are kept in
  lockstep by `scripts/drift-check.sh §10` so they don't diverge.

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
