# Contributing to pg_turbovec

Thanks for your interest in `pg_turbovec`. This is a young project
and every patch helps.

## Development setup

You need a Rust toolchain (≥ 1.85) and `cargo-pgrx` matching the
version pinned in `Cargo.toml` (currently 0.17.0).

```bash
# One-time:
cargo install --locked cargo-pgrx --version 0.17.0
cargo pgrx init                # bootstraps a private PostgreSQL cluster
                                # under ~/.pgrx for development

# Build & install into the dev cluster:
cargo pgrx install --release   # default Postgres major (pg17)
cargo pgrx install --pg-config $(which pg_config)   # alternative

# Open a psql session against the dev cluster:
cargo pgrx run pg17

# Inside psql:
CREATE EXTENSION pg_turbovec;
SELECT '[1, 2, 3]'::turbovec.tvector <=> '[3, 2, 1]'::turbovec.tvector;
```

## Testing

```bash
# pgrx unit tests (boots a temp cluster).
cargo pgrx test pg17

# Stand-alone Rust tests (parser / pure-function code paths).
cargo test
```

When you change the public SQL surface, regenerate the migration
mirror so reviewers see the change:

```bash
cargo pgrx schema --target-dir target/pgrx-schema \
    > migrations/001_pg_turbovec_v0.1.0.sql
```

## Commit style

- Imperative subject ≤ 72 chars.
- Body explains *what* and *why*, not *how*.
- Reference the relevant Phase from `docs/ARCHITECTURE.md`.

## Coding conventions

- All `#[pg_extern]` functions are `immutable, parallel_safe` unless
  there is a documented reason to be otherwise.
- Distance accumulators are `f64`; outputs match pgvector's choice
  of `double precision` for distances and `f32` for elements of a
  `tvector`.
- Errors that should bubble to the SQL caller use `pgrx::error!`.
- Dimension mismatches raise; never silently truncate.

## License

By contributing you agree your work is offered under the project's
Apache-2.0 license.
