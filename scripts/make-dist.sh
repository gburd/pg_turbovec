#!/usr/bin/env bash
# scripts/make-dist.sh — build the PGXN *source* distribution zip for
# pg_turbovec.
#
# pg_turbovec is a pgrx (Rust) extension, so — unlike a PGXS C
# extension — there is no `Makefile` a consumer runs against
# `pg_config`, and a plain `git archive` would ship a dist with no
# installable SQL. So this script:
#   1. reads the single-source-of-truth version from Cargo.toml,
#   2. renders META.json from META.json.in (@VERSION@ substitution),
#   3. runs `cargo pgrx schema` to generate the install SQL
#      (sql/pg_turbovec--X.Y.Z.sql), the actually-useful artifact,
#   4. zips a PGXN-layout SOURCE distribution.
#
# HONEST CAVEAT (documented on PGXN too): this is a SOURCE archive for
# humans, not a `pgxn install`-able package — a pgrx extension is built
# with `cargo pgrx`, not `make`. The dist carries the buildable Rust
# sources + Cargo manifests + the generated install SQL + control +
# docs so a user can `cargo pgrx install` from it, and so the release
# is discoverable + version-pinned on PGXN.
#
# Usage: scripts/make-dist.sh [PG_FEATURE]   (PG_FEATURE default: pg16)
set -euo pipefail

cd "$(dirname "$0")/.."
PG_FEATURE="${1:-pg16}"

VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VERSION" ] || { echo "could not read version from Cargo.toml" >&2; exit 1; }
DIST="pg_turbovec-${VERSION}"
echo "make-dist: version ${VERSION}, feature ${PG_FEATURE}"

# 1. Render META.json from the template.
sed "s/@VERSION@/${VERSION}/g" META.json.in > META.json
echo "make-dist: wrote META.json"

# 2. Generate the install SQL (the artifact PGXN's provides.file points at).
#    cargo-pgrx must be installed and `cargo pgrx init` run for the PG
#    version; the release workflow handles that.
mkdir -p sql
cargo pgrx schema --features "${PG_FEATURE}" \
    --out "sql/pg_turbovec--${VERSION}.sql"
echo "make-dist: generated sql/pg_turbovec--${VERSION}.sql"

# 3. Assemble the PGXN-layout source zip. Include the buildable Rust
#    sources + manifests + control + generated SQL + docs + migrations.
#    Exclude build artifacts, VCS, local config, benches results
#    (large, not needed to build).
rm -f "${DIST}.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "${STAGE}/${DIST}"

# The file set that makes the dist buildable + PGXN-valid.
cp -r \
    META.json \
    Cargo.toml Cargo.lock \
    build.rs \
    pg_turbovec.control \
    README.md CHANGELOG.md LICENSE \
    src sql migrations docs \
    "${STAGE}/${DIST}/" 2>/dev/null || true

# Fail loudly if the two PGXN-critical files aren't present.
test -f "${STAGE}/${DIST}/META.json" || { echo "META.json missing from dist" >&2; exit 1; }
test -f "${STAGE}/${DIST}/sql/pg_turbovec--${VERSION}.sql" || {
    echo "generated install SQL missing from dist" >&2; exit 1; }

( cd "$STAGE" && zip -rq "${OLDPWD}/${DIST}.zip" "${DIST}" )
echo "make-dist: wrote ${DIST}.zip"
ls -l "${DIST}.zip"
