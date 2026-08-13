#!/usr/bin/env bash
# install-upgrade-scripts.sh — copy the tracked ALTER EXTENSION UPDATE
# scripts into the extension sharedir so in-place upgrades apply their
# SQL deltas.
#
# WHY THIS EXISTS: `cargo pgrx install` only writes the full-install
# `pg_turbovec--<version>.sql`; it does NOT copy the hand-written
# `sql/pg_turbovec--<from>--<to>.sql` upgrade scripts. Without them,
# `ALTER EXTENSION pg_turbovec UPDATE` is a .so-only no-op and never
# creates new SQL objects (this is the v1.28.4 turbovec_check regression
# reported by agora 2026-08-11). Run this AFTER `cargo pgrx install`.
#
# Usage: bash scripts/install-upgrade-scripts.sh [PG_CONFIG]
#   PG_CONFIG defaults to `pg_config` on PATH.
set -euo pipefail

PG_CONFIG="${1:-pg_config}"
SHAREDIR="$("$PG_CONFIG" --sharedir)/extension"
REPO_SQL="$(cd "$(dirname "$0")/.." && pwd)/sql"

if [ ! -d "$SHAREDIR" ]; then
    echo "install-upgrade-scripts: sharedir $SHAREDIR not found" >&2
    exit 1
fi

n=0
for f in "$REPO_SQL"/pg_turbovec--*--*.sql; do
    [ -e "$f" ] || continue
    cp -v "$f" "$SHAREDIR/"
    n=$((n + 1))
done
echo "install-upgrade-scripts: installed $n upgrade script(s) into $SHAREDIR"
[ "$n" -gt 0 ] || { echo "WARNING: no sql/pg_turbovec--<from>--<to>.sql found to install" >&2; }
