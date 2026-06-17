#!/usr/bin/env bash
# scripts/compile-matrix.sh — fast cross-PG-version compile gate.
#
# Local dev typically targets one PG (pg16). PostgreSQL's C API
# differs across major versions (e.g. BufFileReadExact landed in
# pg16; BufFileWrite's pointer type changed; reloptions layout
# shifts), so code that builds on pg16 can fail to COMPILE on
# pg13/14/15/18. `cargo pgrx test pg16` alone never catches this —
# it cost us a silent CI break across the whole pg13/14/15/18 matrix
# from v1.12.0 (the Phase B-4 BufFile spill) through v1.15.0.
#
# This runs `cargo check` (compile only, no test cluster, ~20s each)
# across every PG feature in Cargo.toml so the break is caught
# locally, before tagging. Wired into .githooks/pre-push alongside
# drift-check.
#
# Skips with COMPILE_MATRIX_SKIP=1 (e.g. on a host where not every
# pg-version pgrx toolchain is installed — CI still covers them).
#
# Exit codes: 0 = all features compile; 1 = at least one failed.

set -uo pipefail

if [ "${COMPILE_MATRIX_SKIP:-0}" = "1" ]; then
    echo "compile-matrix: skipped (COMPILE_MATRIX_SKIP=1)"
    exit 0
fi

cd "$(git rev-parse --show-toplevel)"

# The set of pgNN features Cargo.toml declares (single source of
# truth; stays in lock-step with the CI matrix via drift-check § 2).
features=$(grep -oE '^pg1[0-9] =' Cargo.toml | grep -oE 'pg1[0-9]' | sort -u)
if [ -z "$features" ]; then
    echo "compile-matrix: no pgNN features found in Cargo.toml" >&2
    exit 1
fi

rc=0
for f in $features; do
    echo "compile-matrix: cargo check --no-default-features --features $f"
    if ! cargo check --no-default-features --features "$f" --quiet 2>&1 | sed 's/^/  /'; then
        echo "compile-matrix: FAILED on $f"
        rc=1
    fi
done

if [ "$rc" -eq 0 ]; then
    echo "compile-matrix: all features compile ✅"
else
    echo "compile-matrix: one or more features failed to compile ❌"
fi
exit "$rc"
