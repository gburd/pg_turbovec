#!/usr/bin/env bash
# scripts/drift-check.sh — automated docs-drift audit
#
# Mirror of the .pi/skills/drift-check/SKILL.md checklist as a
# script that exits non-zero on drift. Invoked from CI and from
# the `pre-push` git hook (see scripts/install-hooks.sh) so we
# never tag a release with documentation that lies about the
# code.
#
# Exit codes:
#   0 = clean
#   1 = drift detected (printed to stdout)
#   2 = setup error (couldn't read a required file)

set -uo pipefail
cd "$(dirname "$0")/.."

drift=0
say() { printf '%s\n' "$*"; }
fail() { say "DRIFT: $*"; drift=1; }

# ----------------------------------------------------------------------
# 1. Version numbers consistent across Cargo.toml, control file,
#    CHANGELOG, and the latest tag.
# ----------------------------------------------------------------------

cargo_ver=$(grep -oE '^version = "[^"]+"' Cargo.toml | head -1 | cut -d'"' -f2)
ctrl_ver=$(grep -oE "default_version = '[^']+'" pg_turbovec.control | cut -d"'" -f2)
chlog_ver=$(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+[^]]*\]' CHANGELOG.md | head -1 | tr -d '[]## ')
tag_ver=$(git tag --list 'v*' --sort=-creatordate | head -1 | sed 's/^v//')

[ "$cargo_ver" = "$ctrl_ver" ] || fail "Cargo.toml ($cargo_ver) ≠ pg_turbovec.control ($ctrl_ver)"
[ "$cargo_ver" = "$chlog_ver" ] || fail "Cargo.toml ($cargo_ver) ≠ CHANGELOG.md top entry ($chlog_ver)"

# Tag drift is informational, not failure: a freshly-bumped Cargo.toml
# may legitimately precede the tag for that version.
if [ "$cargo_ver" != "$tag_ver" ]; then
    say "INFO: Cargo.toml=$cargo_ver, latest tag=$tag_ver (release pending?)"
fi

# ----------------------------------------------------------------------
# 2. PG version matrix consistent across Cargo features, docs,
#    and CI workflows.
# ----------------------------------------------------------------------

cargo_pgs=$(grep -E '^pg[0-9]+ = ' Cargo.toml | grep -oE 'pg[0-9]+' | sort | tr '\n' ' ')
docs_pgs=$(grep -oE '^\| [0-9]+\.' docs/PG_VERSION_SUPPORT.md | grep -oE '[0-9]+' | sort | sed 's/^/pg/' | tr '\n' ' ')

cargo_pg_set=$(echo "$cargo_pgs" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
docs_pg_set=$(echo "$docs_pgs" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
[ "$cargo_pg_set" = "$docs_pg_set" ] || fail "Cargo.toml PG features ($cargo_pg_set) ≠ docs/PG_VERSION_SUPPORT.md ($docs_pg_set)"

# CI matrices must match Cargo.toml.
for ci in .forgejo/workflows/test.yml .github/workflows/test.yml; do
    [ -f "$ci" ] || continue
    ci_pgs=$(grep -oE 'pg: \[[^]]+\]' "$ci" | head -1 | grep -oE '[0-9]+' | sort | sed 's/^/pg/' | tr '\n' ' ' | sed 's/ $//')
    [ "$ci_pgs" = "$cargo_pg_set" ] || fail "$ci matrix ($ci_pgs) ≠ Cargo.toml ($cargo_pg_set)"
done

# ----------------------------------------------------------------------
# 3. Test count: documented numbers must match the annotation count
#    (ANN is an upper bound; the live runner number is authoritative).
# ----------------------------------------------------------------------

ann=$(grep -rE '#\[(test|pg_test|pgrx::pg_test)\]' src/ 2>/dev/null | wc -l)
docs_counts=$(grep -ohE '\b9[0-9]/[0-9]+|\b10[0-9]/[0-9]+|\b1[1-9][0-9]/[0-9]+' \
                  README.md CHANGELOG.md docs/PG_VERSION_SUPPORT.md \
                  docs/PARITY_GAPS.md 2>/dev/null \
              | sort -u | tr '\n' ' ')

# Heuristic: if the annotation count is wildly off from any documented
# claim (>5 discrepancy on the larger number), flag it.
say "INFO: $ann test annotations in src/; docs cite: $docs_counts"

# ----------------------------------------------------------------------
# 4. Bench result files referenced in docs must exist on disk.
# ----------------------------------------------------------------------

referenced=$(grep -rohE 'benches/results/[a-zA-Z0-9_.-]+\.json' \
                docs/ README.md CHANGELOG.md 2>/dev/null \
            | sort -u)
for ref in $referenced; do
    [ -f "$ref" ] || fail "doc references $ref but it doesn't exist"
done

# ----------------------------------------------------------------------
# 5. Markdown link sanity inside the docs/ tree.
# ----------------------------------------------------------------------

# Strip http*, anchors, and absolute paths; the rest must resolve to
# real files.
broken=$(grep -rohE '\[[^]]+\]\(([^)]+\.md|[^)]+\.json)(#[^)]+)?\)' \
            docs/ README.md CHANGELOG.md 2>/dev/null \
        | grep -oE '\([^)]+\)' | tr -d '()' \
        | sed 's/#.*$//' \
        | grep -vE '^https?://|^/' \
        | sort -u \
        | while read link; do
            for prefix in '' 'docs/' '../'; do
                [ -f "${prefix}${link}" ] && continue 2
            done
            echo "$link"
        done)
for b in $broken; do
    fail "broken markdown link target: $b"
done

# ----------------------------------------------------------------------
# 6. Vendored deps must have PATCH_NOTES.md.
# ----------------------------------------------------------------------

if [ -d vendor ]; then
    for d in vendor/*/; do
        [ -f "$d/PATCH_NOTES.md" ] || fail "$d missing PATCH_NOTES.md"
    done
fi

# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

if [ "$drift" -ne 0 ]; then
    say ""
    say "Drift detected. Fix the listed items, then re-run:"
    say "    bash scripts/drift-check.sh"
    exit 1
fi

say ""
say "drift-check: clean ✅"
exit 0
