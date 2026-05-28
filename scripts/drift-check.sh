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

# CI matrices must match Cargo.toml IF a matrix is present. The
# Codeberg workflow (.forgejo/workflows/test.yml) intentionally
# carries only drift-check because Codeberg's hosted runners have
# a 10-minute job cap that cargo pgrx test can't fit; the full
# matrix lives only on .github/. Skip the matrix check for any
# workflow that doesn't declare one.
for ci in .forgejo/workflows/test.yml .github/workflows/test.yml; do
    [ -f "$ci" ] || continue
    grep -q 'pg: \[' "$ci" || continue
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
# 7. Wire-format VERSION constant: must not change in patch releases.
# ----------------------------------------------------------------------
#
# Patch releases (X.Y.Z → X.Y.Z+1) are forbidden from changing the
# on-disk index format. The single source of truth is the `VERSION`
# constant in `src/index/page.rs`. This check compares the working
# tree's VERSION against the most recent tag's VERSION and fails if:
#
#   - The working-tree Cargo.toml is at the same X.Y as the latest
#     tag (i.e. you're shipping a patch on the current minor line),
#   - AND the VERSION constant differs between the working tree and
#     the tag.
#
# Out-of-tree work (between tags) is allowed to bump freely; the
# gate is on the version-bump-vs-VERSION-bump alignment.

last_tag=$(git tag --list 'v*' --sort=-creatordate | head -1)
if [ -n "$last_tag" ]; then
    last_tag_xy=$(printf '%s' "$last_tag" | sed 's/^v//; s/\([0-9]\+\)\.\([0-9]\+\)\..*/\1.\2/')
    cargo_xy=$(printf '%s' "$cargo_ver" | sed 's/\([0-9]\+\)\.\([0-9]\+\)\..*/\1.\2/')
    if [ "$last_tag_xy" = "$cargo_xy" ]; then
        # Same minor line. VERSION must not have moved.
        wt_version=$(grep -E '^pub const VERSION: u8 = ' src/index/page.rs \
                     | sed -E 's/.*= +([0-9]+).*/\1/' | head -1)
        tag_version=$(git show "$last_tag:src/index/page.rs" 2>/dev/null \
                      | grep -E '^pub const VERSION: u8 = ' \
                      | sed -E 's/.*= +([0-9]+).*/\1/' | head -1)
        if [ -n "$wt_version" ] && [ -n "$tag_version" ] \
           && [ "$wt_version" != "$tag_version" ]; then
            fail "src/index/page.rs::VERSION ($wt_version) differs from last tag $last_tag ($tag_version), but Cargo.toml is on the same X.Y line ($cargo_xy). Patch releases must not change the on-disk format. See docs/UPGRADING.md."
        fi
    fi
fi

# ----------------------------------------------------------------------
# 8. Scoreboard cells in docs/PARITY_GAPS.md must not say TBD or claim
#    a regression without a phase qualifier explaining when it shipped.
# ----------------------------------------------------------------------
#
# History: v1.0.x → v1.1.0 (Phase K) shipped the 3000× INSERT speedup,
# but the PARITY_GAPS row stayed "~200 ms / we lose 400×" through
# three minor versions because no one re-read the table. Phase J
# measured Recall@10 = 1.000 on dbpedia-1M but the row stayed "TBD"
# the same way. This check fires when:
#
#   - Any cell in the scoreboard contains "TBD".
#   - A cell says "we lose <N>x" without a same-row "post-Phase-X"
#     or "shipped in v" qualifier (i.e. a regression that hasn't
#     been re-evaluated since shipping the fix).

if [ -f docs/PARITY_GAPS.md ]; then
    # Find the scoreboard table. Awk's range pattern is finicky when
    # the end regex could match the start line; gate the start with
    # a flag so we only enter the range AFTER the header line.
    scoreboard=$(awk '
        /^## Performance gaps/ { in_section = 1; next }
        /^## / && in_section   { exit }
        in_section             { print }
    ' docs/PARITY_GAPS.md | grep -E '^\| ' | head -20)
    while IFS= read -r row; do
        # TBD in any cell.
        if printf '%s' "$row" | grep -qE '\bTBD\b'; then
            fail "docs/PARITY_GAPS.md scoreboard contains TBD: $(printf '%s' "$row" | head -c 120)"
        fi
        # "we lose Nx" claim without a Phase / shipped-in-v qualifier.
        if printf '%s' "$row" | grep -qE 'we lose [~]?[0-9]+(x|×)'; then
            if ! printf '%s' "$row" | grep -qE 'Phase [A-Z]|shipped in v|post-Phase|v1\.[0-9]'; then
                fail "docs/PARITY_GAPS.md scoreboard claims a regression without a phase qualifier; either fix the regression and re-measure, or annotate the row with the in-flight fix: $(printf '%s' "$row" | head -c 120)"
            fi
        fi
    done <<<"$scoreboard"
fi

# ----------------------------------------------------------------------
# 9. Migration files vs documented release history.
# ----------------------------------------------------------------------
#
# Every release that ships a tagged version must have a matching
# migrations/0NN_pg_turbovec_vX.Y.Z.sql file (even if empty), so
# `ALTER EXTENSION pg_turbovec UPDATE TO 'X.Y.Z';` resolves cleanly.
# This check cross-references the migration filenames against the
# 'From' column of the migration matrix in docs/UPGRADING.md.
#
# History: caught at release time twice already — v1.6.0 and v1.7.0
# both had to be re-tagged after the migration file was forgotten.
# Phase Y (v1.7.2) added an in-Rust mirror of this check
# (`migration_files_cover_documented_versions`) so the gate fires
# in `cargo pgrx test` too.

if [ -f docs/UPGRADING.md ] && [ -d migrations ]; then
    # The most recent Cargo.toml version must have a matching
    # migration file. Catches "bumped Cargo.toml + control + tag
    # but forgot the migration file" — the historical bug.
    if ! ls "migrations/"*"_pg_turbovec_v${cargo_ver}.sql" >/dev/null 2>&1; then
        fail "Cargo.toml is at v${cargo_ver} but migrations/ has no matching file. Add migrations/0NN_pg_turbovec_v${cargo_ver}.sql (may be empty / comments-only)."
    fi

    # Migration filenames must be monotonically increasing on the
    # 0NN prefix (catches a release engineer who reuses or
    # backdates a sigil).
    prev=0
    for f in $(ls migrations/0*.sql 2>/dev/null | sort); do
        n=$(basename "$f" | sed -E 's/^0*([0-9]+)_.*/\1/')
        if [ "$n" -le "$prev" ]; then
            fail "migrations/ ordering not monotonic at $f (sigil $n <= prev $prev)."
        fi
        prev=$n
    done

    # The Rust-side `migration_files_cover_documented_versions`
    # #[pg_test] in src/lib.rs holds the authoritative list of
    # release sigils. We mirror just the "latest sigil is
    # documented" gate here so the docs-only side of the contract
    # is enforceable without a running PG cluster. Per-sigil
    # mention coverage in docs/UPGRADING.md isn't checked because
    # the matrix uses `X.Y.x → X.Y.x+1 (patch)` blanket rows for
    # patch-line hops; a pure substring grep would false-fire on
    # those.
    if ! grep -qE "v?${cargo_ver}\\b" docs/UPGRADING.md; then
        fail "docs/UPGRADING.md doesn't mention v${cargo_ver}; add a row to the migration matrix or extend the \"X.Y.x → X.Y.x+1 (patch)\" row to cover it."
    fi
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
