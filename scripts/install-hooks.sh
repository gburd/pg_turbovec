#!/usr/bin/env bash
# scripts/install-hooks.sh — install local git hooks from .githooks/
#
# Run once per clone. Invoked by `make hooks` or directly:
#     bash scripts/install-hooks.sh

set -euo pipefail
cd "$(dirname "$0")/.."

git config core.hooksPath .githooks

# Make sure the hook scripts are executable.
chmod +x .githooks/* 2>/dev/null || true

echo "git hooks installed (core.hooksPath = .githooks)"
echo "pre-push will run scripts/drift-check.sh"
