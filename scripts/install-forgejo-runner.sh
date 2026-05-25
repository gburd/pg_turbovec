#!/usr/bin/env bash
# scripts/install-forgejo-runner.sh
#
# One-shot installer for a self-hosted Forgejo Actions runner that
# picks up Codeberg CI jobs for this repo.
#
# Why not Codeberg's shared runners? You can apply for shared-runner
# access at https://codeberg.org/Codeberg-CI/request-access (free,
# requires review). Until that lands, a self-hosted runner on a
# machine you already trust (e.g. arnold) is a one-command install.
#
# Usage (on the host that will run the runner; default: arnold):
#
#   1. Visit https://codeberg.org/gregburd/pg_turbovec/settings/actions/runners
#   2. Click "Create new runner" → copy the token.
#   3. ssh to the runner host and run:
#        export FORGEJO_RUNNER_TOKEN=<paste-the-token>
#        bash scripts/install-forgejo-runner.sh
#
# What it does:
#   - Downloads the latest forgejo-runner static binary.
#   - Registers it against the repo using the token.
#   - Creates a systemd user service so it auto-starts on boot.
#
# Privileges: needs Docker (for jobs that use containers like our
# `image: docker.io/library/rust:1-bookworm` workflow). On arnold,
# Docker is already installed.
#
# Idempotent: re-running upgrades the binary in place and re-registers.

set -euo pipefail

: "${FORGEJO_RUNNER_TOKEN:?Set FORGEJO_RUNNER_TOKEN to the token from Codeberg's UI}"
FORGEJO_RUNNER_VERSION="${FORGEJO_RUNNER_VERSION:-latest}"
FORGEJO_RUNNER_DIR="${FORGEJO_RUNNER_DIR:-$HOME/.local/share/forgejo-runner}"
FORGEJO_RUNNER_NAME="${FORGEJO_RUNNER_NAME:-$(hostname -s)-pg_turbovec}"
FORGEJO_INSTANCE="${FORGEJO_INSTANCE:-https://codeberg.org}"

mkdir -p "$FORGEJO_RUNNER_DIR"
cd "$FORGEJO_RUNNER_DIR"

# 1. Download the latest binary.
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  RUNNER_ARCH=amd64 ;;
    aarch64) RUNNER_ARCH=arm64 ;;
    *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
esac

if [ "$FORGEJO_RUNNER_VERSION" = "latest" ]; then
    # Resolve "latest" via the public release API.
    LATEST=$(curl -fsSL "https://code.forgejo.org/api/v1/repos/forgejo/runner/releases?limit=1" \
             | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['tag_name'])")
    FORGEJO_RUNNER_VERSION=$LATEST
fi

echo "[install] forgejo-runner $FORGEJO_RUNNER_VERSION ($RUNNER_ARCH)"
URL="https://code.forgejo.org/forgejo/runner/releases/download/${FORGEJO_RUNNER_VERSION}/forgejo-runner-${FORGEJO_RUNNER_VERSION#v}-linux-${RUNNER_ARCH}"
curl -fsSL -o forgejo-runner "$URL"
chmod +x forgejo-runner

# 2. Register against the repo (idempotent; re-registering with the
#    same token replaces the previous entry in the repo's runner list).
echo "[register] runner '$FORGEJO_RUNNER_NAME' against $FORGEJO_INSTANCE"
./forgejo-runner register \
    --no-interactive \
    --instance "$FORGEJO_INSTANCE" \
    --token "$FORGEJO_RUNNER_TOKEN" \
    --name "$FORGEJO_RUNNER_NAME" \
    --labels "docker,ubuntu-latest:docker://docker.io/library/ubuntu:latest"

# 3. systemd user unit so it auto-starts.
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/forgejo-runner.service <<EOF
[Unit]
Description=Forgejo runner for pg_turbovec CI
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory=$FORGEJO_RUNNER_DIR
ExecStart=$FORGEJO_RUNNER_DIR/forgejo-runner daemon
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now forgejo-runner.service
loginctl enable-linger "$USER" 2>/dev/null || true

echo "[done] runner installed at $FORGEJO_RUNNER_DIR"
echo "[done] systemd unit: ~/.config/systemd/user/forgejo-runner.service"
echo ""
echo "Verify:"
echo "  systemctl --user status forgejo-runner.service"
echo "  https://codeberg.org/gregburd/pg_turbovec/settings/actions/runners"
echo ""
echo "Next push to origin/main triggers a CI run."
