#!/usr/bin/env bash
# =============================================================================
# vessel-agent setup — run this on the droplet as the deploy user
#
# Installs Claude Code CLI + Python Agent SDK, then registers a systemd
# service that listens on port 18790 (where Caddy routes /api/agent/stream).
#
# Usage:
#   cd /opt/sylvessel && bash docker/vessel-agent/setup.sh
# =============================================================================
set -euo pipefail

APP_DIR="/opt/sylvessel"
AGENT_DIR="${APP_DIR}/docker/vessel-agent"
SERVICE_NAME="vessel-agent"
PYTHON_BIN="$(command -v python3)"
VENV_DIR="${AGENT_DIR}/.venv"

echo ">>> vessel-agent setup starting..."
echo ""

# ── 1. Node.js (needed for Claude Code CLI) ──────────────────────────────────
if ! command -v node &>/dev/null; then
  echo ">>> Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
echo "  ✓ Node.js $(node -v)"

# ── 2. Claude Code CLI ────────────────────────────────────────────────────────
if ! command -v claude &>/dev/null; then
  echo ">>> Installing Claude Code CLI..."
  sudo npm install -g @anthropic-ai/claude-code
fi
echo "  ✓ Claude Code CLI: $(claude --version 2>/dev/null || echo 'installed')"

# ── 3. Python venv + agent SDK ────────────────────────────────────────────────
echo ">>> Setting up Python venv..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${AGENT_DIR}/requirements.txt"
echo "  ✓ Python deps installed"

# ── 4. Workspace directory ───────────────────────────────────────────────────
mkdir -p /tmp/vessel-workspace
echo "  ✓ Workspace: /tmp/vessel-workspace"

# ── 5. Remove OpenClaw if present ────────────────────────────────────────────
if systemctl is-active --quiet openclaw-gateway 2>/dev/null; then
  echo ">>> Disabling openclaw-gateway..."
  sudo systemctl disable --now openclaw-gateway
  sudo rm -f /etc/systemd/system/openclaw-gateway.service
  sudo systemctl daemon-reload
  echo "  ✓ OpenClaw removed"
fi

# ── 6. Systemd service ───────────────────────────────────────────────────────
echo ">>> Installing systemd service: ${SERVICE_NAME}..."

# Read ANTHROPIC_API_KEY from the app .env if available
ENV_FILE="${APP_DIR}/.env"

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Sylana Vessel Agent (Claude Agent SDK)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${AGENT_DIR}
ExecStart=${VENV_DIR}/bin/python server.py
Restart=always
RestartSec=5
EnvironmentFile=${ENV_FILE}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
echo "  ✓ Service installed and started"

# ── 7. Verify ────────────────────────────────────────────────────────────────
echo ""
echo ">>> Waiting for service to come up..."
sleep 3
if curl -sf http://localhost:18790/api/health > /dev/null; then
  echo "  ✓ vessel-agent is healthy at http://localhost:18790/api/health"
else
  echo "  ✗ Health check failed — check logs:"
  echo "    sudo journalctl -u vessel-agent -n 40"
fi

echo ""
echo "======================================================"
echo "  vessel-agent running on port 18790."
echo "  Caddy routes /api/agent/stream here automatically."
echo ""
echo "  Manage:"
echo "    sudo systemctl status vessel-agent"
echo "    sudo journalctl -u vessel-agent -f"
echo "    sudo systemctl restart vessel-agent"
echo "======================================================"
