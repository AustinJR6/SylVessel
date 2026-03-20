#!/usr/bin/env bash
# =============================================================================
# OpenClaw Post-Install Setup
# Run this on the droplet AFTER filling in .env and after bootstrap.
# It copies agent definitions and verifies the gateway is working.
#
# Usage (on the droplet as the deploy user):
#   cd /opt/sylvessel && bash docker/openclaw/setup-openclaw.sh
# =============================================================================
set -euo pipefail

APP_DIR="/opt/sylvessel"
OPENCLAW_DIR="${HOME}/.openclaw"

echo ">>> Setting up OpenClaw agents..."

# Create OpenClaw config directory structure
mkdir -p "${OPENCLAW_DIR}/agents"

# Copy agent definitions into OpenClaw's config dir
cp "${APP_DIR}/docker/openclaw/agents/sylana.md" "${OPENCLAW_DIR}/agents/sylana.md"
cp "${APP_DIR}/docker/openclaw/agents/claude.md"  "${OPENCLAW_DIR}/agents/claude.md"
echo "  ✓ Agent files copied"

# Run doctor to verify installation
echo ""
echo ">>> Running openclaw doctor..."
openclaw doctor || echo "  (doctor reported warnings — check above)"

echo ""
echo "======================================================"
echo "  OpenClaw agents installed."
echo "  Next: run  openclaw onboard  interactively"
echo "  (Select Anthropic, enter your ANTHROPIC_API_KEY,"
echo "   enable sandbox when prompted)"
echo "======================================================"
echo ""
echo "After onboarding, start the gateway:"
echo "  sudo systemctl enable --now openclaw-gateway"
echo "  sudo systemctl status openclaw-gateway"
echo ""
echo "Test the gateway:"
echo "  curl http://localhost:18789/api/health"
