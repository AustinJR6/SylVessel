#!/usr/bin/env bash
# =============================================================================
# Sylana Vessel — DigitalOcean Droplet Bootstrap
# Run this ONCE on the droplet via the DO web console or SSH.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/AustinJR6/SylVessel/main/scripts/setup-droplet.sh | sudo bash
#   — OR —
#   sudo bash setup-droplet.sh
# =============================================================================
set -euo pipefail

REPO="https://github.com/AustinJR6/SylVessel.git"
APP_DIR="/opt/SylVessel"
DEPLOY_USER="deploy"

echo "======================================================"
echo "  Sylana Vessel — Droplet Bootstrap"
echo "======================================================"

# ── 1. System update ──────────────────────────────────────
echo ""
echo ">>> [1/8] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

# ── 2. Install dependencies ───────────────────────────────
echo ""
echo ">>> [2/8] Installing core dependencies..."
apt-get install -y -qq \
  git curl wget unzip ufw fail2ban \
  ca-certificates gnupg lsb-release

# ── 3. Install Docker ─────────────────────────────────────
echo ""
echo ">>> [3/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
  echo "Docker installed."
else
  echo "Docker already installed, skipping."
fi

# ── 4. Install Node.js 24 ─────────────────────────────────
echo ""
echo ">>> [4/8] Installing Node.js 24..."
if ! node --version 2>/dev/null | grep -q "^v24"; then
  curl -fsSL https://deb.nodesource.com/setup_24.x | bash -
  apt-get install -y -qq nodejs
  echo "Node.js $(node --version) installed."
else
  echo "Node.js 24 already installed, skipping."
fi

# ── 5. Install OpenClaw ───────────────────────────────────
echo ""
echo ">>> [5/8] Installing OpenClaw..."
npm install -g openclaw@latest
echo "OpenClaw $(openclaw --version 2>/dev/null || echo 'installed') ready."

# ── 6. Create deploy user ─────────────────────────────────
echo ""
echo ">>> [6/8] Setting up deploy user..."
if ! id "$DEPLOY_USER" &>/dev/null; then
  useradd -m -s /bin/bash "$DEPLOY_USER"
  usermod -aG docker "$DEPLOY_USER"
  echo "User '$DEPLOY_USER' created and added to docker group."
else
  usermod -aG docker "$DEPLOY_USER"
  echo "User '$DEPLOY_USER' already exists."
fi

# Set up SSH authorized_keys for deploy user (GitHub Actions will use this)
mkdir -p "/home/${DEPLOY_USER}/.ssh"
chmod 700 "/home/${DEPLOY_USER}/.ssh"
touch "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chmod 600 "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "/home/${DEPLOY_USER}/.ssh"

echo ""
echo "  !! ACTION REQUIRED: Paste your GitHub Actions deploy SSH public key into:"
echo "     /home/${DEPLOY_USER}/.ssh/authorized_keys"
echo "  !! You will generate this key pair in the GitHub setup step below."

# ── 7. Clone repo ─────────────────────────────────────────
echo ""
echo ">>> [7/8] Cloning repository..."
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO" "$APP_DIR"
  chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "$APP_DIR"
  echo "Repo cloned to $APP_DIR"
else
  echo "Repo already present at $APP_DIR, pulling latest..."
  cd "$APP_DIR" && git pull origin main
fi

# Create .env file placeholder (user must fill this in)
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.production.template" "$APP_DIR/.env" 2>/dev/null || \
    touch "$APP_DIR/.env"
  echo ""
  echo "  !! ACTION REQUIRED: Fill in your secrets at $APP_DIR/.env"
  echo "     See .env.production.template for all required variables."
fi

# ── 8. Firewall ───────────────────────────────────────────
echo ""
echo ">>> [8/8] Configuring firewall (UFW)..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh           # 22
ufw allow 80/tcp        # HTTP (Caddy redirect)
ufw allow 443/tcp       # HTTPS (Caddy)
# Port 18789 (OpenClaw) is NOT exposed externally — only internal Docker network
ufw --force enable
echo "Firewall configured. Open ports: 22, 80, 443"

# ── OpenClaw systemd service ──────────────────────────────
echo ""
echo ">>> Setting up OpenClaw gateway as systemd service..."

# OpenClaw will be configured after you fill in .env
# It runs as a service alongside Docker Compose

cat > /etc/systemd/system/openclaw-gateway.service << 'UNIT'
[Unit]
Description=OpenClaw AI Agent Gateway
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=deploy
WorkingDirectory=/opt/SylVessel
EnvironmentFile=/opt/SylVessel/.env
ExecStartPre=/bin/sh -c 'mkdir -p /home/deploy/.openclaw'
ExecStart=/usr/bin/openclaw gateway start
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
# NOTE: Do NOT start openclaw yet — fill .env first, then run openclaw onboard
echo "OpenClaw systemd unit created (not started yet — see next steps)."

# ── Completion message ────────────────────────────────────
echo ""
echo "======================================================"
echo "  Bootstrap complete! Next steps:"
echo "======================================================"
echo ""
echo "STEP 1 — Fill in secrets:"
echo "  nano $APP_DIR/.env"
echo "  (Copy from .env.production.template, fill in all API keys)"
echo ""
echo "STEP 2 — Configure OpenClaw (interactive, run once):"
echo "  cd $APP_DIR"
echo "  source .env && openclaw onboard"
echo "  # Select: Anthropic, enter your ANTHROPIC_API_KEY"
echo "  # Enable sandbox mode when prompted"
echo "  # This saves config to /home/deploy/.openclaw/"
echo ""
echo "STEP 3 — Start OpenClaw gateway:"
echo "  systemctl enable --now openclaw-gateway"
echo "  systemctl status openclaw-gateway"
echo ""
echo "STEP 4 — Start the app stack:"
echo "  cd $APP_DIR"
echo "  docker compose -f docker-compose.prod.yml up -d --build"
echo "  docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "STEP 5 — GitHub Actions auto-deploy setup:"
echo "  On your LOCAL machine, generate a deploy SSH key pair:"
echo "    ssh-keygen -t ed25519 -f ~/.ssh/SylVessel_deploy -N ''"
echo "  Then:"
echo "    cat ~/.ssh/SylVessel_deploy.pub"
echo "    >> paste into /home/deploy/.ssh/authorized_keys on this droplet"
echo "    cat ~/.ssh/SylVessel_deploy"
echo "    >> add as GitHub secret: DO_SSH_KEY"
echo "  Add these GitHub repo secrets (Settings > Secrets > Actions):"
echo "    DO_HOST     = 167.99.127.14"
echo "    DO_USERNAME = deploy"
echo "    DO_SSH_KEY  = (contents of ~/.ssh/SylVessel_deploy)"
echo ""
echo "  After that, every push to 'main' auto-deploys."
echo ""
echo "Your app will be live at: https://167-99-127-14.sslip.io"
echo "OpenClaw gateway: http://localhost:18789 (internal only)"
echo "======================================================"
