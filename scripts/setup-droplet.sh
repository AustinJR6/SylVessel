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
echo ">>> [1/6] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

# ── 2. Install dependencies ───────────────────────────────
echo ""
echo ">>> [2/6] Installing core dependencies..."
apt-get install -y -qq \
  git curl wget unzip ufw fail2ban \
  ca-certificates gnupg lsb-release

# ── 3. Install Docker ─────────────────────────────────────
echo ""
echo ">>> [3/6] Installing Docker..."
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

# ── 4. Create deploy user ─────────────────────────────────
echo ""
echo ">>> [4/6] Setting up deploy user..."
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

# ── 5. Clone repo ─────────────────────────────────────────
echo ""
echo ">>> [5/6] Cloning repository..."
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

# ── 6. Firewall ───────────────────────────────────────────
echo ""
echo ">>> [6/6] Configuring firewall (UFW)..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh           # 22
ufw allow 80/tcp        # HTTP (Caddy redirect)
ufw allow 443/tcp       # HTTPS (Caddy)
ufw --force enable
echo "Firewall configured. Open ports: 22, 80, 443"

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
echo "STEP 2 — Start the app stack:"
echo "  cd $APP_DIR"
echo "  docker compose -f docker-compose.prod.yml up -d --build"
echo "  docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "STEP 3 — GitHub Actions auto-deploy setup:"
echo "  On your LOCAL machine, generate a deploy SSH key pair:"
echo "    ssh-keygen -t ed25519 -f ~/.ssh/SylVessel_deploy"
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
echo "======================================================"
