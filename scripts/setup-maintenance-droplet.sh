#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/AustinJR6/SylVessel.git"
APP_DIR="/opt/SylVesselMaintenance"
DEPLOY_USER="deploy"
SERVICE_NAME="sylana-maintenance-controller"

echo "======================================================"
echo "  Sylana Maintenance Controller Bootstrap"
echo "======================================================"

echo ""
echo ">>> [1/7] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

echo ""
echo ">>> [2/7] Installing core dependencies..."
apt-get install -y -qq \
  git curl ca-certificates \
  python3 python3-venv python3-pip \
  ufw fail2ban

echo ""
echo ">>> [3/7] Setting up deploy user..."
if ! id "$DEPLOY_USER" &>/dev/null; then
  useradd -m -s /bin/bash "$DEPLOY_USER"
  echo "User '$DEPLOY_USER' created."
fi
usermod -aG sudo "$DEPLOY_USER"

mkdir -p "/home/${DEPLOY_USER}/.ssh"
chmod 700 "/home/${DEPLOY_USER}/.ssh"
touch "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chmod 600 "/home/${DEPLOY_USER}/.ssh/authorized_keys"
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "/home/${DEPLOY_USER}/.ssh"
cat >/etc/sudoers.d/${SERVICE_NAME} <<EOF
${DEPLOY_USER} ALL=(ALL) NOPASSWD:/bin/systemctl,/usr/bin/systemctl
EOF
chmod 440 /etc/sudoers.d/${SERVICE_NAME}

echo ""
echo ">>> [4/7] Cloning repository..."
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO" "$APP_DIR"
else
  cd "$APP_DIR"
  git fetch origin
  git checkout main
  git pull origin main
fi
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "$APP_DIR"

echo ""
echo ">>> [5/7] Creating maintenance virtualenv..."
sudo -u "$DEPLOY_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$DEPLOY_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$DEPLOY_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.maintenance.txt"

if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.maintenance.template" "$APP_DIR/.env"
  chown "${DEPLOY_USER}:${DEPLOY_USER}" "$APP_DIR/.env"
  echo ""
  echo "  !! ACTION REQUIRED: Fill in controller secrets at $APP_DIR/.env"
fi

echo ""
echo ">>> [6/7] Verifying Codex CLI..."
if ! sudo -u "$DEPLOY_USER" env PATH="/usr/local/bin:/usr/bin:/bin:$PATH" codex exec --help >/dev/null 2>&1; then
  echo "Codex CLI is not available for the deploy user."
  echo "Install/authenticate Codex on this host, then rerun this script."
  exit 1
fi

echo ""
echo ">>> [7/7] Installing systemd service..."
install -m 0644 "$APP_DIR/deploy/systemd/${SERVICE_NAME}.service" "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
systemctl --no-pager --full status "${SERVICE_NAME}" || true

echo ""
echo "======================================================"
echo "Bootstrap complete."
echo "Next steps:"
echo "  1. Fill in $APP_DIR/.env"
echo "  2. Add your GitHub Actions maintenance deploy SSH key to /home/${DEPLOY_USER}/.ssh/authorized_keys"
echo "  3. Add MAINTENANCE_DO_HOST / MAINTENANCE_DO_USERNAME / MAINTENANCE_DO_SSH_KEY repo secrets"
echo "  4. Set repo variable MAINTENANCE_DEPLOY_ENABLED=true when you want auto-deploys"
echo "======================================================"
