#!/bin/bash
# =============================================================================
# Sylana Vessel - New Pod Setup (run from YOUR LOCAL machine)
# =============================================================================
#
# USAGE:
#   bash setup_new_pod.sh <pod-ssh-address>
#
# EXAMPLE:
#   bash setup_new_pod.sh abc123@ssh.runpod.io
#
# This script (from your local machine):
#   1. Transfers GPTZIP.zip to the pod
#   2. SSHs into the pod
#   3. Clones the repo & runs deploy_runpod.sh
#
# PREREQUISITES:
#   - GPTZIP.zip in the same directory as this script (or ~/Sylana_Vessel/)
#   - SSH key set up for RunPod (~/.ssh/id_ed25519)
#   - Port 7860 exposed in RunPod pod HTTP settings
#
# ENVIRONMENT VARIABLES (optional, to auto-configure .env):
#   export HF_TOKEN=hf_your_token_here
#   export RUNPOD_API_KEY=rpa_your_key_here
# =============================================================================

set -e

# Parse arguments
POD_SSH="$1"
SSH_KEY="${2:-$HOME/.ssh/id_ed25519}"

if [ -z "$POD_SSH" ]; then
    echo ""
    echo "============================================================"
    echo "  SYLANA VESSEL - NEW POD SETUP"
    echo "============================================================"
    echo ""
    echo "  Usage: bash setup_new_pod.sh <pod-ssh-address>"
    echo ""
    echo "  Find your pod SSH address in the RunPod dashboard."
    echo "  It looks like: abc123xyz@ssh.runpod.io"
    echo ""
    echo "  Example:"
    echo "    bash setup_new_pod.sh h2mctb8xc5ik0w-64411282@ssh.runpod.io"
    echo ""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo "  SYLANA VESSEL - NEW POD SETUP"
echo "============================================================"
echo ""
echo "  Pod: $POD_SSH"
echo "  SSH Key: $SSH_KEY"
echo ""

# ------------------------------------------------------------------
# STEP 1: Find and transfer GPTZIP.zip
# ------------------------------------------------------------------
GPTZIP=""
if [ -f "$SCRIPT_DIR/GPTZIP.zip" ]; then
    GPTZIP="$SCRIPT_DIR/GPTZIP.zip"
elif [ -f "$HOME/Sylana_Vessel/GPTZIP.zip" ]; then
    GPTZIP="$HOME/Sylana_Vessel/GPTZIP.zip"
elif [ -f "./GPTZIP.zip" ]; then
    GPTZIP="./GPTZIP.zip"
fi

if [ -n "$GPTZIP" ]; then
    FILESIZE=$(du -h "$GPTZIP" | cut -f1)
    echo "[1/3] Transferring GPTZIP.zip ($FILESIZE) to pod..."
    echo "  This may take a few minutes depending on file size..."
    cat "$GPTZIP" | ssh -i "$SSH_KEY" "$POD_SSH" 'cat > /workspace/GPTZIP.zip'
    echo "  Transfer complete!"
else
    echo "[1/3] No GPTZIP.zip found locally — skipping transfer."
    echo "  Looked in: $SCRIPT_DIR/, $HOME/Sylana_Vessel/, ./"
    echo "  Sylana will deploy without memories."
fi
echo ""

# ------------------------------------------------------------------
# STEP 2: Clone repo on pod (or pull if exists)
# ------------------------------------------------------------------
echo "[2/3] Setting up repo on pod..."
ssh -i "$SSH_KEY" "$POD_SSH" << 'REMOTEOF'
cd /workspace
if [ -d "Sylana_Vessel/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd Sylana_Vessel
    git pull
else
    echo "  Cloning repo..."
    git clone https://github.com/AustinJR6/SylVessel.git Sylana_Vessel
    cd Sylana_Vessel
fi

# Move GPTZIP into repo if it was uploaded to /workspace
if [ -f "/workspace/GPTZIP.zip" ] && [ ! -f "/workspace/Sylana_Vessel/GPTZIP.zip" ]; then
    mv /workspace/GPTZIP.zip /workspace/Sylana_Vessel/GPTZIP.zip
fi
REMOTEOF
echo "  Done."
echo ""

# ------------------------------------------------------------------
# STEP 3: Run deploy script on pod
# ------------------------------------------------------------------
echo "[3/3] Running deployment on pod..."
echo "  (This will install deps, import soul, and start the server)"
echo ""
echo "============================================================"
echo "  Handing off to pod — output below is from RunPod"
echo "============================================================"
echo ""

# Pass tokens as env vars so deploy_runpod.sh can auto-configure .env
ssh -t -i "$SSH_KEY" "$POD_SSH" "cd /workspace/Sylana_Vessel && HF_TOKEN='${HF_TOKEN:-}' RUNPOD_API_KEY='${RUNPOD_API_KEY:-}' bash deploy_runpod.sh"
