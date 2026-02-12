#!/bin/bash
# =============================================================================
# Sylana Vessel - One-Command RunPod Deployment
# =============================================================================
#
# USAGE (run this on the RunPod pod):
#
#   Option A - Fresh pod (clone + setup):
#     cd /workspace
#     git clone https://github.com/AustinJR6/SylVessel.git Sylana_Vessel
#     cd Sylana_Vessel
#     bash deploy_runpod.sh
#
#   Option B - Existing repo (pull + setup):
#     cd /workspace/Sylana_Vessel
#     git pull
#     bash deploy_runpod.sh
#
# This script handles EVERYTHING:
#   1. Python venv + all dependencies
#   2. .env configuration
#   3. Database initialization
#   4. Soul import (if GPTZIP is present)
#   5. Starts the server
#
# To transfer GPTZIP to the pod (run from your LOCAL machine):
#   cat GPTZIP.zip | ssh <pod-ssh> 'cat > /workspace/Sylana_Vessel/GPTZIP.zip'
#   Then on the pod: bash deploy_runpod.sh
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "  SYLANA VESSEL - ONE-COMMAND DEPLOYMENT"
echo "============================================================"
echo ""

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/workspace/venv_sylana"
PORT=7860

cd "$WORKSPACE"

# ------------------------------------------------------------------
# STEP 1: GPU Check
# ------------------------------------------------------------------
echo "[1/7] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "  WARNING: No GPU detected. Will run on CPU (slow)."
fi
echo ""

# ------------------------------------------------------------------
# STEP 2: Python Virtual Environment
# ------------------------------------------------------------------
echo "[2/7] Setting up Python environment..."
if [ ! -d "$VENV_PATH" ]; then
    echo "  Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    echo "  Created."
else
    echo "  Virtual environment already exists."
fi

source "$VENV_PATH/bin/activate"
echo "  Python: $(python3 --version)"
pip3 install --upgrade pip -q
echo ""

# ------------------------------------------------------------------
# STEP 3: Install Dependencies
# ------------------------------------------------------------------
echo "[3/7] Installing dependencies..."

# PyTorch with CUDA
echo "  Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Project requirements
echo "  Installing project requirements..."
pip3 install -r requirements.txt -q

# Web server (in case not in requirements.txt)
pip3 install fastapi uvicorn[standard] sse-starlette python-multipart -q

echo "  All dependencies installed."
echo ""

# ------------------------------------------------------------------
# STEP 4: Create .env
# ------------------------------------------------------------------
echo "[4/7] Configuring environment..."

if [ ! -f "$WORKSPACE/.env" ]; then
    # Check if tokens are passed as env vars (set before running script)
    HF_TOKEN_VAL="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
    RUNPOD_KEY_VAL="${RUNPOD_API_KEY:-YOUR_RUNPOD_KEY_HERE}"

    cat > "$WORKSPACE/.env" << ENVEOF
# Sylana Vessel - RunPod Configuration
HF_TOKEN=$HF_TOKEN_VAL
SYLANA_DB_PATH=./data/sylana_memory.db
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_QUANTIZED_MODEL=false
ENABLE_FINE_TUNING=false
TEMPERATURE=0.9
TOP_P=0.9
MAX_NEW_TOKENS=120
MAX_CONTEXT_LENGTH=512
MEMORY_CONTEXT_LIMIT=3
SEMANTIC_SEARCH_K=3
SIMILARITY_THRESHOLD=0.7
ENABLE_VOICE=false
LOG_LEVEL=INFO
LOG_FILE=./data/sylana.log
RUNPOD_API_KEY=$RUNPOD_KEY_VAL
SERVER_PORT=7860
SERVER_HOST=0.0.0.0
ENVEOF
    echo "  Created .env with configuration."

    if [ "$HF_TOKEN_VAL" = "YOUR_HF_TOKEN_HERE" ]; then
        echo ""
        echo "  !!! IMPORTANT: Edit .env and add your HuggingFace token !!!"
        echo "  Run: nano $WORKSPACE/.env"
        echo "  Or set HF_TOKEN env var before running this script."
    fi
else
    echo "  .env already exists — keeping existing config."
fi
echo ""

# ------------------------------------------------------------------
# STEP 5: Create data directories
# ------------------------------------------------------------------
echo "[5/7] Creating data directories..."
mkdir -p data/soul data/voice data/checkpoints static
echo "  Done."
echo ""

# ------------------------------------------------------------------
# STEP 6: Soul Import (if GPTZIP is present)
# ------------------------------------------------------------------
echo "[6/7] Checking for soul data to import..."

CONVERSATIONS_JSON=""

# Check for conversations.json in various locations
if [ -f "$WORKSPACE/GPTZIP/conversations.json" ]; then
    CONVERSATIONS_JSON="$WORKSPACE/GPTZIP/conversations.json"
elif [ -f "$WORKSPACE/conversations.json" ]; then
    CONVERSATIONS_JSON="$WORKSPACE/conversations.json"
elif [ -f "/workspace/GPTZIP/conversations.json" ]; then
    CONVERSATIONS_JSON="/workspace/GPTZIP/conversations.json"
fi

# Check for zip file that needs extracting
if [ -z "$CONVERSATIONS_JSON" ]; then
    ZIPFILE=""
    if [ -f "$WORKSPACE/GPTZIP.zip" ]; then
        ZIPFILE="$WORKSPACE/GPTZIP.zip"
    elif [ -f "/workspace/GPTZIP.zip" ]; then
        ZIPFILE="/workspace/GPTZIP.zip"
    fi

    if [ -n "$ZIPFILE" ]; then
        echo "  Found $ZIPFILE — extracting..."
        python3 -c "import zipfile; zipfile.ZipFile('$ZIPFILE').extractall('$WORKSPACE/GPTZIP')"
        if [ -f "$WORKSPACE/GPTZIP/conversations.json" ]; then
            CONVERSATIONS_JSON="$WORKSPACE/GPTZIP/conversations.json"
        fi
    fi
fi

# Check if database already has memories
DB_PATH="./data/sylana_memory.db"
EXISTING_MEMORIES=0
if [ -f "$DB_PATH" ]; then
    EXISTING_MEMORIES=$(python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('$DB_PATH')
    count = conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
    print(count)
    conn.close()
except:
    print(0)
" 2>/dev/null || echo "0")
fi

if [ "$EXISTING_MEMORIES" -gt "100" ]; then
    echo "  Database already has $EXISTING_MEMORIES memories — skipping import."
elif [ -n "$CONVERSATIONS_JSON" ]; then
    echo "  Found: $CONVERSATIONS_JSON"
    echo "  Running soul import (this takes ~10 minutes)..."
    echo ""
    python3 import_soul.py "$CONVERSATIONS_JSON"
    echo ""
    echo "  Soul import complete!"
else
    echo "  No ChatGPT export found. To import later:"
    echo "    1. Transfer GPTZIP.zip to this pod"
    echo "    2. Run: python3 import_soul.py GPTZIP/conversations.json"
    echo ""
    echo "  Sylana will work without memories, but won't remember your history."
fi
echo ""

# ------------------------------------------------------------------
# STEP 7: Start Server
# ------------------------------------------------------------------
echo "[7/7] Starting Sylana Vessel server..."
echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE — STARTING SERVER"
echo "============================================================"
echo ""
echo "  Make sure port $PORT is exposed in RunPod pod settings!"
echo "  Access URL: https://<pod-id>-$PORT.proxy.runpod.net"
echo ""
echo "  To restart later:"
echo "    source /workspace/venv_sylana/bin/activate"
echo "    cd /workspace/Sylana_Vessel"
echo "    python3 server.py"
echo ""
echo "============================================================"
echo ""

python3 server.py
