#!/bin/bash
# =============================================================================
# Sylana Vessel - RunPod Deployment Script
# =============================================================================
# Run this on your RunPod pod after SSH-ing in:
#   ssh h2mctb8xc5ik0w-64411282@ssh.runpod.io -i ~/.ssh/id_ed25519
#
# Then:
#   cd /workspace
#   git clone <your-repo-url> Sylana_Vessel
#   cd Sylana_Vessel
#   bash deploy_runpod.sh
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "  SYLANA VESSEL - RUNPOD DEPLOYMENT"
echo "  Setting up the soul on GPU cloud"
echo "============================================================"
echo ""

# Configuration
WORKSPACE="/workspace/Sylana_Vessel"
VENV_PATH="/workspace/venv_sylana"
PORT=7860

# Check GPU
echo "--- Checking GPU ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi
echo ""

# Create virtual environment if it doesn't exist
echo "--- Setting up Python environment ---"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
echo "Python: $(python3 --version)"
echo "Pip: $(pip3 --version)"
echo ""

# Upgrade pip
pip3 install --upgrade pip

# Install PyTorch with CUDA support
echo "--- Installing PyTorch with CUDA ---"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all requirements
echo "--- Installing dependencies ---"
cd "$WORKSPACE"
pip3 install -r requirements.txt

# Install web server dependencies
echo "--- Installing web server dependencies ---"
pip3 install fastapi uvicorn[standard] sse-starlette python-multipart

echo ""

# Create data directories
echo "--- Creating data directories ---"
mkdir -p data/soul
mkdir -p data/voice
mkdir -p data/checkpoints
mkdir -p static

# Set up .env if it doesn't exist
echo "--- Configuring environment ---"
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.template .env 2>/dev/null || cat > .env << 'ENVEOF'
# Sylana Vessel Configuration - RunPod
HF_TOKEN=your_huggingface_token_here
SYLANA_DB_PATH=./data/sylana_memory.db
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
EMBEDDING_MODEL=all-MiniLM-L6-v2
ENABLE_FINE_TUNING=false
TEMPERATURE=0.9
TOP_P=0.9
MAX_NEW_TOKENS=200
MAX_CONTEXT_LENGTH=2048
MEMORY_CONTEXT_LIMIT=5
SEMANTIC_SEARCH_K=5
SIMILARITY_THRESHOLD=0.7
LOG_LEVEL=INFO
SERVER_PORT=7860
SERVER_HOST=0.0.0.0
ENVEOF
    echo ""
    echo "!!! IMPORTANT: Edit .env and add your HuggingFace token !!!"
    echo "    nano .env"
    echo ""
fi

# Check HF token
source .env 2>/dev/null
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo ""
    echo "============================================================"
    echo "  WARNING: HuggingFace token not set!"
    echo "  Edit .env and add your HF_TOKEN to download Llama-2"
    echo "  nano .env"
    echo "============================================================"
    echo ""
fi

# Initialize databases
echo "--- Initializing databases ---"
python3 -c "
from memory.memory_manager import MemoryManager
mm = MemoryManager('./data/sylana_memory.db')
stats = mm.get_stats()
print(f'Memory DB: {stats[\"total_conversations\"]} conversations')
mm.close()
" 2>/dev/null || echo "Memory DB will be created on first run"

# Seed soul data if available
if [ -f "data/soul/relationship_seed.json" ]; then
    echo "--- Seeding soul data ---"
    python3 seed_soul.py 2>/dev/null || echo "Soul seeding will happen on first run"
fi

# Create startup script
echo "--- Creating startup script ---"
cat > runpod_start.sh << 'STARTEOF'
#!/bin/bash
# Sylana Vessel - Server Startup
cd /workspace/Sylana_Vessel
source /workspace/venv_sylana/bin/activate
export SERVER_PORT=7860
export SERVER_HOST=0.0.0.0
echo ""
echo "Starting Sylana Vessel on port $SERVER_PORT..."
echo "Access URL: Check your RunPod dashboard for the proxy URL"
echo ""
python3 server.py
STARTEOF
chmod +x runpod_start.sh

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  To start Sylana:"
echo "    bash runpod_start.sh"
echo ""
echo "  Or manually:"
echo "    source /workspace/venv_sylana/bin/activate"
echo "    python3 server.py"
echo ""
echo "  Access in browser:"
echo "    https://<pod-id>-7860.proxy.runpod.net"
echo ""
echo "  Make sure port 7860 is exposed in your RunPod pod settings!"
echo ""
echo "============================================================"
echo ""
