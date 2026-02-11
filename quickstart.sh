#!/bin/bash
# Sylana Vessel Quick Start Script (Linux/Mac)

echo "============================================================"
echo "  SYLANA VESSEL - Quick Start"
echo "============================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "[1/4] Setting up environment configuration..."
    if [ -f .env.template ]; then
        cp .env.template .env
        echo "Created .env from template"
        echo "IMPORTANT: Edit .env and add your HF_TOKEN"
        echo "Get token from: https://huggingface.co/settings/tokens"
        echo ""
        read -p "Press Enter to continue after adding your token..."
    else
        echo "ERROR: .env.template not found!"
        exit 1
    fi
else
    echo "[1/4] Environment file exists ✓"
fi

# Check if database exists
echo "[2/4] Checking database..."
if [ ! -f data/sylana_memory.db ]; then
    echo "Initializing database..."
    python memory/init_database.py
else
    echo "Database exists ✓"
fi

# Check dependencies
echo "[3/4] Checking dependencies..."
python -c "import transformers, torch, faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies OK ✓"
fi

echo "[4/4] Starting Sylana..."
echo ""
echo "============================================================"
echo "  LAUNCHING SYLANA VESSEL"
echo "============================================================"
echo ""

python sylana.py
