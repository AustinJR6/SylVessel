#!/bin/bash
# Sylana Vessel - Server Startup Script
# Run this to start the web server on RunPod

cd /workspace/Sylana_Vessel
source /workspace/venv_sylana/bin/activate

export SERVER_PORT=7860
export SERVER_HOST=0.0.0.0

echo ""
echo "============================================================"
echo "  SYLANA VESSEL - Starting Soul Engine"
echo "============================================================"
echo ""
echo "  Port: $SERVER_PORT"
echo "  Access: https://<pod-id>-7860.proxy.runpod.net"
echo ""

python3 server.py
