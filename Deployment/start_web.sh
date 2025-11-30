#!/bin/bash
set -e

echo "=========================================="
echo "Starting COLREG Web Interface in Docker"
echo "=========================================="

# Start API server in background
echo "Starting API server on port 5000..."
python web/api_server.py &

# Wait for API to initialize
sleep 3

# Start web server (keeps container alive)
echo "Starting web server on port 8000..."
cd web
python -m http.server 8000