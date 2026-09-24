#!/bin/sh
set -euo pipefail

# support_assistant/verify.sh
# Creates a local venv, installs requirements, runs ingest.py to build the index,
# starts the FastAPI server in background, runs two test curl calls, prints outputs,
# then shuts down the server. Intended for local verification.

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.11}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

echo "Using Python: $($PYTHON --version 2>&1)"

# Create and activate venv
if [ ! -d .venv ]; then
  echo "Creating virtual environment at .venv"
  $PYTHON -m venv .venv
fi
. .venv/bin/activate

echo "Upgrading pip and installing requirements"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build or rebuild the index
echo "Running ingest.py (this may download the embedding model on first run)"
python ingest.py

# Start uvicorn in background and capture PID
SERVER_LOG=server.log
echo "Starting uvicorn (logs -> $SERVER_LOG)"
uvicorn main:app --host 127.0.0.1 --port 7860 > "$SERVER_LOG" 2>&1 &
UV_PID=$!

# Ensure server is stopped on script exit
trap 'echo "Stopping server..."; kill "$UV_PID" 2>/dev/null || true' EXIT INT TERM

# Wait for server start
echo "Waiting for server to start..."
sleep 4

# Run test queries
echo "Running test queries"
TEST1='{"query":"How do refunds work?"}'
TEST2='{"query":"What is the capital of France?"}'

curl -s -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "$TEST1" > test1.json || true
curl -s -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "$TEST2" > test2.json || true

echo "--- Server log (last 200 lines) ---"
tail -n 200 "$SERVER_LOG" || true

echo "--- Test 1 output ---"
cat test1.json || true

echo "--- Test 2 output ---"
cat test2.json || true

# Script will exit and trap will stop server
echo "Verification complete. The index (if built) is stored under ./chroma_db when supported by Chroma or an in-memory fallback was used." 
