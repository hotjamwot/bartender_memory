#!/usr/bin/env bash
set -euo pipefail
# Start the bartender FastAPI server in the background and write its PID to .bartender.pid
# Uses the project's venv uvicorn if available, otherwise falls back to system uvicorn.

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$BASE_DIR/.bartender.pid"
LOG_FILE="$BASE_DIR/bartender.log"
UVICORN="$BASE_DIR/venv/bin/uvicorn"

if [ ! -x "$UVICORN" ]; then
  UVICORN="uvicorn"
fi

if [ -f "$PID_FILE" ]; then
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Bartender already running (PID $(cat "$PID_FILE"))."
    exit 1
  else
    echo "Removing stale PID file."
    rm -f "$PID_FILE"
  fi
fi

nohup "$UVICORN" server.main:app --host 127.0.0.1 --port 5001 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Started bartender at http://127.0.0.1:5001, PID $(cat "$PID_FILE"), logs: $LOG_FILE"
