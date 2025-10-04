#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$BASE_DIR/.bartender.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found; is bartender running?"
  exit 1
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped bartender (PID $PID)."
else
  echo "Process $PID not running. Cleaning PID file."
fi
rm -f "$PID_FILE"
