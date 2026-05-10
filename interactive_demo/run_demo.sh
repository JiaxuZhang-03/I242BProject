#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/littleotter/miniconda3/envs/nnenv2/bin/python}"
PORT="${PORT:-7860}"
HOST="${HOST:-127.0.0.1}"

cd "$(dirname "$0")/.."

exec "$PYTHON_BIN" interactive_demo/app.py --host "$HOST" --port "$PORT" "$@"
