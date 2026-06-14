#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/FLUXoFORUM}"
DATA_ROOT="${FLUXOFORUM_DATA_ROOT:-/workspace/fluxoforum-data}"
VENV="${FLUXOFORUM_VENV:-/workspace/fluxoforum-venv}"

echo "[1/6] Installing system dependencies"
apt-get update -qq
apt-get install -y -qq ffmpeg git curl python3 python3-pip python3-venv

echo "[2/6] Creating persistent directories"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/jobs" "$DATA_ROOT/outputs"

echo "[3/6] Creating Python environment"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing FLUXoFORUM"
cd "$PROJECT_DIR"
python -m pip install --upgrade -r requirements.txt
python -m pip install --no-deps -e .

echo "[5/6] Running diagnostics"
export FLUXOFORUM_DATA_ROOT="$DATA_ROOT"
export HF_HOME="$DATA_ROOT/models"
export HUGGINGFACE_HUB_CACHE="$DATA_ROOT/models/hub"
fluxoforum --diagnostics

echo "[6/6] Ready"
echo "Accept the BFL model terms and set HF_TOKEN before downloading."
echo "Pre-download: source $VENV/bin/activate && python scripts/predownload.py"
echo "Launch: source $VENV/bin/activate && bash scripts/launch.sh"
