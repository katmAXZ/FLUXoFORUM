#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${FLUXOFORUM_DATA_ROOT:-/workspace/fluxoforum-data}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DATA_ROOT/models/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/jobs" "$DATA_ROOT/outputs"
exec fluxoforum \
    --host "${FLUXOFORUM_HOST:-0.0.0.0}" \
    --port "${FLUXOFORUM_PORT:-7860}" \
    --data-root "$DATA_ROOT"

