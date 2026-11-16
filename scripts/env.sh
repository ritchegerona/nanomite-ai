#!/usr/bin/env bash
# Resolve the Python interpreter that has mlx_lm installed, plus model paths.
# Source this from other scripts: source "$(dirname "$0")/env.sh"

if [ -n "${NANOMITE_PY:-}" ]; then
  PY="$NANOMITE_PY"
elif python3 -c "import mlx_lm" >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif [ -x "$HOME/.local/pipx/venvs/mlx-lm/bin/python" ]; then
  PY="$HOME/.local/pipx/venvs/mlx-lm/bin/python"
else
  echo "mlx_lm not found; set NANOMITE_PY to a python with mlx_lm installed" >&2
  exit 1
fi

if [ -x "$(dirname "$PY")/mlx_lm" ]; then
  MLX_LM="$(dirname "$PY")/mlx_lm"
else
  MLX_LM="$(command -v mlx_lm)"
fi

export PY MLX_LM
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_BASE="${MODEL_BASE:-$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-4bit/snapshots/*/ 2>/dev/null | tail -1)}"
export DRAFT="${DRAFT:-$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--LFM2-350M-4bit/snapshots/*/ 2>/dev/null | tail -1)}"

if [ -z "$MODEL_BASE" ]; then
  echo "base model not found in HF cache; set MODEL_BASE" >&2
  exit 1
fi
