#!/usr/bin/env bash
# Compare base vs tuned decode speed and peak memory.
set -euo pipefail
source "$(dirname "$0")/env.sh"

ART="$ROOT/artifacts"
echo "=== base: $MODEL_BASE ==="
"$MLX_LM" benchmark --model "$MODEL_BASE" -p 256 -g 128 2>&1 | tail -8 || true

if [ -d "$ART/final-4bit" ]; then
  echo
  echo "=== tuned: $ART/final-4bit ==="
  "$MLX_LM" benchmark --model "$ART/final-4bit" -p 256 -g 128 2>&1 | tail -8 || true
else
  echo "no tuned model yet; run fuse_quantize.sh"
fi
