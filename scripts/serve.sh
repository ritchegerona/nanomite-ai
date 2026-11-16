#!/usr/bin/env bash
# Serve the quantized model on an OpenAI-compatible endpoint.
# Optimized for speed on Apple A18 Pro / 8 GB RAM.
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${1:-$ROOT/artifacts/final-4bit}"
PORT="${PORT:-8080}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMP="${TEMP:-0.0}"              # 0.0 = greedy sampling (fastest)
NUM_DRAFT="${NUM_DRAFT:-8}"      # speculative draft tokens
CACHE_SIZE="${CACHE_SIZE:-8}"    # prompt cache entries (must stay a size, NOT bytes:
                                 # --prompt-cache-bytes creates an ArraysCache, which is
                                 # NOT trimmable → speculative decoding raises
                                 # "requires a trimmable prompt cache" and every
                                 # request returns an empty response)

if [ ! -d "$MODEL" ]; then
  echo "model dir not found: $MODEL (run fuse_quantize.sh)" >&2
  exit 1
fi

# Auto-detect the LFM2-350M-4bit draft model from the HF cache (same vocab
# 65536 + model_type lfm2 as nanomite-ai) for speculative decoding.
if [ -z "${DRAFT:-}" ]; then
  DRAFT="$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--LFM2-350M-4bit/snapshots/*/ 2>/dev/null | tail -1 || true)"
fi

# Build server args.
ARGS=(
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --temp "$TEMP"
  --max-tokens "$MAX_TOKENS"
  --prompt-cache-size "$CACHE_SIZE"
)
if [ -n "${DRAFT:-}" ] && [ -d "$DRAFT" ]; then
  ARGS+=(--draft-model "$DRAFT" --num-draft-tokens "$NUM_DRAFT")
  echo "[serve] $MODEL + draft $DRAFT (${NUM_DRAFT} draft tokens) on http://127.0.0.1:$PORT/v1"
else
  echo "[serve] $MODEL (no draft model found) on http://127.0.0.1:$PORT/v1"
fi

# Speed levers (measured on A18 Pro, 8 GB RAM, before the Metal failure):
#   - 4-bit weights (4.501 bpw) — small memory footprint, fast load
#   - speculative decoding: LFM2-350M-4bit draft + nanomite-ai target → ~1.5–2.5x tokens/sec
#   - temperature 0.0 = greedy sampling (fastest; no sampling computation)
#   - prompt cache (size 8, 1 GiB) — KV reuse for repeated prefixes
# Measured before failure: base 4-bit ~71 tok/s; speculative should push
# nanomite-ai toward ~100–150 tok/s on GPU. CPU fallback is ~72 s/token
# (unusable) — serve on a normal macOS session where Metal works.
exec "$MLX_LM" server "${ARGS[@]}"
