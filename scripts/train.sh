#!/usr/bin/env bash
# QLoRA fine-tune LFM2.5-1.2B on the verified STaR dataset.
set -euo pipefail
source "$(dirname "$0")/env.sh"

ITERS="${ITERS:-250}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-2}"
LR="${LR:-2e-5}"
SEQ="${SEQ:-512}"
LAYERS="${LAYERS:-8}"

mkdir -p "$ROOT/artifacts/adapter"
cd "$ROOT"

echo "[train] base=$MODEL_BASE"
echo "[train] data=$ROOT/data/sft iters=$ITERS batch=$BATCH accum=$ACCUM seq=$SEQ layers=$LAYERS"

"$MLX_LM" lora \
  --model "$MODEL_BASE" \
  --train \
  --data "$ROOT/data/sft" \
  --fine-tune-type lora \
  --num-layers "$LAYERS" \
  --batch-size "$BATCH" \
  --iters "$ITERS" \
  --learning-rate "$LR" \
  --max-seq-length "$SEQ" \
  --mask-prompt \
  --grad-checkpoint \
  --grad-accumulation-steps "$ACCUM" \
  --adapter-path "$ROOT/artifacts/adapter" \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --save-every 50 \
  --val-batches 1 \
  --test
