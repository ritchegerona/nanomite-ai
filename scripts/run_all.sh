#!/usr/bin/env bash
# End-to-end pipeline: generate -> build_sft -> train -> fuse/quantize -> eval.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROOT"

ROUND="${ROUND:-1}"
SAMPLES="${SAMPLES:-6}"
ITERS="${ITERS:-250}"

echo "== 1/5 generate =="
"$PY" scripts/generate.py --round "$ROUND" --samples "$SAMPLES" ${DRAFT:+--draft}

echo "== 2/5 build_sft =="
"$PY" scripts/build_sft.py --round "$ROUND"

echo "== 3/5 train =="
ITERS="$ITERS" scripts/train.sh

echo "== 4/5 fuse + quantize =="
scripts/fuse_quantize.sh

echo "== 5/5 evaluate =="
"$PY" scripts/evaluate.py --model "$ROOT/artifacts/final-4bit" | tail -20

echo "done. serve with: scripts/serve.sh"
