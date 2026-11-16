#!/usr/bin/env bash
# Merge the LoRA adapter into the base, then re-quantize to 4-bit.
set -euo pipefail
source "$(dirname "$0")/env.sh"

ART="$ROOT/artifacts"
ADAPTER="${ADAPTER:-$ART/adapter}"
if [ ! -d "$ADAPTER" ]; then
  echo "missing $ADAPTER; run scripts/train.sh first" >&2
  exit 1
fi

rm -rf "$ART/fused" "$ART/final-4bit"
echo "[fuse] dequantized merge from $ADAPTER"
"$MLX_LM" fuse \
  --model "$MODEL_BASE" \
  --adapter-path "$ADAPTER" \
  --save-path "$ART/fused" \
  --dequantize

echo "[quant] 4-bit affine, group 64 (CPU backend)"
# CPU avoids MTLCompilerService, which is unavailable in headless sessions.
"$PY" "$ROOT/scripts/convert_cpu.py" \
  --model "$ART/fused" \
  --mlx-path "$ART/final-4bit" \
  --q-bits 4 --q-group-size 64

du -sh "$ART/final-4bit" "$ART/fused"
"$PY" - "$ART/final-4bit" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
cfg = json.loads((p / "config.json").read_text())
print("model_type:", cfg.get("model_type"), "quant:", cfg.get("quantization"))
size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
print(f"on-disk: {size/1e6:.0f} MB")
EOF
