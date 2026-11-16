#!/usr/bin/env bash
# Import the fused model into Ollama via the legacy (llama.cpp/GGUF) path and serve it.
# The --experimental/-q path uses MLX and fails in this environment; the legacy path
# produces a valid LFM2 GGUF (stack: llama_model_lfm2::load_arch_tensors).
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_NAME="${MODEL_NAME:-nanomite-ai}"
FUSED="${FUSED:-$ROOT/artifacts/fused}"

if [ ! -d "$FUSED" ]; then
  echo "missing $FUSED; run scripts/fuse_quantize.sh first" >&2
  exit 1
fi

if ! curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "[ollama] starting server"
  (ollama serve > /tmp/ollama.log 2>&1 &)
  sleep 3
fi

cat > "$ROOT/artifacts/Modelfile" <<EOF
FROM $FUSED
EOF

echo "[ollama] importing $FUSED as GGUF ($MODEL_NAME-bf16)"
ollama create "$MODEL_NAME-bf16" -f "$ROOT/artifacts/Modelfile"
ollama list
echo
echo "bf16 model ($MODEL_NAME-bf16) created (~2.3 GB)."
echo "To quantize to Q4_0 (~696 MB) with the bundled llama-quantize (CPU, no Metal):"
echo "  llama-quantize <bf16 GGUF blob> nano_q40.gguf Q4_0"
echo "  ollama create nanomite-ai-q40 -f <Modelfile FROM nano_q40.gguf>"
echo "OpenAI endpoint: http://127.0.0.1:11434/v1   model=$MODEL_NAME-bf16 (or -q40)"
echo "NOTE: serving requires a working Metal session (llama-server aborts on MTLCompilerService failure)."
