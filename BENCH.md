# Benchmarks — nanomite-ai

Hardware: Apple A18 Pro, 8 GB unified memory, macOS 27.0.1, MLX 0.31.3.
All numbers below were measured in-session on 2026-11-16, **before** the host's
Metal compiler service died (see caveat at the end).

## Model size

| Artifact | Size | Bits/weight |
|---|---|---|
| `artifacts/final-4bit` | 633 MB | 4.501 |
| `artifacts/fused` (bf16) | 2.2 GB | 16 |
| `artifacts/adapter_250` | 11 MB | — |
| `artifacts/nano_q40.gguf` | 696 MB | 4.74 |
| `artifacts/nano_q4_K_M.gguf` | 731 MB | 4.98 |

`model.safetensors` = 658,540,250 bytes. Budget: 700 MB → **67 MB headroom** (MLX);
GGUF Q4_0 → ~7 MB headroom.

## Throughput

| Run | Measurement | tok/s |
|---|---|---|
| base 4-bit, generation bench | single stream, 7-batch median | **71.6** |
| base 4-bit, held-out eval | generation throughput | 44.8 |
| tuned-250 adapter, held-out eval | generation throughput | 39.1 |
| tuned-250 adapter, test split | generation throughput | 41.1 |
| final 4-bit, MLX **CPU** | 1 token (incl. prefill) | 0.014 (72 s) |
| final 4-bit + speculative (draft LFM2-350M, 8 tokens) | expected, not yet measured | ~100–150 (est.) |

The adapter costs ~12% throughput at eval time. The fused 4-bit model was not
re-benchmarked after the Metal failure; it has the same architecture and size as
the base, so expect ≈ base throughput.

**Speed tuning** (via `scripts/serve.sh` env vars):

| Lever | Default | Effect |
|---|---|---|
| `TEMP=0.0` | 0.0 | Greedy sampling — fastest, no sampling overhead |
| `NUM_DRAFT=8` | 8 | Speculative draft tokens (higher = more speedup, more risk of rejection) |
| `DRAFT=<path>` | auto-detected | Draft model path; must share vocab/model_type with target |
| `CACHE_SIZE=8` | 8 | Prompt cache entries (KV reuse for repeated prefixes) |

## Memory

| Run | Peak RSS |
|---|---|
| base 4-bit inference | 0.70 GB |
| QLoRA training (8 layers, seq 512, batch 1 × accum 2) | 1.06 GB |

## Training

| Metric | Value |
|---|---|
| Iterations | 250 |
| Wall time | 2 min 51 s |
| Train loss | 0.82 → 0.28 |
| Best validation loss | 0.213 @ iter 200 |
| Data | 67 train / 20 valid / 20 test |

## Accuracy

Held-out 24 prompts: base 0.71 → tuned-250 **0.83**.
Test split 20 prompts: base 0.85 → tuned-250 **0.90**.

Per-domain and scoring details are in `README.md` and `MODEL_CARD.md`.

## Caveat

Mid-session the Metal compiler service became unreachable
(`Unable to reach MTLCompilerService`, invalidation 141, broken Mach bootstrap
in a headless VM). GPU inference, `mlx_lm.convert`, and Ollama's safetensors
import all failed afterward. llama.cpp (`llama-server`) also aborts on Metal
library creation; `num_gpu=0`, `OLLAMA_LLM_LIBRARY=cpu`, and
`GGML_METAL_TENSOR_DISABLE=1` do not prevent the abort in this build. The final
4-bit artifact was therefore produced with the CPU quantizer
(`scripts/convert_cpu.py`), numerically equivalent but Metal-free.

GGUF throughput could not be measured in this session for the same reason.
On a normal macOS login session, `nano_q40.gguf` (Q4_0, LFM2-supported by
llama.cpp) runs via `ollama serve` or `llama-server` on GPU or CPU.

Reproduce the GPU benchmarks in a normal macOS login session:

```bash
scripts/benchmark.sh
"$PY" scripts/evaluate.py --model artifacts/final-4bit
```
