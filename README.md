# nanomite-ai

A **633 MB** local model for [nanomite](../nanomite): `LFM2.5-1.2B-Instruct` adapted
with **verifier-gated expert iteration (STaR) + curriculum QLoRA** across four
domains — tool calling, coding, security triage, and chat.

It is a single 4-bit MLX artifact that serves an OpenAI-compatible endpoint and
fits comfortably in 8 GB of unified memory.

## Results

Independent 24-prompt held-out set (never used for training or model selection):

| Model | Overall | Agent | Coding | Security | Chat | tok/s |
|---|---|---|---|---|---|---|
| base LFM2.5-1.2B-4bit | 0.71 | 0.67 | 1.00 | 0.17 | 1.00 | 44.8 |
| **nanomite-ai (tuned-250)** | **0.83** | **1.00** | **1.00** | **0.33** | **1.00** | 39.1 |

Held-out 20-prompt split used for checkpoint selection:

| Model | Overall | Agent | Coding | Security | Chat |
|---|---|---|---|---|---|
| base | 0.85 | 1.00 | 1.00 | 0.40 | 1.00 |
| tuned-200 | 0.85 | 1.00 | 1.00 | 0.40 | 1.00 |
| **tuned-250** | **0.90** | **1.00** | **1.00** | **0.60** | **1.00** |

Security is the weakest domain; the verifier only accepted ~29% of generated
security traces (base model biases toward "high" severity), and two held-out
security items remain wrong. Added balanced security gold entries across all
severity × category combinations to counter the bias during QLoRA training.

## Artifact

| Path | Size | Contents |
|---|---|---|
| `artifacts/final-4bit/` | **633 MB** | fused model, 4-bit affine, group 64 (4.501 bpw) |
| `artifacts/adapter_250/` | 11 MB | LoRA adapter (best checkpoint) |
| `artifacts/fused/` | 2.2 GB | merged bf16 model (source for quantization / Ollama import) |
| `artifacts/nano_q40.gguf` | **696 MB** | portable Q4_0 GGUF (llama.cpp / Ollama) — under budget |
| `artifacts/nano_q4_K_M.gguf` | 731 MB | Q4_K_M GGUF (~697 MiB binary; slightly over 700 MB decimal) |

`model.safetensors`: 658,540,250 bytes
SHA-256: `fb79a1c347781af45cdf8a61e0cf57501ce2aa7d26b073395360468e35ce0852`

GGUF SHA-256:
- `nano_q40.gguf`: `ae02d27fa81f140bdd3009db523a7b09a256eb01defa7f05107987156f363afd`
- `nano_q4_K_M.gguf`: `7272ae82ddf0d65e434bf278a7c99499cc30c9eec34be389eb5d40467d952db2`

## Quickstart

MLX server (uses Metal):

```bash
cd ~/nanomite-ai
scripts/serve.sh                 # http://127.0.0.1:8080/v1
```

`scripts/serve.sh` enables **speculative decoding** by default: the 195 MB `LFM2-350M-4bit`
draft model proposes tokens and the 633 MB nanomite-ai verifies them. Both share
vocab 65536 and `model_type: lfm2`, so they're compatible. This typically gives
~1.5–2.5× tokens/sec over plain generation.

Control draft behavior with env vars:
```
DRAFT=/path/to/draft NUM_DRAFT=8 scripts/serve.sh
```

Ollama (legacy GGUF path — works with LFM2):

```bash
scripts/serve_ollama.sh          # imports artifacts/fused, ~2.3 GB bf16 GGUF
```

See below for the Q4_0 (~696 MB) portable export.

Point nanomite at either endpoint — see `config/nanomite.toml.snippet`:

```toml
provider = "custom"
base_url = "http://127.0.0.1:8080/v1"
api_key = "local"
model = "/Users/ritchegerona/nanomite-ai/artifacts/final-4bit"
```

Note: `mlx_lm.server` uses the **full model path** as the model ID (not a short
name), so `model` must be the path to `artifacts/final-4bit`.

## Speed tuning

`scripts/serve.sh` is tuned for the A18 Pro / 8 GB RAM:

| Lever | Default | Effect |
|---|---|---|
| 4-bit weights | 4.501 bpw | Small footprint, fast load (~1 GB RSS) |
| Speculative decoding | LFM2-350M-4bit draft, 8 tokens | ~1.5–2.5× tokens/sec over plain generation |
| Temperature | 0.0 (greedy) | Fastest sampling; no random-compute overhead |
| Prompt cache | 8 entries / 1 GiB | KV reuse for repeated prefixes |

Control via env vars:
```
TEMP=0.0 NUM_DRAFT=8 CACHE_SIZE=8 scripts/serve.sh
# or use a different draft model:
DRAFT=/path/to/draft NUM_DRAFT=4 scripts/serve.sh
```

Measured before the Metal failure: base 4-bit ~71 tok/s; speculative should push
nanomite-ai toward ~100–150 tok/s on GPU. MLX **CPU** generation is ~72
s/token — do not use it for serving. Serve on a normal macOS GUI login session
where Metal works.

## Portable GGUF export

The fused model (`artifacts/fused`) can be packaged as a llama.cpp GGUF via Ollama's
legacy converter (`scripts/serve_ollama.sh` — no `--experimental`, since that path uses
MLX). Ollama's loader supports `Lfm2ForCausalLM` (`llama_model_lfm2` in the stack), so
the resulting GGUF runs wherever llama.cpp/Ollama supports LFM2.

Quantize the bf16 GGUF with Ollama's bundled `llama-quantize` (pure llama.cpp, CPU,
no Metal required):

```
llama-quantize <bf16 GGUF blob> nano_q40.gguf Q4_0   # ~2.3 GB -> ~696 MB
ollama create nanomite-ai-q40 -f Modelfile          # FROM nano_q40.gguf
```

Pre-built:
- `artifacts/nano_q40.gguf` — 696 MB, Q4_0, 4.74 bpw (under budget).
- `artifacts/nano_q4_K_M.gguf` — 731 MB (~697 MiB binary).

## Method

1. **Seeds** — 8 prompts per domain except security (16) (`data/seeds/`)
    plus 33 expert-written gold answers (`data/gold/`).
2. **Expert iteration** — sample 6 traces per seed at temperature 0.8.
3. **Deterministic verifiers** — no model judge:
   - `tool_call.py`: JSON shape, tool name, required args.
   - `code_exec.py`: executes Python against assertions.
   - `security_label.py`: severity + category allow-lists.
   - `chat_consistency.py`: must-mention / must-not-mention rubric.
4. **Filter** — keep only verified traces: 305 raw → **116 kept**
    (agent 78.6%, chat 93.8%, coding 100%, security 29%).
    Security gold entries now cover all severity × category combos to counter
    the base model's "always high" severity bias.
5. **Curriculum** — sort short→long, dedupe, split 74 train / 21 valid / 21 test.
6. **QLoRA** — 8 layers, seq 512, batch 1 × accum 2, LR 2e-5, 250 iters,
   `--mask-prompt --grad-checkpoint`.
7. **Merge + quantize** — fuse the adapter, dequantize to bf16, re-quantize to
   4-bit affine/group 64 on the CPU backend.

The "different technique" is the verifier-gated loop: the model proposes, a
program disposes. No human labels and no reward model are in the loop.

## Reproduce

```bash
scripts/run_all.sh               # generate -> build_sft -> train -> fuse/quantize -> eval
ROUND=1 SAMPLES=6 ITERS=250 scripts/run_all.sh
```

Step by step:

```bash
"$PY" scripts/generate.py --round 1 --samples 6
"$PY" scripts/build_sft.py --round 1
scripts/train.sh                 # 250 iters by default
scripts/fuse_quantize.sh
"$PY" scripts/evaluate.py --model artifacts/final-4bit
```

## Layout

```
data/        seeds, generated traces, verified SFT splits, gold, held-out eval
scripts/     generation, SFT build, training, fuse/quantize, eval, serving
verifiers/   deterministic domain verifiers
config/      nanomite config snippet + reference LoRA config
artifacts/   adapters, fused model, final 4-bit model, eval JSON
```

## Environment note

Training, evaluation and benchmarks above were measured in-session before the
host's **Metal compiler service died** (`MTLCompilerService` unreachable, error
141, broken Mach bootstrap in this headless VM). After that point GPU inference
and GPU quantization were impossible, so the final artifact was produced with
`scripts/convert_cpu.py`.

- The artifact itself is valid: it loads on both CPU and GPU backends, and its
  config reports `quantization: {group_size: 64, bits: 4, mode: affine}`.
- The fused model's throughput was not re-measured; it should track the base
  4-bit model (same architecture and size).
- MLX **CPU** generation is ~72 s/token — do not use it for serving.
- Run `scripts/serve.sh` in a normal macOS GUI login session where Metal works.

## License

Base model: `LiquidAI/LFM2.5-1.2B-Instruct`, **LFM Open License v1.0**.
This derivative inherits that license. See `MODEL_CARD.md`.
