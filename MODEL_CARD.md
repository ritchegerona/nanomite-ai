# Model Card — nanomite-ai

## Model details

| Field | Value |
|---|---|
| Name | `nanomite-ai` |
| Base | `mlx-community/LFM2.5-1.2B-Instruct-4bit` (from `LiquidAI/LFM2.5-1.2B-Instruct`) |
| Architecture | LFM2 (`Lfm2ForCausalLM`): 16 layers, hidden 2048, vocab 65536, conv + full-attention mix |
| Adaptation | QLoRA, 8 layers, rank/alpha from mlx-lm defaults |
| Final format | MLX safetensors, 4-bit affine, group size 64, 4.501 bits/weight |
| Size | 633 MB on disk; `model.safetensors` = 658,540,250 bytes |
| SHA-256 | `fb79a1c347781af45cdf8a61e0cf57501ce2aa7d26b073395360468e35ce0852` |
| License | LFM Open License v1.0 (inherited) |
| Language | Predominantly English training data; base is multilingual |

## Training

**Method: verifier-gated expert iteration (STaR).** The base model samples its
own traces; deterministic programs accept or reject them; only accepted traces
are trained on.

- Seeds: 8 per domain except security (16); 40 total.
- Gold: 33 expert-written answers used as few-shot anchors and eval references.
- Sampling: 6 per seed, temperature 0.8.
- Verification:
  - tool calls parsed as JSON, checked against the tool schema and required args;
  - Python code executed against unit assertions;
  - security answers checked for an allowed severity and category;
  - chat answers checked against must-mention / must-not-mention rubrics.
- Yield: 305 traces → 116 verified (74 train / 21 valid / 21 test).
- Security gold entries now cover all severity × category combinations to
  counter the base model's "always high" severity bias.
- Curriculum: sorted by length, deduplicated.
- Optimization: 250 iterations, batch 1 × grad-accum 2, LR 2e-5, seq 512,
  `mask_prompt`, `grad_checkpoint`; peak memory 1.06 GB; 2 min 51 s.
- Loss: train 0.82 → 0.28; best validation 0.213 at iteration 200.

## Evaluation

Independent held-out set (24 prompts, never trained on and not used for
checkpoint selection):

| Model | Overall | Agent | Coding | Security | Chat |
|---|---|---|---|---|---|
| base | 0.71 | 0.67 | 1.00 | 0.17 | 1.00 |
| tuned-250 | **0.83** | **1.00** | **1.00** | **0.33** | **1.00** |

Checkpoint-selection split (20 prompts):

| Model | Overall | Security |
|---|---|---|
| base | 0.85 | 0.40 |
| tuned-200 | 0.85 | 0.40 |
| tuned-250 | **0.90** | **0.60** |

Scoring is exact-match / rubric-based and intentionally strict. Coding and chat
saturate at 1.00; agent calling improves from 0.67 to 1.00; security remains the
bottleneck.

## Portable export

The model exports cleanly to llama.cpp GGUF — Ollama's legacy converter supports
`Lfm2ForCausalLM` (confirmed via `llama_model_lfm2::load_arch_tensors` in the run
stack). Pre-built:

| Artifact | Size | Format | Notes |
|---|---|---|---|
| `artifacts/nano_q40.gguf` | 696 MB | Q4_0 GGUF | under 700 MB budget |
| `artifacts/nano_q4_K_M.gguf` | 731 MB | Q4_K_M GGUF | ~697 MiB binary |

Quantize with Ollama's bundled `llama-quantize` (pure llama.cpp, CPU, no Metal):

```
llama-quantize <bf16 GGUF blob> nano_q40.gguf Q4_0
ollama create nanomite-ai-q40 -f Modelfile   # FROM nano_q40.gguf
```

## Intended use

- Local, offline assistant inside the `nanomite` terminal agent.
- Tool-call proposal (JSON arguments), short coding tasks, security **triage**
  (severity/category suggestions), general chat.
- Resource-constrained machines: 4-bit weights, <1 GB model memory.

## Out of scope

- Autonomous exploitation, payload generation, or any privileged action. Model
  output is **untrusted data**; nanomite's `ToolPolicy` and confirmation gates
  remain the authority.
- High-stakes security decisions without human review.
- Production code changes without tests and review.
- Medical, legal, or financial advice.

## Limitations

- **Small, synthetic dataset** (107 traces): narrow coverage, easy to overfit.
- **Security accuracy is low** (33% held-out): severity/category labels are the
  main failure mode.
- **Verifier bias**: the model learns to satisfy the verifiers, not necessarily
  ground truth; over-strict verifiers discard good traces.
- **Distribution shift**: prompts unlike the seeds will degrade.
- **No safety/alignment tuning** beyond the base model.
- **English-centric** training data.
- **Throughput**: ~39–45 tok/s on the A18 Pro GPU (measured); CPU MLX is
  ~72 s/token and unusable.

## Ethical and security considerations

- Treat every generation as attacker-controlled input.
- Never let the model authorize tools, scope changes, or credential access.
- The training data contains no credentials, captured traffic, or real target
  data; security examples are synthetic.
- Redistribution must comply with the LFM Open License v1.0.

## Citation

```
Base: LiquidAI/LFM2.5-1.2B-Instruct (LFM Open License v1.0)
Adaptation: nanomite-ai, verifier-gated STaR + QLoRA, 2026-11-16
```
