# nanomite-ai — AGENTS.md

Local fast hybrid agent (Brent 1.0-4b) — LFM2.5-1.2B-Instruct QLoRA-finetuned MLX model for agentic/security tasks. Served via `mlx_lm.server` on port 8080. Python agent at `scripts/agent.py`; Rust harness at `scripts/` (fuse/quantize/eval/train helpers).

## Prerequisites

- **MLX**: `pip install mlx-lm` (includes `mlx` and `mlx-lm`).
- **Hardware**: Apple Silicon Mac with Metal GPU strongly recommended. CPU-only generation is ~72 s/token — do not use for serving.
- **Base model**: `artifacts/brent-1-0-4b-base/` (LFM2.5-1.2B-Instruct 4-bit, ~633 MB).
- **Adapter**: `artifacts/adapter_dox_meta/` (fine-tuned LoRA, 250 iters).
- **MLX server port**: 8080 (`SERVER` const in `scripts/agent.py`).

## Essential Commands

### Start server (run once per session)
```bash
cd /Users/ritchegerona/nanomite-ai
scripts/serve.sh          # mlx_lm.server on 127.0.0.1:8080, model = artifacts/brent-1-0-4b-base, adapter = artifacts/adapter_dox_meta
# env overrides: MODEL, ADAPTER, TEMP, NUM_DRAFT, CACHE_SIZE
```

### Single-shot agent query
```bash
python3 scripts/agent.py --prompt "your request here" --max-tokens 256 --no-thinking
```

Examples:
```
python3 scripts/agent.py --prompt "run nmap example.com" --no-thinking
python3 scripts/agent.py --prompt "run nikto -h example.com" --no-thinking
python3 scripts/agent.py --prompt "write a python script that prints hello" --no-thinking
```

- `--no-thinking` hides the reasoning block (default: shown).
- `--max-tokens` default 256; raise for longer answers.
- Output tags: `[run] <cmd>` then raw stdout; `[write] <path>`; `[read] <path>`.

### Interactive chat
```bash
python3 scripts/agent.py --prompt "..."   # single turn
```
There is no built-in REPL loop; call once per turn. Use `--prompt` for each request.

## Deterministic Tool Execution (no model turn)

`scripts/agent.py` matches intent patterns in the user prompt and executes tools directly — zero model turns for tool work. Supported actions:

| Pattern | Tool | Example |
|---|---|---|
| `run`/`execute` + command | `run_command` | `run nmap example.com` |
| `read`/`read file` | `read_file` | `read file README.md` |
| `list`/`list directory` | `list_dir` | `list directory` |
| `write`/`create`/`make` + code hint | `write_file` | `write a python script that prints hello` |
| `install` + tool name | `install_tool` | `install nmap` |
| `check`/`is ... installed` | `check_tool` | `is nmap installed` |
| `web`/`search` + URL | `web_fetch` | `search the web for https://example.com` |

Tool aliases: `nmap`, `nikto`, `sqlmap`, `hydra`, `john`, `aircrack-ng`, `hashcat`, `wifite`, `masscan`, `gobuster`, `dirb`, `zaproxy`, `burpsuite`, `metasploit`, `ncat`, `netcat`, `curl`, `wget`.

## Tool Calling from Model Output

The model may emit native tool calls in its response:

```json
{"name": "run_command", "arguments": {"command": "nmap example.com"}}
```

These are auto-parsed and executed. Raw JSON tool tags (`{"tool": ..., "arguments": ...}`) are also accepted.

## Model Artifacts

- `artifacts/brent-1-0-4b-base/` — base LFM2.5-1.2B-Instruct 4-bit (base weights).
- `artifacts/adapter_dox_meta/` — fine-tuned LoRA adapter (use this one).
- `artifacts/adapter_norefuse/` — older adapter (no-refusal tuned).
- `artifacts/adapter_multilingual/` — multilingual adapter.
- `artifacts/fused/` — fused bf16 model (2.2 GB, for Ollama export).
- `artifacts/nano_q40.gguf` / `nano_q4_K_M.gguf` — prebuilt GGUF exports.

### Rebuild adapter
```bash
scripts/train.sh              # 250 iters, LR 2e-5, batch 1×2, seq 512
scripts/build_sft.py --round 1   # rebuild SFT dataset from data/
scripts/fuse_quantize.sh         # fuse adapter into fused/
```

### Training data layout
- `data/seeds/` — expert-written gold answers (33 entries)
- `data/gold/` — verified gold traces
- `data/generated/` — round-N generated traces
- `data/sft/` — final train/valid/test JSONL

### Verifiers (`scripts/verifiers/`)
- `tool_call.py` — JSON tool-call schema validation
- `code_exec.py` — executes Python against assertions
- `security_label.py` — severity/category label check
- `chat_consistency.py` — must-mention / must-not-mention rubric

## Environment Variables

| Var | Default | Purpose |
|---|---|---|
| `MODEL` | `artifacts/brent-1-0-4b-base` | Base model path |
| `ADAPTER` | `artifacts/adapter_dox_meta` | LoRA adapter path |
| `TEMP` | `0.0` | Sampling temperature |
| `NUM_DRAFT` | `8` | Speculative draft tokens |
| `CACHE_SIZE` | `8` | KV cache entries |
| `DRAFT` | `LFM2-350M-4bit` | Draft model for speculative decoding |
| `ROOT` | (auto) | Project root (auto-detected) |

## Verified Behaviors

- Tool execution: nmap, nikto, sqlmap, hydra, john, aircrack-ng, hashcat, wifite, masscan, gobuster, dirb, zaproxy, burpsuite, metasploit, ncat, netcat, curl, wget — all run via `run_command` with allow-by-default policy.
- Code writing: writes raw code directly to file (no markdown fences) when prompt contains `write`/`create`/`make` + code hint.
- Web: `web_fetch` for any `http(s)://` URL.
- No deny-by-default: every registered tool is available; no path confinement; no command allow-list.

## Known Gotchas

- `mlx_lm.server` uses the **full model path** as model ID (not a short name): `model = "/Users/ritchegerona/nanomite-ai/artifacts/brent-1-0-4b-base"`.
- MLX **CPU** generation is ~72 s/token — do not use for serving. Run `scripts/serve.sh` in a macOS GUI login session where Metal works.
- MLX Compiler service can die in headless VMs (`MTLCompilerService` error); then fall back to `scripts/convert_cpu.py`.
- `mlx_lm` server must be running on port 8080 before `scripts/agent.py` will work (in-process fallback exists but is slow).
- Default max-tokens is 256; raise for longer answers.
- Output is plain text, no JSON (except embedded tool calls).
- Deterministic tool detection is prompt-based: `run nmap example.com` triggers `run_command`; `read file README.md` triggers `read_file`, etc.

## Key Files

| File | Purpose |
|---|---|
| `scripts/agent.py` | Main agent loop: intent detection, tool dispatch, chat fallback |
| `scripts/tools.py` | Tool registry: read_file, write_file, list_dir, run_command, web_fetch, check_tool, install_tool, extract_metadata, osint_username |
| `scripts/build_sft.py` | Build SFT dataset from data/gold + data/seeds |
| `scripts/train.sh` | QLoRA training (250 iters) |
| `scripts/fuse_quantize.sh` | Fuse adapter + quantize to 4-bit |
| `scripts/evaluate.py` | Evaluate model against held-out test |
| `scripts/run_all.sh` | Full pipeline: generate → SFT → train → fuse → eval |
| `scripts/serve.sh` | Start mlx_lm server with model + adapter |
| `scripts/serve_ollama.sh` | Serve fused GGUF via Ollama |
| `scripts/generate.py` | Generate traces (STaR step 1) |
| `scripts/convert_cpu.py` | CPU fallback conversion |
| `scripts/smoke.py` | Smoke test model across domains |
| `verifiers/` | Deterministic verifiers (tool_call, code_exec, security_label, chat_consistency) |
| `data/seeds/` | Expert gold answers (33 entries) |
| `data/gold/` | Verified gold traces |
| `data/sft/` | Final train/valid/test JSONL |
| `artifacts/brent-1-0-4b-base/` | Base LFM2.5-1.2B-Instruct 4-bit |
| `artifacts/adapter_dox_meta/` | Fine-tuned LoRA adapter (use this one) |
| `artifacts/fused/` | Fused bf16 model (for Ollama export) |
| `artifacts/nano_q40.gguf` | Q4_0 GGUF (~696 MB, portable) |
| `artifacts/nano_q4_K_M.gguf` | Q4_K_M GGUF (~731 MB) |
| `MODEL_CARD.md` | Full model documentation |
| `RUNBOOK.md` | Operational runbook |
| `BENCH.md` | Benchmark results |