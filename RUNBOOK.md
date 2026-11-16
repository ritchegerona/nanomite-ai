# Runbook — Brent 1.0-4b local model

Serve the quantized model on an OpenAI-compatible endpoint, then call it.

## Prerequisites

- Python with `mlx_lm` installed:
  ```
  ~/.local/pipx/venvs/mlx-lm/bin/python -c "import mlx_lm"
  ```
- Model artifact present: `artifacts/brent-1-0-4b/` (4-bit affine, group 64)

## 1. Serve the model

Run once; it stays up serving `http://127.0.0.1:8080/v1`.

**Single command:**
```bash
nanomite-ai
```
This installs via `~/.local/bin/nanomite-ai` (on your PATH) and serves
`artifacts/brent-1-0-4b` on port 8080 with speed optimizations:
- `temp 0.0` — greedy sampling (fastest, no sampling overhead)
- `max-tokens 64` — bounded generation for fast responses (override with `NANOMITE_MAX_TOKENS`)
- `prompt-cache-size 8` — KV cache reuse for repeated prefixes

**Manual:**
```bash
cd /Users/ritchegerona/nanomite-ai
~/.local/pipx/venvs/mlx-lm/bin/mlx_lm server \
  --model artifacts/brent-1-0-4b \
  --host 127.0.0.1 \
  --port 8080 \
  --temp 0.0 \
  --max-tokens 64 \
  --prompt-cache-size 8
```

Only `artifacts/brent-1-0-4b` is served. Other models were removed from the
HuggingFace cache so `/v1/models` returns only Brent 1.0-4b.

## 2. Call the endpoint (OpenAI-compatible chat)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/Users/ritchegerona/nanomite-ai/artifacts/brent-1-0-4b",
    "messages": [{"role": "user", "content": "What port does DNS use by default?"}],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

Expected output (non-empty, fast):
```
{"...","choices":[{"message":{"role":"assistant","content":"By default, DNS uses the UDP port 53."}}]}
```

List served models:
```bash
curl http://127.0.0.1:8080/v1/models
```

## 3. Point nanomite at it

Merge `config/nanomite.toml.snippet` into `~/.config/nanomite/config.toml`:

```toml
provider = "custom"
base_url = "http://127.0.0.1:8080/v1"
api_key = "local"
model = "/Users/ritchegerona/nanomite-ai/artifacts/brent-1-0-4b"
```

## 4. Quick validation (direct MLX, CPU — slow ~72 s/token, not for serving)

```bash
~/.local/pipx/venvs/mlx-lm/bin/python scripts/smoke.py \
  --model artifacts/brent-1-0-4b --cpu --max-tokens 24
```

## 5. Tool-using agent (read / write / run / search web)

Give Brent 1.0-4b tool-calling ability so it can read files, write code to disk,
run any command, and fetch the web — all tasks allowed, no restrictions:

```bash
~/.local/pipx/venvs/mlx-lm/bin/python scripts/agent.py \
  --model artifacts/brent-1-0-4b \
  --prompt "Read README.md and tell me the quickstart command." \
  --max-steps 6
```

Examples:
```bash
# Write code anywhere, then run it
~/.local/pipx/venvs/mlx-lm/bin/python scripts/agent.py \
  --model artifacts/brent-1-0-4b \
  --prompt "Write square(n) to /tmp/demo.py, run it, confirm square(4)==16."

# Run nmap or any command
~/.local/pipx/venvs/mlx-lm/bin/python scripts/agent.py \
  --model artifacts/brent-1-0-4b \
  --prompt "Run 'nmap --version' and tell me the version."

# Fetch a web page
~/.local/pipx/venvs/mlx-lm/bin/python scripts/agent.py \
  --model artifacts/brent-1-0-4b \
  --prompt "Use web_fetch to get the title from https://example.com"
```

Available tools (registered in `scripts/tools.py`, all tasks allowed):
- `read_file(path)` — read any file anywhere
- `write_file(path, content)` — write to any path
- `list_dir(path)` — list any directory
- `run_command(cmd)` — run any command (shell enabled, no allow-list)
- `web_fetch(url)` — http/https GET, 15s timeout

Every call is redacted (secrets stripped) and logged to
`artifacts/agent_log.jsonl` for evidence. Tool bugs or network failures never
crash the loop.

## 6. Thinking (visible reasoning)

The agent emits a `<thoughts>...</thoughts>` reasoning block before each tool
call or final answer, shown as `[thoughts]`:

```bash
~/.local/pipx/venvs/mlx-lm/bin/python scripts/agent.py \
  --model artifacts/brent-1-0-4b \
  --prompt "What is 2+2? Think step by step."
```

Example output:
```
[thoughts] The operation is straightforward addition.
Let's calculate 2 + 2 step by step.
...
The final answer is 4.
```

Thoughts are on by default. Hide them with `--no-thinking`. The parser also
accepts `<thinking>` and `<reasoning>` tags. Note: thought quality depends on
the 1.2B base model — reasoning is best-effort, not guaranteed.

## Notes

- Metal GPU is required for usable speed (~70 tok/s). CPU fallback is ~72 s/token.
- `brent-1-0-4b` config reports `quantization: {group_size: 64, bits: 4, mode: affine}`.
- The model ID passed to the endpoint must be the **full path** to `artifacts/brent-1-0-4b` (not a short name).
- For fastest responses, keep `max_tokens` small (default 64) and `temperature 0.0`.
- Response speed is bounded by the 1.2B model on Metal GPU (~35–50 tok/s).
  Thoughts add extra tokens, so disable them (`--no-thinking`) for maximum speed.
