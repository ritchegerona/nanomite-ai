#!/usr/bin/env bash
# Smoke test: run Brent 1.0 inside nanomite-ai and verify all tools work.
set -euo pipefail

ROOT="/Users/ritchegerona/nanomite-ai"
PY="$HOME/.local/pipx/venvs/mlx-lm/bin/python"
MODEL="$ROOT/artifacts/brent-1-0-4b-base"

echo "=== nanomite-ai Brent 1.0 smoke test ==="
echo

# 1. Ensure server is running
if ! curl -s http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "Starting mlx_lm server with Brent 1.0 + fine-tuned adapter..."
  nohup "$PY" -m mlx_lm server \
    --model "$MODEL" \
    --adapter-path "$ROOT/artifacts/adapter_lang" \
    --host 127.0.0.1 --port 8080 --temp 0.0 \
    > /tmp/nanomite_smoke_server.log 2>&1 &
  for i in $(seq 1 30); do
    curl -s http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
fi

echo "Server models:"
curl -s http://127.0.0.1:8080/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin); [print('  ', m['id']) for m in d['data']]"
echo

FAIL=0

run_test() {
  local name="$1"; shift
  echo "--- $name ---"
  "$@" 2>&1 | grep -v debug
  echo
}

# 2. Chat test with thinking
run_test "Chat test (with thoughts)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "Explain why the sky is blue, step by step." --max-tokens 256

# 3. write_file test
run_test "write_file test" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "Write 'smoke test ok' to /tmp/nanomite_smoke.txt" --max-tokens 128 --no-thinking
cat /tmp/nanomite_smoke.txt
echo

# 4. read_file test
run_test "read_file test" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "Read /tmp/nanomite_smoke.txt" --max-tokens 64 --no-thinking
echo

# 5. list_dir test
run_test "list_dir test" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "List the current directory" --max-tokens 64 --no-thinking | head -15
echo

# 6. run_command test (nmap)
run_test "run_command test (nmap)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt 'Run "nmap --version | head -1"' --max-tokens 64 --no-thinking
echo

# 7. web_fetch test
run_test "web_fetch test" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "Get the title from https://example.com" --max-tokens 64 --no-thinking
echo

# 8. check_tool test
run_test "check_tool test (nmap)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "is nmap installed" --max-tokens 64 --no-thinking
echo

# 9. install_tool test
run_test "install_tool test (hydra)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "install hydra" --max-tokens 64 --no-thinking
echo

# 10. Persistence memory test
run_test "Memory test (part 1)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "Remember that my favorite pentest tool is nmap" --max-tokens 128
run_test "Memory test (part 2)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "What is my favorite pentest tool?" --max-tokens 128
echo

# 11. Code write test (multi-language)
run_test "Code write test (Python)" \
  "$PY" "$ROOT/scripts/agent.py" --model "$MODEL" --prompt "write a python function that checks if a number is prime to /tmp/prime_test.py" --max-tokens 128 --no-thinking
cat /tmp/solution.py 2>/dev/null || cat /tmp/prime_test.py 2>/dev/null
echo

echo "=== All smoke tests completed ==="
