#!/usr/bin/env python3
"""Smoke-test a model dir (optionally on CPU) across all four domains."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import apply_chat, strip_stop  # noqa: E402
from verifiers import verify  # noqa: E402

CASES = [
    {
        "domain": "agent",
        "prompt": "You are a tool-calling agent. Tools: read_file(path), list_dir(path), run_command(cmd). Reply with ONLY a JSON object containing keys 'tool' and 'arguments'. The arguments object MUST include the required parameter. User: 'List the tests directory'.",
        "meta": {"allowed_tools": ["read_file", "list_dir", "run_command"], "required_args": ["path"]},
    },
    {
        "domain": "coding",
        "prompt": "Write only the Python function `square(n)` that returns n squared. Reply with code only, no explanation.",
        "meta": {"tests": "assert square(3) == 9\nassert square(-4) == 16"},
    },
    {
        "domain": "security",
        "prompt": "You are a security triage assistant. Categories: remote-access, cleartext, tls, access-control, vulnerable-software, dns, information-disclosure, authentication, reachability. Severity is high, medium, or low. Respond with a single JSON object with exactly these keys in this order: category, severity, confidence. Finding: 'The Kubernetes API server allows anonymous authentication.'",
        "meta": {"expected": {"severity": "high", "category": "access-control"}, "required_keys": ["severity", "category", "confidence"]},
    },
    {
        "domain": "chat",
        "prompt": "What port does DNS use by default?",
        "meta": {"must_contain": ["53"]},
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    if args.cpu:
        import contextlib
        import sys

        import mlx.core as mx

        mx.set_default_device(mx.cpu)

        # mlx-lm's wired_limit reads GPU device_info; CPU has no such key.
        @contextlib.contextmanager
        def _wired_limit(*_a, **_k):
            yield

        _mlxgen = sys.modules.get("mlx_lm.generate")
        if _mlxgen is None:
            import mlx_lm.generate  # noqa: F401

            _mlxgen = sys.modules["mlx_lm.generate"]
        _mlxgen.wired_limit = _wired_limit

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)
    ok_count = 0
    for case in CASES:
        prompt = apply_chat(tok, case["prompt"])
        t = time.time()
        out = strip_stop(
            generate(model, tok, prompt, max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        )
        dt = time.time() - t
        ok, why = verify(case["domain"], out, case["meta"])
        ok_count += ok
        print(f"\n[{case['domain']}] {'PASS' if ok else 'FAIL:' + why}  ({dt:.1f}s)")
        print(out[:500])
    print(f"\nSmoke: {ok_count}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
