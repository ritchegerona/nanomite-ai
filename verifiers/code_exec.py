"""Verify generated Python code by executing it against meta['tests'].

Runs code in a throwaway directory with a short timeout. This is a local-data
sanity check, not a hardened sandbox.
"""
import os
import re
import subprocess
import sys
import tempfile
from typing import Any


def extract_code(text: str) -> str:
    """Extract code from markdown fences or return raw text."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def verify(output: str, meta: dict) -> tuple[bool, str]:
    code = extract_code(output)
    if not code.strip():
        return False, "no_code"

    tests = meta.get("tests", "")
    if not tests:
        return False, "no_tests"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution_test.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests + "\n")
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=10,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:
            return False, f"error:{type(exc).__name__}"

    if proc.returncode == 0:
        return True, "ok"
    detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[-200:]
    return False, f"tests_failed:{detail}"