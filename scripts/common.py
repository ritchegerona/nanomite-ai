"""Shared helpers for the nanomite-ai training pipeline."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def base_model() -> str:
    env = os.environ.get("MODEL_BASE")
    if env:
        return env
    pats = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-4bit/snapshots/*/"
        )
    )
    if not pats:
        raise SystemExit("base model not found in HF cache; set MODEL_BASE")
    return sorted(pats)[-1]


def draft_model() -> str | None:
    env = os.environ.get("DRAFT")
    if env:
        return env or None
    pats = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--LFM2-350M-4bit/snapshots/*/"
        )
    )
    return sorted(pats)[-1] if pats else None


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_chat(tok, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


def strip_stop(text: str) -> str:
    for s in ("<|im_end|>", "<|endoftext|>", "<|startoftext|>"):
        text = text.split(s)[0]
    return text.strip()
