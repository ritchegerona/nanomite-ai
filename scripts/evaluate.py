#!/usr/bin/env python3
"""Measure verifier pass-rate and decode speed for a model on held-out data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import ART, DATA, apply_chat, base_model, read_jsonl, strip_stop  # noqa: E402
from verifiers import verify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["test", "valid"], default="test")
    ap.add_argument("--file", default=None, help="explicit eval jsonl path")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        import contextlib
        import sys as _sys

        import mlx.core as mx
        import mlx_lm.generate as _mlxgen  # noqa: F401

        mx.set_default_device(mx.cpu)

        @contextlib.contextmanager
        def _wired_limit(*_a, **_k):
            yield

        _sys.modules["mlx_lm.generate"].wired_limit = _wired_limit

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=args.temp, top_p=0.95)
    model_path = args.model or base_model()
    eval_file = Path(args.file) if args.file else DATA / "sft" / f"{args.split}_eval.jsonl"
    rows = read_jsonl(eval_file)
    if args.limit:
        rows = rows[: args.limit]

    model, tok = load(model_path, adapter_path=args.adapter)
    total = Counter()
    passed = Counter()
    tokens = 0
    t0 = time.time()
    failures: list[dict] = []

    for i, row in enumerate(rows, 1):
        prompt = apply_chat(tok, row["prompt"])
        out = strip_stop(
            generate(
                model,
                tok,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
        )
        ok, why = verify(row["domain"], out, row.get("meta", {}))
        total[row["domain"]] += 1
        if ok:
            passed[row["domain"]] += 1
        else:
            failures.append(
                {"id": row["id"], "domain": row["domain"], "reason": why, "output": out[:300]}
            )
        tokens += len(tok.encode(out)) if out else 0
        print(f"[eval] {i}/{len(rows)} {row['domain']} {'PASS' if ok else 'FAIL:' + why}", flush=True)

    elapsed = max(1e-9, time.time() - t0)
    overall = sum(passed.values()) / max(1, sum(total.values()))
    report = {
        "model": model_path,
        "adapter": args.adapter,
        "split": args.split,
        "file": str(eval_file),
        "n": sum(total.values()),
        "pass_rate": {d: round(passed[d] / total[d], 3) for d in sorted(total)},
        "overall": round(overall, 3),
        "tokens_per_sec": round(tokens / elapsed, 1),
        "seconds": round(elapsed, 1),
        "failures": failures,
    }
    print("\n" + json.dumps(report, indent=2))
    out = Path(args.json_out) if args.json_out else ART / f"eval_{Path(model_path).name}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
