#!/usr/bin/env python3
"""STaR step 2: verify, dedupe, curriculum-order and split traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import DATA, read_jsonl, write_jsonl  # noqa: E402
from verifiers import verify  # noqa: E402


def to_chat(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def to_eval(row: dict) -> dict:
    return {
        "id": row["id"],
        "domain": row["domain"],
        "prompt": row["prompt"],
        "meta": row.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    raw: list[dict] = []
    for gen_path in sorted((DATA / "generated").glob(f"round{args.round}*.jsonl")):
        raw += read_jsonl(gen_path)
    for gold_path in sorted((DATA / "gold").glob("*.jsonl")):
        raw += read_jsonl(gold_path)
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    attempts = Counter()
    passed = Counter()
    reasons = Counter()

    for row in raw:
        attempts[row["domain"]] += 1
        ok, why = verify(row["domain"], row["output"], row.get("meta", {}))
        reasons[f"{row['domain']}:{why}"] += 1
        if not ok:
            continue
        passed[row["domain"]] += 1
        digest = hashlib.sha1(row["output"].strip().encode("utf-8")).hexdigest()
        key = (row["id"], digest)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    if not kept:
        raise SystemExit("no verified traces; loosen seeds or increase --samples")

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        by_domain[row["domain"]].append(row)

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []
    for domain, rows in by_domain.items():
        # curriculum proxy: shorter answers first (easier)
        rows.sort(key=lambda r: len(r["output"]))
        n = len(rows)
        if n >= 3:
            n_test = max(1, n // 5)
            n_valid = max(1, n // 5)
            test += rows[-n_test:]
            valid += rows[-(n_test + n_valid) : -n_test]
            train += rows[: -(n_test + n_valid)]
        else:
            train += rows

    # guarantee mlx_lm has at least one validation/test example
    if not valid:
        valid = train[:1]
    if not test:
        test = train[-1:]

    train.sort(key=lambda r: len(r["output"]))  # curriculum: easy -> hard
    sft = DATA / "sft"
    write_jsonl(sft / "train.jsonl", [to_chat(r) for r in train])
    write_jsonl(sft / "valid.jsonl", [to_chat(r) for r in valid])
    write_jsonl(sft / "test.jsonl", [to_chat(r) for r in test])
    write_jsonl(sft / "valid_eval.jsonl", [to_eval(r) for r in valid])
    write_jsonl(sft / "test_eval.jsonl", [to_eval(r) for r in test])

    stats = {
        "raw": len(raw),
        "kept": len(kept),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "attempts": dict(attempts),
        "passed": dict(passed),
        "pass_rate": {
            d: round(passed[d] / attempts[d], 3) for d in sorted(attempts)
        },
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:40]),
    }
    (ROOT / "artifacts" / "data_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
