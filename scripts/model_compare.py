#!/usr/bin/env python3
"""Compare model performance across self-improvement rounds.

Reads eval JSON files and generates a comparison report showing
how performance has changed across rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ART  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare model eval results across rounds")
    ap.add_argument("eval_files", nargs="+", help="Eval JSON files to compare")
    ap.add_argument("--sort-by", default="overall", choices=["overall", "coding", "security", "agent", "chat"],
                    help="Sort by this metric")
    ap.add_argument("--output", default=None, help="Output JSON file")
    args = ap.parse_args()

    results = []
    for f in args.eval_files:
        with open(f) as fh:
            data = json.load(fh)
        data["file"] = f
        results.append(data)

    # Sort by the specified metric
    def sort_key(r):
        pr = r.get("pass_rate", {})
        return pr.get(args.sort_by, 0)

    results.sort(key=sort_key, reverse=True)

    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON (sorted by {args.sort_by})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Model':<40} {'Overall':>8} {'Agent':>8} {'Coding':>8} {'Security':>9} {'Chat':>6}")
    print(f"{'-'*80}")

    for i, r in enumerate(results, 1):
        pr = r.get("pass_rate", {})
        model_name = r.get("model", r.get("file", "unknown"))[-50:]
        print(f"{i:<6} {model_name:<40} "
              f"{pr.get('overall', 'N/A'):>8} "
              f"{pr.get('agent', 'N/A'):>8} "
              f"{pr.get('coding', 'N/A'):>8} "
              f"{pr.get('security', 'N/A'):>9} "
              f"{pr.get('chat', 'N/A'):>6}")

    print(f"\n{'='*80}")
    print("Best model:", results[0].get("model", results[0].get("file", "unknown")))
    print(f"{'='*80}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[model_compare] wrote {args.output}")


if __name__ == "__main__":
    main()