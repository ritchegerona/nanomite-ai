#!/usr/bin/env python3
"""Self-improvement training loop for Brent 1.0.

Implements STaR (Self-Taught Reasoner) with iterative self-improvement:
1. Generate candidate answers using current model
2. Verify answers with deterministic verifiers
3. Retrain on verified traces
4. Repeat for N iterations with increasing difficulty

Usage:
    python3 scripts/self_train.py --rounds 5 --iters 250 --samples 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, apply_chat, base_model, draft_model, read_jsonl, write_jsonl


def run_cmd(cmd: list[str], description: str) -> None:
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"[self-train] {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        raise SystemExit(1)
    print(f"\n[OK] {description} completed")


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-improvement training loop for Brent 1.0"
    )
    ap.add_argument("--rounds", type=int, default=3, help="Self-improvement rounds")
    ap.add_argument("--iters", type=int, default=250, help="Training iterations per round")
    ap.add_argument("--samples", type=int, default=4, help="Samples per seed per round")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--skip-generate", action="store_true", help="Skip generation step")
    ap.add_argument("--skip-build", action="store_true", help="Skip build_sft step")
    ap.add_argument("--skip-train", action="store_true", help="Skip training step")
    ap.add_argument("--domain", type=str, default=None, help="Specific domain to train on")
    args = ap.parse_args()

    ROOT_DATA = DATA
    sft_dir = DATA / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    # Track model path across rounds
    current_model = str(base_model())
    print(f"[self-train] Base model: {current_model}")
    print(f"[self-train] Rounds: {args.rounds}, Iters per round: {args.iters}")
    print(f"[self-train] Samples per seed: {args.samples}")

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*70}")
        print(f"SELF-IMPROVEMENT ROUND {round_idx}/{args.rounds}")
        print(f"{'='*70}")

        # --- Step 1: Generate traces ---
        if not args.skip_generate:
            gen_cmd = [
                "python3",
                "scripts/generate.py",
                "--round", str(round_idx),
                "--samples", str(args.samples),
                "--max-tokens", str(args.max_tokens),
                "--temp", str(args.temp),
            ]
            if args.domain:
                gen_cmd += ["--domain", args.domain]
            run_cmd(gen_cmd, f"Round {round_idx}: Generate traces")

        # Count generated rows
        gen_dir = DATA / "generated"
        gen_files = list(gen_dir.glob(f"round{round_idx}*.jsonl")) if gen_dir.exists() else []
        total_gen = sum(count_rows(p) for p in gen_files)
        print(f"[self-train] Round {round_idx}: {total_gen} generated traces")

        # --- Step 2: Build SFT dataset ---
        if not args.skip_build:
            run_cmd(
                ["python3", "scripts/build_sft.py", "--round", str(round_idx)],
                f"Round {round_idx}: Build SFT dataset",
            )

        sft_dir = DATA / "sft"
        train_count = count_rows(sft_dir / "train.jsonl")
        valid_count = count_rows(sft_dir / "valid.jsonl")
        test_count = count_rows(sft_dir / "test.jsonl")
        print(
            f"[self-train] Round {round_idx}: {train_count} train / "
            f"{valid_count} valid / {test_count} test"
        )

        if train_count == 0:
            print("[ERROR] No training data. Stopping.")
            break

        # --- Step 3: Train ---
        if not args.skip_train:
            train_cmd = [
                "python3",
                "scripts/train.sh",
                "--iters", str(args.iters),
                "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum),
                "--lr", str(args.lr),
                "--max-seq-length", str(args.seq_len),
                "--layers", str(args.layers),
            ]
            run_cmd(train_cmd, f"Round {round_idx}: Train model (iters={args.iters})")

            # Update current model path to latest adapter
            current_model = str(ROOT / "artifacts" / "adapter")
            print(f"[self-train] New model: {current_model}")

        # --- Step 4: Evaluate ---
        eval_cmd = [
            "python3",
            "scripts/evaluate.py",
            "--model",
            str(current_model if not args.skip_train else base_model()),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.domain:
            eval_cmd += ["--domain", args.domain]
        try:
            run_cmd(eval_cmd, f"Round {round_idx}: Evaluate model")
        except SystemExit:
            print("[WARN] Evaluation failed, continuing to next round")

        print(f"\n[self-train] Round {round_idx} complete\n")

    print(f"\n{'='*70}")
    print("SELF-IMPROVEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Final model: {current_model}")
    print(f"To use: scripts/agent.py --model {current_model} --prompt '...'")


if __name__ == "__main__":
    main()