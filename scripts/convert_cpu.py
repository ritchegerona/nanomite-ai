#!/usr/bin/env python3
"""Quantize on the CPU backend.

mlx quantizes on the GPU by default, which requires MTLCompilerService. In
headless / restricted sessions that service can be unreachable; the CPU path
avoids Metal entirely (slower, but works).
"""
from __future__ import annotations

import argparse

import mlx.core as mx
from mlx_lm.convert import convert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    mx.set_default_device(mx.cpu)
    print(f"[convert] CPU backend, {args.q_bits}-bit group {args.q_group_size}")
    convert(
        hf_path=args.model,
        mlx_path=args.mlx_path,
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
    )
    print(f"[convert] done -> {args.mlx_path}")


if __name__ == "__main__":
    main()
