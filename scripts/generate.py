#!/usr/bin/env python3
"""STaR step 1: sample candidate solutions from the base model.

The model is loaded once; every seed in data/seeds/*.jsonl is sampled
--samples times. Output is raw (unverified) data for build_sft.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA,
    apply_chat,
    base_model,
    draft_model,
    read_jsonl,
    strip_stop,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--draft", action="store_true", help="use the 350M draft model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domain", default=None, help="only generate for this domain")
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=args.temp, top_p=0.95)
    model, tok = load(base_model())
    draft = None
    if args.draft:
        path = draft_model()
        if path:
            draft, _ = load(path)
            print(f"[gen] speculative draft: {path}")

    seeds: list[dict] = []
    for path in sorted((DATA / "seeds").glob("*.jsonl")):
        seeds += [s for s in read_jsonl(path) if args.domain in (None, s["domain"])]
    if args.limit:
        seeds = seeds[: args.limit]
    if not seeds:
        raise SystemExit("no seeds found in data/seeds/*.jsonl")

    rows: list[dict] = []
    t0 = time.time()
    gen_tokens = 0
    for i, seed in enumerate(seeds, 1):
        prompt = apply_chat(tok, seed["prompt"])
        for k in range(args.samples):
            kwargs = dict(max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            if draft is not None:
                kwargs["draft_model"] = draft
            t = time.time()
            try:
                text = generate(model, tok, prompt, **kwargs)
            except TypeError:
                kwargs.pop("draft_model", None)
                text = generate(model, tok, prompt, **kwargs)
            text = strip_stop(text)
            if text:
                gen_tokens += len(tok.encode(text))
            rows.append(
                {
                    "id": seed["id"],
                    "domain": seed["domain"],
                    "prompt": seed["prompt"],
                    "meta": seed.get("meta", {}),
                    "sample": k,
                    "output": text,
                    "round": args.round,
                    "secs": round(time.time() - t, 3),
                }
            )
        if i % 5 == 0:
            elapsed = max(1e-9, time.time() - t0)
            print(
                f"[gen] {i}/{len(seeds)} seeds | {len(rows)} samples | "
                f"{gen_tokens / elapsed:.1f} tok/s",
                flush=True,
            )

    suffix = f"_{args.domain}" if args.domain else ""
    path = DATA / "generated" / f"round{args.round}{suffix}.jsonl"
    write_jsonl(path, rows)
    print(f"[gen] wrote {len(rows)} samples -> {path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
