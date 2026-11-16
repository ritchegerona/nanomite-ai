"""Verify chat answers using must-mention / must-not-mention rubrics.

The meta dict can contain:
  - must_mention: list of strings that must appear in the answer
  - must_not_mention: list of strings that must NOT appear
  - min_length: minimum answer length
  - max_length: maximum answer length
"""
import re


def verify(output: str, meta: dict) -> tuple[bool, str]:
    text = output.strip()
    if not text:
        return False, "empty"

    # Check must_mention
    for phrase in meta.get("must_mention", []):
        if phrase.lower() not in text.lower():
            return False, f"missing:{phrase}"

    # Check must_not_mention
    for phrase in meta.get("must_not_mention", []):
        if phrase.lower() in text.lower():
            return False, f"forbidden:{phrase}"

    # Check min_length
    min_len = meta.get("min_length", 0)
    if len(text) < min_len:
        return False, f"too_short:{len(text)}<{min_len}"

    # Check max_length
    max_len = meta.get("max_length", 10000)
    if len(text) > max_len:
        return False, f"too_long:{len(text)}>{max_len}"

    # Check must_match_regex
    for pattern in meta.get("must_match_regex", []):
        if not re.search(pattern, text, re.I):
            return False, f"no_match:{pattern}"

    return True, "ok"